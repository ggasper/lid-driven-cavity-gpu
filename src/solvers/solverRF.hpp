#pragma once
#include <iostream>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <Eigen/Sparse>
#include <cusolverSp.h>
#include <cusolverRf.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
class SolverRF {
  public:
    typedef double scal_t;
    cusolverSpHandle_t cusolverSp_handle;
    cusolverRfHandle_t cusolverRf_handle;
    cusparseHandle_t cusparse_handle;
    cudaStream_t stream;
    cusparseMatDescr_t descrA;
    csrluInfoHost_t info;

    // Matrix A
    int m;    // rows/columns of A
    int nnz;  // nonzeros of A
    Eigen::SparseMatrix<scal_t, Eigen::RowMajor> *A;
    std::vector<int> h_Qreorder;

    size_t size_perm = 0;
    size_t size_internal = 0;
    size_t size_lu = 0;
    std::vector<size_t> buffer_cpu;

    // Plu * B * Qlu^T = L * U
    std::vector<int> h_Plu;
    std::vector<int> h_Qlu;

    int nnzL = 0;
    std::vector<scal_t> h_L_value_ptr;
    std::vector<int> h_L_row_ptr;
    std::vector<int> h_L_col_ind;

    int nnzU = 0;
    std::vector<scal_t> h_U_value_ptr;
    std::vector<int> h_U_row_ptr;
    std::vector<int> h_U_col_ind;

    std::vector<int> h_P;  // P = Plu * Qreorder
    std::vector<int> h_Q;  // Q = Qlu * Qreorder

    scal_t* d_value_ptr;
    int* d_row_ptr;
    int* d_col_ind;
    scal_t* d_b;
    scal_t* d_x;

    int* d_P;
    int* d_Q;
    scal_t* d_T;  // working space in cusolverRfSolve

    bool initialized = false;
    SolverRF(Eigen::SparseMatrix<scal_t, Eigen::RowMajor>& mat) {
        // Matrix B = Q * A * Q^T
        std::vector<scal_t> h_B_value_ptr;
        std::vector<int> h_B_row_ptr;
        std::vector<int> h_B_col_ind;
        std::vector<int> h_map_B_from_A;
        initialized = false;
        A = &mat;
        cusparseCreate(&cusparse_handle);
        cusolverSpCreate(&cusolverSp_handle);
        cudaStreamCreate(&stream);

        cusolverRfCreate(&cusolverRf_handle);
        cusolverRfSetNumericProperties(cusolverRf_handle, 0, 0);
        cusolverRfSetAlgs(cusolverRf_handle, CUSOLVERRF_FACTORIZATION_ALG0,
                          CUSOLVERRF_TRIANGULAR_SOLVE_ALG1);
        cusolverRfSetMatrixFormat(cusolverRf_handle, CUSOLVERRF_MATRIX_FORMAT_CSR,
                                  CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L);
        cusolverRfSetResetValuesFastMode(cusolverRf_handle, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);

        cusolverSpSetStream(cusolverSp_handle, stream);
        cusparseSetStream(cusparse_handle, stream);

        cusparseCreateMatDescr(&descrA);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

        m = mat.rows();
        nnz = mat.nonZeros();


        h_Qreorder.resize(m);

        std::vector<scal_t> h_x(m);
        std::vector<scal_t> h_b(m);

        cudaMalloc((void**)&d_col_ind, sizeof(int) * nnz);
        cudaMalloc((void**)&d_row_ptr, sizeof(int) * (m + 1));
        cudaMalloc((void**)&d_value_ptr, sizeof(scal_t) * nnz);

        cudaMalloc((void**)&d_x, sizeof(scal_t) * m);
        // cudaMalloc((void**)&d_b, sizeof(scal_t) * m);

        cudaMalloc((void**)&d_P, sizeof(scal_t) * m);
        cudaMalloc((void**)&d_Q, sizeof(scal_t) * m);
        cudaMalloc((void**)&d_T, sizeof(scal_t) * m);

        auto status = cusolverSpXcsrsymamdHost(cusolverSp_handle, m, nnz, descrA, A->outerIndexPtr(),
                                A->innerIndexPtr(), h_Qreorder.data());
        if (status != CUSOLVER_STATUS_SUCCESS) {
            std::cout << "Failed to permute matrix A" << std::endl;
        }
        for (int j = 0; j < m + 1; ++j) {
            h_B_row_ptr.push_back(A->outerIndexPtr()[j]);
        }
        for (int j = 0; j < nnz; ++j) {
            h_B_col_ind.push_back(A->innerIndexPtr()[j]);
        }

        status = cusolverSpXcsrperm_bufferSizeHost(cusolverSp_handle, m, m, nnz, descrA, h_B_row_ptr.data(),
                                          h_B_col_ind.data(), h_Qreorder.data(), h_Qreorder.data(),
                                          &size_perm);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            std::cout << "Failed to allocate buffer" << std::endl;
        }
        buffer_cpu.resize(size_perm);
        for (int j = 0; j < nnz; ++j) {
            h_map_B_from_A.push_back(j);
        }

        status = cusolverSpXcsrpermHost(cusolverSp_handle, m, m, nnz, descrA, h_B_row_ptr.data(),
                               h_B_col_ind.data(), h_Qreorder.data(), h_Qreorder.data(),
                               h_map_B_from_A.data(), buffer_cpu.data());
        if (status != CUSOLVER_STATUS_SUCCESS) {
            std::cout << "Failed to permute matrix A" << std::endl;
        }
        for (int j = 0; j < nnz; ++j) {
            h_B_value_ptr.push_back(mat.valuePtr()[h_map_B_from_A[j]]);
        }
        status = cusolverSpCreateCsrluInfoHost(&info);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            std::cout << "Failed to create info" << std::endl;
        }
        status = cusolverSpXcsrluAnalysisHost(cusolverSp_handle, m, nnz, descrA, h_B_row_ptr.data(),
                                     h_B_col_ind.data(), info);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            std::cout << "Failed to analyze matrix B " << status << std::endl;
            if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
                std::cout << "internal error..." << std::endl;
                std::cout << h_B_col_ind.size() << " " << nnz << std::endl;
                std::cout << h_B_row_ptr.size() << " " << (m + 1) << std::endl;
            }
        }
        cusolverSpDcsrluBufferInfoHost(cusolverSp_handle, m, nnz, descrA, h_B_value_ptr.data(),
                                       h_B_row_ptr.data(), h_B_col_ind.data(), info, &size_internal,
                                       &size_lu);
        cusolverSpDcsrluFactorHost(cusolverSp_handle, m, nnz, descrA, h_B_value_ptr.data(),
                                   h_B_row_ptr.data(), h_B_col_ind.data(), info, 1.0, buffer_cpu.data());
        auto singularity = -1;
        cusolverSpDcsrluZeroPivotHost(cusolverSp_handle, info, 1e-10, &singularity);
        if (0 <= singularity) {
            std::cout << "Error: matrix is not invertible, has singularity " << singularity
                      << std::endl;
        }
    }
    Eigen::VectorXd solveInitial(Eigen::VectorXd b) {
        std::vector<scal_t> h_b_hat(m);
        std::vector<scal_t> h_x_hat(m);
        Eigen::VectorXd solution(m);
        // b_hat = Q*b
        for (int j = 0; j < m; ++j) {
            h_b_hat[j] = b[h_Qreorder[j]];
        }
        // B * x_hat = b_hat
        auto status = cusolverSpDcsrluSolveHost(cusolverSp_handle, m, h_b_hat.data(), h_x_hat.data(), info,
                                  buffer_cpu.data());
        if (status == CUSOLVER_STATUS_INVALID_VALUE) {
            std::cout << "Failed to solve system" << std::endl;
        }
        // x = Q^T * x_hat
        for (int j = 0; j < m; ++j) {
            solution[h_Qreorder[j]] = h_x_hat[j];
        }

        // Extract P, Q, L and U from P * B * Q^T = L * U
        cusolverSpXcsrluNnzHost(cusolverSp_handle, &nnzL, &nnzU, info);

        h_Plu.resize(m);
        h_Qlu.resize(m);

        h_L_value_ptr.resize(nnzL);
        h_L_col_ind.resize(nnzL);
        h_L_row_ptr.resize(m + 1);

        h_U_value_ptr.resize(nnzU);
        h_U_col_ind.resize(nnzU);
        h_U_row_ptr.resize(m + 1);

        cusolverSpDcsrluExtractHost(cusolverSp_handle, h_Plu.data(), h_Qlu.data(), descrA,
                                    h_L_value_ptr.data(), h_L_row_ptr.data(), h_L_col_ind.data(),
                                    descrA, h_U_value_ptr.data(), h_U_row_ptr.data(),
                                    h_U_col_ind.data(), info, buffer_cpu.data());
        h_P.resize(m);
        h_Q.resize(m);

        // P = Plu * Qreorder
        for (int j = 0; j < m; ++j) {
            h_P[j] = h_Qreorder[h_Plu[j]];
        }
        // Q = Qlu * Qreorder
        for (int j = 0; j < m; ++j) {
            h_Q[j] = h_Qreorder[h_Qlu[j]];
        }

        cusolverRfSetupHost(m, nnz, A->outerIndexPtr(), A->innerIndexPtr(), A->valuePtr(), nnzL,
                            h_L_row_ptr.data(), h_L_col_ind.data(), h_L_value_ptr.data(), nnzU,
                            h_U_row_ptr.data(), h_U_col_ind.data(), h_U_value_ptr.data(),
                            h_P.data(), h_Q.data(), cusolverRf_handle);
        cusolverRfAnalyze(cusolverRf_handle);

        cudaMemcpy(d_row_ptr, A->outerIndexPtr(), sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ind, A->innerIndexPtr(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_value_ptr, A->valuePtr(), sizeof(scal_t) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_P, h_P.data(), sizeof(int) * m, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q, h_Q.data(), sizeof(int) * m, cudaMemcpyHostToDevice);

        cusolverRfResetValues(m, nnz, d_row_ptr, d_col_ind, d_value_ptr, d_P, d_Q,
                              cusolverRf_handle);
        cusolverRfRefactor(cusolverRf_handle);
        initialized = true;
        return solution;
    }
    Eigen::VectorXd solve(Eigen::VectorXd b) {
        if (!initialized) {
            return solveInitial(b);
        }
        cudaMemcpy(d_x, b.data(), sizeof(scal_t) * m, cudaMemcpyHostToDevice);
        // cusolverRfResetValues(m, nnz, d_row_ptr, d_col_ind, d_value_ptr, d_P, d_Q,
                            //   cusolverRf_handle);
        // cusolverRfRefactor(cusolverRf_handle);
        cusolverRfSolve(cusolverRf_handle, d_P, d_Q, 1, d_T, m, d_x, m);
        Eigen::VectorXd solution(m);
        cudaMemcpy(solution.data(), d_x, sizeof(scal_t) * m, cudaMemcpyDeviceToHost);
        return solution;
    }
    ~SolverRF() {
        cusolverRfDestroy(cusolverRf_handle);
        cusolverSpDestroy(cusolverSp_handle);
        cusparseDestroy(cusparse_handle);
        cudaStreamDestroy(stream);
        cusparseDestroyMatDescr(descrA);
        cusolverSpDestroyCsrluInfoHost(info);
        cudaFree(d_row_ptr);
        cudaFree(d_col_ind);
        cudaFree(d_value_ptr);
        cudaFree(d_x);
        cudaFree(d_P);
        cudaFree(d_Q);
        cudaFree(d_T);
    }
};