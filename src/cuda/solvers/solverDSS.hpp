#pragma once
#include <cuda_runtime_api.h>
#include <cuDSS.h>
#include <Eigen/Sparse>

class SolverDSS {
  public:
    typedef double scal_t;
    cudaStream_t cuda_stream;
    cudssHandle_t handle;
    cudssConfig_t solver_config;
    cudssData_t solver_data;
    int m;
    int nnz;
    cudssMatrix_t A;
    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;

    cudssMatrix_t x, b;

    // Device pointers
    scal_t* value_ptr;
    int* row_ptr;
    int* col_ind;
    scal_t* b_values;
    scal_t* x_values;

    SolverDSS(const Eigen::SparseMatrix<scal_t, Eigen::RowMajor>& A_eig) {
        // Initialize cuDSS
        cudssCreate(&handle);
        cudaStreamCreate(&cuda_stream);
        cudssSetStream(handle, cuda_stream);

        m = A_eig.rows();
        nnz = A_eig.nonZeros();

        // Initialize the memory on the gpu
        cudaMalloc((void**)&value_ptr, sizeof(scal_t) * nnz);
        cudaMalloc((void**)&row_ptr, (m + 1) * sizeof(int));
        cudaMalloc((void**)&col_ind, sizeof(int) * nnz);
        cudaMalloc((void**)&b_values, sizeof(scal_t) * m);
        cudaMalloc((void**)&x_values, sizeof(scal_t) * m);
        // Copy from host to device
        cudaMemcpy(value_ptr, A_eig.valuePtr(), nnz * sizeof(scal_t), cudaMemcpyHostToDevice);
        cudaMemcpy(col_ind, A_eig.innerIndexPtr(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(row_ptr, A_eig.outerIndexPtr(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Setup cuDSS
        cudssConfigCreate(&solver_config);
        cudssDataCreate(handle, &solver_data);
        cudssMatrixCreateCsr(&A, m, m, nnz, row_ptr, nullptr, col_ind, value_ptr, CUDA_R_32I,
                             CUDA_R_64F, mtype, mview, base);
    }
    ~SolverDSS() {
        cudssMatrixDestroy(A);
        cudssMatrixDestroy(b);
        cudssMatrixDestroy(x);
        cudssDataDestroy(handle, solver_data);
        cudssConfigDestroy(solver_config);
        cudaFree(value_ptr);
        cudaFree(row_ptr);
        cudaFree(col_ind);
        cudaFree(b_values);
        cudaFree(x_values);
        cudssDestroy(handle);
        cudaStreamSynchronize(cuda_stream);
    }
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) {
        cudaMemcpy(b_values, rhs.data(), sizeof(scal_t) * m, cudaMemcpyHostToDevice);

        cudssMatrixCreateDn(&b, m, 1, m, b_values, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
        cudssMatrixCreateDn(&x, m, 1, m, x_values, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);

        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config, solver_data, A, x, b);
        cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solver_config, solver_data, A, x, b);
        cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data, A, x, b);

        cudaStreamSynchronize(cuda_stream);
        Eigen::VectorXd solution(m);
        cudaMemcpy(solution.data(), x_values, sizeof(scal_t) * m, cudaMemcpyDeviceToHost);
        return solution;
    }
};