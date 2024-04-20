#pragma once
#include <iostream>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <Eigen/Sparse>

class SolverQR {
  public:
    typedef double scal_t;
    cusolverSpHandle_t cusolver_handle;
    int m;
    int nnz;
    cusparseMatDescr_t descrA;

    // Device pointers
    scal_t* value_ptr;
    int* row_ptr;
    int* col_ind;
    scal_t* b;
    scal_t* x;

    SolverQR(const Eigen::SparseMatrix<scal_t, Eigen::RowMajor>& A) {
        cusolverSpCreate(&cusolver_handle);
        m = A.rows();
        nnz = A.nonZeros();
        // Allocate space on device
        cudaMalloc((void**)&value_ptr, sizeof(scal_t) * nnz);
        cudaMalloc((void**)&row_ptr, (m + 1) * sizeof(int));
        cudaMalloc((void**)&col_ind, sizeof(int) * nnz);
        cudaMalloc((void**)&b, sizeof(scal_t) * m);
        cudaMalloc((void**)&x, sizeof(scal_t) * m);
        // Copy from host to device
        cudaMemcpy(value_ptr, A.valuePtr(), nnz * sizeof(scal_t), cudaMemcpyHostToDevice);
        cudaMemcpy(col_ind, A.innerIndexPtr(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(row_ptr, A.outerIndexPtr(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Setup cusparse data
        cusparseCreateMatDescr(&descrA);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    }
    ~SolverQR() {
        cudaFree(value_ptr);
        cudaFree(row_ptr);
        cudaFree(col_ind);
        cudaFree(b);
        cudaFree(x);
        cusparseDestroyMatDescr(descrA);
        cusolverSpDestroy(cusolver_handle);
    }
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) {
        cudaMemcpy(b, rhs.data(), sizeof(scal_t) * m, cudaMemcpyHostToDevice);

        int singularity = -1;
        cusolverSpDcsrlsvqr(cusolver_handle, m, nnz, descrA, value_ptr, row_ptr, col_ind, b, 1e-12, 1,
                            x, &singularity);
        if (singularity != -1) {
            std::cout << "Singularity is " << singularity << std::endl;
        }

        Eigen::VectorXd solution(m);
        cudaMemcpy(solution.data(), x, sizeof(scal_t) * m, cudaMemcpyDeviceToHost);
        return solution;
    }
};