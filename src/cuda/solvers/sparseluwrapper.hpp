#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
class SparseLUWrapper {
    public:
    typedef double scal_t;
    Eigen::SparseLU<Eigen::SparseMatrix<scal_t>> solver;
    SparseLUWrapper(const Eigen::SparseMatrix<scal_t> &A) {
        solver.analyzePattern(A);
        solver.factorize(A);
    }
    Eigen::VectorXd solve(Eigen::VectorXd b) {
        return solver.solve(b);
    }
};