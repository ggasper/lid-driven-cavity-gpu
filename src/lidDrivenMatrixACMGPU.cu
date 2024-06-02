#include <iostream>
#include <medusa/Medusa.hpp>
#include <Eigen/SparseLU>
#include <Eigen/Sparse>
#include <chrono>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <medusa/Utils.hpp>

#define EPS 1e-10

using namespace mm;
struct axpy_functor {
    const double a;
    axpy_functor(double _a) : a(_a) {}
    __host__ __device__ double operator()(const double& x, const double& y) const {
        return a * x + y;
    }
};
struct abs_functor {
    __host__ __device__ double operator()(const double& x) { return fabs(x); }
};
struct u_tuple_functor {
    double* u;
    int N;
    int dim;
    u_tuple_functor(double* _u, int _N, int _dim) : u(_u), N(_N), dim(_dim){};
    __host__ __device__ thrust::tuple<int, double> operator()(const int& i) {
        return thrust::make_tuple(i, u[i + (dim - 1) * N]);
    }
};
struct tuple_max_functor {
    __host__ __device__ thrust::tuple<int, double> operator()(
        const thrust::tuple<int, double>& lhs, const thrust::tuple<int, double>& rhs) {
        double max_lhs = thrust::get<1>(lhs);
        double max_rhs = thrust::get<1>(rhs);
        if (max_lhs < max_rhs) {
            return rhs;
        } else {
            return lhs;
        }
    }
};
class VectorGPU {
  public:
    typedef double scal_t;
    int m;
    thrust::device_vector<scal_t> value_ptr;
    VectorGPU(Eigen::VectorXd vec) {
        m = vec.size();
        value_ptr.assign(vec.begin(), vec.end());
    }
    VectorGPU(int size) {
        m = size;
        value_ptr.resize(m);
        thrust::fill(value_ptr.begin(), value_ptr.end(), 0);
    }
    void to_eigen(Eigen::VectorXd& vec) {
        thrust::copy(value_ptr.begin(), value_ptr.end(), vec.begin());
        // for (int i = 0; i < m; ++i) {
        //     vec[i] = value_ptr[i];
        // }
    }
};
class MatrixGPU {
  public:
    typedef double scal_t;
    int nnz;
    int m;
    thrust::device_vector<int> row_ptr;
    thrust::device_vector<int> col_ind;
    thrust::device_vector<scal_t> value_ptr;
    MatrixGPU(const Eigen::SparseMatrix<scal_t, Eigen::RowMajor>& mat) {
        nnz = mat.nonZeros();
        m = mat.rows();
        row_ptr.assign(mat.outerIndexPtr(), mat.outerIndexPtr() + (m + 1));
        col_ind.assign(mat.innerIndexPtr(), mat.innerIndexPtr() + nnz);
        value_ptr.assign(mat.valuePtr(), mat.valuePtr() + nnz);
    }
};
class Multiply {
    MatrixGPU* mat;
    VectorGPU* vec;
    VectorGPU* out;
    cusparseSpMatDescr_t descrMat;
    cusparseDnVecDescr_t descrVec;
    cusparseDnVecDescr_t descrOut;
    cusparseHandle_t handle;
    size_t buffer_size;
    size_t* buffer;
    double alpha;
    double beta;

  public:
    Multiply(MatrixGPU* _mat, VectorGPU* _vec, VectorGPU* _out, double _alpha, double _beta,
             cusparseHandle_t _handle)
        : mat(_mat), vec(_vec), out(_out), alpha(_alpha), beta(_beta), handle(_handle) {
            cusparseCreateCsr(&descrMat, mat->m, mat->m, mat->nnz,
                              thrust::raw_pointer_cast(mat->row_ptr.data()),
                              thrust::raw_pointer_cast(mat->col_ind.data()),
                              thrust::raw_pointer_cast(mat->value_ptr.data()), CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
            cusparseCreateDnVec(&descrVec, mat->m, thrust::raw_pointer_cast(vec->value_ptr.data()),
                                CUDA_R_64F);
            cusparseCreateDnVec(&descrOut, mat->m, thrust::raw_pointer_cast(out->value_ptr.data()),
                                CUDA_R_64F);
            cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrMat,
                                    descrVec, &beta, descrOut, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
            cudaMalloc((void**)&buffer, buffer_size);
            cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrMat,
                                    descrVec, &beta, descrOut, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, buffer);

        }
    void operator()(cudaStream_t stream) {
        cusparseSetStream(handle, stream);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrMat, descrVec, &beta,
                     descrOut, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    }
    ~Multiply() {
        cudaFree(buffer);
        cusparseDestroySpMat(descrMat);
        cusparseDestroyDnVec(descrVec);
        cusparseDestroyDnVec(descrOut);
    }
};
template <int dim>
class LidDrivenMatrixACM {
  public:
    typedef double scal_t;
    typedef Vec<scal_t, dim> vec_t;

    explicit LidDrivenMatrixACM(XML& param_file) {
        Stopwatch s;
        s.start("time");
        s.start("setup");
        HDF::Mode mode = param_file.get<bool>("output.clear_hdf5") ? HDF::DESTROY : HDF::APPEND;
        HDF hdf_out(param_file.get<std::string>("output.hdf5_name"), mode);

        auto Re = param_file.get<scal_t>("case.Re");
        auto h = param_file.get<scal_t>("num.h");
        auto cfl = param_file.get<scal_t>("num.cfl");
        auto seed = param_file.get<int>("case.seed");

        scal_t dt1 = cfl * h / dim;
        scal_t dt2 = cfl / 10 * h * h * Re;
        std::cout << "dt1:" << dt1 << ", dt2:" << dt2 << std::endl;
        scal_t dt = std::min(dt1, dt2);

        const auto start{std::chrono::steady_clock::now()};

        BoxShape<vec_t> box(vec_t::Zero(), vec_t::Constant(1));
        DomainDiscretization<vec_t> domain = box.discretizeBoundaryWithStep(h);
        GeneralFill<vec_t> fill;
        fill.seed(seed);
        fill(domain, h);

        auto borders = box.bbox();
        Range<int> corner = domain.positions().filter([&](const vec_t& p) {
            // remove nodes that are EPS close to more than 1 border
            return (((p - borders.first) < EPS).size() + ((borders.second - p) < EPS).size()) > 1;
        });
        domain.removeNodes(corner);

        hdf_out.writeDomain("domain", domain);
        hdf_out.writeXML("", param_file, true);
        hdf_out.close();

        int N = domain.size();
        Range<int> interior = domain.interior();
        Range<int> boundary = domain.boundary();
        Range<int> lid, wall;
        for (int i : boundary) {
            if (borders.second[dim - 1] - domain.pos(i, dim - 1) < EPS)
                lid.push_back(i);
            else
                wall.push_back(i);
        }

        scal_t midpoint = (borders.second(1) - borders.first(1)) / 2;
        Range<int> midplane = domain.positions().filter(
            [&](const vec_t& p) { return std::abs(p(1) - midpoint) < h; });

        std::cout << "h:" << h << ", N: " << N << ", dt:" << dt << std::endl;

        VectorField<scal_t, dim> u_initial(N);
        u_initial.setZero();
        for (int i : lid) {
            u_initial(i, 0) = 1;
        }
        ScalarField<scal_t> u_partial, u;
        u = u_initial.asLinear();
        u_partial = u;
        ScalarField<scal_t> p(N);
        p.setZero();

        auto k = param_file.get<int>("num.phs_order");
        Monomials<vec_t> mon(param_file.get<int>("num.mon_order"));
        int support_size =
            std::round(param_file.get<scal_t>("num.support_size_factor") * mon.size());
        mm::RBFFD<Polyharmonic<scal_t>, vec_t, ScaleToClosest> rbffd(k, mon);

        domain.findSupport(FindClosest(support_size).forNodes(interior));
        domain.findSupport(
            FindClosest(support_size).forNodes(boundary).searchAmong(interior).forceSelf(true));

        auto storage = domain.template computeShapes<sh::lap | sh::grad>(rbffd);

        Eigen::SparseMatrix<double, Eigen::RowMajor> grad_p(dim * N, N);
        grad_p.reserve(Eigen::VectorXi::Constant(dim * N, support_size));
        Eigen::SparseMatrix<double, Eigen::RowMajor> lap(dim * N, dim * N);
        lap.reserve(Eigen::VectorXi::Constant(dim * N, support_size));
        Eigen::SparseMatrix<double, Eigen::RowMajor> div(N, dim * N);
        div.reserve(Eigen::VectorXi::Constant(N, dim * support_size));
        Eigen::SparseMatrix<double, Eigen::RowMajor> neumann(N, N);
        neumann.reserve(Eigen::VectorXi::Constant(N, support_size));
        Eigen::SparseMatrix<double, Eigen::RowMajor> component_sum(N, dim * N);
        component_sum.reserve(Eigen::VectorXi::Constant(N, dim));
        Range<Eigen::SparseMatrix<double, Eigen::RowMajor>> derivative, stack;
        for (int var = 0; var < dim; ++var) {
            // derivative.emplace_back(dim * N, dim * N);
            // stack.emplace_back(dim * N, dim * N);
            stack.emplace_back(dim * N, dim * N);
            stack.back().reserve(Eigen::VectorXi::Constant(dim * N, 1));
        }
         // workaround
        Eigen::SparseMatrix<double, Eigen::RowMajor> d0(dim * N, dim * N), d1(dim * N, dim * N);
        d0.reserve(Eigen::VectorXi::Constant(dim * N, support_size));
        d1.reserve(Eigen::VectorXi::Constant(dim * N, support_size));
        for (int i = 0; i < domain.size(); ++i) {
            if (domain.type(i) > 0) {
                neumann.insert(i, i) = 1;
                for (int var = 0; var < dim; ++var) {
                    component_sum.insert(i, i + var * domain.size()) = 1;
                    for (int s = 0; s < storage.supportSize(i); ++s) {
                        grad_p.insert(i + var * domain.size(), storage.support(i, s)) =
                            storage.d1(var, i, s);
                        lap.insert(i + var * domain.size(),
                                   storage.support(i, s) + var * domain.size()) =
                            storage.laplace(i, s);
                        div.insert(i, storage.support(i, s) + var * domain.size()) =
                            storage.d1(var, i, s);
                        // workaround
                        d0.insert(i + var * domain.size(), storage.support(i, s) + var * domain.size()) = storage.d1(0, i, s);
                        d1.insert(i + var * domain.size(), storage.support(i, s) + var * domain.size()) = storage.d1(1, i, s);
                        for (int der_var = 0; der_var < dim; ++der_var) {
                            // derivative[der_var].insert(
                                // i + var * domain.size(),
                                // storage.support(i, s) + var * domain.size()) =
                                // storage.d1(der_var, i, s);
                            // This could be replaced with some sort of stacking operation if
                            // available
                            if (s == 0) {
                                stack[der_var].insert(i + var * domain.size(),
                                                      i + der_var * domain.size()) = 1;
                            }
                        }
                    }
                }
            } else {
                // Works only for neumman 0
                vec_t normal = domain.normal(i);
                scal_t central_coeff = 0;
                for (int s = 0; s < storage.supportSize(i); ++s) {
                    scal_t neumann_coeff = 0;
                    for (int var = 0; var < dim; ++var) {
                        neumann_coeff += normal(var) * storage.d1(var, i, s);
                    }
                    if (s == 0)
                        central_coeff = neumann_coeff;
                    else {
                        neumann.insert(i, storage.support(i, s)) = -neumann_coeff / central_coeff;
                    }
                }
            }
        }
        grad_p.makeCompressed();
        lap.makeCompressed();
        div.makeCompressed();
        neumann.makeCompressed();
        component_sum.makeCompressed();
        derivative.push_back(d0); // workaround
        derivative.push_back(d1); // workaround
        for (int var = 0; var < dim; ++var) {
            derivative[var].makeCompressed();
            stack[var].makeCompressed();
        }

        cusparseHandle_t cusparse_handle;
        cublasHandle_t cublas_handle;

        cusparseCreate(&cusparse_handle);
        cublasCreate(&cublas_handle);

        std::vector<MatrixGPU> d_derivative;
        std::vector<MatrixGPU> d_stack;
        MatrixGPU d_lap{lap};
        d_derivative.reserve(derivative.size());
        for (int i = 0; i < derivative.size(); ++i) {
            d_derivative.emplace_back(derivative[i]);
        }
        d_stack.reserve(stack.size());
        for (int i = 0; i < stack.size(); ++i) {
            d_stack.emplace_back(stack[i]);
        }
        s.start("gpu setup");
        VectorGPU d_u{u};
        VectorGPU d_u_partial{u_partial};
        MatrixGPU d_grad_p{grad_p};
        VectorGPU d_p{p};
        VectorGPU d_p_temp{p};
        VectorGPU d_u2{u};
        VectorGPU d_component_sum_u{u};
        MatrixGPU d_component_sum{component_sum};
        MatrixGPU d_div{div};
        VectorGPU d_divergence{p};
        MatrixGPU d_neumann{neumann};
        std::vector<cudaStream_t> streams_lhs(dim);
        std::vector<cudaStream_t> streams_rhs(dim);
        for (int i = 0; i < dim; ++i) {
            cudaStreamCreate(&streams_lhs[i]);
            cudaStreamCreate(&streams_rhs[i]);
        }
        std::vector<VectorGPU> u_partial_lhs;
        std::vector<VectorGPU> u_partial_rhs;
        for (int i = 0; i < dim; ++i) {
            u_partial_lhs.emplace_back(u.size());
            u_partial_rhs.emplace_back(u.size());
        }
        thrust::device_vector<scal_t> d_midplane(midplane.begin(), midplane.end());

        Multiply lap_u(&d_lap, &d_u, &d_u_partial, dt / Re, 1, cusparse_handle);
        std::vector<Multiply> derivative_u;
        std::vector<Multiply> stack_u;
        derivative_u.reserve(dim);  // workaround for lack of copy/move constructors (TODO)
        stack_u.reserve(dim);
        Multiply grad_p_p(&d_grad_p, &d_p, &d_u, -dt, 1, cusparse_handle);
        Multiply component_sum_u2(&d_component_sum, &d_u2, &d_component_sum_u, 1, 0,
                                  cusparse_handle);
        Multiply div_u(&d_div, &d_u, &d_divergence, 1, 0, cusparse_handle);
        Multiply neumann_p(&d_neumann, &d_p_temp, &d_p, 1, 0, cusparse_handle);
        for (int var = 0; var < dim; ++var) {
            derivative_u.emplace_back(&d_derivative[var], &d_u, &u_partial_lhs[var], dt, 0,
                                      cusparse_handle);
            stack_u.emplace_back(&d_stack[var], &d_u, &u_partial_rhs[var], 1, 0, cusparse_handle);
        }
        scal_t t = 0;
        auto end_time = param_file.get<scal_t>("case.end_time");
        auto printout_interval = param_file.get<int>("output.printout_interval");
        auto max_p_iter = param_file.get<int>("acm.max_p_iter");
        auto div_limit = param_file.get<scal_t>("num.max_divergence");
        auto compress = param_file.get<scal_t>("acm.compressibility");
        auto v_ref = param_file.get<scal_t>("acm.reference_velocity");
        int num_print = end_time / (dt * printout_interval);
        Eigen::VectorXd max_u_y(num_print);
        int iteration = 0;
        s.stop("gpu setup");
        s.stop("setup");
        while (t < end_time) {
            s.start("u = u + dt / Re * lap * u");
            thrust::copy(d_u.value_ptr.begin(), d_u.value_ptr.end(), d_u_partial.value_ptr.begin());
            lap_u(streams_lhs[0]);
            cudaDeviceSynchronize();
            s.stop("u = u + dt / Re * lap * u");

            s.start("derivative stack correction");
            for (int var = 0; var < dim; ++var) {  // Complication to remain general in dimensions.
                                                   // Can be written explicitly.
                derivative_u[var](streams_lhs[var]);
                cudaEvent_t ev;
                cudaEventCreate(&ev);
                stack_u[var](streams_rhs[var]);
                cudaEventRecord(ev, streams_rhs[var]);
                cudaStreamWaitEvent(streams_lhs[var], ev);

                auto op = thrust::multiplies<double>();
                thrust::transform(
                    thrust::cuda::par.on(streams_lhs[var]), u_partial_rhs[var].value_ptr.begin(),
                    u_partial_rhs[var].value_ptr.end(), u_partial_lhs[var].value_ptr.begin(),
                    u_partial_lhs[var].value_ptr.begin(), op);
            }
            cudaDeviceSynchronize();
            for (int var = 0; var < dim; ++var) {  // Complication to remain general in dimensions.
                thrust::transform(u_partial_lhs[var].value_ptr.begin(),
                                  u_partial_lhs[var].value_ptr.end(), d_u_partial.value_ptr.begin(),
                                  d_u_partial.value_ptr.begin(), axpy_functor(-1));
            }
            s.stop("derivative stack correction");
            scal_t max_norm, max_div;
            int p_iter;
            cudaDeviceSynchronize();
            for (p_iter = 0; p_iter < max_p_iter; ++p_iter) {
                s.start("PV correction");
                cudaStream_t default_stream = 0;
                thrust::copy(d_u_partial.value_ptr.begin(), d_u_partial.value_ptr.end(),
                             d_u.value_ptr.begin());

                grad_p_p(default_stream);
                s.stop("PV correction");
                s.start("max_norm");
                thrust::copy(d_u.value_ptr.begin(), d_u.value_ptr.end(), d_u2.value_ptr.begin());
                auto op = thrust::multiplies<double>();
                thrust::transform(d_u.value_ptr.begin(), d_u.value_ptr.end(),
                                  d_u2.value_ptr.begin(), d_u2.value_ptr.begin(), op);
                component_sum_u2(default_stream);
                max_norm = std::sqrt(*thrust::max_element(d_component_sum_u.value_ptr.begin(),
                                                          d_component_sum_u.value_ptr.end()));
                s.stop("max_norm");
                s.start("p");
                scal_t C = compress * std::max(max_norm, v_ref);
                div_u(default_stream);
                thrust::transform(d_divergence.value_ptr.begin(), d_divergence.value_ptr.end(),
                                  d_p.value_ptr.begin(), d_p_temp.value_ptr.begin(),
                                  axpy_functor(-C * C * dt));
                neumann_p(default_stream);
                s.stop("p");
                s.start("max_div");
                thrust::transform(d_divergence.value_ptr.begin(), d_divergence.value_ptr.end(),
                                  d_divergence.value_ptr.begin(), abs_functor());
                max_div = *thrust::max_element(d_divergence.value_ptr.begin(),
                                               d_divergence.value_ptr.end());
                s.stop("max_div");
                if (max_div < div_limit) break;
            }
            t += dt;
            if (++iteration % printout_interval == 0) {
                s.start("print");
                scal_t max = 0, pos;
                thrust::tuple<int, double> result = thrust::transform_reduce(
                    d_midplane.begin(), d_midplane.end(),
                    u_tuple_functor(thrust::raw_pointer_cast(d_u.value_ptr.data()), N, dim),
                    thrust::make_tuple<int, double>(0, 0), tuple_max_functor());
                cudaDeviceSynchronize();
                max = thrust::get<1>(result);
                pos = domain.pos(thrust::get<0>(result), 0);
                int print_iter = (iteration - 1) / printout_interval;
                max_u_y[print_iter] = max;
                std::cout << iteration << " - t:" << t << " max u_y:" << max << " @ x:" << pos
                          << "  (max div:" << max_div << " @ " << p_iter << ")" << std::endl;
                s.stop("print");
            }
        }

        // const auto end{std::chrono::steady_clock::now()};
        // const std::chrono::duration<double> elapsed_time{end - start};

        cublasDestroy(cublas_handle);
        cusparseDestroy(cusparse_handle);
        s.stop("time");
        d_u.to_eigen(u);
        d_p.to_eigen(p);

        hdf_out.reopen();
        hdf_out.openGroup("/");
        hdf_out.writeEigen("velocity", VectorField<scal_t, dim>::fromLinear(u));
        hdf_out.writeEigen("pressure", p);
        // hdf_out.writeDoubleAttribute("time", elapsed_time.count());
        hdf_out.writeEigen("max_u_y", max_u_y);
        std::vector<std::string> labels{"time",
                                        "setup",
                                        "gpu setup",
                                        "u = u + dt / Re * lap * u",
                                        "derivative stack correction",
                                        "PV correction",
                                        "max_norm",
                                        "p",
                                        "max_div",
                                        "print"};
        for (std::string label : labels) {
            hdf_out.writeDoubleAttribute(label, s.cumulativeTime(label));
        }
        hdf_out.close();
    }
};

// The path to .xml parameter file required as a command line parameter
int main(int arg_num, char* arg[]) {
    if (arg_num < 2) {
        throw std::invalid_argument(
            "Missing command line argument. Provide path to .xml configuration.");
    } else if (arg_num > 2) {
        throw std::invalid_argument(
            "Too many command line arguments. Only path to .xml configuration is required.");
    }
    std::string parameter_file(arg[1]);

    XML params(parameter_file);
    if (params.get<bool>("output.use_xml_name")) {
        std::string output_name =
            mm::join({mm::split(mm::split(parameter_file, "/").back(), ".").front(), "h5"}, ".");
        params.set("output.hdf5_name", output_name, true);
    }
    omp_set_num_threads(params.get<int>("sys.num_threads"));

    switch (params.get<int>("case.dim")) {
        case 2: {
            LidDrivenMatrixACM<2> solution(params);
            break;
        }
        case 3: {
            LidDrivenMatrixACM<3> solution(params);
            break;
        }
        default:
            std::cout << params.get<int>("case.dim") << "is not a supported dimension" << std::endl;
    }
}

// int main(int arg_num, char* arg[]) {}