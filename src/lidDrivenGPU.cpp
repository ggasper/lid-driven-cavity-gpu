#include <iostream>
#include <medusa/Medusa.hpp>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <time.h>

#define EPS 1e-10

using namespace mm;
template <int dim>
class LidDriven {
  public:
    typedef double scal_t;
    typedef Vec<scal_t, dim> vec_t;

    explicit LidDriven(XML& param_file) {
        HDF::Mode mode = param_file.get<bool>("output.clear_hdf5") ? HDF::DESTROY : HDF::APPEND;
        HDF hdf_out(param_file.get<std::string>("output.hdf5_name"), mode);

        auto Re = param_file.get<scal_t>("case.Re");
        auto h = param_file.get<scal_t>("num.h");
        auto cfl = param_file.get<scal_t>("num.cfl");
        auto seed = param_file.get<int>("case.seed");

        scal_t dt = cfl * h / dim;

        clock_t start_time = clock();
        clock_t preprocessing_start = clock();
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

        VectorField<scal_t, dim> u(N);
        u.setZero();
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
        auto op_e_v = storage.explicitVectorOperators();
        auto op_e_s = storage.explicitOperators();

        // pressure correction matrix
        // Eigen::SparseMatrix<scal_t, Eigen::ColMajor> M_p(N + 1, N + 1);
        Eigen::SparseMatrix<scal_t, Eigen::RowMajor> M_p(N + 1, N + 1);
        Eigen::VectorXd rhs_p(N + 1);
        rhs_p.setZero();
        auto p_op = storage.implicitOperators(M_p, rhs_p);

        for (int i : interior) {
            p_op.lap(i).eval(1);
        }
        for (int i : boundary) {
            p_op.neumann(i, domain.normal(i)).eval(1);
        }
        // regularization
        for (int i = 0; i < N; ++i) {
            M_p.coeffRef(N, i) = 1;
            M_p.coeffRef(i, N) = 1;
        }
        // set the sum of all values
        rhs_p[N] = 0.0;
        M_p.makeCompressed();
        double preprocessing_time =
            static_cast<double>(clock() - preprocessing_start) / CLOCKS_PER_SEC;
        clock_t lu_start = clock();
        std::cout << "Problem size: " << M_p.rows() << " " << M_p.cols()
                  << ", nonzeros: " << M_p.nonZeros() << std::endl;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_p;
        solver_p.compute(M_p);
        if (solver_p.info() != Eigen::Success) {
            std::cout << "LU factorization failed with error:" << solver_p.lastErrorMessage()
                      << std::endl;
        }
        double lu_time = static_cast<double>(clock() - lu_start) / CLOCKS_PER_SEC;

        // Prepare CUDA
        cusparseHandle_t cusparse_handle;
        cusolverSpHandle_t cusolver_handle;
        cusparseCreate(&cusparse_handle);
        cusolverSpCreate(&cusolver_handle);
        cusparseMatDescr_t descrM_u;
        cusparseCreateMatDescr(&descrM_u);
        cusparseSetMatIndexBase(descrM_u, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descrM_u, CUSPARSE_MATRIX_TYPE_GENERAL);

        scal_t t = 0;
        auto end_time = param_file.get<scal_t>("case.end_time");
        auto printout_interval = param_file.get<int>("output.printout_interval");
        auto max_p_iter = param_file.get<int>("num.max_p_iter");
        auto div_limit = param_file.get<scal_t>("num.max_divergence");
        int iteration = 0;
        double M_u_construct_time = 0;
        double M_u_solve_time = 0;
        double PV_construction_time = 0;
        double PV_solve_time = 0;
        while (t < end_time) {
            clock_t M_u_start = clock();
            Eigen::SparseMatrix<double, Eigen::RowMajor> M_u(dim * N, dim * N);
            Eigen::VectorXd rhs_u(dim * N);
            rhs_u.setZero();
            // Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>,
            //     Eigen::IncompleteLUT<double>> solver_u;
            // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_u;

            Range<int> per_row_u(2 * N, support_size);
            M_u.reserve(per_row_u);

            auto u_op = storage.implicitVectorOperators(M_u, rhs_u);

            for (int i : interior) {
                1 / dt* u_op.value(i) + u_op.grad(i, u[i]) + (-1 / Re) * u_op.lap(i) =
                    -1 * op_e_s.grad(p, i) + u[i] / dt;
            }

            for (int i : lid) {
                u_op.value(i) = vec_t{1, 0};
            }
            for (int i : wall) {
                u_op.value(i) = 0;
            }

            M_u.makeCompressed();
            M_u_construct_time += static_cast<double>(clock() - M_u_start) / CLOCKS_PER_SEC;

            clock_t M_u_solve_start = clock();
            //  Eigen::VectorXd solution = solver_u.solveWithGuess(rhs_u, u.asLinear());
            // Eigen::VectorXd solution = solver_u.solve(rhs_u);
            Eigen::VectorXd solution(M_u.rows());
            int singularity = -1;
            cusolverSpDcsrlsvluHost(cusolver_handle, M_u.rows(), M_u.nonZeros(), descrM_u,
                                    M_u.valuePtr(), M_u.outerIndexPtr(), M_u.innerIndexPtr(),
                                    rhs_u.data(), 1e-12, 1, solution.data(), &singularity);
            if (singularity == -1) {
                std::cout << "Singularity is -1" << std::endl;
            }
            u = VectorField<scal_t, dim>::fromLinear(solution);
            M_u_solve_time += static_cast<double>(clock() - M_u_solve_start) / CLOCKS_PER_SEC;
            // P-V correction iteration -- PVI
            scal_t max_div = 0;
            for (int p_iter = 0; p_iter < max_p_iter; ++p_iter) {
                clock_t PV_construction_start = clock();
                // clock_t start_correction = clock();
                for (int i : interior) rhs_p(i) = dt * op_e_v.div(u, i);
                for (int i : boundary) rhs_p(i) = dt * u[i].dot(domain.normal(i));
                PV_construction_time +=
                    static_cast<double>(clock() - PV_construction_start) / CLOCKS_PER_SEC;
                clock_t PV_solve_start = clock();
                ScalarFieldd p_c = solver_p.solve(rhs_p).head(N);
                PV_solve_time += static_cast<double>(clock() - PV_solve_start) / CLOCKS_PER_SEC;
                // ScalarFieldd p_c = solver_cuda.solve(rhs_p);//.head(N);
                p += p_c;

#pragma omp parallel for default(none) schedule(static) \
    shared(interior, u, op_e_s, op_e_v, p_c, dt) reduction(max : max_div)
                for (int _ = 0; _ < interior.size(); ++_) {
                    int i = interior[_];
                    u[i] -= dt * op_e_s.grad(p_c, i);
                    max_div = std::max(std::abs(op_e_v.div(u, i)), max_div);
                }

                if (max_div < div_limit) break;
                // std::cout << float(clock() - start_correction)/CLOCKS_PER_SEC << std::endl;
            }
            t += dt;
            if (++iteration % printout_interval == 0) {
                scal_t max = 0, pos;
                for (int i : midplane) {
                    if (u(i, 1) > max) {
                        max = u(i, 1);
                        pos = domain.pos(i, 0);
                    }
                }
                std::cout << iteration << " - t:" << t << " max u_y:" << max << " @ x:" << pos
                          << "  (max div:" << max_div << ")" << std::endl;
            }
        }

        hdf_out.reopen();
        hdf_out.openGroup("/");
        hdf_out.writeEigen("velocity", u);
        hdf_out.writeEigen("pressure", p);
        hdf_out.writeDoubleAttribute("time", static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC);
        hdf_out.close();
        cusparseDestroy(cusparse_handle);
        cusolverSpDestroy(cusolver_handle);
        std::cout << "Total time: " << static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
        std::cout << "  Preprocessing time: " << preprocessing_time << std::endl;
        std::cout << "  LU time: " << lu_time << std::endl;
        std::cout << "  M_u construct time: " << M_u_construct_time << std::endl;
        std::cout << "  M_u solve time: " << M_u_solve_time << std::endl;
        std::cout << "  PV construction time: " << PV_construction_time << std::endl;
        std::cout << "  PV solve time: " << PV_solve_time << std::endl;
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
            LidDriven<2> solution(params);
            break;
        }
        case 3: {
            LidDriven<3> solution(params);
            break;
        }
        default:
            std::cout << params.get<int>("case.dim") << "is not a supported dimension" << std::endl;
    }
}