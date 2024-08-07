#include <iostream>
#include <medusa/Medusa.hpp>
#include <medusa/Utils.hpp>
#include <Eigen/SparseLU>
#include <sparseluwrapper.hpp>
#include <solverQR.hpp>
#include <solverDSS.hpp>
#include <solverRF.hpp>
#include <chrono>

#define EPS 1e-10

using namespace mm;

template <class SolverU, class SolverP, Eigen::StorageOptions ordering_u,
          Eigen::StorageOptions ordering_p, int dim>
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

        const auto start{std::chrono::steady_clock::now()};
        Stopwatch s;
        s.start("time");
        s.start("domain setup");

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

        s.stop("domain setup");
        // pressure correction matrix
        s.start("M_p construction");
        Eigen::SparseMatrix<scal_t, ordering_p> M_p(N + 1, N + 1);
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
        // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_p;
        SolverP solver_p(M_p);
        s.stop("M_p construction");
        // solver_p.compute(M_p);
        // if (solver_p.info() != Eigen::Success) {
            // std::cout << "LU factorization failed with error:" << solver_p.lastErrorMessage()
                    //   << std::endl;
        // }

        scal_t t = 0;
        auto end_time = param_file.get<scal_t>("case.end_time");
        auto printout_interval = param_file.get<int>("output.printout_interval");
        auto max_p_iter = param_file.get<int>("num.max_p_iter");
        auto div_limit = param_file.get<scal_t>("num.max_divergence");
        int num_print = end_time / (dt * printout_interval);
        Eigen::VectorXd max_u_y(num_print);
        int iteration = 0;
        while (t < end_time) {
            s.start("M_u construction");
            Eigen::SparseMatrix<double, ordering_u> M_u(dim * N, dim * N);
            Eigen::VectorXd rhs_u(dim * N);
            rhs_u.setZero();
            // Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>,
            //     Eigen::IncompleteLUT<double>> solver_u;
            // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_u;
            // SolverU solver_u;

            Range<int> per_row_u(dim * N, support_size);
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
            s.stop("M_u construction");
            s.start("M_u solve");
            SolverU solver_u(M_u);
            // solver_u.compute(M_u);
            // Eigen::VectorXd solution = solver_u.solveWithGuess(rhs_u, u.asLinear());
            Eigen::VectorXd solution = solver_u.solve(rhs_u);
            u = VectorField<scal_t, dim>::fromLinear(solution);
            s.stop("M_u solve");

            // P-V correction iteration -- PVI
            scal_t max_div;
            int p_iter;
            for (p_iter = 0; p_iter < max_p_iter; ++p_iter) {
                s.start("rhs_p");
#pragma omp parallel for default(none) schedule(static) shared(interior, u, rhs_p, op_e_v, dt)
                for (int _ = 0; _ < interior.size(); ++_) {
                    int i = interior[_];
                    rhs_p(i) = op_e_v.div(u, i) / dt;
                }
                for (int i : boundary) rhs_p(i) = dt * u[i].dot(domain.normal(i));
                s.stop("rhs_p");
                s.start("M_p solve");
                ScalarFieldd p_c = solver_p.solve(rhs_p).head(N);
                p += p_c;
                s.stop("M_p solve");

                max_div = 0;
                s.start("pressure correction");
#pragma omp parallel for default(none) schedule(static) \
    shared(interior, u, op_e_s, op_e_v, p_c, dt) reduction(max : max_div)
                for (int _ = 0; _ < interior.size(); ++_) {
                    int i = interior[_];
                    u[i] -= dt * op_e_s.grad(p_c, i);
                    max_div = std::max(std::abs(op_e_v.div(u, i)), max_div);
                }
                s.stop("pressure correction");
                if (max_div < div_limit) break;
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
                int print_iter = (iteration - 1) / printout_interval;
                max_u_y[print_iter] = max;
                std::cout << iteration << " - t:" << t << " max u_y:" << max << " @ x:" << pos
                          << "  (max div:" << max_div << " @ " << p_iter + 1 << ")" << std::endl;
            }
        }

        s.stop("time");
        auto times = {"time", "domain setup", "M_p construction", "M_u construction", "M_u solve", "rhs_p", "M_p solve", "pressure correction"};
        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_time{end - start};

        hdf_out.reopen();
        hdf_out.openGroup("/");
        hdf_out.writeEigen("velocity", u);
        hdf_out.writeEigen("pressure", p);
        hdf_out.writeDoubleAttribute("time", elapsed_time.count());
        hdf_out.writeEigen("max_u_y", max_u_y);
        hdf_out.openGroup("times");
        for (std::string time : times) {
            hdf_out.writeDoubleAttribute(time, s.cumulativeTime(time));
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

    std::string solver_u = params.get<std::string>("case.solver_u");
    std::string solver_p = params.get<std::string>("case.solver_p");
    switch (params.get<int>("case.dim")) {
        case 2: {
            if (solver_u == "QR" && solver_p == "SparseLUWrapper") {
                LidDriven<SolverQR, SparseLUWrapper, Eigen::RowMajor, Eigen::ColMajor, 2>
                    solution(params);
            } else if (solver_u == "DSS" && solver_p == "SparseLUWrapper") {
                LidDriven<SolverDSS, SparseLUWrapper, Eigen::RowMajor, Eigen::ColMajor, 2>
                    solution(params);
            } else if (solver_u == "DSS" && solver_p == "RF") {
                LidDriven<SolverDSS, SolverRF, Eigen::RowMajor, Eigen::RowMajor, 2>
                    solution(params);
            } else {
                LidDriven<SparseLUWrapper, SparseLUWrapper, Eigen::RowMajor, Eigen::ColMajor, 2>
                    solution(params);
            }
            break;
        }
        case 3: {
            if (solver_u == "QR" && solver_p == "SparseLUWrapper") {
                LidDriven<SolverQR, SparseLUWrapper, Eigen::RowMajor, Eigen::ColMajor, 3>
                    solution(params);
            } else if (solver_u == "DSS" && solver_p == "SparseLUWrapper") {
                LidDriven<SolverDSS, SparseLUWrapper, Eigen::RowMajor, Eigen::ColMajor, 3>
                    solution(params);
            } else if (solver_u == "DSS" && solver_p == "RF") {
                LidDriven<SolverDSS, SolverRF, Eigen::RowMajor, Eigen::RowMajor, 3>
                    solution(params);
            } else {
                LidDriven<SparseLUWrapper, SparseLUWrapper, Eigen::RowMajor, Eigen::ColMajor, 3>
                    solution(params);
            }
            break;
        }
        default:
            std::cout << params.get<int>("case.dim") << "is not a supported dimension" << std::endl;
    }
}
