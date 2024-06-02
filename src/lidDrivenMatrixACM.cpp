#include <iostream>
#include <medusa/Medusa.hpp>
#include <medusa/Utils.hpp>
#include <Eigen/SparseLU>
#include <chrono>

#define EPS 1e-10

using namespace mm;

template <int dim>
class LidDrivenMatrixACM {
  public:
    typedef double scal_t;
    typedef Vec<scal_t, dim> vec_t;

    explicit LidDrivenMatrixACM(XML &param_file) {
        Stopwatch s;
        s.start("time");
        s.start("setup");
        HDF::Mode mode = param_file.get<bool>("output.clear_hdf5")
                         ? HDF::DESTROY
                         : HDF::APPEND;
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
        Range<int> corner = domain.positions().filter([&](const vec_t &p) {
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
          if (borders.second[dim - 1] - domain.pos(i, dim - 1) < EPS) lid.push_back(i);
          else wall.push_back(i);
        }

        scal_t midpoint = (borders.second(1) - borders.first(1)) / 2;
        Range<int> midplane = domain.positions().filter([&](const vec_t &p) {
            return std::abs(p(1) - midpoint) < h;
        });

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
        int support_size = std::round(param_file.get<scal_t>("num.support_size_factor") * mon.size());
        mm::RBFFD<Polyharmonic<scal_t>, vec_t, ScaleToClosest> rbffd(k, mon);

        domain.findSupport(FindClosest(support_size).forNodes(interior));
        domain.findSupport(FindClosest(support_size).forNodes(boundary).searchAmong(interior).forceSelf(true));
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
            // derivative.back().reserve(Eigen::VectorXi::Constant(dim * N, dim * support_size));
            stack.emplace_back(dim * N, dim * N);
            stack.back().reserve(Eigen::VectorXi::Constant(dim * N, 1));
        }
        // workaround
        Eigen::SparseMatrix<double, Eigen::RowMajor> d0(dim * N, dim * N), d1(dim * N, dim * N);
        d0.reserve(Eigen::VectorXi::Constant(dim * N, support_size));
        d1.reserve(Eigen::VectorXi::Constant(dim * N, support_size));
        for (int i = 0; i < domain.size(); ++i) {
            if (domain.type(i) > 0 ) {
                neumann.insert(i, i) = 1;
                for (int var = 0; var < dim; ++var) {
                    component_sum.insert(i, i + var * domain.size()) = 1;
                    for (int s = 0; s < storage.supportSize(i); ++s) {
                        grad_p.insert(i + var * domain.size(), storage.support(i, s)) = storage.d1(var, i, s);
                        lap.insert(i + var * domain.size(), storage.support(i, s) + var * domain.size()) = storage.laplace(i, s);
                        div.insert(i, storage.support(i, s) + var * domain.size()) = storage.d1(var, i, s);
                        // workaround
                        d0.insert(i + var * domain.size(), storage.support(i, s) + var * domain.size()) = storage.d1(0, i, s);
                        d1.insert(i + var * domain.size(), storage.support(i, s) + var * domain.size()) = storage.d1(1, i, s);
                        for (int der_var = 0; der_var < dim; ++der_var) {
                            // derivative[der_var].insert(i + var * domain.size(), storage.support(i, s) + var * domain.size()) = storage.d1(der_var, i, s);
                            //This could be replaced with some sort of stacking operation if available
                            if (s == 0) {
                                stack[der_var].insert(i + var * domain.size(), i + der_var * domain.size()) = 1;
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
                    if (s == 0) central_coeff = neumann_coeff;
                    else {
                        neumann.insert(i, storage.support(i, s)) = -neumann_coeff / central_coeff;
                    }
                }
            }
        }
        // std::cout << derivative[0].nonZeros() / double(derivative[0].rows() * derivative[0].cols()) << std::endl;
        s.start("compression");
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
        std::cout << derivative[0].nonZeros() / double((derivative[0].rows() * derivative[0].cols())) << std::endl;
        s.stop("compression");

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
        s.stop("setup");
        while (t < end_time) {
            s.start("u = u + dt / Re * lap * u");
            u_partial = u + dt / Re * lap * u;
            s.stop("u = u + dt / Re * lap * u");
            s.start("advection");
            for (int var = 0; var < dim; ++var) { // Complication to remain general in dimensions. Can be written explicitly.
                u_partial -= dt * (derivative[var] * u).cwiseProduct(stack[var] * u);
            }
            s.stop("advection");
            scal_t max_norm, max_div;
            int p_iter;
            for(p_iter=0; p_iter < max_p_iter; ++p_iter) {
                s.start("PV correction");
                u = u_partial - dt * grad_p * p;
                s.stop("PV correction");
                s.start("max_norm");
                max_norm = std::sqrt((component_sum * u.cwiseProduct(u)).maxCoeff());
                s.stop("max_norm");
                s.start("p");
                scal_t C = compress * std::max(max_norm, v_ref);
                Eigen::VectorXd divergence =  div * u;
                p = p - C * C * dt * divergence;
                p = neumann * p;
                s.stop("p");
                s.start("max_div");
                max_div = divergence.cwiseAbs().maxCoeff();
                s.stop("max_div");
                if (max_div < div_limit) break;
            }
            t += dt;
            if (++iteration % printout_interval == 0) {
                s.start("print");
                scal_t max = 0, pos;
                for (int i : midplane) {
                    if (u(i + (dim - 1) * N) > max) {
                        max = u(i + (dim - 1) * N);
                        pos = domain.pos(i, 0);
                    }
                }
                int print_iter = (iteration - 1) / printout_interval;
                max_u_y[print_iter] = max;
                std::cout << iteration << " - t:" << t << " max u_y:" << max << " @ x:" << pos
                          << "  (max div:" << max_div << " @ " << p_iter << ")" << std::endl;
                s.stop("print");
            }
        }

        // const auto end{std::chrono::steady_clock::now()};
        // const std::chrono::duration<double> elapsed_time{end - start};

        s.stop("time");
        hdf_out.reopen();
        hdf_out.openGroup("/");
        hdf_out.writeEigen("velocity", VectorField<scal_t, dim>::fromLinear(u));
        hdf_out.writeEigen("pressure", p);
        // hdf_out.writeDoubleAttribute("time", elapsed_time.count());
        std::vector<std::string> labels{"time",
                                        "setup",
                                        "compression",
                                        "u = u + dt / Re * lap * u",
                                        "derivative stack correction",
                                        "PV correction",
                                        "max_norm",
                                        "p",
                                        "max_div",
                                        "print"};
        hdf_out.writeEigen("max_u_y", max_u_y);
        hdf_out.openGroup("times");
        for (std::string label : labels) {
            hdf_out.writeDoubleAttribute(label, s.cumulativeTime(label));
        }
        hdf_out.close();
    }
};

// The path to .xml parameter file required as a command line parameter
int main(int arg_num, char* arg[]) {
    if (arg_num < 2) {
        throw std::invalid_argument("Missing command line argument. Provide path to .xml configuration.");
    } else if (arg_num > 2) {
        throw std::invalid_argument("Too many command line arguments. Only path to .xml configuration is required.");
    }
    std::string parameter_file(arg[1]);

    XML params(parameter_file);
    if (params.get<bool>("output.use_xml_name")) {
        std::string output_name = mm::join({mm::split(mm::split(parameter_file, "/").back(),".").front(), "h5"}, ".");
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