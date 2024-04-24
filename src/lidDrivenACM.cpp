#include <iostream>
#include <medusa/Medusa.hpp>
#include <Eigen/SparseLU>
#include <chrono>

#define EPS 1e-10

using namespace mm;

template <int dim>
class LidDrivenACM {
  public:
    typedef double scal_t;
    typedef Vec<scal_t, dim> vec_t;

    explicit LidDrivenACM(XML &param_file) {
        HDF::Mode mode = param_file.get<bool>("output.clear_hdf5")
                         ? HDF::DESTROY
                         : HDF::APPEND;
        HDF hdf_out(param_file.get<std::string>("output.hdf5_name"), mode);
        
        auto Re = param_file.get<scal_t>("case.Re");
        auto h = param_file.get<scal_t>("num.h");
        auto cfl = param_file.get<scal_t>("num.cfl");
        auto seed = param_file.get<int>("case.seed");

        scal_t dt = cfl * h / dim;

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

        VectorField<scal_t, dim> u(N), u_partial(N);
        u.setZero();
        for (int i : lid) {
            u(i, 0) = 1;
        }
        u_partial = u;
        Range<ScalarField<scal_t>> p(2);
        p[0].resize(N);
        p[0].setZero();
        p[1].resize(N);
        p[1].setZero();
        int p_old = 0, p_new = 1;

        auto k = param_file.get<int>("num.phs_order");
        Monomials<vec_t> mon(param_file.get<int>("num.mon_order"));
        int support_size = std::round(param_file.get<scal_t>("num.support_size_factor") * mon.size());
        mm::RBFFD<Polyharmonic<scal_t>, vec_t, ScaleToClosest> rbffd(k, mon);

        domain.findSupport(FindClosest(support_size).forNodes(interior));
        domain.findSupport(FindClosest(support_size).forNodes(boundary).searchAmong(interior).forceSelf(true));

        auto storage = domain.template computeShapes<sh::lap | sh::grad>(rbffd);
        auto op_e_v = storage.explicitVectorOperators();
        auto op_e_s = storage.explicitOperators();

        scal_t t = 0;
        auto end_time = param_file.get<scal_t>("case.end_time");
        auto printout_interval = param_file.get<int>("output.printout_interval");
        auto max_p_iter = param_file.get<int>("acm.max_p_iter");
        auto div_limit = param_file.get<scal_t>("num.max_divergence");
        auto compress = param_file.get<scal_t>("acm.compressibility");
        auto v_ref = param_file.get<scal_t>("acm.reference_velocity");
        int num_print = end_time/(dt * printout_interval);
        Eigen::VectorXd max_u_y(num_print);
        int iteration = 0;
        while (t < end_time) {
            #pragma omp parallel for default(none) schedule(static) shared(interior, u_partial, u, dt, op_e_v, Re)
            for (int _ = 0; _ < interior.size(); ++_) {
                int i = interior[_];
                u_partial[i] = u[i] + dt * (op_e_v.lap(u, i) / Re
                                         -  op_e_v.grad(u, i) * u[i]);
            }
            scal_t max_norm;
            scal_t max_div;
            int p_iter;
            for(p_iter=0; p_iter < max_p_iter; ++p_iter) {
                std::swap(p_old, p_new);
                max_norm = 0;
                #pragma omp parallel for default(none) schedule(static) shared(interior, u, u_partial, dt, op_e_s, p, p_old) reduction(max : max_norm)
                for (int _ = 0; _ < interior.size(); ++_) {
                        int i = interior[_];
                        u[i] = u_partial[i] - dt * op_e_s.grad(p[p_old], i);
                        max_norm = std::max(u[i].norm(), max_norm);
                }
                scal_t C = compress * std::max(max_norm, v_ref);
                max_div = 0;
                #pragma omp parallel for default(none) schedule(static) shared(interior, op_e_v, u, p, p_new, p_old, C, dt, op_e_s, Re) reduction(max : max_div)
                for (int _ = 0; _ < interior.size(); ++_) {
                    int i = interior[_];
                    scal_t div = op_e_v.div(u, i);
                    max_div = std::max(std::abs(div), max_div);
                    p[p_new][i] = p[p_old][i] - C * C * dt * div;// + dt * op_e_s.lap(p[p_old], i) / Re;
                }

                #pragma omp parallel for default(none) schedule(static) shared(boundary, p, p_new, op_e_s, domain)
                for (int _ = 0; _ < boundary.size(); ++_) {
                    // Consider leaving one node constant to "fix" the pressure invariance
                    int i = boundary[_];
                    p[p_new][i] = op_e_s.neumann(p[p_new], i, domain.normal(i), 0);
                }
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
                int print_iter = (iteration - 1)/printout_interval;
                max_u_y[print_iter] = max;
                std::cout << iteration << " - t:" << t << " max u_y:" << max << " @ x:" << pos
                          << "  (max div:" << max_div << " @ " << p_iter << ")" << std::endl;
            }
        }

        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_time{end - start};

        hdf_out.reopen();
        hdf_out.openGroup("/");
        hdf_out.writeEigen("velocity", u);
        hdf_out.writeEigen("pressure", p[p_new]);
        hdf_out.writeDoubleAttribute("time", elapsed_time.count());
        hdf_out.writeEigen("max_u_y", max_u_y);
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
        LidDrivenACM<2> solution(params);
        break;
      }
      case 3: {
        LidDrivenACM<3> solution(params);
        break;
      }
      default:
        std::cout << params.get<int>("case.dim") << "is not a supported dimension" << std::endl; 
    }
}

// int main(int arg_num, char* arg[]) {}