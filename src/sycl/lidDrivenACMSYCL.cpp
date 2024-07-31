#include <iostream>
#include <medusa/Medusa.hpp>
// #include <omp.h>
#include <Eigen/SparseLU>
#include <chrono>
#include <sycl/sycl.hpp>
#include <numeric>
#include <vector>
#include <mkl.h>
#define EPS 1e-10

using namespace mm;
template <typename scal_t, int dim>
class VectorFieldDevice {
    scal_t* u_;
    int n_;

  public:
    VectorFieldDevice(const VectorField<scal_t, dim>& u, sycl::queue& q) : n_(u.rows()) {
        u_ = sycl::malloc_shared<scal_t>(u.size(), q);
        q.submit([&](sycl::handler& h) {
             h.memcpy(u_, &u.begin()[0], u.size() * sizeof(scal_t));
         }).wait();
    }
    int rows() const { return n_; }
    scal_t* begin() const { return u_; }
    int size() { return dim * n_; }
};
template <typename scalar_t>
sycl::event lap_device(sycl::queue& q, int* support, int support_start, int support_size,
                       scalar_t* shape, scalar_t* u, scalar_t* res, int dim, int u_size,
                       const std::vector<sycl::event>& depends_on = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(depends_on);
        *res = 0.0;
        auto red = sycl::reduction(res, sycl::plus<>());
        h.parallel_for(support_size, red, [=](sycl::id<1> idx, auto& tmp) {
            tmp += shape[support_start + idx] * u[support[support_start + idx] + u_size * dim];
            // tmp += support[support_start + idx];
            // tmp += 1;  // shape[support_start + idx];
        });
    });
}
template <typename vec_t,
          typename OpFamilies = std::tuple<Lap<vec_t::dim>, Der1s<vec_t::dim>, Der2s<vec_t::dim>>>
class RaggedShapeStorageDevice {
    typedef vec_t vector_t;
    typedef typename vec_t::scalar_t scalar_t;
    enum { dim = vec_t::dim };
    typedef OpFamilies op_families_tuple;
    constexpr static int num_operators = std::tuple_size<op_families_tuple>::value;
    sycl::queue* q_;
    int domain_size_;
    int* support_;
    std::vector<scalar_t*> shapes_;

    Range<int> support_starts_;
    Range<int> support_sizes_;
    int total_size_;

  public:
    RaggedShapeStorageDevice(RaggedShapeStorage<vec_t, OpFamilies> storage, sycl::queue& q) {
        q_ = &q;
        auto support_sizes = storage.supportSizes();
        support_sizes_ = support_sizes;
        int size = std::reduce(support_sizes.begin(), support_sizes.end());
        total_size_ = size;
        support_ = sycl::malloc_shared<int>(size, *q_);
        int j = 0;
        for (int i = 0; i < storage.size(); ++i) {
            auto vec = storage.support(i);
            support_starts_.push_back(j);
            q.submit([&](sycl::handler& h) {
                h.memcpy(&support_[j], &vec.begin()[0], vec.size() * sizeof(int));
            });
            j += vec.size();
        }
        for (int i = 0; i < storage.shapes().size(); ++i) {
            auto& shape = storage.shapes()[i];
            shapes_.emplace_back(sycl::malloc_shared<scalar_t>(shape.size(), *q_));
            q.submit([&](sycl::handler& h) {
                h.memcpy(shapes_[i], &shape[0], shape.size() * sizeof(scalar_t));
            });
        }

        q.wait();
    }
    scalar_t laplace(int node, int j) const {
        int laplace_index = tuple_index<Lap<vector_t::dim>, op_families_tuple>::value;
        return shapes_[laplace_index][support_starts_[node] + j];
    }
    scalar_t d1(int var, int node, int j) const {
        int d1_index = tuple_index<Der1s<vector_t::dim>, op_families_tuple>::value;
        auto op = Der1s<vector_t::dim>::index(Der1<vector_t::dim>(var));
        return shapes_[d1_index][op * total_size_ + support_starts_[node] + j];
    }
    // scalar_t* lap(scalar_t* u, int u_size, int i) {
    //     int laplace_index = tuple_index<Lap<vector_t::dim>, op_families_tuple>::value;
    //     // for (int j = 0; j < support_sizes_[i]; ++j)
    //     scalar_t* lap_res = sycl::malloc_shared<scalar_t>(dim, *q_);
    //     int support_start = support_starts_[i];
    //     size_t support_size = support_sizes_[i];
    //     scalar_t* shape = shapes_[laplace_index];
    //     for (int d = 0; d < dim; ++d) {
    //         lap_device(*q_, support_, support_start, support_size, shape, u, &lap_res[d], d,
    //         u_size)
    //             .wait();
    //     }
    //     return lap_res;
    // }
    std::vector<sycl::event> lap(VectorFieldDevice<scalar_t, dim>& u, int i, scalar_t* res,
                                 const std::vector<sycl::event>& depends_on = {}) const {
        std::vector<sycl::event> events = {};
        for (int d = 0; d < dim; ++d) {
            int laplace_index = tuple_index<Lap<vector_t::dim>, op_families_tuple>::value;
            int support_start = support_starts_[i];
            size_t support_size = support_sizes_[i];
            scalar_t* shape = shapes_[laplace_index];
            int* support = support_;
            events.push_back(q_->submit([&](sycl::handler& h) {
                h.depends_on(depends_on);
                res[d] = 0;
                auto scalar_prod = sycl::reduction(&res[d], sycl::plus<>());
                h.parallel_for(support_size, scalar_prod, [=](sycl::id<1> idx, auto& tmp) {
                    tmp += shape[support_start + idx] *
                           u.begin()[support[support_start + idx] + u.rows() * d];
                });
            }));
        }
        return events;
    }
    int support(int node, int j) const { return support_[support_starts_[node] + j]; }
    ~RaggedShapeStorageDevice() {
        sycl::free(support_, *q_);
        for (int i = 0; i < num_operators; ++i) {
            sycl::free(shapes_[i], *q_);
        }
    }
};

template <int dim>
class LidDrivenACM {
  public:
    typedef double scal_t;
    typedef Vec<scal_t, dim> vec_t;

    explicit LidDrivenACM(XML& param_file) {
        HDF::Mode mode = param_file.get<bool>("output.clear_hdf5") ? HDF::DESTROY : HDF::APPEND;
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
        int support_size =
            std::round(param_file.get<scal_t>("num.support_size_factor") * mon.size());

        mm::RBFFD<Polyharmonic<scal_t>, vec_t, ScaleToClosest> rbffd(k, mon);

        domain.findSupport(FindClosest(support_size).forNodes(interior));
        domain.findSupport(
            FindClosest(support_size).forNodes(boundary).searchAmong(interior).forceSelf(true));

        auto storage = domain.template computeShapes<sh::lap | sh::grad>(rbffd);
        auto op_e_v = storage.explicitVectorOperators();
        auto op_e_s = storage.explicitOperators();

        // Move storage to GPU version
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
                  << "\n";
        RaggedShapeStorageDevice storage_device{storage, q};

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
        while (t < end_time) {
            // #pragma omp parallel for default(none) schedule(static) shared(interior,
            // u_partial, u, dt, op_e_v, Re)
            for (int _ = 0; _ < interior.size(); ++_) {
                int i = interior[_];
                // scal_t* u_device = sycl::malloc_shared<scal_t>(u.size(), q);
                // q.submit([&](sycl::handler& h) {
                //      h.memcpy(u_device, &u.begin()[0], u.size() * sizeof(scal_t));
                //  }).wait();
                VectorFieldDevice<scal_t, dim> u_device{u, q};
                std::cout << op_e_v.lap(u, i) << std::endl;
                // scal_t* lap_device = storage_device.lap(u_device.begin(), u.rows(), i);
                scal_t* lap_device = sycl::malloc_shared<scal_t>(dim, q);

                auto events = storage_device.lap(u_device, i, lap_device);

                for (auto event : events) {
                    event.wait();
                }
                std::cout << "[";
                for (int j = 0; j < dim; ++j) {
                    std::cout << lap_device[j] << " ";
                }
                std::cout << "]\n";
                u_partial[i] = u[i] + dt * (op_e_v.lap(u, i) / Re - op_e_v.grad(u, i) * u[i]);
            }
            scal_t max_norm;
            scal_t max_div;
            int p_iter;
            for (p_iter = 0; p_iter < max_p_iter; ++p_iter) {
                std::swap(p_old, p_new);
                max_norm = 0;
                // #pragma omp parallel for default(none) schedule(static) shared(interior, u,
                // u_partial, dt, op_e_s, p, p_old) reduction(max : max_norm)
                for (int _ = 0; _ < interior.size(); ++_) {
                    int i = interior[_];
                    u[i] = u_partial[i] - dt * op_e_s.grad(p[p_old], i);
                    max_norm = std::max(u[i].norm(), max_norm);
                }
                scal_t C = compress * std::max(max_norm, v_ref);
                max_div = 0;
                // #pragma omp parallel for default(none) schedule(static) shared(interior,
                // op_e_v, u, p, p_new, p_old, C, dt, op_e_s, Re) reduction(max : max_div)
                for (int _ = 0; _ < interior.size(); ++_) {
                    int i = interior[_];
                    scal_t div = op_e_v.div(u, i);
                    max_div = std::max(std::abs(div), max_div);
                    p[p_new][i] =
                        p[p_old][i] - C * C * dt * div;  // + dt * op_e_s.lap(p[p_old], i) / Re;
                }

                // #pragma omp parallel for default(none) schedule(static) shared(boundary, p,
                // p_new, op_e_s, domain)
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
                int print_iter = (iteration - 1) / printout_interval;
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
    // omp_set_num_threads(params.get<int>("sys.num_threads"));

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