#include <iostream>
#include <medusa/Medusa.hpp>
// #include <omp.h>
#include <Eigen/SparseLU>
#include <chrono>
#include <sycl/sycl.hpp>
#include <sycl/marray.hpp>
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
        u_ = sycl::malloc_device<scal_t>(u.size(), q);
        q.submit([&](sycl::handler& h) {
             h.memcpy(u_, &u.begin()[0], u.size() * sizeof(scal_t));
         }).wait();
    }
    int rows() const { return n_; }
    scal_t* col(int var) const { return &u_[var * n_]; }
    scal_t* begin() const { return u_; }
    int size() { return dim * n_; }
    scal_t& operator()(int var, int i) const { return u_[var * n_ + i]; }
};
// template <typename scal_t, int dim>
// struct sycl::is_device_copyable<VectorFieldDevice<scal_t, dim>> : std::true_type {};

template <typename scal_t>
class ScalarFieldDevice {
    scal_t* u_;
    int n_;

  public:
    ScalarFieldDevice(const ScalarField<scal_t>& u, sycl::queue& q) : n_(u.size()) {
        u_ = sycl::malloc_device<scal_t>(u.size(), q);
        q.submit([&](sycl::handler& h) {
             h.memcpy(u_, &u.begin()[0], u.size() * sizeof(scal_t));
         }).wait();
    }
    // ScalarFieldDevice(scal_t* u, int rows) {
    //     n_ = rows;
    //     u_ = u;
    // }
    scal_t* begin() const { return u_; }
    int size() { return n_; }
};
// template <typename scal_t>
// struct sycl::is_device_copyable<ScalarFieldDevice<scal_t>> : std::true_type {};
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
    Range<scalar_t*> shapes_;

    // Range<int> support_starts_;
    // Range<int> support_sizes_;
    int* support_starts_;
    int* support_sizes_;
    int total_size_;

  public:
    int support_starts_size_;
    int support_sizes_size_;
    RaggedShapeStorageDevice(RaggedShapeStorage<vec_t, OpFamilies> storage, sycl::queue& q) {
        q_ = &q;
        auto support_sizes = storage.supportSizes();
        // support_sizes_ = support_sizes;
        support_sizes_ = sycl::malloc_shared<int>(support_sizes.size(), q);
        q.submit([&](sycl::handler& h) {
             h.memcpy(&support_sizes_[0], &support_sizes.begin()[0],
                      support_sizes.size() * sizeof(int));
         }).wait();
        support_sizes_size_ = support_sizes.size();
        int size = std::reduce(support_sizes.begin(), support_sizes.end());
        total_size_ = size;
        support_ = sycl::malloc_device<int>(size, *q_);
        int j = 0;
        support_starts_ = sycl::malloc_shared<int>(storage.size(), q);
        support_starts_size_ = storage.size();
        for (int i = 0; i < storage.size(); ++i) {
            support_starts_[i] = j;
            auto vec = storage.support(i);
            q.submit([&](sycl::handler& h) {
                h.memcpy(&support_[j], &vec.begin()[0], vec.size() * sizeof(int));
            });
            j += vec.size();
        }
        for (int i = 0; i < storage.shapes().size(); ++i) {
            auto& shape = storage.shapes()[i];
            shapes_.emplace_back(sycl::malloc_device<scalar_t>(shape.size(), *q_));
            q.submit([&](sycl::handler& h) {
                h.memcpy(shapes_[i], &shape[0], shape.size() * sizeof(scalar_t));
            });
        }

        q.wait();
    }
    scalar_t* laplace_shape() {
        int laplace_index = tuple_index<Lap<vector_t::dim>, op_families_tuple>::value;
        return shapes_[laplace_index];
    }
    int* support_starts() { return support_starts_; }
    int* support_sizes() { return support_sizes_; }
    Range<scalar_t*> shapes() { return shapes_; }
    int* support() { return support_; }
    int total_size() { return total_size_; }
    int support(int node, int j) const { return support_[support_starts_[node] + j]; }
    ~RaggedShapeStorageDevice() {
        sycl::free(support_, *q_);
        for (int i = 0; i < num_operators; ++i) {
            sycl::free(shapes_[i], *q_);
        }
    }
};
namespace operators {
template <typename vec_t,
          typename OpFamilies = std::tuple<Lap<vec_t::dim>, Der1s<vec_t::dim>, Der2s<vec_t::dim>>>
class Operator {
    typedef vec_t vector_t;
    typedef typename vec_t::scalar_t scalar_t;
    enum { dim = vec_t::dim };
    typedef OpFamilies op_families_tuple;

  protected:
    int shape_index;
    scalar_t* shape_;
    int* support_;
    int* support_starts_;
    int support_starts_size_;
    int* support_sizes_;
    int support_sizes_size_;

  public:
    Operator(RaggedShapeStorageDevice<vector_t, op_families_tuple>& storage, int shape_index) {
        shape_ = storage.shapes()[shape_index];
        support_ = storage.support();
        support_starts_size_ = storage.support_starts_size_;
        support_starts_ = storage.support_starts();
        support_sizes_size_ = storage.support_sizes_size_;
        support_sizes_ = storage.support_sizes();
    }
};
template <typename vec_t,
          typename OpFamilies = std::tuple<Lap<vec_t::dim>, Der1s<vec_t::dim>, Der2s<vec_t::dim>>>
class LapOp : public Operator<vec_t, OpFamilies> {
    typedef vec_t vector_t;
    typedef typename vec_t::scalar_t scalar_t;
    enum { dim = vec_t::dim };
    typedef OpFamilies op_families_tuple;

  public:
    LapOp(RaggedShapeStorageDevice<vec_t, OpFamilies>& storage)
        : Operator<vector_t, op_families_tuple>(
              storage, tuple_index<Lap<vec_t::dim>, op_families_tuple>::value) {}
    scalar_t laplace(int node, int j) const {
        return this->shape_[this->support_starts_[node] + j];
    }
    void lap(const VectorFieldDevice<scalar_t, dim>& u, int i, int var, scalar_t alpha,
             scalar_t& res) const {
        for (int j = 0; j < this->support_sizes_[i]; ++j) {
            res += alpha * laplace(i, j) * u(var, this->support_[this->support_starts_[i] + j]);
        }
    }
    sycl::event lap(sycl::queue q, const VectorFieldDevice<scalar_t, dim>& u, scalar_t alpha,
                    VectorFieldDevice<scalar_t, dim>& res, int* interior, size_t interior_size,
                    const std::vector<sycl::event>& depends_on = {}) {
        return q.submit([&](sycl::handler& h) {
            h.depends_on(depends_on);
            h.parallel_for(
                interior_size, [=, support_sizes = this->support_sizes_, support = this->support_,
                                support_starts = this->support_starts_, shape = this->shape_,
                                res_ptr = res.begin(), rows = res.rows()](sycl::id<1> idx) {
                    int i = interior[idx];
                    for (int var = 0; var < dim; ++var) {
                        for (int j = 0; j < support_sizes[i]; ++j) {
                            res_ptr[rows * var + j] += alpha * shape[support_starts[i] + j] *
                                                       u(var, support[support_starts[i] + j]);
                        }
                    }
                });
        });
    }
};
template <typename vec_t,
          typename OpFamilies = std::tuple<Lap<vec_t::dim>, Der1s<vec_t::dim>, Der2s<vec_t::dim>>>
class DivOp : public Operator<vec_t, OpFamilies> {
    typedef vec_t vector_t;
    typedef typename vec_t::scalar_t scalar_t;
    enum { dim = vec_t::dim };
    typedef OpFamilies op_families_tuple;
    int total_size_;

  public:
    DivOp(RaggedShapeStorageDevice<vec_t, OpFamilies>& storage)
        : Operator<vector_t, op_families_tuple>(
              storage, tuple_index<Der1s<vec_t::dim>, op_families_tuple>::value) {
        total_size_ = storage.total_size();
    }
    scalar_t d1(int var, int node, int j) const {
        return this->shape_[var * total_size_ + this->support_starts_[node] + j];
    }
    scalar_t div(VectorFieldDevice<scalar_t, dim> u, int i) const {
        scalar_t res = 0;
        for (int var = 0; var < dim; ++var) {
            for (int j = 0; j < this->support_sizes_[j]; ++j) {
                res += d1(var, i, j) *
                       u.begin()[this->support_[this->support_starts_[i] + j] + u.rows() * var];
            }
        }
        return res;
    }
};
template <typename vec_t,
          typename OpFamilies = std::tuple<Lap<vec_t::dim>, Der1s<vec_t::dim>, Der2s<vec_t::dim>>>
class GradOp : public Operator<vec_t, OpFamilies> {
    typedef vec_t vector_t;
    typedef typename vec_t::scalar_t scalar_t;
    enum { dim = vec_t::dim };
    typedef OpFamilies op_families_tuple;
    int total_size_;
    int op;

  public:
    GradOp(RaggedShapeStorageDevice<vec_t, OpFamilies>& storage)
        : Operator<vector_t, op_families_tuple>(
              storage, tuple_index<Der1s<vec_t::dim>, op_families_tuple>::value) {
        total_size_ = storage.total_size();
    }
    scalar_t d1(int var, int node, int j) const {
        return this->shape_[var * total_size_ + this->support_starts_[node] + j];
    }
    // grad(d, var)
    scalar_t grad(VectorFieldDevice<scalar_t, dim> u, int i, int d, int var) const {
        scalar_t res = 0;
        for (int j = 0; j < this->support_sizes_[i]; ++j) {
            res += d1(var, i, j) * u(d, this->support_[this->support_starts_[i] + j]);
        }
        return res;
    }
    void grad(VectorFieldDevice<scalar_t, dim> u, int i, int d, scalar_t alpha,
              scalar_t& res) const {
        for (int var = 0; var < dim; ++var) {
            res += alpha * grad(u, i, d, var) * u(var, i);
        }
    }
    void grad(ScalarFieldDevice<scalar_t> u, int i, int var, scalar_t alpha, scalar_t& res) const {
        for (int j = 0; j < this->support_sizes_[i]; ++j) {
            res += alpha * d1(var, i, j) * u.begin()[this->support_[this->support_starts_[i] + j]];
        }
    }
};
template <typename vec_t,
          typename OpFamilies = std::tuple<Lap<vec_t::dim>, Der1s<vec_t::dim>, Der2s<vec_t::dim>>>
class NeumannOp : public Operator<vec_t, OpFamilies> {
    typedef vec_t vector_t;
    typedef typename vec_t::scalar_t scalar_t;
    enum { dim = vec_t::dim };
    typedef OpFamilies op_families_tuple;
    int total_size_;

  public:
    NeumannOp(RaggedShapeStorageDevice<vec_t, OpFamilies>& storage)
        : Operator<vector_t, op_families_tuple>(
              storage, tuple_index<Der1s<vec_t::dim>, op_families_tuple>::value) {
        total_size_ = storage.total_size();
    }
    scalar_t d1(int var, int node, int j) const {
        return this->shape_[var * total_size_ + this->support_starts_[node] + j];
    }
    scalar_t neumann(ScalarFieldDevice<scalar_t> u, int i, scalar_t* normal) const {
        // scalar_t res = 0;
        scalar_t val = 0;
        scalar_t denominator = 0;
        for (int var = 0; var < dim; ++var) {
            for (int j = 1; j < this->support_sizes_[j]; ++j) {
                val -= normal[var] * d1(var, i, j) *
                       u.begin()[this->support_[this->support_starts_[i] + j]];
            }
            denominator += normal[var] * d1(var, i, 0);
        }
        return val / denominator;
    }
};
}  // namespace operators
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

        // scal_t dt = cfl * h / dim;
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
        auto device = param_file.get<std::string>("sys.device");

        sycl::queue q(device == "gpu" ? sycl::gpu_selector_v : sycl::cpu_selector_v);
        std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>()
                  << "\n";
        RaggedShapeStorageDevice storage_device{storage, q};

        int* boundary_map = sycl::malloc_shared<int>(domain.bmap().size(), q);
        q.submit([&](sycl::handler& h) {
             h.memcpy(&boundary_map[0], &domain.bmap().begin()[0],
                      domain.bmap().size() * sizeof(int));
         }).wait();
        scal_t* normals_device = sycl::malloc_shared<scal_t>(domain.normals().size() * dim, q);
        int j = 0;
        for (auto normal : domain.normals()) {
            // normals_device.push_back(sycl::malloc_shared<scal_t>(normal.size(), q));
            q.submit([&](sycl::handler& h) {
                 h.memcpy(&normals_device[j], &normal.begin()[0], normal.size() * sizeof(scal_t));
             }).wait();
            j += normal.size();
        }
        VectorFieldDevice<scal_t, dim> u_device{u, q}, u_partial_device{u_partial, q};
        Range<ScalarFieldDevice<scal_t>> p_device{{p[p_old], q}, {p[p_new], q}};
        int* interior_device = sycl::malloc_shared<int>(interior.size(), q);
        q.submit([&](sycl::handler& h) {
             h.memcpy(interior_device, &interior.begin()[0], interior.size() * sizeof(int));
         }).wait();
        scal_t t = 0;
        int* boundary_device = sycl::malloc_shared<int>(boundary.size(), q);
        q.submit([&](sycl::handler& h) {
             h.memcpy(boundary_device, &boundary.begin()[0], boundary.size() * sizeof(int));
         }).wait();
        int* midplane_device = sycl::malloc_shared<int>(midplane.size(), q);
        q.submit([&](sycl::handler& h) {
             h.memcpy(midplane_device, &midplane.begin()[0], midplane.size() * sizeof(int));
         }).wait();
        auto end_time = param_file.get<scal_t>("case.end_time");
        auto printout_interval = param_file.get<int>("output.printout_interval");
        auto max_p_iter = param_file.get<int>("acm.max_p_iter");
        auto div_limit = param_file.get<scal_t>("num.max_divergence");
        auto compress = param_file.get<scal_t>("acm.compressibility");
        auto v_ref = param_file.get<scal_t>("acm.reference_velocity");
        int num_print = end_time / (dt * printout_interval);
        Eigen::VectorXd max_u_y(num_print);
        int iteration = 0;
        scal_t* max_norm_device = sycl::malloc_shared<scal_t>(1, q);
        scal_t* max_div_device = sycl::malloc_shared<scal_t>(1, q);
        while (t < end_time) {
            auto ev = q.submit([&](sycl::handler& h) {
                auto lap_op = operators::LapOp(storage_device);
                auto grad_op = operators::GradOp(storage_device);
                int interior_size = interior.size();
                sycl::range global{static_cast<unsigned long>(interior_size), 2};
                h.parallel_for(global, [=](sycl::id<2> idx) {
                    int i = interior_device[idx[0]];
                    int var = idx[1];
                    u_partial_device(var, i) = u_device(var, i);
                    lap_op.lap(u_device, i, var, dt / Re, u_partial_device(var, i));
                    grad_op.grad(u_device, i, var, -dt, u_partial_device(var, i));
                });
            });
            int p_iter;
            for (p_iter = 0; p_iter < max_p_iter; ++p_iter) {
                std::swap(p_old, p_new);
                *max_norm_device = 0;
                auto ev2 = q.submit([&](sycl::handler& h) {
                    h.depends_on(ev);
                    auto grad_op = operators::GradOp(storage_device);
                    auto p_dev = p_device[p_old];
                    sycl::range global{static_cast<unsigned long>(interior.size()), 2};
                    h.parallel_for(global, [=](sycl::id<2> idx) {
                        int i = interior_device[idx[0]];
                        int var = idx[1];
                        u_device(var, i) = u_partial_device(var, i);
                        grad_op.grad(p_dev, i, var, -dt, u_device(var, i));
                    });
                });
                auto ev3 = q.submit([&](sycl::handler& h) {
                    h.depends_on(ev2);
                    auto max = sycl::reduction(max_norm_device, sycl::maximum<>());
                    h.parallel_for(interior.size(), max, [=](sycl::id<1> idx, auto& max) {
                        int i = interior_device[idx];
                        max.combine(std::hypot(u_device(0, i), u_device(1, i)));
                    });
                });
                ev3.wait();
                scal_t C = compress * std::max(*max_norm_device, v_ref);
                *max_div_device = 0;
                auto ev4 = q.submit([&](sycl::handler& h) {
                    h.depends_on(ev3);
                    auto max = sycl::reduction(max_div_device, sycl::maximum<>());
                    auto div_op = operators::DivOp(storage_device);
                    auto& p_new_dev = p_device[p_new];
                    auto& p_old_dev = p_device[p_old];
                    h.parallel_for(interior.size(), max, [=](sycl::id<1> idx, auto& max) {
                        int i = interior_device[idx];
                        scal_t div = div_op.div(u_device, i);
                        max.combine(std::abs(div));
                        p_new_dev.begin()[i] = p_old_dev.begin()[i] - C * C * dt * div;
                    });
                });

                auto ev5 = q.submit([&](sycl::handler& h) {
                    h.depends_on(ev4);
                    auto p_dev = p_device[p_new];
                    operators::NeumannOp neu_op(storage_device);
                    h.parallel_for(boundary.size(), [=](sycl::id<1> idx) {
                        int i = boundary_device[idx];
                        p_dev.begin()[i] =
                            neu_op.neumann(p_dev, i, &normals_device[boundary_map[i] * dim]);
                    });
                });
                ev5.wait();
                if (*max_div_device < div_limit) break;
            }
            t += dt;
            if (++iteration % printout_interval == 0) {
                std::pair<scal_t, int>* max_pos_device =
                    sycl::malloc_shared<std::pair<scal_t, int>>(1, q);
                std::pair<scal_t, int> max_id_device{0, 0};

                *max_pos_device = max_id_device;
                q.submit([&](sycl::handler& h) {
                     auto max_pos = sycl::reduction(max_pos_device, max_id_device,
                                                    sycl::maximum<std::pair<scal_t, int>>());
                     h.parallel_for(midplane.size(), max_pos, [=](sycl::id<1> idx, auto& max_loc) {
                         int i = midplane_device[idx];
                         std::pair<scal_t, int> partial{u_device(1, i), i};
                         max_loc.combine(partial);
                     });
                 }).wait();
                int print_iter = (iteration - 1) / printout_interval;
                max_u_y[print_iter] = max_pos_device->first;
                std::cout << iteration << " - t:" << t << " max u_y:" << max_pos_device->first
                          << " @ x:" << domain.pos(max_pos_device->second, 0)
                          << "  (max div:" << *max_div_device << " @ " << p_iter << ")"
                          << std::endl;
            }
        }

        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_time{end - start};
        q.submit([&](sycl::handler& h) {
             h.memcpy(&u.begin()[0], u_device.begin(), u.size() * sizeof(scal_t));
         }).wait();
        q.submit([&](sycl::handler& h) {
             h.memcpy(&p[p_new].begin()[0], p_device[p_new].begin(),
                      p[p_new].size() * sizeof(scal_t));
         }).wait();
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