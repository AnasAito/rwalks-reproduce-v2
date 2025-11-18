#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <faiss/Index.h>
#include <faiss/IndexACORN.h>
#include <faiss/impl/platform_macros.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

inline void ensure_c_contiguous(const py::buffer_info& buf, const char* name) {
    if (buf.strides.size() > 0) {
        // Any stride of 0 or non-ascending order is suspicious for our use
        // We only accept C-contiguous arrays here
        ssize_t expected = buf.itemsize;
        for (ssize_t i = static_cast<ssize_t>(buf.ndim) - 1; i >= 0; --i) {
            if (buf.shape[i] == 0)
                break;
            if (buf.strides[i] != expected) {
                throw std::invalid_argument(
                        std::string(name) + " must be C-contiguous");
            }
            expected *= buf.shape[i];
        }
    }
}

inline faiss::MetricType parse_metric(const std::string& metric) {
    if (metric == "l2" || metric == "L2")
        return faiss::METRIC_L2;
    if (metric == "ip" || metric == "inner_product")
        return faiss::METRIC_INNER_PRODUCT;
    throw std::invalid_argument(
            "Unsupported metric: " + metric + ". Use 'l2' or 'ip'.");
}

} // namespace

class ACORNIndexWrapper {
   public:
    ACORNIndexWrapper(
            int dim,
            int M,
            int gamma,
            int M_beta,
            const std::string& metric)
            : dim_(dim),
              M_(M),
              gamma_(gamma),
              M_beta_(M_beta),
              mt_(parse_metric(metric)),
              capacity_(0),
              preallocated_(false) {}

    ACORNIndexWrapper(
            int dim,
            int M,
            int gamma,
            int M_beta,
            const std::string& metric,
            long long capacity)
            : dim_(dim),
              M_(M),
              gamma_(gamma),
              M_beta_(M_beta),
              mt_(parse_metric(metric)),
              capacity_(capacity),
              preallocated_(true) {
        if (capacity_ <= 0) {
            throw std::invalid_argument("capacity must be > 0");
        }
        metadata_.assign(static_cast<size_t>(capacity_), 0);
        index_ = std::make_unique<faiss::IndexACORNFlat>(
                dim_, M_, gamma_, metadata_, M_beta_, mt_);
    }

    void add(py::array_t<float, py::array::c_style | py::array::forcecast> xb) {
        auto buf = xb.request();
        if (buf.ndim != 2) {
            throw std::invalid_argument(
                    "xb must be a 2D float32 array of shape (N, d)");
        }
        if (buf.shape[1] != dim_) {
            throw std::invalid_argument(
                    "xb.shape[1] must equal dim passed to constructor");
        }
        ensure_c_contiguous(buf, "xb");
        size_t n = static_cast<size_t>(buf.shape[0]);
        const float* x_ptr = static_cast<const float*>(buf.ptr);
        if (!index_) {
            // Lazy path: allocate metadata exactly for first batch
            capacity_ = static_cast<long long>(n);
            metadata_.assign(n, 0);
            index_ = std::make_unique<faiss::IndexACORNFlat>(
                    dim_, M_, gamma_, metadata_, M_beta_, mt_);
            preallocated_ = false;
        } else if (preallocated_) {
            long long after = static_cast<long long>(index_->ntotal) +
                    static_cast<long long>(n);
            if (after > capacity_) {
                throw std::runtime_error(
                        "add would exceed capacity; construct with larger capacity");
            }
        } else {
            // Non-preallocated: further adds are unsafe for metadata pointer
            // lifetime
            throw std::runtime_error(
                    "Sequential add not allowed without capacity; construct with capacity to enable growth");
        }
        index_->add(n, x_ptr);
    }

    py::tuple search(
            py::array_t<float, py::array::c_style | py::array::forcecast> xq,
            int64_t k,
            py::object filter_ids_opt = py::none(),
            py::object ef_search_opt = py::none()) {
        auto qbuf = xq.request();
        if (qbuf.ndim != 2) {
            throw std::invalid_argument(
                    "xq must be a 2D float32 array of shape (nq, d)");
        }
        if (qbuf.shape[1] != dim_) {
            throw std::invalid_argument("xq.shape[1] must equal dim");
        }
        ensure_c_contiguous(qbuf, "xq");
        if (!index_) {
            throw std::runtime_error(
                    "Index not initialized. Call add() first.");
        }
        size_t nq = static_cast<size_t>(qbuf.shape[0]);
        const float* q_ptr = static_cast<const float*>(qbuf.ptr);

        if (!ef_search_opt.is_none()) {
            int ef = ef_search_opt.cast<int>();
            index_->acorn.efSearch = ef;
        }

        std::vector<float> distances(static_cast<size_t>(k) * nq);
        std::vector<faiss::idx_t> labels(static_cast<size_t>(k) * nq);

        if (!filter_ids_opt.is_none()) {
            // Accept uint8/byte or bool; shape can be (nq, ntotal) or flat
            // nq*ntotal
            py::array filter_arr = py::array(filter_ids_opt);
            auto fbuf = filter_arr.request();
            if (fbuf.ndim != 1 && fbuf.ndim != 2) {
                throw std::invalid_argument(
                        "filter_ids must be 1D or 2D array (nq, ntotal) of uint8/bool");
            }
            ensure_c_contiguous(fbuf, "filter_ids");
            size_t expected = nq * static_cast<size_t>(index_->ntotal);
            size_t got = 1;
            for (ssize_t i = 0; i < fbuf.ndim; ++i)
                got *= static_cast<size_t>(fbuf.shape[i]);
            if (got != expected) {
                throw std::invalid_argument(
                        "filter_ids total size must be nq * ntotal");
            }
            char* filter_ptr = static_cast<char*>(fbuf.ptr);
            index_->search(
                    nq,
                    q_ptr,
                    static_cast<faiss::idx_t>(k),
                    distances.data(),
                    labels.data(),
                    filter_ptr);
        } else {
            index_->search(
                    nq,
                    q_ptr,
                    static_cast<faiss::idx_t>(k),
                    distances.data(),
                    labels.data());
        }

        // Prepare numpy outputs with shape (nq, k)
        py::array_t<float> D(
                {static_cast<ssize_t>(nq), static_cast<ssize_t>(k)});
        py::array_t<long long> I(
                {static_cast<ssize_t>(nq), static_cast<ssize_t>(k)});
        std::memcpy(
                D.mutable_data(),
                distances.data(),
                sizeof(float) * distances.size());
        // faiss::idx_t is typically int64
        auto* Iptr = I.mutable_data();
        for (size_t i = 0; i < labels.size(); ++i) {
            Iptr[i] = static_cast<long long>(labels[i]);
        }
        return py::make_tuple(D, I);
    }

    void set_ef_search(int ef) {
        if (index_)
            index_->acorn.efSearch = ef;
    }
    int get_ef_search() const {
        return index_ ? index_->acorn.efSearch : 0;
    }

    void set_num_threads(int n) {
#ifdef _OPENMP
        if (n > 0) {
            omp_set_num_threads(n);
        }
#else
        (void)n;
#endif
    }

    long long ntotal() const {
        return index_ ? static_cast<long long>(index_->ntotal) : 0;
    }

    void set_labels(
            py::array_t<long long, py::array::c_style | py::array::forcecast>
                    ids,
            py::array_t<int, py::array::c_style | py::array::forcecast>
                    labels) {
        if (!index_) {
            throw std::runtime_error("Index not initialized");
        }
        auto ib = ids.request();
        auto lb = labels.request();
        if (ib.ndim != 1 || lb.ndim != 1 || ib.shape[0] != lb.shape[0]) {
            throw std::invalid_argument(
                    "ids and labels must be 1D with same length");
        }
        ensure_c_contiguous(ib, "ids");
        ensure_c_contiguous(lb, "labels");
        size_t m = static_cast<size_t>(ib.shape[0]);
        const long long* ip = static_cast<const long long*>(ib.ptr);
        const int* lp = static_cast<const int*>(lb.ptr);
        for (size_t i = 0; i < m; ++i) {
            long long id = ip[i];
            if (id < 0 || (preallocated_ && id >= capacity_)) {
                throw std::out_of_range("id out of range for capacity");
            }
            if (static_cast<size_t>(id) >= metadata_.size()) {
                throw std::out_of_range("id out of current metadata range");
            }
            metadata_[static_cast<size_t>(id)] = lp[i];
        }
    }

   private:
    int dim_;
    int M_;
    int gamma_;
    int M_beta_;
    faiss::MetricType mt_;
    std::unique_ptr<faiss::IndexACORNFlat> index_;
    std::vector<int> metadata_;
    long long capacity_;
    bool preallocated_;
};

PYBIND11_MODULE(_acorn, m) {
    m.doc() = "Python bindings for ACORN (Faiss-based)";

    py::class_<ACORNIndexWrapper>(m, "ACORNIndex")
            .def(py::init<int, int, int, int, const std::string&>(),
                 py::arg("dim"),
                 py::arg("M"),
                 py::arg("gamma"),
                 py::arg("M_beta"),
                 py::arg("metric") = "l2")
            .def(py::init<int, int, int, int, const std::string&, long long>(),
                 py::arg("dim"),
                 py::arg("M"),
                 py::arg("gamma"),
                 py::arg("M_beta"),
                 py::arg("metric"),
                 py::arg("capacity"))
            .def("add",
                 &ACORNIndexWrapper::add,
                 py::arg("xb"),
                 "Add base vectors of shape (N, dim), dtype=float32")
            .def("set_labels",
                 &ACORNIndexWrapper::set_labels,
                 py::arg("ids"),
                 py::arg("labels"))
            .def("search",
                 &ACORNIndexWrapper::search,
                 py::arg("xq"),
                 py::arg("k"),
                 py::arg("filter_ids").none(true) = py::none(),
                 py::arg("ef_search").none(true) = py::none(),
                 "Search with optional filter bitmap (nq x ntotal) and ef override")
            .def_property(
                    "ef_search",
                    &ACORNIndexWrapper::get_ef_search,
                    &ACORNIndexWrapper::set_ef_search)
            .def("set_num_threads",
                 &ACORNIndexWrapper::set_num_threads,
                 py::arg("num_threads"))
            .def_property_readonly("ntotal", &ACORNIndexWrapper::ntotal);
}
