#include <ctime>

#include <tchem/linalg.hpp>

#include <initial/harmonic.hpp>

namespace initial {

Harmonic::Harmonic() {}
// `x`: equilibirum Cartesian coordinate
// `mass`: mass of each atom
// `Hessian`: Cartesian coordinate Hessian at `x`
Harmonic::Harmonic(const at::Tensor & _x, const std::vector<double> & _mass, const at::Tensor & _Hessian) : x_(_x.clone()) {
    if (_x.sizes().size() != 1) throw std::invalid_argument(
    "initial::Harmonic:Harmonic: x must be a vector");
    if (_Hessian.sizes().size() != 2) throw std::invalid_argument(
    "initial::Harmonic:Harmonic: Hessian must be a matrix");
    if (_Hessian.size(0) != _Hessian.size(1)) throw std::invalid_argument(
    "initial::Harmonic:Harmonic: Hessian must be a square matrix");
    if (_x.size(0) != _mass.size() * 3) throw std::invalid_argument(
    "initial::Harmonic:Harmonic: inconsistent dimension between x and mass");
    if (_x.size(0) != _Hessian.size(0)) throw std::invalid_argument(
    "initial::Harmonic:Harmonic: inconsistent dimension between x and Hessian");
    at::Tensor Hessian = _Hessian.clone();
    // Cartesian coordinate -> mass weighed Cartesian coordinate
       sqrt_mass_ = _x.new_zeros({_x.size(0), _x.size(0)}),
    invsqrt_mass_ = _x.new_zeros({_x.size(0), _x.size(0)});
    for (size_t i = 0; i < _mass.size(); i++) {
        double sqrt_mass = sqrt(_mass[i]);
        int64_t start = 3 * i;
        Hessian.slice(0, start, start + 3) /= sqrt_mass;
        Hessian.slice(1, start, start + 3) /= sqrt_mass;
           sqrt_mass_.slice(0, start, start + 3).slice(1, start, start + 3).fill_diagonal_(sqrt_mass);
        invsqrt_mass_.slice(0, start, start + 3).slice(1, start, start + 3).fill_diagonal_(1.0 / sqrt_mass);
    }
    // obtain frequency^2 and mass weighed normal modes
    at::Tensor freqs;
    std::tie(freqs, normal_mode_) = Hessian.symeig(true);
    if (freqs[6].item<double>() < 0.0) throw std::invalid_argument(
    "Harmonic::Harmonic: imaginary frequency at initial geometry");
    // initialize Wigner sampling
    srand(time(NULL));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_ = new std::default_random_engine(seed);
    size_t vibdim = _x.size(0) - 6;
    wigners_.resize(vibdim);
    for (size_t i = 0; i < vibdim; i++) {
        double freq = sqrt(freqs[i + 6].item<double>());
        double sigma_q = sqrt(0.5 / freq);
        double sigma_p = 0.5 / sigma_q;
        wigners_[i].first  = new std::normal_distribution<double>(0.0, sigma_q);
        wigners_[i].second = new std::normal_distribution<double>(0.0, sigma_p);
    }
}
Harmonic::~Harmonic() {}

const at::Tensor & Harmonic::sqrt_mass() const {return sqrt_mass_;}
const at::Tensor & Harmonic::invsqrt_mass() const {return invsqrt_mass_;}
const at::Tensor & Harmonic::normal_mode() const {return normal_mode_;}

// generate initial coordinate and momentum
std::tuple<at::Tensor, at::Tensor> Harmonic::operator()() {
    // Wigner sampling in normal mode
    at::Tensor q = x_.new_empty(x_.sizes()),
               p = x_.new_empty(x_.sizes());
    q.slice(0, 0, 6).fill_(0.0);
    p.slice(0, 0, 6).fill_(0.0);
    for (size_t i = 0; i < wigners_.size(); i++) {
        q[i + 6] = wigners_[i].first ->operator()(*generator_);
        p[i + 6] = wigners_[i].second->operator()(*generator_);
    }
    // normal mode -> Cartesian coordinate
    q = invsqrt_mass_.mv(normal_mode_.mv(q)) + x_;
    p =    sqrt_mass_.mv(normal_mode_.mv(p));
    return std::make_tuple(q, p);
}

} // namespace initial