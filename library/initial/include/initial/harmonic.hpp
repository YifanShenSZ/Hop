#ifndef initial_harmonic_hpp
#define initial_harmonic_hpp

#include <random>

#include <torch/torch.h>

namespace initial {

// randomly generate initial coordinate and momentum
// from Wigner sampling of harmonic oscillator ground level
class Harmonic {
    protected:
        at::Tensor x_;

        // normal mode -> Cartesian coordinate
        at::Tensor sqrt_mass_, invsqrt_mass_, normal_mode_;

        // Wigner sampling
        std::default_random_engine * generator_;
        std::vector<std::pair<std::normal_distribution<double> *, std::normal_distribution<double> *>> wigners_;
    public:
        Harmonic();
        // `x`: equilibirum Cartesian coordinate
        // `mass`: mass of each atom
        // `Hessian`: Cartesian coordinate Hessian at `x`
        Harmonic(const at::Tensor & _x, const std::vector<double> & _mass, const at::Tensor & _Hessian);
        ~Harmonic();

        const at::Tensor & sqrt_mass() const;
        const at::Tensor & invsqrt_mass() const;
        const at::Tensor & normal_mode() const;

        // generate initial coordinate and momentum
        std::tuple<at::Tensor, at::Tensor> operator()();
};

} // namespace initial

#endif