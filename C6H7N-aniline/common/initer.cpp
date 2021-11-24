#include "Hd.hpp"
#include "initer.hpp"

Initer::Initer() {}
// `x`: equilibirum Cartesian coordinate
// `mass`: mass of each atom
// `Hessian`: Cartesian coordinate Hessian at `x`
Initer::Initer(const at::Tensor & _x, const std::vector<double> & _mass, const at::Tensor & _Hessian) : initial::Harmonic(_x, _mass, _Hessian) {}
Initer::~Initer() {}

// generate initial active state, coordinate and momentum
std::tuple<size_t, at::Tensor, at::Tensor> Initer::operator()() {
    // initial coordinate and momentum
    at::Tensor x, p;
    std::tie(x, p) = this->initial::Harmonic::operator()();
    // initial electronic state
    at::Tensor Hd = compute_Hd(x);
    at::Tensor energy, state;
    std::tie(energy, state) = Hd.symeig(true);
    size_t NStates = Hd.size(0);
    std::vector<double> population(NStates);
    for (size_t i = 0; i < NStates; i++) {
        // the 3rd diabatic state is the pi->pi* state
        double c = state[2][i].item<double>();
        population[i] = c * c;
    }
    double random = (double)(*generator_)() / (double)(generator_->max() - generator_->min() + 1);
    for (size_t i = 0; i < NStates - 1; i++) {
        if (random < population[i]) return std::make_tuple(i, x, p);
        else random -= population[i];
    }
    return std::make_tuple(NStates - 1, x, p);
}