#include <hop/FSSH.hpp>

namespace hop {

std::vector<double> FSSH::compute_probabilities(const double & dt) const {
    // calculate fewest switches flux
    std::vector<double> probabilities(NStates_);
    for (size_t i = 0; i < active_state_; i++) {
        at::Tensor nac = dHa_old_[i][active_state_] / (energy_old_[active_state_] - energy_old_[i]);
        probabilities[i] = -2.0 * dt *
                       ( (w_old_[active_state_][0] * w_old_[i][0] + w_old_[active_state_][1] * w_old_[i][1])
                       * (p_old_ / mass_).dot(nac)).item<double>();
    }
    probabilities[active_state_] = 0.0;
    for (size_t i = active_state_ + 1; i < NStates_; i++) {
        at::Tensor nac = dHa_old_[active_state_][i] / (energy_old_[active_state_] - energy_old_[i]);
        probabilities[i] = -2.0 * dt *
                       ( (w_old_[active_state_][0] * w_old_[i][0] + w_old_[active_state_][1] * w_old_[i][1])
                       * (p_old_ / mass_).dot(nac)).item<double>();
    }
    // convert flux to probability
    double real = w_old_[active_state_][0].item<double>(),
           imag = w_old_[active_state_][1].item<double>();
    double population = real * real + imag * imag;
    for (double & probability : probabilities) probability = std::max(0.0, probability / population);
    double total = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    if (total > 1.0) {
        std::cerr << "hop::FSSH::compute_probabilities warning: the sum of hopping probability > 1,\n"
                  << "                                          please consider decreasing time step";
        for (double & probability : probabilities) probability /= total;
    }
    return probabilities;
}

FSSH::FSSH() {}

FSSH::FSSH(const int64_t & _NStates, const at::Tensor & _mass,
at::Tensor (*_compute_Hd)(const at::Tensor &),
std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &))
: Base(_NStates, _mass, _compute_Hd, _compute_Hd_dHd) {}

FSSH::~FSSH() {}

// perform a surface hopping loop with time step dt
void FSSH::step(const double & dt) {
    // propagate nuclear coordinate and momentum
    propagate_nucleus(dt);
    // propagate electronic wave function
    propagate_electron(dt);
    // switch active surface by hopping probability and momentum rescaling
    std::vector<double> probabilities = compute_probabilities(dt);
    int64_t target_state = determine_target(probabilities);
    rescale_momentum(target_state);
    // Prepare for the next loop
    prepare_next();
}
// specialization for 1-dimensional case
void FSSH::step_1D(const double & dt) {
    // propagate nuclear coordinate and momentum
    propagate_nucleus(dt);
    // propagate electronic wave function
    propagate_electron(dt);
    // switch active surface by hopping probability and momentum rescaling
    std::vector<double> probabilities = compute_probabilities(dt);
    int64_t target_state = determine_target(probabilities);
    rescale_scalar(target_state);
    // prepare for the next loop
    prepare_next();
}

} // namespace hop