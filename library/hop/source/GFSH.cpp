#include <hop/GFSH.hpp>

namespace hop {

std::vector<double> GFSH::compute_probabilities(const double & dt) const {
    std::vector<double> probabilities(NStates_);
    // calculate population
    double real = w_old_[active_state_][0].item<double>(),
           imag = w_old_[active_state_][1].item<double>();
    double population = real * real + imag * imag;
    at::Tensor wsq = w_.pow(2), woldsq = w_old_.pow(2);
    at::Tensor pop_change = wsq.select(1, 0) + wsq.select(1, 1)
                          - (woldsq.select(1, 0) + woldsq.select(1, 1));
    // no hop if active state is gaining population
    if (pop_change[active_state_].item<double>() > 0.0) {
        std::fill(probabilities.begin(), probabilities.end(), 0.0);
        return probabilities;
    }
    // in 2-state case global flux is equivalent to the well-defined unique flux
    if (NStates_ == 2) {
        if (active_state_ == 0) {
            probabilities[0] = 0.0;
            probabilities[1] = pop_change[1].item<double>() / population;
        }
        else {
            probabilities[0] = pop_change[0].item<double>() / population;
            probabilities[1] = 0.0;
        }
    }
    // calculate global flux for other cases
    else {
        // determine which states raise / reduce population
        // i.e. have net in / out flux
        std::forward_list<size_t> in_states;
        double net_out = 0.0;
        for (size_t i = 0; i < NStates_; i++)
        if (pop_change[i].item<double>() > 0.0) in_states.push_front(i);
        else net_out += pop_change[i].item<double>();
        // calculate global flux
        std::fill(probabilities.begin(), probabilities.end(), 0.0);
        // no hop if net outward flux is too small
        if (net_out > -1e-14) return probabilities;
        for (const size_t & state : in_states) probabilities[state] = (pop_change[state] * pop_change[active_state_]).item<double>() / net_out;
    }
    return probabilities;
}

GFSH::GFSH() {}

GFSH::GFSH(const int64_t & _NStates, const at::Tensor & _mass,
at::Tensor (*_compute_Hd)(const at::Tensor &),
std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &))
: Base(_NStates, _mass, _compute_Hd, _compute_Hd_dHd) {}

GFSH::~GFSH() {}

// perform a surface hopping loop with time step dt
void GFSH::step(const double & dt) {
    // Propagate nuclear coordinate and momentum
    propagate_nucleus(dt);
    // Propagate electronic wave function
    propagate_electron(dt);
    // Switch active surface by hopping probability and momentum rescaling
    std::vector<double> probabilities = compute_probabilities(dt);
    int64_t target_state = determine_target(probabilities);
    rescale_momentum(target_state);
    // Prepare for the next loop
    prepare_next();
}
// specialization for 1-dimensional case
void GFSH::step_1D(const double & dt) {
    // Propagate nuclear coordinate and momentum
    propagate_nucleus(dt);
    // Propagate electronic wave function
    propagate_electron(dt);
    // Switch active surface by hopping probability and momentum rescaling
    std::vector<double> probabilities = compute_probabilities(dt);
    int64_t target_state = determine_target(probabilities);
    rescale_scalar(target_state);
    // Prepare for the next loop
    prepare_next();
}

} // namespace hop