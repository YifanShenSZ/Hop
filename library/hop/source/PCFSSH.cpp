#include <hop/PCFSSH.hpp>

namespace hop {

PCFSSH::PCFSSH() {}

PCFSSH::PCFSSH(const int64_t & _NStates, const at::Tensor & _mass,
at::Tensor (*_compute_Hd)(const at::Tensor &),
std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &))
: FSSH(_NStates, _mass, _compute_Hd, _compute_Hd_dHd) {}

PCFSSH::~PCFSSH() {}

// perform a surface hopping loop with time step dt
void PCFSSH::step(const double & dt) {
    // propagate nuclear coordinate and momentum
    propagate_nucleus(dt);
    // propagate electronic wave function
    propagate_PC(dt);
    // switch active surface by hopping probability and momentum rescaling
    std::vector<double> probabilities = compute_probabilities(dt);
    int64_t target_state = determine_target(probabilities);
    rescale_momentum(target_state);
    // prepare for the next loop
    prepare_next();
}
// specialization for 1-dimensional case
void PCFSSH::step_1D(const double & dt) {
    // propagate nuclear coordinate and momentum
    propagate_nucleus(dt);
    // propagate electronic wave function
    propagate_PC(dt);
    // switch active surface by hopping probability and momentum rescaling
    std::vector<double> probabilities = compute_probabilities(dt);
    int64_t target_state = determine_target(probabilities);
    rescale_scalar(target_state);
    // prepare for the next loop
    prepare_next();
}

} // namespace hop