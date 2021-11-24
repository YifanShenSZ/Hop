#ifndef hop_base_hpp
#define hop_base_hpp

#include <torch/torch.h>

namespace hop {

// the base class of all hoppers
// the surface hopping loop consists of:
// 1. propagate nuclear coordinate and momentum
// 2. propagate electronic wave function
// 3. switch active surface by hopping probability and momentum rescaling
// 4. decoherence if necessary
// 5. prepare for the next loop
class Base {
    protected:
        // basic information
        int64_t NStates_;
        at::Tensor mass_;
        // standard interface for Hd and ▽Hd
        at::Tensor (*compute_Hd_)(const at::Tensor &);
        std::tuple<at::Tensor, at::Tensor> (*compute_Hd_dHd_)(const at::Tensor &);

        // dynamics variables
        int64_t active_state_;
        at::Tensor x_, p_,
                   c_, // diabatic wave function
                   w_; // adiabatic wave function

        // derived variables
        double kinetic_energy_;
        at::Tensor energy_, states_, dHa_;

        // old variables
        at::Tensor x_old_, p_old_, w_old_, c_old_,
                   energy_old_, states_old_, dHa_old_;
    public:
        Base();
        Base(const int64_t & _NStates, const at::Tensor & _mass,
        at::Tensor (*_compute_Hd)(const at::Tensor &),
        std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &));
        ~Base();

        const int64_t & active_state() const;
        const at::Tensor & x() const;
        const at::Tensor & p() const;
        // diabatic wave function
        const at::Tensor & c() const;
        // adiabatic wave function
        const at::Tensor & w() const;

        // set initial condition: active state and all old variables
        // `_c`: real diabatic wave function
        void initialize(const int64_t & _active_state, const at::Tensor & _x, const at::Tensor & _p, const at::Tensor & _c);
        // initial electronic wave function is set to adiabatic_states[_active_state]
        void initialize(const int64_t & _active_state, const at::Tensor & _x, const at::Tensor & _p);

        // propagate nuclear coordinate and momentum with velocity verlet
        void propagate_nucleus(const double & dt);

        // propagate electronic wave function with exact propagator
        void propagate_electron(const double & dt);
        // propagate electronic wave function with phase correction
        // N. Shenvi, J. Subotnik, W. Yang 2011 JCP
        void propagate_PC(const double & dt);

        // determine target state to hop to
        // return current active state if no hopping occurs
        int64_t determine_target(const std::vector<double> & probabilities) const;

        // rescale momentum to conserve energy, then set active state to target state
        // return whether momentum has been rescaled successfully
        // the momentum change is along nonadiabatic coupling
        bool rescale_momentum(const int64_t & target_state);
        // the momentum stays along the original direction
        bool rescale_scalar(const int64_t & target_state);

        // update all old variables for the next loop
        void prepare_next();
};

} // namespace hop

#endif