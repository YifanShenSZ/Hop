#ifndef hop_GFSH_hpp
#define hop_GFSH_hpp

#include <hop/base.hpp>

namespace hop {

// global flux surface hopping
// L. Wang, D. Trivedi, O. Prezhdo 2014 JCTC
class GFSH : public Base {
    protected:
        std::vector<double> compute_probabilities(const double & dt) const;
    public:
        GFSH();
        GFSH(const int64_t & _NStates, const at::Tensor & _mass,
        at::Tensor (*_compute_Hd)(const at::Tensor &),
        std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &));
        ~GFSH();

        // perform a surface hopping loop with time step dt
        void step(const double & dt);
        // specialization for 1-dimensional case
        void step_1D(const double & dt);
};

} // namespace hop

#endif