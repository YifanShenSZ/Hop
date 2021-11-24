#ifndef hop_MFSH_hpp
#define hop_MFSH_hpp

#include <hop/base.hpp>

namespace hop {

// Minimum flux surface hopping
class MFSH : public Base {
    protected:
        std::vector<double> compute_probabilities(const double & dt) const;
    public:
        MFSH();
        MFSH(const int64_t & _NStates, const at::Tensor & _mass,
        at::Tensor (*_compute_Hd)(const at::Tensor &),
        std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &));
        ~MFSH();

        // Perform a surface hopping loop with time step dt
        void step(const double & dt);
        // Specialization for 1-dimensional case
        void step_1D(const double & dt);
};

} // namespace hop

#endif