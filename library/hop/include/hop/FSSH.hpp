#ifndef hop_FSSH_hpp
#define hop_FSSH_hpp

#include <hop/base.hpp>

namespace hop {

// fewest switches surface hopping
// J. Tully 1990 JCP
class FSSH : public Base {
    protected:
        std::vector<double> compute_probabilities(const double & dt) const;
    public:
        FSSH();
        FSSH(const int64_t & _NStates, const at::Tensor & _mass,
        at::Tensor (*_compute_Hd)(const at::Tensor &),
        std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &));
        ~FSSH();

        // perform a surface hopping loop with time step dt
        void step(const double & dt);
        // specialization for 1-dimensional case
        void step_1D(const double & dt);
};

} // namespace hop

#endif