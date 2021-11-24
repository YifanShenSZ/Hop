#ifndef hop_PCFSSH_hpp
#define hop_PCFSSH_hpp

#include <hop/FSSH.hpp>

namespace hop {

// phase corrected fewest switches surface hopping
class PCFSSH : public FSSH {
    public:
        PCFSSH();
        PCFSSH(const int64_t & _NStates, const at::Tensor & _mass,
        at::Tensor (*_compute_Hd)(const at::Tensor &),
        std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &));
        ~PCFSSH();

        // perform a surface hopping loop with time step dt
        void step(const double & dt);
        // specialization for 1-dimensional case
        void step_1D(const double & dt);
};

} // namespace hop

#endif