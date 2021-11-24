#ifndef hop_PCGFSH_hpp
#define hop_PCGFSH_hpp

#include <hop/GFSH.hpp>

namespace hop {

// phase corrected global flux surface hopping
class PCGFSH : public GFSH {
    public:
        PCGFSH();
        PCGFSH(const int64_t & _NStates, const at::Tensor & _mass,
        at::Tensor (*_compute_Hd)(const at::Tensor &),
        std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &));
        ~PCGFSH();

        // perform a surface hopping loop with time step dt
        void step(const double & dt);
        // specialization for 1-dimensional case
        void step_1D(const double & dt);
};

} // namespace hop

#endif