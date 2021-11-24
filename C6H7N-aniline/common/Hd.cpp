#include <tchem/linalg.hpp>

#include "Hd.hpp"

std::shared_ptr<Hd::kernel> HdKernel;

at::Tensor compute_Hd(const at::Tensor & x) {
    return (*HdKernel)(x);
}

std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & x) {
    return HdKernel->compute_Hd_dHd(x);
}

// here ddHa is ▽[(▽H)a], computed by finite difference of (▽H)a
at::Tensor compute_ddHa(const at::Tensor & r) {
    const double dr = 1e-3;
    std::vector<at::Tensor> plus(r.size(0)), minus(r.size(0));
    #pragma omp parallel for
    for (size_t i = 0; i < r.size(0); i++) {
        at::Tensor Hd, dHd, energy, states;
        plus[i] = r.clone();
        plus[i][i] += dr;
        std::tie(Hd, dHd) = compute_Hd_dHd(plus[i]);
        std::tie(energy, states) = Hd.symeig(true);
        plus[i] = tchem::linalg::UT_sy_U(dHd, states);
        minus[i] = r.clone();
        minus[i][i] -= dr;
        std::tie(Hd, dHd) = compute_Hd_dHd(minus[i]);
        std::tie(energy, states) = Hd.symeig(true);
        minus[i] = tchem::linalg::UT_sy_U(dHd, states);
    }
    at::Tensor ddHa = r.new_empty({plus[0].size(0), plus[0].size(1), r.size(0), r.size(0)});
    for (size_t i = 0; i < r.size(0); i++) ddHa.select(2, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    return ddHa;
}