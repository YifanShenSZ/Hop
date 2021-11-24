#include "Hd.hpp"

at::Tensor compute_Hd(const at::Tensor & _x) {
    const double A = 0.1, B = 0.28, C = 0.015, D = 0.06, E0 = 0.05;
    double x = _x.item<double>();
    at::Tensor Hd = _x.new_empty({2, 2});
    Hd[0][0].fill_(0.0);
    Hd[1][1].fill_(-A * exp(-B * x * x) + E0);
    Hd[0][1].fill_( C * exp(-D * x * x));
    return Hd;
}

std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & _x) {
    const double A = 0.1, B = 0.28, C = 0.015, D = 0.06, E0 = 0.05;
    double x = _x.item<double>();
    // Hd
    at::Tensor Hd = _x.new_empty({2, 2});
    Hd[0][0].fill_(0.0);
    Hd[1][1].fill_(-A * exp(-B * x * x) + E0);
    Hd[0][1].fill_( C * exp(-D * x * x));
    // ▽Hd
    at::Tensor dHd = _x.new_empty({2, 2, 1});
    dHd[0][0].fill_(0.0);
    dHd[1][1].fill_( 2.0 * A * B * x * exp(-B * x * x));
    dHd[0][1].fill_(-2.0 * C * D * x * exp(-D * x * x));
    return std::make_tuple(Hd, dHd);
}