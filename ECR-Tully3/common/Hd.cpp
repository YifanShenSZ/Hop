#include "Hd.hpp"

at::Tensor compute_Hd(const at::Tensor & _x) {
    const double A = 6e-4, B = 0.1, C = 0.9;
    double x = _x.item<double>();
    at::Tensor Hd = _x.new_empty({2, 2});
    Hd[0][0].fill_( A);
    Hd[1][1].fill_(-A);
    if (x < 0.0) Hd[0][1].fill_( B * exp( C * x));
    else         Hd[0][1].fill_(-B * exp(-C * x) + 2.0 * B);
    return Hd;
}

std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & _x) {
    const double A = 6e-4, B = 0.1, C = 0.9;
    double x = _x.item<double>();
    // Hd
    at::Tensor Hd = _x.new_empty({2, 2});
    Hd[0][0].fill_( A);
    Hd[1][1].fill_(-A);
    if (x < 0.0) Hd[0][1].fill_( B * exp( C * x));
    else         Hd[0][1].fill_(-B * exp(-C * x) + 2.0 * B);
    // ▽Hd
    at::Tensor dHd = _x.new_empty({2, 2, 1});
    dHd[0][0].fill_(0.0);
    dHd[1][1].fill_(0.0);
    if (x < 0.0) dHd[0][1].fill_(B * C * exp( C * x));
    else         dHd[0][1].fill_(B * C * exp(-C * x));
    return std::make_tuple(Hd, dHd);
}