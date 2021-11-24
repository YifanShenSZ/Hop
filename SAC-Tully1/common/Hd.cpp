#include "Hd.hpp"

at::Tensor compute_Hd(const at::Tensor & _x) {
    const double A = 0.01, B = 1.6, C = 0.005, D = 1.0;
    double x = _x.item<double>();
    at::Tensor Hd = _x.new_empty({2, 2});
    double Hd00;
    if (x > 0.0) Hd00 =  A * (1.0 - exp(-B * x));
    else         Hd00 = -A * (1.0 - exp( B * x));
    Hd[0][0].fill_( Hd00);
    Hd[1][1].fill_(-Hd00);
    Hd[0][1].fill_(C * exp(-D * x * x));
    return Hd;
}

std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & _x) {
    const double A = 0.01, B = 1.6, C = 0.005, D = 1.0;
    double x = _x.item<double>();
    // Hd
    at::Tensor Hd = _x.new_empty({2, 2});
    double Hd00;
    if (x > 0.0) Hd00 =  A * (1.0 - exp(-B * x));
    else         Hd00 = -A * (1.0 - exp( B * x));
    Hd[0][0].fill_( Hd00);
    Hd[1][1].fill_(-Hd00);
    Hd[0][1].fill_(C * exp(-D * x * x));
    // ▽Hd
    at::Tensor dHd = _x.new_empty({2, 2, 1});
    double dHd00;
    if (x > 0.0) dHd00 = A * B * exp(-B * x);
    else         dHd00 = A * B * exp( B * x);
    dHd[0][0].fill_( dHd00);
    dHd[1][1].fill_(-dHd00);
    dHd[0][1].fill_(-2.0 * C * D * x * exp(-D * x * x));
    return std::make_tuple(Hd, dHd);
}