#include <tchem/intcoord.hpp>

#include <initial/harmonic.hpp>

int main() {
    // provide a water molecule
    std::vector<double> mass = {16.0 * 1822.888486192, 1822.888486192, 1822.888486192};
    at::Tensor r = at::tensor({0.0000000,    0.0000000,    0.0000000,
                               0.0000000,   -0.7493682,   -0.6000000,
                               0.0000000,    0.7493682,   -0.6000000}),
               intgrad = at::tensor({0.0, 0.0, 0.0}),
               inthess = at::tensor({1.0, 0.1, 0.0,
                                     0.1, 0.6, 0.0,
                                     0.0, 0.0, 10.0}).reshape({3, 3});
    std::ofstream ofs; ofs.open("IntCoordDef");
    ofs << "     1    1.000000    stretching     1     2\n"
        << "          1.000000    stretching     1     3\n"
        << "     2    1.000000       bending     2     1     3\n"
        << "     3    1.000000    stretching     1     2\n"
        << "         -1.000000    stretching     1     3\n";
    ofs.close();
    tchem::IC::IntCoordSet intcoordset("default", "IntCoordDef");
    at::Tensor carthess = intcoordset.Hessian_int2cart(r, intgrad, inthess);
    // compare sampling covariance with analytical result
    initial::Harmonic harmonic(r, mass, carthess);
    at::Tensor Exx = r.new_zeros({9, 9}),
               Epp = r.new_zeros({9, 9});
    size_t NAvg = 10000;
    for (size_t i = 0; i < 10000; i++) {
        at::Tensor x, p;
        std::tie(x, p) = harmonic();
        x -= r;
        x = harmonic.sqrt_mass().mv(x);
        p = harmonic.invsqrt_mass().mv(p);
        Exx += (x).outer(x);
        Epp += p.outer(p);
    }
    Exx /= (double)NAvg;
    Epp /= (double)NAvg;
    at::Tensor eigval_xx, eigvec_xx, eigval_pp, eigvec_pp;
    std::tie(eigval_xx, eigvec_xx) = Exx.symeig();
    std::tie(eigval_pp, eigvec_pp) = Epp.symeig();
    std::cerr << (eigval_xx[6] * eigval_pp[8]).item<double>() - 0.25 << ' '
              << (eigval_xx[7] * eigval_pp[7]).item<double>() - 0.25 << ' '
              << (eigval_xx[8] * eigval_pp[6]).item<double>() - 0.25 << '\n';
}