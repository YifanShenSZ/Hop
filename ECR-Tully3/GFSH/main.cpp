#include <ctime>
#include <random>

#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include <hop/GFSH.hpp>

#include "../common/Hd.hpp"

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("Extended coupling with reflection - Tully 3rd model by global flux surface hopping");

    // required arguments
    parser.add_argument("-p","--p", 1, false, "initial momemtum");

    // optional arguments
    parser.add_argument("-n","--num_traj",  1, true, "number of trajectories, default = 10000");
    parser.add_argument("-t","--time_step", 1, true, "time step in fs, default = 1");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Extended coupling with reflection - Tully 3rd model by global flux surface hopping\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    srand(time(NULL));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    at::Tensor mass = at::tensor({2000.0});
    hop::GFSH hopper(2, mass, compute_Hd, compute_Hd_dHd);

    double miu_p = args.retrieve<double>("p");
    double miu_x = -10.0 - 50.0 / miu_p,
           sigma_x = 10.0 / miu_p,
           sigma_p = miu_p / 20.0;
    std::normal_distribution<double> wigner_x(miu_x, sigma_x),
                                     wigner_p(miu_p, sigma_p);

    size_t NTrajectories = 10000;
    if (args.gotArgument("num_traj")) NTrajectories = args.retrieve<size_t>("num_traj");
    double dt = 1.0;
    if (args.gotArgument("time_step")) dt = args.retrieve<double>("time_step");
    dt *= 41.341373336561354; // fs -> atomic unit

    size_t T1 = 0, R1 = 0, T2 = 0, R2 = 0;
    for (size_t i = 0; i < NTrajectories; i++) {
        at::Tensor x = at::tensor({wigner_x(generator)}),
                   p = at::tensor({wigner_p(generator)});
        hopper.initialize(0, x, p);
        while (true) {
            hopper.step_1D(dt);
            if (std::abs(hopper.x().item<double>()) > 30.0) break;
        }
        if (hopper.active_state() == 0) {
            if (hopper.x().item<double>() > 0.0) T1++;
            else R1++;
        }
        else {
            if (hopper.x().item<double>() > 0.0) T2++;
            else R2++;
        }
    }

    std::cout << "T1 = " << (double)T1 / NTrajectories << '\n'
              << "R1 = " << (double)R1 / NTrajectories << '\n'
              << "T2 = " << (double)T2 / NTrajectories << '\n'
              << "R2 = " << (double)R2 / NTrajectories << '\n';

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}