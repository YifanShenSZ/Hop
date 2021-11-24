#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>
#include <CppLibrary/chemistry.hpp>

#include <hop/FSSH.hpp>

#include <initial/harmonic.hpp>

#include "../common/Hd.hpp"
#include "../common/initer.hpp"

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("C6H7N-aniline: aniline photodissociation");

    // required arguments
    parser.add_argument("-d","--diabatz", '+', false, "diabatz definition files");
    parser.add_argument("-x","--xyz",       1, false, "initial xyz geometry");
    parser.add_argument("-m","--mass",      1, false, "the masses of atoms");

    // optional arguments
    parser.add_argument("-t","--time_step",   1, true, "time step in fs, default = 0.1");
    parser.add_argument("-o","--output_step", 1, true, "output step in fs, default = 1");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "C6H7N-aniline: aniline photodissociation\n"
              << "Yifan Shen 2021\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::vector<std::string> diabatz_inputs = args.retrieve<std::vector<std::string>>("diabatz");
    HdKernel = std::make_shared<Hd::kernel>(diabatz_inputs);

    CL::chem::xyz<double> geom(args.retrieve<std::string>("xyz"), true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), at::TensorOptions().dtype(torch::kFloat64));
    auto mass = CL::utility::read_vector(args.retrieve<std::string>("mass"));
    for (double & atom : mass) atom *= 1822.888486192;
    at::Tensor Hessian = compute_ddHa(r)[0][0];
    Initer initer(r, mass, Hessian);

    at::Tensor mass_vector = r.new_empty(r.sizes());
    for (size_t i = 0; i < mass.size(); i++) mass_vector.slice(0, 3 * i, 3 * i + 3).fill_(mass[i]);
    hop::FSSH hopper(HdKernel->NStates(), mass_vector, compute_Hd, compute_Hd_dHd);

    double dt = 0.1 * 41.341373336561354;
    if (args.gotArgument("time_step")) dt = args.retrieve<double>("time_step") * 41.341373336561354;
    double output_step = 1.0 * 41.341373336561354;
    if (args.gotArgument("output_step")) output_step = args.retrieve<double>("output_step") * 41.341373336561354;
    size_t output_freq = output_step / dt;

    // initial condition
    size_t active_state;
    at::Tensor x, p;
    std::tie(active_state, x, p) = initer();
std::cerr << "got initial condition\n";
    hopper.initialize(active_state, x, p, at::tensor({0.0, 0.0, 1.0, 0.0}));
std::cerr << "pass initialize\n";
    // output preparation
    size_t step_count = 0;
    auto symbols = geom.symbols();
std::cerr << "pass symbols\n";
    std::ofstream traj_ofs, state_ofs;
    traj_ofs.open("trajectory.txt");
    state_ofs.open("state.txt");
    // main loop
    while (true) {
        // output every `output_freq` steps
        if (step_count % output_freq == 0) {
            // output geometry
            const at::Tensor & r = hopper.x();
std::cerr << r << '\n';
            auto coords = geom.coords();
            for (size_t i = 0; i < coords.size(); i++) coords[i] = r[i].item<double>();
std::cerr << "pass coords\n";
            CL::chem::xyz<double> out_geom(symbols, coords, true);
            out_geom.print(traj_ofs);
            // output active state
            state_ofs << hopper.active_state() << '\n';
        }
        // propagate
        hopper.step(dt);
        step_count++;
        // stop when one of the N-H bonds > 3 A
        const at::Tensor & r = hopper.x();
        at::Tensor NH1 = r.slice(0, 21, 24) - r.slice(0, 0, 3),
                   NH2 = r.slice(0, 24, 27) - r.slice(0, 0, 3);
        double max_NH = std::max(NH1.norm().item<double>(), NH2.norm().item<double>());
        if (max_NH > 3 * 1.8897261339212517) break;
    }
    traj_ofs.close();
    state_ofs.close();

    std::cout << '\n';
    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}
