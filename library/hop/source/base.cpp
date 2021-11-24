#include <tchem/linalg.hpp>

#include <hop/base.hpp>

namespace hop {

Base::Base() {}

Base::Base(const int64_t & _NStates, const at::Tensor & _mass,
at::Tensor (*_compute_Hd)(const at::Tensor &),
std::tuple<at::Tensor, at::Tensor> (*_compute_Hd_dHd)(const at::Tensor &))
: NStates_(_NStates), mass_(_mass.clone()),
compute_Hd_(_compute_Hd), compute_Hd_dHd_(_compute_Hd_dHd) {
    x_ = _mass.new_empty(_mass.sizes());
    p_ = _mass.new_empty(_mass.sizes());
    c_ = _mass.new_empty({_NStates, 2});
    w_ = _mass.new_empty({_NStates, 2});
    energy_ = _mass.new_empty(_NStates);
    states_ = _mass.new_empty({_NStates, _NStates});
    dHa_    = _mass.new_empty({_NStates, _NStates, _mass.size(0)});
}

Base::~Base() {}

const int64_t & Base::active_state() const {return active_state_;}
const at::Tensor & Base::x() const {return x_;}
const at::Tensor & Base::p() const {return p_;}
// diabatic wave function
const at::Tensor & Base::c() const {return c_;}
// adiabatic wave function
const at::Tensor & Base::w() const {return w_;}

// set initial condition: active state and all old variables
// `_c`: real diabatic wave function
void Base::initialize(const int64_t & _active_state, const at::Tensor & _x, const at::Tensor & _p, const at::Tensor & _c) {
    active_state_ = _active_state;
    x_old_ = _x.clone();
    p_old_ = _p.clone();
    // adiabatic representation
    at::Tensor Hd, dHd;
    std::tie(Hd, dHd) = compute_Hd_dHd_(_x);
    std::tie(energy_old_, states_old_) = Hd.symeig(true);
    dHa_old_ = tchem::linalg::UT_sy_U(dHd, states_old_);
    // electronic wave function
    c_old_ = _x.new_zeros({NStates_, 2});
    c_old_.select(1, 0).copy_(_c);
    w_old_ = states_old_.transpose(0, 1).mm(c_old_);
}
// initial electronic wave function is set to adiabatic_states[_active_state]
void Base::initialize(const int64_t & _active_state, const at::Tensor & _x, const at::Tensor & _p) {
    active_state_ = _active_state;
    x_old_ = _x.clone();
    p_old_ = _p.clone();
    // adiabatic representation
    at::Tensor Hd, dHd;
    std::tie(Hd, dHd) = compute_Hd_dHd_(_x);
    std::tie(energy_old_, states_old_) = Hd.symeig(true);
    dHa_old_ = tchem::linalg::UT_sy_U(dHd, states_old_);
    // electronic wave function
    w_old_ = _x.new_zeros({NStates_, 2});
    w_old_[_active_state][0].fill_(1.0);
    c_old_ = _x.new_zeros({NStates_, 2});
    c_old_.select(1, 0).copy_(states_old_.select(1, _active_state));
}

// propagate nuclear coordinate and momentum with velocity verlet
// update kinetic_energy_, energy_, states_, dHa_ as well
void Base::propagate_nucleus(const double & dt) {
    double dtd2 = dt / 2.0;
    // determine new coordinate from old momentum and old force
    at::Tensor f_old = -dHa_old_[active_state_][active_state_];
    x_.copy_(x_old_ + (p_old_ + dtd2 * f_old) / mass_ * dt);
    // determine new momentum from old and new forces
    at::Tensor Hd, dHd;
    std::tie(Hd, dHd) = compute_Hd_dHd_(x_);
    std::tie(energy_, states_) = Hd.symeig(true);
    dHa_.copy_(tchem::linalg::UT_sy_U(dHd, states_));
    at::Tensor f_new = -dHa_[active_state_][active_state_];
    p_.copy_(p_old_ + (f_old + f_new) * dtd2);
    // compute kinetic energy
    kinetic_energy_ = (p_ / mass_).dot(p_).item<double>() / 2.0;
}

// propagate electronic wave function with exact propagator
void Base::propagate_electron(const double & dt) {
    // nuclear propagation does not change diabatic wave function,
    // but does change adiabatic wave function
    w_.copy_(states_.transpose(0, 1).mm(c_old_));
    // Propagate adiabatic wave function
    const double * p2e = energy_.data_ptr<double>();
    for (int64_t i = 0; i < NStates_; i++) {
        double w_real = w_[i][0].item<double>(),
               w_imag = w_[i][1].item<double>();
        double angle = -p2e[i] * dt;
        double real = cos(angle), imag = sin(angle);
        w_[i][0].fill_(w_real * real - w_imag * imag);
        w_[i][1].fill_(w_real * imag + w_imag * real);
    }
    // Convert back to diabatic representation
    c_.copy_(states_.mm(w_));
}
// propagate electronic wave function with phase correction
// N. Shenvi, J. Subotnik, W. Yang 2011 JCP
void Base::propagate_PC(const double & dt) {
    // Nuclear propagation does not change diabatic wave function,
    // but does change adiabatic wave function
    w_.copy_(states_.transpose(0, 1).mm(c_old_));
    // Propagate adiabatic wave function
    const double * p2e = energy_.data_ptr<double>();
    for (int64_t i = 0; i < NStates_; i++)
    // We set such a phase factor that leaves the active state component unchanged
    if (i != active_state_) {
        // inactive state kinetic energy
        double T_inactive = kinetic_energy_ + p2e[active_state_] - p2e[i];
        // Let p_inactive = ratio * p
        double ratio = 0.0;
        if (T_inactive > 0.0) ratio = sqrt(T_inactive / kinetic_energy_);
        // inactive state phase is exp(i k . (v_inactive - v) * dt) ahead than active state
        double w_real = w_[i][0].item<double>(),
               w_imag = w_[i][1].item<double>();
        double angle = 2.0 * kinetic_energy_ * (ratio - 1.0) * dt;
        double real = cos(angle), imag = sin(angle);
        w_[i][0].fill_(w_real * real - w_imag * imag);
        w_[i][1].fill_(w_real * imag + w_imag * real);
    }
    // Convert back to diabatic representation
    c_.copy_(states_.mm(w_));
}

// determine target state to hop to
// return current active state if no hopping occurs
int64_t Base::determine_target(const std::vector<double> & probabilities) const {
    double random_number = (double)rand() / RAND_MAX;
    for (size_t i = 0; i < NStates_; i++)
    if (random_number < probabilities[i]) return i;
    else random_number -= probabilities[i];
    return active_state_;
}

// rescale momentum to conserve energy, then set active state to target state
// the momentum change is along nonadiabatic coupling
// true = normal termination, false = frustrated hop
bool Base::rescale_momentum(const int64_t & target_state) {
    // quick return if no hopping
    if (target_state == active_state_) return true;
    // let the momentum change direction be n, magnitude be r:
    //     (p + r n) / 2m . (p + r n) + e[target] = p / 2m . p + e[active]
    // so r satisfies:
    //     n / 2m . n r^2 + p / m . n r + e[target] - e[active] = 0
    const double * p2e = energy_.data_ptr<double>();
    double C = p2e[target_state] - p2e[active_state_];
    if (kinetic_energy_ > C) {
        // here n is chosen to be interstate coupling
        // it is parallel to nonadiabatic coupling and much more stable
        at::Tensor n;
        if (target_state > active_state_) n = dHa_[active_state_][target_state ];
        else                              n = dHa_[target_state ][active_state_];
        double A = (n / mass_).dot(n).item<double>() / 2.0,
               B = (p_ / mass_).dot(n).item<double>();
        double discriminant = B * B - 4.0 * A * C;
        // adjust along n cannot satisfy energy conservation
        if (discriminant < 0.0) return false;
        double sqrt_d = sqrt(discriminant);
        at::Tensor p1 = p_ + (-B + sqrt_d) / (2.0 * A) * n,
                   p2 = p_ + (-B - sqrt_d) / (2.0 * A) * n;
        // choose the one with larger projection along the original direction
        if (p1.dot(p_).item<double>() > p2.dot(p_).item<double>()) p_.copy_(p1);
        else                                                       p_.copy_(p2);
        active_state_ = target_state;
        return true;
    }
    // insufficient kinetic energy, hop frustrated
    else return false;
}
// the momentum stays along the original direction
bool Base::rescale_scalar(const int64_t & target_state) {
    // quick return if no hopping
    if (target_state == active_state_) return true;
    // new kinetic energy
    const double * p2e = energy_.data_ptr<double>();
    double T_new = kinetic_energy_ + p2e[active_state_] - p2e[target_state];
    // let p_new = ratio * p
    if (T_new > 0.0) {
        p_ *= sqrt(T_new / kinetic_energy_);
        active_state_ = target_state;
        return true;
    }
    // insufficient kinetic energy, hop frustrated
    else return false;
}

// update all old variables for the next loop
void Base::prepare_next() {
    x_old_.copy_(x_);
    p_old_.copy_(p_);
    w_old_.copy_(w_);
    c_old_.copy_(c_);
    energy_old_.copy_(energy_);
    states_old_.copy_(states_);
    dHa_old_.copy_(dHa_);
}

} // namespace hop