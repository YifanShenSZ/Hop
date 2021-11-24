#ifndef Hd_hpp
#define Hd_hpp

#include <Hd/kernel.hpp>

extern std::shared_ptr<Hd::kernel> HdKernel;

at::Tensor compute_Hd(const at::Tensor & x);

std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & x);

// here ddHa is ▽[(▽H)a], computed by finite difference of (▽H)a
at::Tensor compute_ddHa(const at::Tensor & r);

#endif