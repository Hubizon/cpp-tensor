#include "Optimizers.hpp"
#include "Tensor.hpp"

namespace cpp_tensor {

// Optimizer operations

void SGD::Step() {
  Tensor::SetUseGrad(false);
  for (auto &p : parameters_)
    for (int i = 0; i < p->Size(); i++)
      p->Data(i) -= lr_ * p->Grad(i);
  Tensor::SetUseGrad(true);
}

void SGD::ZeroGrad() {
  for (auto &p : parameters_)
    p->SetGrad(0);
}

}