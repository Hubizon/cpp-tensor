#include "Losses.hpp"

namespace cpp_tensor {

// Loss computation
Tensor MSELoss::Compute(const Tensor &pred, const Tensor &target) const {
  switch (reduction_) {
    case SUM:return (pred - target).Pow(2).Sum();
    case MEAN:
    default:return (pred - target).Pow(2).Mean();
  }
}

}