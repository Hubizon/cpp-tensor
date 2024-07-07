#ifndef CPPTENSOR_INCLUDE_LOSSES_HPP_
#define CPPTENSOR_INCLUDE_LOSSES_HPP_

#include "Tensor.hpp"

namespace cpp_tensor {

class Loss {
 public:
  // Loss computation
  virtual Tensor Compute(const Tensor &pred, const Tensor &target) const = 0;

  // Overloaded call operator
  Tensor operator()(const Tensor &pred, const Tensor &target) const { return Compute(pred, target); }
};

class MSELoss : public Loss {
 public:
  // Enumeration for specifying the reduction method to apply to the output
  enum Reduction { MEAN, SUM };

  // Constructor
  MSELoss(Reduction reduction = MEAN) : reduction_(reduction) {}

  // Loss computation
  virtual Tensor Compute(const Tensor &pred, const Tensor &target) const override;

 private:
  // The reduction method to apply to the output
  Reduction reduction_;
};

// TODO: create BCELoss

}

#endif // CPPTENSOR_INCLUDE_LOSSES_HPP_