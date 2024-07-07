#ifndef CPPTENSOR_INCLUDE_OPTIMIZERS_HPP_
#define CPPTENSOR_INCLUDE_OPTIMIZERS_HPP_

#include <utility>
#include <vector>

#include "InternalTensor.hpp"

namespace cpp_tensor {

class SGD {
 public:
  // Constructor
  SGD(std::vector<SharedTensor> parameters, double lr) : parameters_(std::move(parameters)), lr_(lr) {}

  // Optimizer operations
  void Step();
  void ZeroGrad();

 private:
  // Member variables
  std::vector<SharedTensor> parameters_;
  double lr_;
};

// todo: create Adam optimizer

}

#endif // CPPTENSOR_INCLUDE_OPTIMIZERS_HPP_