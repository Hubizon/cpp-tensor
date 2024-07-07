#ifndef CPPTENSOR_INCLUDE_INITIALIZATIONS_HPP_
#define CPPTENSOR_INCLUDE_INITIALIZATIONS_HPP_

#include <functional>
#include <utility>
#include <vector>

#include "Tensor.hpp"

namespace cpp_tensor {

class Initialization {
 public:
  // Static factory methods for different initialization
  static Initialization Uniform(double a = -1, double b = 1);
  static Initialization Normal(double mean = 0, double std = 1);
  static Initialization Constant(double val = 0);
  // TODO: add he and xavier initializations

  // Constructor
  Initialization(std::function<Tensor(const std::vector<size_t> &)> init) : init_(std::move(init)) {};

  // Overloaded call operator to create tensors
  Tensor operator()(const std::vector<size_t> &shape) { return init_(shape); };

 private:
  // The initialization method used
  std::function<Tensor(const std::vector<size_t> &)> init_;
};

}

#endif // CPPTENSOR_INCLUDE_INITIALIZATIONS_HPP_