#include <random>

#include "Initializations.hpp"

namespace cpp_tensor {

// Static factory methods for different initialization

Initialization Initialization::Uniform(double a, double b) {
  return Initialization([a, b](const std::vector<size_t> &shape) -> Tensor {
    size_t size = 1;
    for (auto &kS : shape)
      size *= kS;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(a, b);
    std::vector<double> values;

    for (int i = 0; i < size; i++)
      values.push_back(dist(mt));
    return Tensor(values, shape, true);
  });
}

Initialization Initialization::Normal(double mean, double std) {
  return Initialization([mean, std](const std::vector<size_t> &shape) -> Tensor {
    size_t size = 1;
    for (auto &kS : shape)
      size *= kS;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<double> dist(mean, std);

    std::vector<double> values;
    for (int i = 0; i < size; i++)
      values.push_back(dist(mt));
    return Tensor(values, shape, true);
  });
}

Initialization Initialization::Constant(double val) {
  return {[val](const std::vector<size_t> &shape) -> Tensor {
    size_t size = 1;
    for (auto &kS : shape)
      size *= kS;

    std::vector<double> values;
    for (int i = 0; i < size; i++)
      values.push_back(val);
    return Tensor(values, shape, true);
  }};
}

}