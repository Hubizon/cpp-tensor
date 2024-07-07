#ifndef CPPTENSOR_INCLUDE_MODULES_HPP_
#define CPPTENSOR_INCLUDE_MODULES_HPP_

#include <vector>
#include <memory>

#include "Tensor.hpp"
#include "Initializations.hpp"

namespace cpp_tensor {

class Module {
 public:
  // Retrieve Parameters
  virtual std::vector<SharedTensor> Parameters() const & = 0;

  // Perform Forward pass
  virtual Tensor Forward(const Tensor &x) const & = 0;

  // Overloaded call operator
  Tensor operator()(const Tensor &x) const &{ return Forward(x); }
};

class LinearLayer : public Module {
 public:
  // Constructor
  LinearLayer(size_t in_features,
              size_t out_features,
              Initialization init = Initialization::Uniform(),
              bool is_bias = true);

  // Overloaded virtual methods
  virtual std::vector<SharedTensor> Parameters() const & override;
  virtual Tensor Forward(const Tensor &x) const & override;

 private:
  // Member variables
  Tensor weight_;
  Tensor bias_;
  size_t in_features_;
  size_t out_features_;
  bool is_bias_;
};

class ReLU : public Module {
 public:
  // Constructor
  explicit ReLU(double leaky = 0) : leaky_(leaky) {}

  // Overloaded virtual methods
  virtual std::vector<SharedTensor> Parameters() const & override { return {}; }
  virtual Tensor Forward(const Tensor &x) const & override { return x.Relu(leaky_); }

 private:
  // For standard ReLU, this parameter should be 0; for LeakyReLU, it specifies the negative slope
  double leaky_;
};

class Sequential : public Module {
 public:
  // Overloaded virtual methods
  virtual std::vector<SharedTensor> Parameters() const & override;
  virtual Tensor Forward(const Tensor &x) const & override;

  // Adds a module to the sequential model using the provided arguments.
  template<typename T, typename... Args>
  void AddModule(Args &&... args) {
    const auto kModule = (Module *) (new T(std::forward<Args>(args)...));
    modules_.push_back(std::unique_ptr<Module>(kModule));
  }

 private:
  // List of modules in the sequential model
  std::vector<std::unique_ptr<Module>> modules_;
};

}

#endif // CPPTENSOR_INCLUDE_MODULES_HPP_