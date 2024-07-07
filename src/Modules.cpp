#include "InternalTensor.hpp"
#include "Modules.hpp"
#include "Tensor.hpp"

namespace cpp_tensor {

// LinearLayer - Constructor

LinearLayer::LinearLayer(size_t in_features, size_t out_features, Initialization init, bool is_bias)
    : in_features_(in_features), out_features_(out_features), is_bias_(is_bias) {
  weight_ = init({in_features, out_features});
  if (is_bias) bias_ = init({out_features});
}

// LinearLayer - Overloaded virtual methods

Tensor LinearLayer::Forward(const Tensor &x) const &{
  if (x.Size() == in_features_) {  // unbatched
    auto res = x.Clone(false).Reshape({1, in_features_}).Matmul(weight_).Flatten();
    if (is_bias_) res = res + bias_;
    return res;
  } else {  // batched
    auto res = x.Matmul(weight_);
    if (is_bias_) res = res + bias_;
    return res;
  }
}

std::vector<SharedTensor> LinearLayer::Parameters() const &{
  std::vector<SharedTensor> parameters = {weight_.GetTensor()};
  if (is_bias_) parameters.push_back(bias_.GetTensor());
  return parameters;
}

// Sequential - Overloaded virtual methods

std::vector<SharedTensor> Sequential::Parameters() const &{
  std::vector<SharedTensor> parameters;
  for (auto &kModule : modules_) {
    auto module_param = kModule->Parameters();
    parameters.insert(parameters.end(), module_param.begin(), module_param.end());
  }
  return parameters;
}

Tensor Sequential::Forward(const Tensor &x) const &{
  Tensor res = x.Clone(false);
  for (auto &kModule : modules_)
    res = kModule->Forward(res);
  return res;
}

}