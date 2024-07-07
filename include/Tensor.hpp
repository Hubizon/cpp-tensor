#ifndef CPPTENSOR_INCLUDE_TENSOR_HPP_
#define CPPTENSOR_INCLUDE_TENSOR_HPP_

#include <functional>
#include <memory>
#include <vector>
#include <array>

#include "InternalTensor.hpp"

namespace cpp_tensor {

class Tensor {
 public:
  // Constructors
  Tensor(double value = 0, bool requires_grad = false); // for a 0D tensor
  Tensor(std::vector<double> values, bool requires_grad = false); // for a 1D tensor
  Tensor(std::vector<double> values, std::vector<size_t> shape, bool requires_grad = false); // specified Shape
  Tensor(double value, std::vector<size_t> shape, bool requires_grad = false); // specified Shape filled with value
  Tensor(SharedTensor &&tensor); // for internal use

  // Static functions
  static void SetUseGrad(bool use_grad) { InternalTensor::use_grad_ = use_grad; }
  static Tensor Concat(const std::vector<Tensor> &tensors);
  static std::array<Tensor, 4> TrainTestSplit(const Tensor &x, const Tensor &y, double ratio);

  // Data access
  SharedTensor GetTensor() const { return tensor_; };
  // Value(indices): If fewer indices are provided than the number of dimensions,
  // the remaining dimensions are assumed to be zero.
  double Value(const std::vector<int> &indices = {}) const;
  // ValueTensor(indices): If fewer indices are provided than the number of dimensions,
  // the returned tensor contains all the Data in the remaining dimensions and has the corresponding Shape.
  Tensor ValueTensor(const std::vector<int> &indices) const;

  // Shape information
  std::vector<size_t> Shape() const { return tensor_->shape_; }
  size_t Shape(int index) const { return tensor_->shape_[index]; }
  size_t Size() const { return tensor_->data_.size(); }
  size_t NumDimensions() const { return tensor_->shape_.size(); }

  // Tensor operations
  void Backward(bool retain_graph = false);
  Tensor Reshape(std::vector<size_t> new_shape);
  Tensor Flatten();
  Tensor Clone(bool deep_copy = true) const &;

  // Mathematical operations
  Tensor operator+(const Tensor &other) const &;
  Tensor operator-(const Tensor &other) const &;
  Tensor operator*(const Tensor &other) const &;
  Tensor operator/(const Tensor &other) const &;
  Tensor Matmul(const Tensor &other) const &;
  Tensor Pow(int exponent) const &;
  Tensor Sum() const &;
  Tensor Mean() const &;

  // Activation functions
  Tensor Relu(double leaky) const &;

  // Indexing operator - returns the Data at the specified index in the 1D representation of the tensor
  double operator[](int index) const { return tensor_->data_[index]; }

 private:
  // Helper function - the strides are used to determine the position of elements in the
  // 1D representation of the tensor, primarily for use in the ValueTensor() function.
  void CalculateStrides();

  // Member variables
  SharedTensor tensor_;
  std::vector<size_t> strides_;
};

}

#endif // CPPTENSOR_INCLUDE_TENSOR_HPP_