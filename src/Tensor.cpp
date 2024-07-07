#include <random>
#include <utility>

#include "Tensor.hpp"

namespace cpp_tensor {

// Constructors

Tensor::Tensor(double value, bool requires_grad) {
  tensor_ = std::make_shared<InternalTensor>(std::vector<double>({value}), std::vector<size_t>(), requires_grad, true);
  CalculateStrides();
}

Tensor::Tensor(std::vector<double> values, bool requires_grad) {
  tensor_ =
      std::make_shared<InternalTensor>(std::move(values), std::vector<size_t>({values.size()}), requires_grad, true);
  CalculateStrides();
}

Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape, bool requires_grad) {
  tensor_ = std::make_shared<InternalTensor>(std::move(values), std::move(shape), requires_grad, true);
  CalculateStrides();
}

Tensor::Tensor(double value, std::vector<size_t> shape, bool requires_grad) {
  size_t size = 1;
  std::vector<double> values;
  for (auto &s : shape)
    size *= s;

  values.reserve(size);
  for (int i = 0; i < size; i++)
    values.push_back(value);
  tensor_ = std::make_shared<InternalTensor>(values, std::move(shape), requires_grad, true);
  CalculateStrides();
}

Tensor::Tensor(SharedTensor &&tensor) : tensor_(std::move(tensor)) {
  CalculateStrides();
}

// Static functions

Tensor Tensor::Concat(const std::vector<Tensor> &tensors) {
  std::vector<double> data;
  for (auto &kT : tensors)
    data.insert(data.end(), kT.tensor_->data_.begin(), kT.tensor_->data_.end());
  std::vector<size_t> shape = {tensors.size()};
  shape.insert(shape.end(), tensors.begin()->tensor_->shape_.begin(), tensors.begin()->tensor_->shape_.end());
  return Tensor(data, shape);
}

std::array<Tensor, 4> Tensor::TrainTestSplit(const Tensor &x, const Tensor &y, double ratio) {
  size_t size = x.Shape(0);
  std::vector<int> indices;
  for (int i = 0; i < size; i++)
    indices.push_back(i);
  std::random_device rd;
  std::mt19937 mt(rd());
  std::shuffle(indices.begin(), indices.end(), mt);

  size_t train_size = ratio * size;
  std::vector<Tensor> data_x_train, data_x_test, data_y_train, data_y_test;

  for (int i = 0; i < train_size; i++) {
    data_x_train.push_back(x.ValueTensor({indices[i]}));
    data_y_train.push_back(y.ValueTensor({indices[i]}));
  }
  for (int i = train_size; i < size; i++) {
    data_x_test.push_back(x.ValueTensor({indices[i]}));
    data_y_test.push_back(y.ValueTensor({indices[i]}));
  }

  return {Tensor::Concat(data_x_train), Tensor::Concat(data_x_test),
          Tensor::Concat(data_y_train), Tensor::Concat(data_y_test)};
}

// Data access

double Tensor::Value(const std::vector<int> &indices) const {
  int index = 0;
  for (int i = 0; i < indices.size(); i++)
    index += indices[i] * strides_[i];
  return tensor_->data_[index];
}

Tensor Tensor::ValueTensor(const std::vector<int> &indices) const {
  int index = 0;
  for (int i = 0; i < indices.size(); i++)
    index += indices[i] * strides_[i];

  std::vector<size_t> shape_t;
  if (indices.size() == tensor_->shape_.size())
    shape_t = {1};
  else
    shape_t = std::vector<size_t>(tensor_->shape_.begin() + indices.size(), tensor_->shape_.end());

  int next_id = index + strides_[indices.size() - 1];
  const std::vector<double> kTensorT(tensor_->data_.begin() + index, tensor_->data_.begin() + next_id);
  return Tensor(kTensorT, shape_t);
}

// Tensor operations

void Tensor::Backward(bool retain_graph) {
  tensor_->SetGrad(1);
  tensor_->Backward(retain_graph);
}

Tensor Tensor::Reshape(std::vector<size_t> new_shape) {
  tensor_->shape_ = std::move(new_shape);
  CalculateStrides();
  return *this;
}

Tensor Tensor::Flatten() {
  tensor_->shape_ = {Size()};
  CalculateStrides();
  return *this;
}

Tensor Tensor::Clone(bool deep_copy) const &{
  if (!deep_copy)
    return Tensor(SharedTensor(tensor_));
  return Tensor(tensor_->data_, tensor_->shape_, tensor_->requires_grad_);
}

// Mathematical operations

Tensor Tensor::operator+(const Tensor &other) const &{
  // The implementation of this function is slightly different compared to other operators.
  // This allows it to support adding a bias (a situation where one tensor has the same Shape as the other,
  // except with a few additional dimensions at the beginning).
  // (I could implement such functionality for other functions and rename them,
  // but for now it is used only for adding the bias.)
  // Additionally, note that if one tensor has a Size of 1, it does not matter which function is called.

  if (other.Size() == 1)
    return Tensor(AddManyOneInternal(this->tensor_, other.tensor_));
  if (this->Size() == 1)
    return Tensor(AddManyOneInternal(other.tensor_, this->tensor_));
  if (this->NumDimensions() > other.NumDimensions())
    return Tensor(AddBiasInternal(tensor_, other.tensor_));
  if (this->NumDimensions() < other.NumDimensions())
    return Tensor(AddBiasInternal(other.tensor_, tensor_));

  return AddManyManyInternal(this->tensor_, other.tensor_);
}

Tensor Tensor::operator-(const Tensor &other) const &{
  if (other.Size() == 1)
    return Tensor(AddManyOneInternal(this->tensor_, OppositeInternal(other.tensor_)));
  if (this->Size() == 1)
    return Tensor(AddManyOneInternal(OppositeInternal(other.tensor_), this->tensor_));

  return AddManyManyInternal(this->tensor_, OppositeInternal(other.tensor_));
}

Tensor Tensor::operator*(const Tensor &other) const &{
  if (other.Size() == 1)
    return Tensor(MultiplyManyOneInternal(this->tensor_, other.tensor_));
  if (this->Size() == 1)
    return Tensor(MultiplyManyOneInternal(other.tensor_, this->tensor_));

  return MultiplyManyManyInternal(this->tensor_, other.tensor_);
}

Tensor Tensor::operator/(const Tensor &other) const &{
  if (other.Size() == 1)
    return Tensor(MultiplyManyOneInternal(this->tensor_, InverseInternal(other.tensor_)));
  if (this->Size() == 1)
    return Tensor(MultiplyManyOneInternal(InverseInternal(other.tensor_), this->tensor_));

  return MultiplyManyManyInternal(this->tensor_, InverseInternal(other.tensor_));
}

Tensor Tensor::Matmul(const Tensor &other) const &{
  return Tensor(MatmulInternal(this->tensor_, other.tensor_)); // this-> ? czm
}

Tensor Tensor::Sum() const &{
  return Tensor(SumInternal(tensor_));
}

Tensor Tensor::Mean() const &{
  return Tensor(MultiplyManyOneInternal(SumInternal(tensor_),
                                        std::make_shared<InternalTensor>(
                                            std::vector<double>({1.0 / Size()}), std::vector<size_t>({}))));
}

Tensor Tensor::Pow(int exponent) const &{
  return Tensor(PowInternal(tensor_, exponent));
}

// Activation functions

Tensor Tensor::Relu(double leaky) const &{
  return Tensor(ReluInternal(tensor_, leaky));
}

// Helper function

void Tensor::CalculateStrides() {
  size_t stride = 1;
  for (int i = tensor_->shape_.size() - 1; i >= 0; --i) {
    strides_.push_back(stride);
    stride *= tensor_->shape_[i];
  }

  std::reverse(strides_.begin(), strides_.end());
  if (strides_.empty())
    strides_ = {1};
}

}