#ifndef CPPTENSOR_INCLUDE_INTERNALTENSOR_HPP_
#define CPPTENSOR_INCLUDE_INTERNALTENSOR_HPP_

#include <functional>
#include <utility>
#include <vector>
#include <memory>

namespace cpp_tensor {

class InternalTensor;
using SharedTensor = std::shared_ptr<InternalTensor>;

class InternalTensor {
 public:
  // Constructor
  InternalTensor(std::vector<double> data, std::vector<size_t> shape, bool requires_grad = false, bool is_leaf = false);

  // Data and gradient access
  double &Data(int index) { return data_[index]; }
  double &Grad(int index) { return grad_[index]; }
  size_t Size() const &{ return data_.size(); }
  bool RequiresGrad() const &{ return requires_grad_ && use_grad_; }

  // Gradient updates
  void SetGrad(std::vector<double> grad) { grad_ = std::move(grad); }
  void SetGrad(double grad) { SetGrad(std::vector<double>(Size(), grad)); }
  void UpdateGrad(std::vector<double> grad);
  void UpdateGrad(double grad) { UpdateGrad(std::vector<double>(Size(), grad)); }

  // Performs Backward propagation through the computational graph created during the Forward pass.
  // If retain_graph is true, the graph is retained for further Backward passes.
  void Backward(bool retain_graph = false);

  // Static boolean indicating whether to use gradients in every tensor
  static bool use_grad_;

 private:
  // Friend class that needs full access to this one
  friend class Tensor;

  // Member variables
  std::vector<double> data_;
  std::vector<double> grad_;
  std::vector<size_t> shape_;
  std::vector<SharedTensor> parents_;
  std::function<void(InternalTensor *)> backward_op_;
  bool is_leaf_ = false;
  bool requires_grad_ = false;
  int num_children_ = 0;
  int children_processed_ = 0;

  // Friend functions for performing mathematical operations on tensors with gradient calculation support
  friend SharedTensor ApplyOperation(const std::vector<double> &data,
                                     const std::vector<size_t> &shape,
                                     const std::vector<SharedTensor> &parents,
                                     std::function<void(InternalTensor *)> backward_op);
  friend SharedTensor AddManyOneInternal(const SharedTensor &a, const SharedTensor &b);
  friend SharedTensor AddManyManyInternal(const SharedTensor &a, const SharedTensor &b);
  friend SharedTensor AddBiasInternal(const SharedTensor &a, const SharedTensor &b);
  friend SharedTensor MultiplyManyOneInternal(const SharedTensor &a, const SharedTensor &b);
  friend SharedTensor MultiplyManyManyInternal(const SharedTensor &a, const SharedTensor &b);
  friend SharedTensor OppositeInternal(const SharedTensor &a);
  friend SharedTensor InverseInternal(const SharedTensor &a);
  friend SharedTensor MatmulInternal(const SharedTensor &a, const SharedTensor &b);
  friend SharedTensor PowInternal(const SharedTensor &a, int exponent);
  friend SharedTensor SumInternal(const SharedTensor &a);
  friend SharedTensor ReluInternal(const SharedTensor &a, double leaky);
};

}

#endif // CPPTENSOR_INCLUDE_INTERNALTENSOR_HPP_