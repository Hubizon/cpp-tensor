#include <numeric>
#include <utility>

#include "InternalTensor.hpp"

namespace cpp_tensor {

bool InternalTensor::use_grad_ = true;

// Constructor

InternalTensor::InternalTensor(std::vector<double> data, std::vector<size_t> shape, bool requires_grad, bool is_leaf)
    : data_(std::move(data)), shape_(std::move(shape)), requires_grad_(requires_grad), is_leaf_(is_leaf) {}

// Gradient updates

void InternalTensor::UpdateGrad(std::vector<double> grad) {
  if (grad_.empty())
    grad_ = std::move(grad);
  else
    for (int i = 0; i < grad_.size(); i++)
      grad_[i] += grad[i];
}

// Performs Backward propagation through the computational graph created during the Forward pass.
// If retainGraph is true, the graph is retained for further Backward passes.

void InternalTensor::Backward(bool retain_graph) {
  if (!requires_grad_)
    return;

  if (backward_op_)
    backward_op_(this);

  if (!is_leaf_ && !retain_graph)
    grad_.clear();

  for (auto &p : parents_)
    p->Backward(retain_graph);

  if (!retain_graph) {
    backward_op_ = nullptr;
    parents_.clear();
  }
}

// Friend functions for performing mathematical operations on tensors with gradient calculation support

SharedTensor ApplyOperation(const std::vector<double> &data,
                            const std::vector<size_t> &shape,
                            const std::vector<SharedTensor> &parents,
                            std::function<void(InternalTensor *)> backward_op) {
  bool requires_grad = false;
  bool is_leaf = false;
  if (InternalTensor::use_grad_) {
    for (auto &kP : parents) {
      if (kP->requires_grad_)
        requires_grad = true;
      if (kP->is_leaf_)
        is_leaf = true;
    }
  }

  auto res = std::make_shared<InternalTensor>(data, shape, requires_grad, is_leaf);
  if (requires_grad) {
    res->parents_ = parents;
    res->backward_op_ = std::move(backward_op);
  }

  return res;
}

SharedTensor AddManyOneInternal(const SharedTensor &a, const SharedTensor &b) {
  std::vector<double> data;
  for (auto &d : a->data_)
    data.push_back(d + b->data_[0]);

  return ApplyOperation(data, a->shape_, {a, b}, [a, b](InternalTensor *res) {
    if (a->RequiresGrad())
      a->UpdateGrad(res->grad_);
    if (b->RequiresGrad())
      b->UpdateGrad(std::accumulate(res->grad_.begin(), res->grad_.end(), 0.));
  });
}

SharedTensor AddManyManyInternal(const SharedTensor &a, const SharedTensor &b) {
  std::vector<double> data;
  for (int i = 0; i < a->data_.size(); i++)
    data.push_back(a->data_[i] + b->data_[i]);

  return ApplyOperation(data, a->shape_, {a, b}, [a, b](InternalTensor *res) {
    if (a->RequiresGrad())
      a->UpdateGrad(res->grad_);

    if (b->RequiresGrad())
      b->UpdateGrad(res->grad_);
  });
}

SharedTensor AddBiasInternal(const SharedTensor &a, const SharedTensor &b) {
  std::vector<double> data(a->Size());
  for (int i = 0; i < data.size(); i += b->Size())
    for (int j = 0; j < b->Size(); j++)
      data[i + j] = a->data_[i + j] + b->data_[j];

  return ApplyOperation({data}, a->shape_, {a, b}, [a, b](InternalTensor *res) {
    if (a->RequiresGrad())
      a->UpdateGrad(res->grad_);

    if (b->RequiresGrad()) {
      std::vector<double> b_grad(b->Size());
      for (int i = 0; i < res->Size(); i += b->Size())
        for (int j = 0; j < b->Size(); j++)
          b_grad[j] += res->grad_[i + j];
      b->UpdateGrad(b_grad);
    }
  });
}

SharedTensor MultiplyManyOneInternal(const SharedTensor &a, const SharedTensor &b) {
  std::vector<double> data;
  for (auto &d : a->data_)
    data.push_back(d * b->data_[0]);

  return ApplyOperation(data, a->shape_, {a, b}, [a, b](InternalTensor *res) {
    if (a->RequiresGrad()) {
      std::vector<double> a_grad(a->data_.size());
      for (int i = 0; i < a_grad.size(); i++)
        a_grad[i] = res->grad_[i] * b->data_[0];
      a->UpdateGrad(a_grad);
    }

    if (b->RequiresGrad()) {
      double b_grad = 0;
      for (int i = 0; i < res->grad_.size(); i++)
        b_grad += res->grad_[i] * a->data_[i];
      b->UpdateGrad(b_grad);
    }
  });
}

SharedTensor MultiplyManyManyInternal(const SharedTensor &a, const SharedTensor &b) {
  std::vector<double> data;
  for (int i = 0; i < a->data_.size(); i++)
    data.push_back(a->data_[i] * b->data_[i]);

  return ApplyOperation(data, a->shape_, {a, b}, [a, b](InternalTensor *res) {
    if (a->RequiresGrad()) {
      std::vector<double> a_grad(a->data_.size());
      for (int i = 0; i < a_grad.size(); i++)
        a_grad[i] = res->grad_[i] * b->data_[i];
      a->UpdateGrad(a_grad);
    }

    if (b->RequiresGrad()) {
      std::vector<double> b_grad(a->data_.size());
      for (int i = 0; i < b_grad.size(); i++)
        b_grad[i] = res->grad_[i] * a->data_[i];
      b->UpdateGrad(b_grad);
    }
  });
}

SharedTensor OppositeInternal(const SharedTensor &a) {
  return MultiplyManyOneInternal(a,
                                 std::make_shared<InternalTensor>(std::vector<double>({-1}), std::vector<size_t>({})));
}

SharedTensor InverseInternal(const SharedTensor &a) {
  return PowInternal(a, -1);
}

std::vector<double> MatmulVectors(const std::vector<double> &a,
                                  const std::vector<double> &b,
                                  const size_t n,
                                  const size_t m,
                                  const size_t p) {
  std::vector<double> res(n * p);
  int iter_res = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      int iter_a = i * m, iter_b = j;
      for (int k = 0; k < m; k++) {
        res[iter_res] = res[iter_res] + a[iter_a] * b[iter_b];
        iter_a++, iter_b += p;
      }
      iter_res++;
    }
  }

  return res;
}

std::vector<double> Transpose(const std::vector<double> &a, const size_t n, const size_t m) {
  std::vector<double> res(m * n);

  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < m; ++j)
      res[j * n + i] = a[i * m + j];

  return res;
}

SharedTensor MatmulInternal(const SharedTensor &a, const SharedTensor &b) {
  std::vector<double> data = MatmulVectors(a->data_, b->data_, a->shape_[0], a->shape_[1], b->shape_[1]);

  return ApplyOperation(data, {a->shape_[0], b->shape_[1]}, {a, b}, [a, b](InternalTensor *res) {
    const size_t kN = a->shape_[0];
    const size_t kM = a->shape_[1];
    const size_t kP = b->shape_[1];
    if (a->RequiresGrad())
      a->UpdateGrad(MatmulVectors(res->grad_, Transpose(b->data_, kM, kP), kN, kP, kM));

    if (b->RequiresGrad())
      b->UpdateGrad(MatmulVectors(Transpose(a->data_, kN, kM), res->grad_, kM, kN, kP));
  });
}

SharedTensor PowInternal(const SharedTensor &a, int exponent) {
  std::vector<double> data(a->data_.size(), 1);

  int exp = exponent;
  while (exp < 0) {
    for (int i = 0; i < data.size(); i++)
      data[i] /= a->data_[i];
    exp++;
  }

  while (exp > 0) {
    for (int i = 0; i < data.size(); i++)
      data[i] *= a->data_[i];
    exp--;
  }

  return ApplyOperation(data, a->shape_, {a}, [a, exponent, data](InternalTensor *res) { // &?
    if (a->RequiresGrad()) {
      std::vector<double> a_grad(a->data_.size());
      for (int i = 0; i < a_grad.size(); i++)
        a_grad[i] = res->grad_[i] * (exponent * (data[i] / a->data_[i]));
      a->UpdateGrad(a_grad);
    }
  });
}

SharedTensor SumInternal(const SharedTensor &a) {
  double data = std::accumulate(a->data_.begin(), a->data_.end(), 0.);

  return ApplyOperation({data}, {}, {a}, [a, data](InternalTensor *res) {
    if (a->RequiresGrad())
      a->UpdateGrad(std::vector<double>(a->data_.size(), res->grad_[0]));
  });
}

SharedTensor ReluInternal(const SharedTensor &a, double leaky) {
  std::vector<double> data = a->data_;
  for (auto &d : data)
    if (d < 0)
      d *= leaky;

  return ApplyOperation(data, a->shape_, {a}, [a, leaky](InternalTensor *res) {
    if (a->RequiresGrad()) {
      std::vector<double> a_grad(a->data_.size());
      for (int i = 0; i < a_grad.size(); i++)
        a_grad[i] = res->grad_[i] * (a->data_[i] < 0 ? leaky : 1);
      a->UpdateGrad(a_grad);
    }
  });
}

}