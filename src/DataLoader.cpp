#include <algorithm>
#include <random>

#include "DataLoader.hpp"
#include "Tensor.hpp"

namespace cpp_tensor {

// Iterator - Constructors

DataLoader::Iterator::Iterator(std::vector<int>::iterator it,
                               std::vector<int>::iterator end,
                               const Tensor &x,
                               const Tensor &y,
                               int batch_size) :
    it_(it), end_(end), x_(x), y_(y), batch_size_(batch_size) {
  batch_processed_ = false;
  LoadBatch();
}

DataLoader::Iterator::Iterator(std::vector<int>::iterator end, const Tensor &x, const Tensor &y)
    : it_(end), end_(end), x_(x), y_(y) {
  batch_processed_ = true;
}

// Iterator - Operators

bool DataLoader::Iterator::operator!=(const Iterator &other) const {
  return (it_ != other.it_ || batch_processed_ != other.batch_processed_);
}

std::pair<Tensor, Tensor> &DataLoader::Iterator::operator*() {
  batch_processed_ = true;
  return batch_;
}

DataLoader::Iterator &DataLoader::Iterator::operator++() {
  if (it_ != end_) {
    batch_processed_ = false;
    LoadBatch();
  }
  return *this;
}

// Iterator - Helper function to load a new batch of Data

void DataLoader::Iterator::LoadBatch() {
  std::vector<Tensor> x_batch;
  std::vector<Tensor> y_batch;
  for (int i = 0; i < batch_size_ && it_ != end_; i++, it_++) {
    x_batch.push_back(x_.ValueTensor({*it_}));
    y_batch.push_back(y_.ValueTensor({*it_}));
  }

  batch_ = std::make_pair(Tensor::Concat(x_batch), Tensor::Concat(y_batch));
}

// DataLoader - Constructor

DataLoader::DataLoader(const Tensor &x, const Tensor &y, int batch_size, bool shuffle)
    : x_(x), y_(y), batch_size_(batch_size), shuffle_(shuffle) {
  size_ = x.Shape(0);
  for (int i = 0; i < size_; i++)
    indices_.push_back(i);
}

// DataLoader - Iterator functions (used automatically by c++ for loop)

DataLoader::Iterator DataLoader::begin() {
  ShuffleIndices();
  return Iterator(indices_.begin(), indices_.end(), x_, y_, batch_size_);
}

DataLoader::Iterator DataLoader::end() {
  return Iterator(indices_.end(), x_, y_);
}

// DataLoader - Helper function to shuffle (if shuffle_ == true) the indices

void DataLoader::ShuffleIndices() {
  if (shuffle_) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::shuffle(indices_.begin(), indices_.end(), mt);
  }
}

}