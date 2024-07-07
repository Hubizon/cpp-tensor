#ifndef CPPTENSOR_INCLUDE_DATALOADER_HPP_
#define CPPTENSOR_INCLUDE_DATALOADER_HPP_

#include <iterator>
#include <vector>

#include "Tensor.hpp"

namespace cpp_tensor {

class DataLoader {
 public:
  class Iterator {
   public:
    // Constructors
    Iterator(std::vector<int>::iterator it,
             std::vector<int>::iterator end,
             const Tensor &x,
             const Tensor &y,
             int batch_size);
    Iterator(std::vector<int>::iterator end, const Tensor &x, const Tensor &y);

    // Operators
    bool operator!=(const Iterator &other) const;
    std::pair<Tensor, Tensor> &operator*();
    Iterator &operator++();

   private:
    // Helper function to load a new batch of Data
    void LoadBatch();

    // Member variables
    std::vector<int>::iterator it_;
    std::vector<int>::iterator end_;
    const Tensor &x_;
    const Tensor &y_;
    std::pair<Tensor, Tensor> batch_;
    int batch_size_;
    bool batch_processed_;
  };

  // Constructor
  DataLoader(const Tensor &x, const Tensor &y, int batch_size = 32, bool shuffle = true);

  // Iterator functions (used automatically by c++ for loop)
  Iterator begin();
  Iterator end();

  // Size access
  size_t Size() const { return size_; }

 private:
  // Helper function to shuffle (if shuffle_ == true) the indices
  void ShuffleIndices();

  // Member variables
  std::vector<int> indices_;
  const Tensor &x_;
  const Tensor &y_;
  size_t size_;
  int batch_size_;
  bool shuffle_;
};

}

#endif // CPPTENSOR_INCLUDE_DATALOADER_HPP_