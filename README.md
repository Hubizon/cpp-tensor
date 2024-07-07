# CppTensor - C++ Deep Learning Library

A small C++ library inspired by PyTorch, enabling users to create simple models for deep learning. Remarkably, for some simple models, it performs about 2.5x faster than PyTorch (on CPU).
For an example of how to use the library, see `main.cpp`.

## Purpose
This library was developed for learning purposes, providing a hands-on approach to understanding and implementing deep learning models and algorithms in C++. It serves as a simplified framework for experimenting with fundamental concepts of machine learning and neural networks.

## Features
- **InternalTensor**: Encapsulated tensors with automatic differentiation.
- **Tensor**: Multi-dimensional array with automatic differentiation support.
- **DataLoader**: Simplified data loader for batching input data and targets.
- **Initialization**: Various strategies for tensor initialization.
- **MSELoss**: Mean Squared Error loss function.
- **LinearLayer**: Fully connected linear layer.
- **ReLU**: Rectified Linear Unit activation function.
- **Sequential**: Container for sequential model construction.
- **SGD**: Stochastic Gradient Descent optimizer.


## Documentation

### InternalTensor
A class representing an internal tensor with automatic differentiation support.

This class encapsulates a tensor object used internally for automatic differentiation and gradient calculation within a computational graph. It is designed to be used indirectly via shared pointers (SharedTensor).

InternalTensor is intended only for internal use within the library.

### Tensor
Class representing a multi-dimensional array (tensor) with support for automatic differentiation.

**Example**
```cpp
// Define 2D tensor a {{1, 2}, {3, 4}} and 0D tensor b {6}
auto a = Tensor({1, 2, 3, 4}, {2, 2}, true), b = Tensor(6);
auto c = a * b; // Multiply the tensors

// Print the values of the resulting tensor c
// One can also iterate through the tensor in 1D using the operator[]
std::cout << c.Value({0, 0}) << ' ' << c.Value({0, 1}) << ' ' << c[2] << ' ' << c[3] << '\n';

// output: 6 12 18 24
```

### DataLoader 
This class provides functionality to iterate over data in batches, similar to data loaders in frameworks like PyTorch, though simplified. It requires two tensors with the same shape along the first dimension: one for input data (`X`) and one for corresponding targets (`y`).
This class allows iterating over batches of data using a C++ for loop syntax, making it
convenient for training and testing machine learning models with batch processing.

**Example:**
```cpp
auto X = Tensor({1, 2, 3, 4}), y = Tensor({4, 3, 2, 1});
DataLoader loader(X, y, 2, true); // Batch size of 2, shuffle enabled

for (auto& [X_batch, y_batch] : loader) {
  // Process X_batch (input data) and y_batch (targets)
  for (int i = 0; i < X_batch.Size(); i++)
    std::cout << X_batch.Value({i}) << ' ' << y_batch.Value({i}) << '\n';
}

/* output: 
 * 3 2
 * 1 4
 * 2 3
 * 4 1
 */
```


### Initialization
A class for initializing tensors with different strategies.

**Example:**
```cpp
// Linear layer with uniform initialization between 0 and 3
LinearLayer linear(2, 3, Initialization::Uniform(0, 3));

for (int i = 0; i < 2 * 3; i++)
  std::cout << linear.Parameters()[0]->Data(i) << ' ';

for (int i = 0; i < 3; i++)
  std::cout << linear.Parameters()[1]->Data(i) << ' ';
  
/* output:
 * 0.239498 2.21703 0.726748 2.30263 2.10703 2.17733    <- weight
 * 2.589010 1.94051 0.377067                            <- bias
 */
 ```


### MSELoss : Loss
Class representing the Mean Squared Error loss function.

**Example**
```cpp
// Mean squared error loss with default reduction (mean)
MSELoss criterion;

// Calculate the loss on 1D tensors {1, 2, 3, 5} and {1, 2, 3, 4}
auto pred = Tensor({1, 2, 3, 5}, true), y = Tensor({1, 2, 3, 4});
auto loss = criterion(pred, y);

// Backpropagate through the computational graph and get the loss and gradient
loss.Backward();
std::cout << loss.Value() << ' ' << -pred.GetTensor()->Grad(3);

// output: 0.25 -0.5
```
 
### LinearLayer : Module
Fully connected linear layer.

**Example**
```cpp
// Fully connected linear layer with 3 features, 2 outputs, 
// Normal initialization (mean = -5, std = 10) and no bias.
LinearLayer linear(3, 2, Initialization::Normal(-5, 10), false);

// Forward an unbatched tensor { 1, 2, 3 } through it.
auto pred = linear(Tensor({ 1, 2, 3 }));
std::cout << pred.Value({0}) << ' ' << pred.Value({1});

// output: 16.3219 -46.6337
 ```

### ReLU : Module
Rectified Linear Unit (ReLU) activation function.

**Example**
```cpp
ReLU relu(0.2); // LeakyReLU with negative slope equal to 0.2
std::cout << relu(3).Value() << ' ' << relu(-3).Value();

// output: 3 -0.6
```

### Sequential : Module
A container module to hold and manage other modules in sequence.

**Example**
```cpp
Sequential model; // Simple sequential model with 3 layers
model.AddModule<LinearLayer>(1, 2);
model.AddModule<ReLU>();
model.AddModule<LinearLayer>(2, 1);
std::cout << model(Tensor(5)).Value(); // Forward pass through the model with tensor {5}

// output: -2.26804
```

### SGD
Stochastic Gradient Descent (SGD) optimizer.

**Example**
```cpp
// Define features, labels, model, optimizer, and criterion
auto X = Tensor({1, 2, 3, 6}, {4, 1}), y = Tensor({1, 2, 3, 4}, {4, 1});
LinearLayer linear(1, 1);
SGD optimizer(linear.Parameters(), 0.01);
MSELoss criterion;

// Perform backpropagation through the graph using the loss function
criterion(linear(X), y).Backward();
std::cout << linear.Parameters()[0]->Data(0) << ' '; // Weight before the optimizer step

// Perform a single optimizer step to update the model's weights and biases
optimizer.Step();
optimizer.ZeroGrad();
std::cout << linear.Parameters()[0]->Data(0); // Weight after the optimizer step

// output: -0.461208 -0.12852
```

## Acknowledgements
Inspired by the design and functionality of PyTorch.

