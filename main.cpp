// An example of this library's usage

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>

#include "Initializations.hpp"
#include "DataLoader.hpp"
#include "Optimizers.hpp"
#include "Modules.hpp"
#include "Losses.hpp"
#include "Tensor.hpp"

using namespace cpp_tensor;

// Standardizes the input values to have zero Mean and unit variance
void Standardize(std::vector<double> &values, int features) {
  for (int i = 0; i < features; i++) {
    double sum = 0;

    // Calculate Mean for the i-th feature
    for (int j = i; j < values.size(); j += features)
      sum += values[j];
    double mean = sum / (values.size() / features), std = 0;

    // Calculate standard deviation for the i-th feature
    for (int j = i; j < values.size(); j += features)
      std += (values[j] - mean) * (values[j] - mean);
    std = sqrt(std / (values.size() / features));

    // Standardize the i-th feature
    for (int j = i; j < values.size(); j += features)
      values[j] = (values[j] - mean) / std;
  }
}

// Computes the Mean squared error of the model on the given Data loader
double ComputeError(Sequential &model, DataLoader &data_loader) {
  auto criterion = MSELoss(MSELoss::SUM);
  double loss_sum = 0, cnt = 0;
  Tensor::SetUseGrad(false);

  // Accumulate loss over all batches in the Data loader
  for (auto &[x_batch, y_batch] : data_loader) {
    auto pred = model(x_batch);
    auto loss = criterion(pred, y_batch);
    loss_sum += loss.Value();
    cnt += x_batch.Shape(0);
  }
  Tensor::SetUseGrad(true);
  return loss_sum / cnt;
}

int main() {
  // Random number generators for creating sample Data
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> uniform_dist(0, 30);
  std::normal_distribution<double> normal_dist(0, 1);

  // Generate sample Data with 2 features and 3 outputs
  const size_t kDataSize = 2e4, kFeatures = 2, kOutputs = 3;
  std::vector<double> data_x, data_y;
  for (int i = 0; i < kDataSize; i++) {
    double x1 = uniform_dist(mt), x2 = uniform_dist(mt);
    double y1 = -7.0 * x1 + 3.0 * x2;
    double y2 = 0.2 * x1 * x2;
    double y3 = 0.4 * x1 * x1 - 0.5 * x2 * x2;
    data_x.insert(data_x.end(), {x1, x2});
    data_y.insert(data_y.end(), {y1, y2, y3});
  }

  // Standardize the input Data
  Standardize(data_x, 2);

  // Add Gaussian noise to the input Data
  for (auto &x : data_x)
    x += 0.05 * normal_dist(mt);

  // Create tensors from Data and split them into training and test sets
  Tensor X(data_x, {data_x.size() / kFeatures, kFeatures});
  Tensor y(data_y, {data_y.size() / kOutputs, kOutputs});
  auto [x_train, x_test, y_train, y_test] = Tensor::TrainTestSplit(X, y, 0.8);

  // Create Data loaders with batch Size of 32
  DataLoader train_loader(x_train, y_train, 32, true);
  DataLoader test_loader(x_test, y_test, 32, false);

  // Define a simple sequential model
  Sequential model;
  model.AddModule<LinearLayer>(2, 8, Initialization::Uniform(0, 1));
  model.AddModule<ReLU>(0.1);
  model.AddModule<LinearLayer>(8, 8, Initialization::Normal(1, 2));
  model.AddModule<ReLU>(0.2);
  model.AddModule<LinearLayer>(8, 3);

  // Define optimizer and loss function
  SGD optimizer(model.Parameters(), 5e-4);
  MSELoss criterion;

  // Training loop for the model
  int n_epochs = 30;
  for (int epoch = 0; epoch < n_epochs; epoch++) {
    int iter = 0;
    for (auto &[x_batch, y_batch] : train_loader) {
      auto pred = model(x_batch);
      auto loss = criterion(pred, y_batch);
      loss.Backward();

      optimizer.Step();
      optimizer.ZeroGrad();

      iter++;
      if (iter % 100 == 0)
        std::cout << "epoch: " << epoch << " iter: " << iter << " : " << loss.Value() << '\n';
    }
  }

  // Compute and print loss on the test set
  std::cout << "loss on the test set: " << ComputeError(model, test_loader) << "\n\n";

  // Example to visualize model performance
  std::cout << "an example:\n";
  auto sample_x = x_test.ValueTensor({0});
  auto sample_y = y_test.ValueTensor({0});
  auto sample_pred = model(sample_x);
  std::cout << "pred:   ";
  for (int i = 0; i < kOutputs; i++)
    std::cout << std::setw(10) << sample_pred[i];
  std::cout << "\ntrue y: ";
  for (int i = 0; i < kOutputs; i++)
    std::cout << std::setw(10) << sample_y[i];
}

/*
 * Sample output:
 *
 * epoch: 0 iter: 100 : 522.557
 * ...
 * epoch: 29 iter: 500 : 45.2544
 * loss on the test set: 129.499

 * an example:
 * pred:     -134.973   107.011   124.485
 * true y:   -136.655   102.128    129.57
 */