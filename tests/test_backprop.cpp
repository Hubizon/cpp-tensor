// Test to verify backpropagation works correctly with shared tensors

#include <iostream>
#include <cmath>
#include "Tensor.hpp"

using namespace cpp_tensor;

const double EPSILON = 1e-6;

bool test_shared_tensor_backprop() {
    auto x = Tensor(3.0, true);
    auto z = x + x;
    
    z.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 2.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_complex_shared_usage() {
    auto x = Tensor(2.0, true);
    auto x_squared = x * x;
    auto z = x_squared + x;
    
    z.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 5.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_triple_usage() {
    auto x = Tensor(4.0, true);
    auto temp = x + x;
    auto z = temp + x;
    
    z.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 3.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_power_then_add() {
    auto x = Tensor(3.0, true);
    auto y = x.Pow(2);
    auto z = y + y;
    
    z.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 12.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_diamond_pattern() {
    auto x = Tensor(5.0, true);
    auto one = Tensor(1.0);
    auto two = Tensor(2.0);
    
    auto y1 = x + one;
    auto y2 = x + two;
    auto z = y1 + y2;
    
    z.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 2.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_kite_pattern() {
    auto x = Tensor(5.0, true);
    auto y = x + x;
    auto one = Tensor(1.0);
    auto two = Tensor(2.0);
    
    auto y1 = y + one;
    auto y2 = y + two;
    auto z = y1 + y2;
    
    z.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 4.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_multiply_then_add() {
    auto x = Tensor(2.0, true);
    auto two = Tensor(2.0);
    auto three = Tensor(3.0);
    
    auto y1 = x * two;
    auto y2 = x * three;
    auto z = y1 + y2;
    
    z.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 5.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_nested_reuse() {
    auto x = Tensor(2.0, true);
    auto y = x * x;
    auto z = y * y;
    
    z.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 32.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_sum_operation() {
    auto x = Tensor({1.0, 2.0, 3.0}, true);
    auto y = x + x;
    auto z = y.Sum();
    
    z.Backward();
    
    bool pass = std::abs(x.GetTensor()->Grad(0) - 2.0) < EPSILON &&
                std::abs(x.GetTensor()->Grad(1) - 2.0) < EPSILON &&
                std::abs(x.GetTensor()->Grad(2) - 2.0) < EPSILON;
    return pass;
}

bool test_retain_graph_multiple_backward() {
    auto x = Tensor(3.0, true);
    auto z = x + x;
    
    z.Backward(true);
    double grad1 = x.GetTensor()->Grad(0);
    
    x.GetTensor()->SetGrad(0);
    
    z.Backward(true);
    double grad2 = x.GetTensor()->Grad(0);
    
    return std::abs(grad1 - 2.0) < EPSILON && std::abs(grad2 - 2.0) < EPSILON;
}

bool test_non_additive_side_effect() {
    auto x = Tensor(2.0, true);
    auto y = x + x;
    auto z = y + y;
    
    z.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 4.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_gradient_clearing_issue() {
    auto x = Tensor(5.0, true);
    auto y = x * x;
    auto z1 = y + Tensor(1.0);
    auto z2 = y + Tensor(2.0);
    auto loss = z1 + z2;
    
    loss.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 20.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

bool test_reference_counting_behavior() {
    auto x = Tensor(2.0, true);
    auto a = x + x;
    auto b = a + a;
    
    b.Backward();
    
    double grad = x.GetTensor()->Grad(0);
    double expected_grad = 4.0;
    return std::abs(grad - expected_grad) < EPSILON;
}

int main() {
    struct Test {
        std::string name;
        bool (*func)();
    };

    std::vector<Test> tests = {
        {"Shared tensor used multiple times (z = x + x)", test_shared_tensor_backprop},
        {"Complex shared usage (z = x*x + x)", test_complex_shared_usage},
        {"Triple usage (z = x + x + x)", test_triple_usage},
        {"Power then add (y = x^2, z = y + y)", test_power_then_add},
        {"Diamond pattern (y1=x+1, y2=x+2, z=y1+y2)", test_diamond_pattern},
        {"Kite pattern (x->y, y->[y1,y2]->z)", test_kite_pattern},
        {"Multiply then add (z = x*2 + x*3)", test_multiply_then_add},
        {"Nested reuse (y = x*x, z = y*y)", test_nested_reuse},
        {"Sum with shared tensor", test_sum_operation},
        {"Multiple backward passes with retain_graph", test_retain_graph_multiple_backward},
        {"Counting backward operations", test_non_additive_side_effect},
        {"Gradient clearing with shared tensors", test_gradient_clearing_issue},
        {"Reference counting behavior", test_reference_counting_behavior}
    };

    int passed = 0;
    int total = tests.size();

    for (int i = 0; i < total; i++) {
        std::cout << "Test " << (i + 1) << ": " << tests[i].name << ": ";
        if (tests[i].func()) {
            std::cout << "PASS\n";
            passed++;
        } else {
            std::cout << "FAIL\n";
        }
    }

    std::cout << "\nResults: " << passed << "/" << total << " tests passed\n";
    return (passed == total) ? 0 : 1;
}
