#include <vector>
#include <iostream>
#include <nn/nn.hpp>
#define CASES 100
#define BIAS 10

template <typename T>
void print_matrix(const math::matrix<T>& m);

int main() {
    // Define the inputs
    math::matrix<double> x = math::matrix<double>({
            {0.7, 0.4, 0.7},
            {0.65, 0.45, 0.75},
            {0.75, 0.35, 0.65},
            {0.15, 0.45, 0.70},
            {0.20, 0.40, 0.65},
            {0.10, 0.50, 0.75},
            {0.65, 0.90, 0.70},
            {0.70, 0.85, 0.65},
            {0.60, 0.95, 0.75},
            {0.65, 0.45, 0.15},
            {0.70, 0.40, 0.20},
            {0.60, 0.50, 0.10}
        }, 12, 3
    );
    // Define the real output values
    std::vector<double> y = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    // Define the weights
    math::matrix<double> w0 = math::matrix<double>({
            {.5, 0.1, 0.2},
            {.1, 0.7, 2},
            {.2, 0.9, 0.9},
            {.2, 0.9, 0.9},
            {.2, 0.9, 0.9}
        }, 5, 3
    );
    math::matrix<double> w1 = math::matrix<double>({
            {0.12, 0.2, -2, 0.2, 0.1},
            {.34, .4, 2, 0.5, -1.2},
            {.67, -0.1, 0.3, 0.1, 0.4},
            {.7, 0.15, 0.1, 0.8, 1}
        }, 4, 5
    );
    std::vector<double> b0 = {0.3, -0.1, 0.2, -0.4, 0.1};
    std::vector<double> b1 = {1.2, 0.29, 0.12, 0.1};

    // Calculate U1 and A1
    math::matrix<double> u1 = nn::forward_pass(x, math::transpose(w0), b0);
    math::matrix<double> a1 = nn::sigmoid(u1);
    
    // Calculate U2 and A2
    math::matrix<double> u2 = nn::forward_pass(a1, math::transpose(w1), b1);
    math::matrix<double> a2 = nn::softmax(u2);

    print_matrix(a2);
}

template <typename T>
void print_matrix(const math::matrix<T>& m) {
    for (int i = 0; i < m.shape.i; i++) {
        for (int j = 0; j < m.shape.j; j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << "\n";
    }
}