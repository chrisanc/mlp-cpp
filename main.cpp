#include <vector>
#include <iostream>
#include <nn/nn.hpp>
#define CASES 100

template <typename T>
void print_matrix(const math::matrix<T>& m);

int main() {
    size_t in, hidden, out;
    std::cout << "Ingresa los valores de la capa de entrada, oculta y salida: ";
    std::cin >> in >> hidden >> out;

    // Generate fake input data
    math::matrix<double> x = nn::init_weights({CASES, 3});
    // Initialize the weights
    math::matrix<double> in_weights = nn::init_weights({in, hidden});
    math::matrix<double> out_weights = nn::init_weights({hidden, out});
    // Initialize the biases
    std::vector<double> b1 = nn::init_bias(hidden);
    std::vector<double> b2 = nn::init_bias(out);

    math::matrix<double> z1 = nn::forward_pass(x, in_weights, b1);
    std::cout << z1.shape.i << " " << z1.shape.j << "\n";
    return 0;
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