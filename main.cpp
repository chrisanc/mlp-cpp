#include <vector>
#include <iostream>
#include <nn/nn.hpp>
#define CASES 100
#define BIAS 10

/*template <typename T>
void print_matrix(const math::matrix<T>& m);*/

int main() {
    /*size_t in, hidden, out;
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
    math::matrix<double> a1 = nn::sigmoid(z1);
    print_matrix(a1);
    return 0;*/
    std::vector<std::vector<double>> w0 = {
        {-1, 0.4, 2},
        {.3, .4, 2},
        {.5, .7, -0.9}
    };
    std::vector<std::vector<double>> w1 = {
        {.5, 0.1, 0.2},
        {.1, 0.7, 2},
        {.2, 0.9, 0.9}
    };
    std::vector<std::vector<double>> w2 = {
        {0.12, 0.2, -2},
        {.34, .4, 2},
        {.67, -0.1, 0.3}
    };
    std::vector<double> b0 = {0, 0.1, -0.2};
    std::vector<double> b1 = {0.3, -0.1, 0.2};
    std::vector<double> b2 = {1.2, 0.29, 0.12};
    std::vector<double> x = {2, 3, 4};
    // First layer
    double u00 = w0[0][0] * x[0] + w0[0][1] * x[1] + w0[0][2] * x[2] + b0[0];
    double u01 = w0[1][0] * x[0] + w0[1][1] * x[1] + w0[1][2] * x[2] + b0[1];
    double u02 = w0[2][0] * x[0] + w0[2][1] * x[1] + w0[2][2] * x[2] + b0[2];
    double a00 = u00 >= 0 ? 1.0 : 0.0;
    double a01 = u01 >= 0 ? 1.0 : 0.0;
    double a02 = u02 >= 0 ? 1.0 : 0.0;

    // Second layer
    double u10 = w1[0][0] * a00 + w1[0][1] * a01 + w1[0][2] * a02 + b1[0];
    double u11 = w1[1][0] * a00 + w1[1][1] * a01 + w1[1][2] * a02 + b1[1];
    double u12 = w1[2][0] * a00 + w1[2][1] * a01 + w1[2][2] * a02 + b1[2];
    double a10 = u10 >= 0 ? 1.0 : 0.0;
    double a11 = u11 >= 0 ? 1.0 : 0.0;
    double a12 = u12 >= 0 ? 1.0 : 0.0;

    // Third layer
    double u20 = w2[0][0] * a10 + w2[0][1] * a11 + w2[0][2] * a12 + b2[0];
    double u21 = w2[1][0] * a10 + w2[1][1] * a11 + w2[1][2] * a12 + b2[1];
    double u22 = w2[2][0] * a10 + w2[2][1] * a11 + w2[2][2] * a12 + b2[2];
    double a20 = 1 / (1 + std::pow(std::exp(1), -u20));
    double a21 = 1 / (1 + std::pow(std::exp(1), -u21));
    double a22 = 1 / (1 + std::pow(std::exp(1), -u22));

    std::cout << "X1: " << a20 << "\n";
    std::cout << "X2: " << a21 << "\n";
    std::cout << "X3: " << a22 << "\n";
}

/*template <typename T>
void print_matrix(const math::matrix<T>& m) {
    for (int i = 0; i < m.shape.i; i++) {
        for (int j = 0; j < m.shape.j; j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << "\n";
    }
}*/