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
        {.5, 0.1, 0.2},
        {.1, 0.7, 2},
        {.2, 0.9, 0.9},
        {.2, 0.9, 0.9},
        {.2, 0.9, 0.9}
    };
    std::vector<std::vector<double>> w1 = {
        {0.12, 0.2, -2, 0.2, 0.1},
        {.34, .4, 2, 0.5, -1.2},
        {.67, -0.1, 0.3, 0.1, 0.4},
        {.7, 0.15, 0.1, 0.8, 1}
    };
    std::vector<double> b0 = {0.3, -0.1, 0.2, -0.4, 0.1};
    std::vector<double> b1 = {1.2, 0.29, 0.12, 0.1};
    std::vector<double> x = {2, 3, 4, 0.2};

    // Hidden layer
    double u00 = w0[0][0] * x[0] + w0[0][1] * x[1] + w0[0][2] * x[2] + b0[0];
    double u01 = w0[1][0] * x[0] + w0[1][1] * x[1] + w0[1][2] * x[2] + b0[1];
    double u02 = w0[2][0] * x[0] + w0[2][1] * x[1] + w0[2][2] * x[2] + b0[2];
    double u03 = w0[3][0] * x[0] + w0[3][1] * x[1] + w0[3][2] * x[2] + b0[3];
    double u04 = w0[4][0] * x[0] + w0[4][1] * x[1] + w0[4][2] * x[2] + b0[4];
    double a00 = 1 / (1 + std::pow(std::exp(1), -u00));
    double a01 = 1 / (1 + std::pow(std::exp(1), -u01));
    double a02 = 1 / (1 + std::pow(std::exp(1), -u02));
    double a03 = 1 / (1 + std::pow(std::exp(1), -u03));
    double a04 = 1 / (1 + std::pow(std::exp(1), -u04));

    // Output layer
    double u10 = w1[0][0] * a00 + w1[0][1] * a01 + w1[0][2] * a02 + w1[0][3] * a03 + w1[0][4] * a04 + b1[0];
    double u11 = w1[1][0] * a00 + w1[1][1] * a01 + w1[1][2] * a02 + w1[1][3] * a03 + w1[1][4] * a04 + b1[1];
    double u12 = w1[2][0] * a00 + w1[2][1] * a01 + w1[2][2] * a02 + w1[2][3] * a03 + w1[2][4] * a04 + b1[2];
    double u13 = w1[3][0] * a00 + w1[3][1] * a01 + w1[3][2] * a02 + w1[3][3] * a03 + w1[3][4] * a04 + b1[3];
    double a10 = u10 >= 0 ? 1.0 : 0.0;
    double a11 = u11 >= 0 ? 1.0 : 0.0;
    double a12 = u12 >= 0 ? 1.0 : 0.0;
    double a13 = u13 >= 0 ? 1.0 : 0.0;

    std::cout << "X1: " << a10 << "\n";
    std::cout << "X2: " << a11 << "\n";
    std::cout << "X3: " << a12 << "\n";
    std::cout << "X4: " << a13 << "\n";
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