#include <iostream>
#include "utils.hpp"
#define CASES 100

template <typename T>
void print_matrix(const math::matrix<T>& m);

int main() {
    size_t in, hidden, out;
    std::cout << "Ingresa los valores de la capa de entrada, oculta y salida: ";
    std::cin >> in >> hidden >> out;

    math::matrix<double> in_weights = nn::init_weights({in, hidden});
    math::matrix<double> out_weights = nn::init_weights({hidden, out});

    /*print_matrix(in_weights);
    math::matrix<double> v = math::transpose(in_weights);
    std::cout << "\n";
    print_matrix(v);*/
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