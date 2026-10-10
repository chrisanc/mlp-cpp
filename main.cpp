#include <iostream>
#include "utils.hpp"
#define CASES 100

int main() {
    int in, hidden, out;
    std::cout << "Ingresa los valores de la capa de entrada, oculta y salida: ";
    std::cin >> in >> hidden >> out;

    math::matrix<double> values = nn::init_weights({2, 2});
    std::cout << values.values[0][0] << "\n";
    return 0;
}