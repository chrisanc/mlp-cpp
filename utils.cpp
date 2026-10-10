#include <iostream>
#include <random>
#include "utils.hpp"

/*
 Mathematical (math) namespace
 */
void math::random_matrix(math::matrix<double>& m) {
    // Initialize the random devices
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Fill up the matrix
    for (int i = 0; i < m.shape.i; i++) {
        for (int j = 0; j < m.shape.j; j++) {
            m.values[i][j] = dist(rng);
        }
    }
}

/*
 Neural networks (nn) namespace
 */

math::matrix<double> nn::init_weights(const math::shape shape) {
    std::vector<std::vector<double>> rows(shape.i);
    std::vector<double> cols(shape.j);
    math::matrix<double> matrix;

    for (int i = 0; i < shape.i; i++) {
        rows[i] = cols;
    }
    
    matrix.values = rows;
    matrix.shape = shape;

    math::random_matrix(matrix);
    return matrix;
}