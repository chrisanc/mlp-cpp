#include <iostream>
#include <nn/nn.hpp>

/*
 Neural networks (nn) namespace
 */

math::matrix<double> nn::init_weights(const math::shape shape) {
    math::matrix<double> matrix(shape.i, shape.j);
    math::random_generator& gen = math::random_generator::get();

    math::random_matrix(matrix, gen);
    return matrix;
}

std::vector<double> nn::init_bias(const size_t n) {
    std::vector<double> vec(n);
    math::random_generator& gen = math::random_generator::get();

    for (int i = 0; i < n; i++) {
        vec[i] = gen.generate();
    }
    return vec;
}


math::matrix<double> nn::forward_pass(const math::matrix<double>& x, const math::matrix<double>& w, const std::vector<double>& b) {
    if (w.shape.j != b.size()) throw new std::invalid_argument("The given shapes aren't valid...");
    
    math::matrix<double> m = math::matmul(x, w);

    size_t idx = 0;
    for (int j = 0; j < m.shape.j; j++) {
        for (int i = 0; i < m.shape.i; i++) {
            m[i][j] += b[idx];
        }
        idx++;
    }

    return m;
}
