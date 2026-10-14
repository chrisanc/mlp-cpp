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


math::matrix<double> nn::sigmoid(const math::matrix<double>& x) {
    math::matrix<double> m(x.shape.i, x.shape.j);

    for (int i = 0; i < x.shape.i; i++) {
        for (int j = 0; j < x.shape.j; j++) {
            m[i][j] = math::sigmoid(x[i][j]);
        }
    }

    return m;
}


math::matrix<double> nn::softmax(const math::matrix<double>& x) {
    math::matrix<double> r(x.shape.i, x.shape.j);

    // Fill up the sums vector
    for (int i = 0; i < x.shape.i; i++) {
        double sum = 0.0;
        for (int j = 0; j < x.shape.j; j++) {
            sum += std::exp(x[i][j]);
        }

        for (int j = 0; j < x.shape.j; j++) {
            r[i][j] = std::exp(x[i][j]) / sum;
        }
    }

    return r;
}


std::vector<double> nn::error(const std::vector<double>& y, const std::vector<double>& y_hat) {
    if (y.size() != y_hat.size()) throw std::invalid_argument("Invalid argument shapes...");
    std::vector<double> err(y.size());

    for (int i = 0; i < y.size(); i++) {
        err[i] = y[i] - y_hat[i];
    }

    return err;
}