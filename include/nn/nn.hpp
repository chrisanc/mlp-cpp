#pragma once
#include <vector>
#include <math/math.hpp>

namespace nn {
    /**
     * Takes a shape and returns a matrix filled up with random numbers
     */
    math::matrix<double> init_weights(const math::shape shape);

    /**
     * Initializes random biases
     */
    std::vector<double> init_bias(const size_t n);

    /**
     * Based on the given data, performs the forward pass on the neural network
     */
    math::matrix<double> forward_pass(const math::matrix<double>& x, const math::matrix<double>& w, const std::vector<double>& b);

    /**
     * Performs the sigmoid calculation on the given matrix
     */
    math::matrix<double> sigmoid(const math::matrix<double>& x);

    /**
     * Calculates the error of the activated input
     */
    std::vector<double> error(const std::vector<double>& y, const std::vector<double>& y_hat);
};