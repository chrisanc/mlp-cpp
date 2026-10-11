#pragma once
#include <vector>
#include <math/math.hpp>

namespace nn {
    /**
     * Takes a shape and returns a matrix filled up with random numbers
     */
    math::matrix<double> init_weights(const math::shape shape);

    /**
     * Based on the given data, performs the forward pass on the neural network
     */
    math::matrix<double> forward_pass(const math::matrix<double>& x, const math::matrix<double>& w, const std::vector<double>& b);
};