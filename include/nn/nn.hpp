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
     * Performs the softmax function to the given matrix (row-wise sum)
     */
    math::matrix<double> softmax(const math::matrix<double>& x);

    /**
     * Calculates the error of the activated input
     */
    std::vector<double> error(const std::vector<double>& y, const std::vector<double>& y_hat);


    /**
     * Takes a matrix and returns a vector with the max element row-wise
     */
    std::vector<double> max(math::matrix<double> m);


    /**
     * Takes a vector and returns the average loss using the cross-entropy loss formula
     */
    double cross_entropy_loss(const math::matrix<double> a);


    /**
     * Given the values, returns a one-hot matrix with its corresponding values
     */
    math::matrix<double> one_hot_y(std::vector<double> y);
};