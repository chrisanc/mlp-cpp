#pragma once
#include <math/math.hpp>

/**
 * Mathematical object function definition
 */

namespace functions {
    class MathFunc {
    public:
        virtual math::matrix<double> forward(const math::matrix<double>& x);
        virtual ~MathFunc() = default;
    };

    class ReLU : public MathFunc {
        math::matrix<double> forward(const math::matrix<double>& x);
    };

    class Softmax : public MathFunc {
        math::matrix<double> forward(const math::matrix<double>& x);
    };

    class Sigmoid : public MathFunc {
        math::matrix<double> forward(const math::matrix<double>& x);
    };
}
/**
 * Definition of the various kinds of neural networks
 */
namespace models {
    class MultiLayerPerceptron {
        MultiLayerPerceptron() = default;
    };
}