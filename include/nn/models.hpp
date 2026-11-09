#pragma once
#include <math/math.hpp>
#include <initializer_list>
#include <variant>
#include <vector>

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
    using ModuleKind = std::variant<functions::MathFunc, math::matrix<double>>;

    class NeuralNetwork {
    private:
        std::initializer_list<ModuleKind> elements;
        NeuralNetwork() = default;
    public:
        NeuralNetwork(std::initializer_list<ModuleKind> args) : elements(args) {};
        virtual void forward();
        virtual void update();
    };

    class MultiLayerPerceptron : public NeuralNetwork {};
}