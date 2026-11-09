#pragma once
#include <initializer_list>
#include <math/math.hpp>
#include <stdexcept>
#include <variant>
#include <vector>
#include <memory>

/**
 * Mathematical object function definition
 */
namespace functions {
    class MathFunc {
    public:
        virtual ~MathFunc() = default;
        virtual math::matrix<double> forward(const math::matrix<double>& x) = 0;
    };

    class ReLU : public MathFunc {
    public:
        math::matrix<double> forward(const math::matrix<double>& x);
    };

    class Softmax : public MathFunc {
    public:
        math::matrix<double> forward(const math::matrix<double>& x);
    };

    class Sigmoid : public MathFunc {
    public:
        math::matrix<double> forward(const math::matrix<double>& x);
    };
}

/**
 * Definition of the various kinds of neural networks
 */
namespace models {
    using ModuleKind = std::variant<std::shared_ptr<functions::MathFunc>, math::matrix<double>>;

    class NeuralNetwork {
    private:
        NeuralNetwork() = default;
    public:
        std::vector<ModuleKind> elements;
        NeuralNetwork(std::initializer_list<ModuleKind> args) {
            if (args.size() < 1) {
                throw std::invalid_argument(
                    "Invalid arguments... there should be at least one element"
                );
            }
            // Verify the first element is actually a matrix
            std::get<math::matrix<double>>(*args.begin());
            elements.assign(args.begin(), args.end());
        };

        virtual void forward() = 0;
        virtual void update() = 0;
    };

    class MultiLayerPerceptron : public NeuralNetwork {
    public:
        void forward();
        void update();
    };
}