#include <nn/models.hpp>
#include <nn/base.hpp>

math::matrix<double> functions::ReLU::forward(const math::matrix<double>& x) {
    return nn::relu(x);
}

math::matrix<double> functions::Softmax::forward(const math::matrix<double>& x) {
    return nn::softmax(x);
}

math::matrix<double> functions::Sigmoid::forward(const math::matrix<double>& x) {
    return nn::sigmoid(x);
}


void models::MultiLayerPerceptron::forward() {}

void models::MultiLayerPerceptron::update() {}