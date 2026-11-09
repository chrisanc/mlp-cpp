#include <nn/models.hpp>
#include <nn/base.hpp>
#include <variant>
#include <memory>

math::matrix<double> functions::ReLU::forward(const math::matrix<double>& x) {
    return nn::relu(x);
}

math::matrix<double> functions::Softmax::forward(const math::matrix<double>& x) {
    return nn::softmax(x);
}

math::matrix<double> functions::Sigmoid::forward(const math::matrix<double>& x) {
    return nn::sigmoid(x);
}


void models::MultiLayerPerceptron::forward() {
    math::matrix<double> a;
    math::matrix<double> w = std::get<math::matrix<double>>(elements[0]);
    for (int i = 1; i < elements.size(); i++) {
        if (auto func = std::get_if<std::shared_ptr<functions::MathFunc>>(&elements[i])) {
            a = func->get()->forward(w);
        } else {
            w = math::matmul(w, std::get<math::matrix<double>>(elements[i]));
        }
    }
}

void models::MultiLayerPerceptron::update() {}