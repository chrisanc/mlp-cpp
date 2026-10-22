#include <vector>
#include <iostream>
#include <nn/nn.hpp>
#include <files/file.hpp>
#define LR 0.1
#define epochs 1000

template <typename T>
void print_matrix(const math::matrix<T>& m);

math::matrix<double> minus(const int value, const math::matrix<double>& m);
std::vector<double> gradient(const std::vector<double>& b, const math::matrix<double>& delta);
void verify(
    const math::matrix<double>& x, const math::matrix<double>& w0, const std::vector<double> b0,
    const math::matrix<double>& w1, const std::vector<double>& b1, const std::vector<double>& y
);

int main() {
    files::dataframe df = files::read("data/iris_processed.csv");
    std::cout << df.data.shape.i << " - " << df.data.shape.j << "\n";
    // Define the inputs
    math::matrix<double> x = files::to_numeric(df, {"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"});
    // Define the real output values
    std::vector<double> y = files::to_class_indexes(df, "Species");
    // Define the weights
    math::matrix<double> w0 = nn::init_weights({5, 4});
    math::matrix<double> w1 = nn::init_weights({3, 5});
    std::vector<double> b0 = nn::init_bias(5);
    std::vector<double> b1 = nn::init_bias(3);

    std::cout << " === PROCESO DE ENTRENAMIENTO === " << "\n";
    for (int i = 0; i <= epochs; i++) {
        // Calculate U1 and A1
        math::matrix<double> u1 = nn::forward_pass(x, math::transpose(w0), b0);
        math::matrix<double> a1 = nn::sigmoid(u1);
        
        // Calculate U2 and A2
        math::matrix<double> u2 = nn::forward_pass(a1, math::transpose(w1), b1);
        math::matrix<double> a2 = nn::softmax(u2);

        math::matrix<double> hot = nn::one_hot_y(y);
        math::matrix<double> delta1 = a2 - hot;
        math::matrix<double> delta0 = math::matmul(delta1, w1) * (a1 * minus(1, a1));

        // Calculate the gradients
        math::matrix<double> dW1 = math::matmul(math::transpose(delta1), a1) / x.shape.i;
        math::matrix<double> dW0 = math::matmul(math::transpose(delta0), x) / x.shape.i;
        
        // Update the values
        w0 = w0 - (dW0 * LR);
        w1 = w1 - (dW1 * LR);
        b0 = gradient(b0, delta0);
        b1 = gradient(b1, delta1);

        if (i % (epochs / 10) == 0) {
            std::cout << "Perdida (época " << i << "): " << nn::cross_entropy_loss(a2, hot) << "\n";
        }
    }
    std::cout << "\n";

    verify(x, w0, b0, w1, b1, y);
}


/**
 * Prints the given matrix in the console with a pretty format
 */
template <typename T>
void print_matrix(const math::matrix<T>& m) {
    for (int i = 0; i < m.shape.i; i++) {
        for (int j = 0; j < m.shape.j; j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "(" << m.shape.i << ", " << m.shape.j << ")" << "\n";
}


/**
 * Executes a substraction of the given value with the matrix ones
 */
math::matrix<double> minus(const int value, const math::matrix<double>& m) {
    math::matrix<double> r(m.shape.i, m.shape.j);

    for (int i = 0; i < m.shape.i; i++) {
        for (int j = 0; j < m.shape.j; j++) {
            r[i][j] = value - m[i][j];
        }
    }

    return r;
}

/**
 * Calculates the gradient of the given bias
 */
std::vector<double> gradient(const std::vector<double>& b, const math::matrix<double>& delta) {
    std::vector<double> vec(delta.shape.j);

    for (int j = 0; j < delta.shape.j; j++) {
        for (int i = 0; i < delta.shape.i; i++) {
            vec[j] += delta[i][j] / delta.shape.i;
        }
    }

    for (int i = 0; i < b.size(); i++) {
        vec[i] = b[i] - (vec[i] * LR);
    }

    return vec;
}


/**
 * Obtains the index of the maximum value available on the given array
 */
int get_max_idx(const std::vector<double>& m) {
    int max = 0;
    for (int i = 1; i < m.size(); i++) {
        if (m[i] > m[max]) {
            max = i;
        }
    }

    return max;
}


/**
 * Verifies the neural network classifier
 */
void verify(
    const math::matrix<double>& x, const math::matrix<double>& w0, const std::vector<double> b0,
    const math::matrix<double>& w1, const std::vector<double>& b1, const std::vector<double>& y
) {
    math::matrix<double> u0 = nn::forward_pass(x, math::transpose(w0), b0);
    math::matrix<double> a0 = nn::sigmoid(u0);

    math::matrix<double> u1 = nn::forward_pass(a0, math::transpose(w1), b1);
    math::matrix<double> a1 = nn::sigmoid(u1);

    math::matrix<double> confusion(3, 3);

    for (int i = 0; i < y.size(); i++) {
        confusion[y[i]][get_max_idx(a1[i])]++;
    }

    std::cout << " === MATRIZ DE CONFUSIÓN === " << "\n";
    print_matrix(confusion);
}