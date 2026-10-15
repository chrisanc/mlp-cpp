#include <iostream>
#include <cmath>
#include <math/math.hpp>

/*
 Mathematical (math) namespace
 */
void math::random_matrix(math::matrix<double>& m, math::random_generator& generator) {
    // Fill up the matrix
    for (int i = 0; i < m.shape.i; i++) {
        for (int j = 0; j < m.shape.j; j++) {
            m.values[i][j] = generator.generate();
        }
    }
}

math::matrix<double> math::matmul(const math::matrix<double>& m1, const math::matrix<double>& m2) {
    if (m1.shape.j != m2.shape.i) throw std::invalid_argument("Invalid shapes between m1 and m2");
    
    math::matrix<double> m(m1.shape.i, m2.shape.j);
    // Transpose the second matrix for easier calculations
    math::matrix<double> m2T = math::transpose(m2);

    for (int i = 0; i < m1.shape.i; i++) {
        for (int j = 0; j < m2T.shape.i; j++) {
            double value = 0.0;
            for (int k = 0; k < m2T.shape.j; k++) {
                value += m1[i][k] * m2T[j][k];
            }
            m[i][j] = value;
        }
    }

    return m;
}


double math::sigmoid(const double x) {
    return 1 / (1 + std::pow(std::exp(1), -x));
}


double math::max(std::vector<double> v) {
    double max = v[0];

    for (int i = 1; i < v.size(); i++) {
        max = std::max(max, v[i]);
    }

    return max;
}