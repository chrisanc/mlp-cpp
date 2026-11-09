#include <math/math.hpp>
#include <stdexcept>

template <typename T>
std::vector<T>& math::matrix<T>::operator[](int i) {
    return values[i];
}

template <typename T>
const std::vector<T>& math::matrix<T>::operator[](int i) const {
    return values[i];
}

template <typename T>
const math::matrix<T> math::matrix<T>::operator-(const math::matrix<T>& m) {
    if (m.shape.i != shape.i || m.shape.j != shape.j) {
        throw std::invalid_argument("The given shapes aren't valid");
    }
    math::matrix<T> r(m.shape.i, m.shape.j);

    for (int i = 0; i < m.shape.i; i++) {
        for (int j = 0; j < m.shape.j; j++) {
            r[i][j] = values[i][j] - m[i][j];
        }
    }

    return r;
}

template <typename T>
const math::matrix<T> math::matrix<T>::operator*(const math::matrix<T>& m) {
    if (m.shape.i != shape.i || m.shape.j != shape.j) {
        throw std::invalid_argument("The given shapes aren't valid");
    }
    math::matrix<T> r(m.shape.i, m.shape.j);

    for (int i = 0; i < m.shape.i; i++) {
        for (int j = 0; j < m.shape.j; j++) {
            r[i][j] = values[i][j] * m[i][j];
        }
    }
    return r;
}

template <typename T>
const math::matrix<T> math::matrix<T>::operator*(const double m) {
    math::matrix<T> r(shape.i, shape.j);

    for (int i = 0; i < shape.i; i++) {
        for (int j = 0; j < shape.j; j++) {
            r[i][j] = values[i][j] * m;
        }
    }
    return r;
}

template <typename T>
const math::matrix<T> math::matrix<T>::operator/(const double m) {
    math::matrix<T> r(shape.i, shape.j);

    for (int i = 0; i < shape.i; i++) {
        for (int j = 0; j < shape.j; j++) {
            r[i][j] = values[i][j] / m;
        }
    }
    return r;
}

template <typename T>
math::matrix<T> math::transpose(const math::matrix<T>& m) {
    math::matrix<T> matrix(m.shape.j, m.shape.i);

    for (int i = 0; i < m.shape.i; i++) {
        for (int j = 0; j < m.shape.j; j++) {
            matrix[j][i] = m[i][j];
        }
    }

    return matrix;
}