#pragma once
#include <vector>
#include <random>

namespace math {
    class random_generator {
        // Initialize the random devices
        std::random_device dev;
        std::mt19937 rng{dev()};
        std::uniform_real_distribution<double> dist{-1.0, 1.0};

        public:
            double generate() {
                return dist(rng);
            }
    };

    typedef struct shape {
        size_t i, j;

        shape() = default;
        shape(size_t r, size_t c) : i(r), j(c) {}
    } shape;

    template <typename T>
    struct matrix {
        std::vector<std::vector<T>> values;
        math::shape shape = {0, 0};

        matrix() = default;

        matrix(size_t r, size_t c) {
            std::vector<T> cols(c);
            std::vector<std::vector<T>> rows(r);

            for (int i = 0; i < r; i++) {
                rows[i] = cols;
            }

            values = rows;
            shape = {r, c};
        }

        std::vector<T>& operator[](int i) {
            return values[i];
        }

        const std::vector<T>& operator[](int i) const {
            return values[i];
        }
    };

    /**
     * Fills up the given matrix with random numbers
     */
    void random_matrix(math::matrix<double>& m, math::random_generator& generator);

    /**
     * Performs matrix multiplication
     */
    math::matrix<double> matmul(const math::matrix<double>& m1, const math::matrix<double>& m2);

    /**
     * Algorithm used to transpose a matrix
     */
    template <typename T>
    math::matrix<T> transpose(const math::matrix<T>& m) {
        math::matrix<T> matrix(m.shape.j, m.shape.i);

        for (int i = 0; i < m.shape.i; i++) {
            for (int j = 0; j < m.shape.j; j++) {
                matrix[j][i] = m[i][j];
            }
        }

        return matrix;
    }
};

namespace nn {
    /**
     * Takes a shape and returns a matrix filled up with random numbers
     */
    math::matrix<double> init_weights(const math::shape shape);
};