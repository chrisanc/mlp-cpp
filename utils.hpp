#pragma once
#include <vector>

namespace math {
    typedef struct shape {
        size_t i, j;

        shape() = default;
        shape(int r, int c) : i(r), j(c) {}
    } shape;

    template <typename T>
    struct matrix {
        std::vector<std::vector<T>> values;
        math::shape shape = {0, 0};
    };

    /**
     * Fills up the given matrix with random numbers
     */
    void random_matrix(math::matrix<double>& m);
};

namespace nn {
    /**
     * Takes a shape and returns a matrix filled up with random numbers
     */
    math::matrix<double> init_weights(const math::shape shape);
};