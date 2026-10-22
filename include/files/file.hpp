#pragma once
#include <variant>
#include <math/math.hpp>

namespace files {
    struct dataframe {
        std::vector<std::string> columns;
        math::matrix<std::variant<int, double, std::string>> data;
    };

    files::dataframe read(std::string path);
}