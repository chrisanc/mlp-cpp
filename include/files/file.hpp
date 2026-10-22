#pragma once
#include <variant>
#include <iostream>
#include <math/math.hpp>

namespace files {
    struct dataframe {
        std::vector<std::string> columns;
        math::matrix<std::variant<int, double, std::string>> data;

        std::vector<std::variant<int, double, std::string>> operator[](const int& i) {
            return data[i];
        }

        std::vector<std::variant<int, double, std::string>> operator[](const std::string& str) {
            size_t idx = -1;
            // Find the index of the header
            for (int i = 0; i < columns.size(); i++) {
                if (columns[i] == str) {
                    idx = i;
                    break;
                }
            }

            if (idx < 0) throw std::invalid_argument("The given column doesn't exists");

            // Fill the columnar vector
            std::vector<std::variant<int, double, std::string>> vec(data.shape.i);
            for (int i = 0; i < data.shape.i; i++) {
                vec[i] = data[i][idx];
            }
            return vec;
        }
    };

    files::dataframe read(std::string path);
}