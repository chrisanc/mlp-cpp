#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <stdexcept>
#include <variant>
#include <sstream>
#include <string>
#include <vector>
#include <files/file.hpp>
#include <types/type.hpp>

namespace {
    struct to_double {
        double operator()(int value) const { return value; }
        double operator()(double value) const { return value; }
        double operator()(const std::string& value) const {
            throw std::invalid_argument("Non numeric value: " + value);
        }
    };

    size_t column_index(const files::dataframe& df, const std::string& name) {
        auto it = std::find(df.columns.begin(), df.columns.end(), name);
        if (it == df.columns.end()) throw std::invalid_argument("The given column doesn't exists");
        return std::distance(df.columns.begin(), it);
    }
}

files::dataframe files::read(std::string path) {
    std::ifstream file(path);
    std::string line;
    files::dataframe df;

    // Variable to verify if it's the first row (column names)
    bool first = true;
    while (std::getline(file, line)) {
        // Windows line endings leave a trailing '\r' that would end up in the last cell
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::variant<int, double, std::string>> row;

        while (std::getline(ss, cell, ',')) {
            if (first) {
                df.columns.push_back(cell);
            } else {
                type::DType dtype = type::detect_type(cell);
                switch (dtype) {
                case type::DType::String:
                    row.push_back(cell);
                    break;
                case type::DType::Double:
                    row.push_back(std::stod(cell));
                    break;
                case type::DType::Integer:
                    row.push_back(static_cast<int>(std::stoll(cell)));
                    break;
                }
            }
        }
        if (!first) df.data.values.push_back(row);
        first = false;
    }

    df.data.shape.i = df.data.values.size();
    df.data.shape.j = df.columns.size();

    return df;
}

math::matrix<double> files::to_numeric(const dataframe& df, const std::vector<std::string>& names) {
    std::vector<size_t> indexes;
    for (const auto& name : names) {
        indexes.push_back(column_index(df, name));
    }

    math::matrix<double> m(df.data.shape.i, indexes.size());
    for (size_t i = 0; i < df.data.shape.i; i++) {
        for (size_t j = 0; j < indexes.size(); j++) {
            m[i][j] = std::visit(to_double{}, df.data[i][indexes[j]]);
        }
    }
    return m;
}

std::vector<double> files::to_numeric_column(const dataframe& df, const std::string& name) {
    size_t index = column_index(df, name);

    std::vector<double> column(df.data.shape.i);
    for (size_t i = 0; i < df.data.shape.i; i++) {
        column[i] = std::visit(to_double{}, df.data[i][index]);
    }
    return column;
}

std::vector<double> files::to_class_indexes(const dataframe& df, const std::string& name) {
    size_t index = column_index(df, name);
    std::map<std::string, double> class_indexes;

    std::vector<double> classes(df.data.shape.i);
    for (size_t i = 0; i < df.data.shape.i; i++) {
        const auto* value = std::get_if<std::string>(&df.data[i][index]);
        if (!value) throw std::invalid_argument("Expected a text value in column " + name);

        auto it = class_indexes.emplace(*value, class_indexes.size()).first;
        classes[i] = it->second;
    }
    return classes;
}