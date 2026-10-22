#include <iostream>
#include <fstream>
#include <variant>
#include <sstream>
#include <string>
#include <vector>
#include <files/file.hpp>
#include <types/type.hpp>

files::dataframe files::read(std::string path) {
    std::ifstream file(path);
    std::string line;
    files::dataframe df;

    // Variable to verify if it's the first row (column names)
    bool first = true;
    while (std::getline(file, line)) {
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
                case type::DType::Double:
                    row.push_back(std::stod(cell));
                case type::DType::Integer:
                    row.push_back(static_cast<int>(std::stoll(cell)));
                }
                df.data.shape.j++;
            }
        }

        first = false;
        df.data.values.push_back(row);
        df.data.shape.i++;
    }

    df.data.shape.j = df.data.shape.j / df.data.shape.i;

    return df;
}