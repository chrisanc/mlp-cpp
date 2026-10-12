#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <files/file.hpp>

math::matrix<double> files::read(std::string path) {
    std::ifstream file(path);
    std::string line;
    math::matrix<double> m;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
            m.shape.j++;
        }

        m.values.push_back(row);
        m.shape.i++;
    }

    m.shape.j = m.shape.j / m.shape.i;

    return m;
}