#pragma once
#include <iostream>

namespace type {
    enum DType {
        Integer,
        Double,
        String
    };

    /**
     * Takes a string value and returns if it contains any kind of number or a string itself
     */
    DType detect_type(const std::string& str);
};