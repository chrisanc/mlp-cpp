#include <types/type.hpp>

bool is_int(const std::string& str);
bool is_double(const std::string& str);

type::DType type::detect_type(const std::string& str) {
    if (str.empty()) return DType::String;
    if (is_int(str)) return DType::Integer;
    if (is_double(str)) return DType::Double;

    return DType::String;
}

bool is_int(const std::string& str) {
    try {
        size_t pos = 0;
        std::stoi(str, &pos);
        return pos == str.size();
    } catch (const std::exception& e) {}
    return false;
}

bool is_double(const std::string& str) {
    try {
        size_t pos = 0;
        std::stod(str, &pos);
        return pos == str.size();
    } catch (const std::exception& e) {}
    return false;
}