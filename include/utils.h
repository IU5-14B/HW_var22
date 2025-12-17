#pragma once
#include <string>
#include <vector>

namespace utils {

// Сплит по строковому разделителю (у нас "::")
std::vector<std::string> split(const std::string& s, const std::string& delim);

// Трим пробелов по краям (на всякий случай)
std::string trim(const std::string& s);

}
