#include "utils.h"
#include <cctype>

namespace utils {

std::vector<std::string> split(const std::string& s, const std::string& delim) {
    std::vector<std::string> out;
    if (delim.empty()) {
        out.push_back(s);
        return out;
    }
    size_t pos = 0;
    while (true) {
        size_t next = s.find(delim, pos);
        if (next == std::string::npos) {
            out.push_back(s.substr(pos));
            break;
        }
        out.push_back(s.substr(pos, next - pos));
        pos = next + delim.size();
    }
    return out;
}

std::string trim(const std::string& s) {
    size_t l = 0;
    while (l < s.size() && std::isspace(static_cast<unsigned char>(s[l]))) l++;
    if (l == s.size()) return "";
    size_t r = s.size() - 1;
    while (r > l && std::isspace(static_cast<unsigned char>(s[r]))) r--;
    return s.substr(l, r - l + 1);
}

}
