#include "data_loader.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <limits>

static bool safe_stoi(const std::string& s, int& out) {
    try {
        size_t idx = 0;
        int v = std::stoi(s, &idx);
        if (idx != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

static bool safe_stof(const std::string& s, float& out) {
    try {
        size_t idx = 0;
        float v = std::stof(s, &idx);
        if (idx != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool DataLoader::load_movies(const std::string& movies_path) {
    std::ifstream in(movies_path);
    if (!in) {
        std::cerr << "Ошибка: не удалось открыть файл movies.dat: " << movies_path << "\n";
        return false;
    }

    movie_titles_.clear();

    std::string line;
    size_t line_no = 0;
    size_t bad = 0;

    while (std::getline(in, line)) {
        line_no++;
        line = utils::trim(line);
        if (line.empty()) continue;

        // Формат: MovieID::Title::Genres
        auto parts = utils::split(line, "::");
        if (parts.size() < 2) { bad++; continue; }

        int movie_id = 0;
        if (!safe_stoi(parts[0], movie_id)) { bad++; continue; }

        std::string title = parts[1];
        movie_titles_[movie_id] = title;
    }

    std::cout << "Загружено фильмов (справочник): " << movie_titles_.size()
              << " (некорректных строк: " << bad << ")\n";
    return true;
}

bool DataLoader::load_ratings(const std::string& ratings_path, bool print_preview, int preview_lines) {
    std::ifstream in(ratings_path);
    if (!in) {
        std::cerr << "Ошибка: не удалось открыть файл ratings.dat: " << ratings_path << "\n";
        return false;
    }

    ratings_.clear();
    movie_counts_.clear();
    user_counts_.clear();

    std::string line;
    size_t line_no = 0;
    size_t bad = 0;

    if (print_preview) {
        std::cout << "Предварительный просмотр файла ratings.dat (первые строки):\n";
    }

    while (std::getline(in, line)) {
        line_no++;
        line = utils::trim(line);
        if (line.empty()) continue;

        // Формат: UserID::MovieID::Rating::Timestamp
        auto parts = utils::split(line, "::");
        if (parts.size() != 4) {
            bad++;
            continue;
        }

        int user_id = 0, movie_id = 0, ts = 0;
        float r = 0.0f;

        if (!safe_stoi(parts[0], user_id)) { bad++; continue; }
        if (!safe_stoi(parts[1], movie_id)) { bad++; continue; }
        if (!safe_stof(parts[2], r))        { bad++; continue; }
        if (!safe_stoi(parts[3], ts))       { bad++; continue; }

        Rating rr;
        rr.user_id = user_id;
        rr.movie_id = movie_id;
        rr.rating = r;
        rr.timestamp = ts;

        ratings_.push_back(rr);
        movie_counts_[movie_id] += 1;
        user_counts_[user_id] += 1;

        if (print_preview && (int)line_no <= preview_lines) {
            std::cout << "  Пользователь " << user_id
                      << " -> фильм " << movie_id
                      << ", рейтинг=" << r
                      << ", время=" << ts << "\n";
        }
    }

    std::cout << "Загружено пользовательских оценок: " << ratings_.size()
              << " (некорректных строк: " << bad << ")\n";
    return true;
}

DatasetStats DataLoader::compute_stats() const {
    DatasetStats st;
    st.num_ratings = ratings_.size();
    st.num_users = user_counts_.size();
    st.num_movies = movie_counts_.size();

    if (ratings_.empty()) return st;

    long double sum = 0.0;
    int min_u = std::numeric_limits<int>::max();
    int max_u = std::numeric_limits<int>::min();
    int min_m = std::numeric_limits<int>::max();
    int max_m = std::numeric_limits<int>::min();

    for (const auto& rr : ratings_) {
        sum += rr.rating;
        if (rr.user_id < min_u) min_u = rr.user_id;
        if (rr.user_id > max_u) max_u = rr.user_id;
        if (rr.movie_id < min_m) min_m = rr.movie_id;
        if (rr.movie_id > max_m) max_m = rr.movie_id;
    }

    st.mean_rating = static_cast<double>(sum / (long double)ratings_.size());
    st.min_user_id = min_u; st.max_user_id = max_u;
    st.min_movie_id = min_m; st.max_movie_id = max_m;
    return st;
}

bool DataLoader::save_stats_csv(const std::string& path, const DatasetStats& st) const {
    std::ofstream out(path);
    if (!out) return false;

    out << "metric,value\n";
    out << "num_ratings," << st.num_ratings << "\n";
    out << "num_users," << st.num_users << "\n";
    out << "num_movies," << st.num_movies << "\n";
    out << "mean_rating," << st.mean_rating << "\n";
    out << "min_user_id," << st.min_user_id << "\n";
    out << "max_user_id," << st.max_user_id << "\n";
    out << "min_movie_id," << st.min_movie_id << "\n";
    out << "max_movie_id," << st.max_movie_id << "\n";
    return true;
}

bool DataLoader::save_counts_csv(const std::string& path,
                                const std::unordered_map<int,int>& counts,
                                const std::string& id_col,
                                const std::string& count_col) const {
    std::ofstream out(path);
    if (!out) return false;

    out << id_col << "," << count_col << "\n";
    for (const auto& kv : counts) {
        out << kv.first << "," << kv.second << "\n";
    }
    return true;
}
