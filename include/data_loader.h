#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct Rating {
    int user_id = 0;
    int movie_id = 0;
    float rating = 0.0f;
    int timestamp = 0; // можно игнорировать дальше, но полезно сохранить
};

struct DatasetStats {
    size_t num_ratings = 0;
    size_t num_users = 0;
    size_t num_movies = 0;
    double mean_rating = 0.0;

    int min_user_id = 0, max_user_id = 0;
    int min_movie_id = 0, max_movie_id = 0;
};

class DataLoader {
public:
    bool load_movies(const std::string& movies_path);
    bool load_ratings(const std::string& ratings_path, bool print_preview = true, int preview_lines = 5);

    const std::unordered_map<int, std::string>& movie_titles() const { return movie_titles_; }
    const std::vector<Rating>& ratings() const { return ratings_; }

    const std::unordered_map<int, int>& movie_rating_counts() const { return movie_counts_; }
    const std::unordered_map<int, int>& user_rating_counts() const { return user_counts_; }

    DatasetStats compute_stats() const;

    // Сохранение CSV для отчёта
    bool save_stats_csv(const std::string& path, const DatasetStats& st) const;
    bool save_counts_csv(const std::string& path,
                         const std::unordered_map<int,int>& counts,
                         const std::string& id_col,
                         const std::string& count_col) const;

private:
    std::unordered_map<int, std::string> movie_titles_;
    std::vector<Rating> ratings_;

    std::unordered_map<int, int> movie_counts_;
    std::unordered_map<int, int> user_counts_;
};
