#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>          // <-- ВАЖНО: для uint32_t
#include "data_loader.h"

struct MFParams {
    int k = 30;              // число латентных факторов
    int epochs = 25;         // число эпох
    double lr = 0.01;        // скорость обучения
    double reg = 0.05;       // регуляризация
    uint32_t seed = 42;      // для воспроизводимости
};

struct MFEvalInfo {
    double rmse = 0.0;
    size_t cold_user_cases = 0;   // в test пользователь не встречался в train
    size_t cold_movie_cases = 0;  // в test фильм не встречался в train
};

class MFModel {
public:
    bool fit(const std::vector<Rating>& train, const MFParams& p);
    double predict(int user_id, int movie_id) const;

    MFEvalInfo evaluate(const std::vector<Rating>& test) const;

    // Логи обучения (для отчёта)
    bool save_training_log_csv(const std::string& path) const;

    double mu() const { return mu_; }
    int num_users_train() const { return (int)u_id_to_idx_.size(); }
    int num_movies_train() const { return (int)m_id_to_idx_.size(); }
    MFParams params() const { return params_; }

private:
    // mapping построен по train (важно для cold start)
    std::unordered_map<int,int> u_id_to_idx_;
    std::unordered_map<int,int> m_id_to_idx_;

    MFParams params_;
    double mu_ = 0.0;

    // параметры модели
    std::vector<double> bu_; // bias пользователя
    std::vector<double> bi_; // bias фильма
    std::vector<double> P_;  // U x k
    std::vector<double> Q_;  // M x k

    // лог по эпохам
    struct LogRow {
        int epoch = 0;
        double train_rmse = 0.0;
        double train_time_ms = 0.0;
    };
    std::vector<LogRow> log_;

private:
    void build_mappings_from_train(const std::vector<Rating>& train);
    void init_params();
    double dot_uv(int u_idx, int m_idx) const;
};
