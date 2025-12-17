#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <random>   // std::mt19937
#include "data_loader.h"

struct MFParams {
    int k = 50;
    int epochs = 60;
    double lr = 0.007;
    double reg = 0.05;
    uint32_t seed = 42;

    int patience = 5; // early stopping
};

struct MFEvalInfo {
    double rmse = 0.0;
    size_t cold_user_cases = 0;
    size_t cold_movie_cases = 0;
};

class MFModel {
public:
    bool fit_with_early_stopping(const std::vector<Rating>& train,
                                 const std::vector<Rating>& val,
                                 const MFParams& p);

    double predict(int user_id, int movie_id) const;
    MFEvalInfo evaluate(const std::vector<Rating>& test) const;

    bool save_training_log_csv(const std::string& path) const;

    double mu() const { return mu_; }
    int num_users_train() const { return (int)u_id_to_idx_.size(); }
    int num_movies_train() const { return (int)m_id_to_idx_.size(); }

private:
    std::unordered_map<int,int> u_id_to_idx_;
    std::unordered_map<int,int> m_id_to_idx_;

    MFParams params_;
    double mu_ = 0.0;

    std::vector<double> bu_;
    std::vector<double> bi_;
    std::vector<double> P_;
    std::vector<double> Q_;

    struct LogRow {
        int epoch = 0;
        double train_rmse = 0.0;
        double val_rmse = 0.0;
        double epoch_time_ms = 0.0;
    };
    std::vector<LogRow> log_;

private:
    void build_mappings_from_train(const std::vector<Rating>& train);
    void init_params();

    double dot_uv(int u_idx, int m_idx) const;
    double predict_mapped(int u_idx, int m_idx) const;

    long double rmse_on_set_mapped_ld(const std::vector<Rating>& data,
                                     size_t& cold_u,
                                     size_t& cold_m) const;

    void sgd_one_epoch(const std::vector<Rating>& train,
                       std::vector<size_t>& order,
                       std::mt19937& rng);
};
