#include "mf_model.h"
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>
#include <algorithm>

static double clamp_rating(double x) {
    if (x < 1.0) return 1.0;
    if (x > 5.0) return 5.0;
    return x;
}

void MFModel::build_mappings_from_train(const std::vector<Rating>& train) {
    u_id_to_idx_.clear();
    m_id_to_idx_.clear();

    int u_next = 0;
    int m_next = 0;

    u_id_to_idx_.reserve(train.size() / 50);
    m_id_to_idx_.reserve(train.size() / 200);

    for (const auto& r : train) {
        if (u_id_to_idx_.find(r.user_id) == u_id_to_idx_.end()) {
            u_id_to_idx_[r.user_id] = u_next++;
        }
        if (m_id_to_idx_.find(r.movie_id) == m_id_to_idx_.end()) {
            m_id_to_idx_[r.movie_id] = m_next++;
        }
    }
}

void MFModel::init_params() {
    const int U = (int)u_id_to_idx_.size();
    const int M = (int)m_id_to_idx_.size();
    const int k = params_.k;

    bu_.assign(U, 0.0);
    bi_.assign(M, 0.0);
    P_.assign((size_t)U * (size_t)k, 0.0);
    Q_.assign((size_t)M * (size_t)k, 0.0);

    std::mt19937 rng(params_.seed);

    // чуть более "мелкая" инициализация помогает стабильности
    std::uniform_real_distribution<double> dist(-0.02, 0.02);

    for (auto& x : P_) x = dist(rng);
    for (auto& x : Q_) x = dist(rng);
}

double MFModel::dot_uv(int u_idx, int m_idx) const {
    const int k = params_.k;
    const size_t u_off = (size_t)u_idx * (size_t)k;
    const size_t m_off = (size_t)m_idx * (size_t)k;

    double s = 0.0;
    for (int f = 0; f < k; ++f) {
        s += P_[u_off + (size_t)f] * Q_[m_off + (size_t)f];
    }
    return s;
}

bool MFModel::fit(const std::vector<Rating>& train, const MFParams& p) {
    params_ = p;
    log_.clear();
    if (train.empty()) return false;

    long double sum = 0.0;
    for (const auto& r : train) sum += r.rating;
    mu_ = (double)(sum / (long double)train.size());

    build_mappings_from_train(train);
    init_params();

    const int k = params_.k;
    const double lr0 = params_.lr;
    const double reg = params_.reg;

    // decay: чем больше, тем быстрее уменьшается lr
    // это помогает "доползти" ниже по RMSE после грубой подгонки
    const double decay = 0.08;

    std::vector<size_t> order(train.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;

    std::mt19937 rng(params_.seed);

    for (int ep = 1; ep <= params_.epochs; ++ep) {
        auto t0 = std::chrono::steady_clock::now();

        std::shuffle(order.begin(), order.end(), rng);

        // lr уменьшаем по эпохам
        const double lr = lr0 / (1.0 + decay * (double)(ep - 1));

        long double se = 0.0;
        size_t n = 0;

        for (size_t jj = 0; jj < order.size(); ++jj) {
            const Rating& r = train[order[jj]];

            auto it_u = u_id_to_idx_.find(r.user_id);
            auto it_m = m_id_to_idx_.find(r.movie_id);
            if (it_u == u_id_to_idx_.end() || it_m == m_id_to_idx_.end()) continue;

            int u = it_u->second;
            int m = it_m->second;

            double pred = mu_ + bu_[u] + bi_[m] + dot_uv(u, m);
            pred = clamp_rating(pred);

            double err = (double)r.rating - pred;

            se += err * err;
            n++;

            bu_[u] += lr * (err - reg * bu_[u]);
            bi_[m] += lr * (err - reg * bi_[m]);

            const size_t u_off = (size_t)u * (size_t)k;
            const size_t m_off = (size_t)m * (size_t)k;

            for (int f = 0; f < k; ++f) {
                double pu = P_[u_off + (size_t)f];
                double qi = Q_[m_off + (size_t)f];

                P_[u_off + (size_t)f] = pu + lr * (err * qi - reg * pu);
                Q_[m_off + (size_t)f] = qi + lr * (err * pu - reg * qi);
            }
        }

        double train_rmse = (n == 0) ? 0.0 : std::sqrt((double)(se / (long double)n));

        auto t1 = std::chrono::steady_clock::now();
        double ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        LogRow row;
        row.epoch = ep;
        row.train_rmse = train_rmse;
        row.train_time_ms = ms;
        log_.push_back(row);
    }

    return true;
}

double MFModel::predict(int user_id, int movie_id) const {
    auto it_u = u_id_to_idx_.find(user_id);
    auto it_m = m_id_to_idx_.find(movie_id);

    double pred = mu_;
    if (it_u != u_id_to_idx_.end()) pred += bu_[it_u->second];
    if (it_m != m_id_to_idx_.end()) pred += bi_[it_m->second];
    if (it_u != u_id_to_idx_.end() && it_m != m_id_to_idx_.end()) {
        pred += dot_uv(it_u->second, it_m->second);
    }

    return clamp_rating(pred);
}

MFEvalInfo MFModel::evaluate(const std::vector<Rating>& test) const {
    MFEvalInfo info;
    if (test.empty()) return info;

    long double se = 0.0;
    size_t n = 0;

    for (const auto& r : test) {
        bool cold_u = (u_id_to_idx_.find(r.user_id) == u_id_to_idx_.end());
        bool cold_m = (m_id_to_idx_.find(r.movie_id) == m_id_to_idx_.end());
        if (cold_u) info.cold_user_cases++;
        if (cold_m) info.cold_movie_cases++;

        double pred = predict(r.user_id, r.movie_id);
        double err = (double)r.rating - pred;

        se += err * err;
        n++;
    }

    info.rmse = (n == 0) ? 0.0 : std::sqrt((double)(se / (long double)n));
    return info;
}

bool MFModel::save_training_log_csv(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return false;

    out << "epoch,train_rmse,train_time_ms\n";
    for (const auto& r : log_) {
        out << r.epoch << "," << r.train_rmse << "," << r.train_time_ms << "\n";
    }
    return true;
}
