#include "mf_model.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>

void MFModel::build_mappings_from_train(const std::vector<Rating>& train) {
    u_id_to_idx_.clear();
    m_id_to_idx_.clear();

    int u_next = 0;
    int m_next = 0;

    for (const auto& r : train) {
        if (u_id_to_idx_.find(r.user_id) == u_id_to_idx_.end())
            u_id_to_idx_[r.user_id] = u_next++;
        if (m_id_to_idx_.find(r.movie_id) == m_id_to_idx_.end())
            m_id_to_idx_[r.movie_id] = m_next++;
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
    std::uniform_real_distribution<double> dist(-0.05, 0.05);

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

double MFModel::predict_mapped(int u_idx, int m_idx) const {
    return mu_ + bu_[u_idx] + bi_[m_idx] + dot_uv(u_idx, m_idx);
}

double MFModel::predict(int user_id, int movie_id) const {
    auto it_u = u_id_to_idx_.find(user_id);
    auto it_m = m_id_to_idx_.find(movie_id);

    double pred = mu_;
    if (it_u != u_id_to_idx_.end()) pred += bu_[it_u->second];
    if (it_m != m_id_to_idx_.end()) pred += bi_[it_m->second];
    if (it_u != u_id_to_idx_.end() && it_m != m_id_to_idx_.end())
        pred += dot_uv(it_u->second, it_m->second);

    return pred;
}

void MFModel::sgd_one_epoch(const std::vector<Rating>& train,
                            std::vector<size_t>& order,
                            std::mt19937& rng) {
    const int k = params_.k;
    const double lr = params_.lr;
    const double reg = params_.reg;

    std::shuffle(order.begin(), order.end(), rng);

    for (size_t jj = 0; jj < order.size(); ++jj) {
        const Rating& r = train[order[jj]];

        auto it_u = u_id_to_idx_.find(r.user_id);
        auto it_m = m_id_to_idx_.find(r.movie_id);
        if (it_u == u_id_to_idx_.end() || it_m == m_id_to_idx_.end()) continue;

        int u = it_u->second;
        int m = it_m->second;

        double pred = predict_mapped(u, m);
        double err = (double)r.rating - pred;

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
}

long double MFModel::rmse_on_set_mapped_ld(const std::vector<Rating>& data,
                                          size_t& cold_u,
                                          size_t& cold_m) const {
    cold_u = 0;
    cold_m = 0;
    if (data.empty()) return 0.0L;

    long double se = 0.0L;
    size_t n = 0;

    for (const auto& r : data) {
        bool cu = (u_id_to_idx_.find(r.user_id) == u_id_to_idx_.end());
        bool cm = (m_id_to_idx_.find(r.movie_id) == m_id_to_idx_.end());
        if (cu) cold_u++;
        if (cm) cold_m++;

        double pred = predict(r.user_id, r.movie_id);
        long double err = (long double)r.rating - (long double)pred;

        se += err * err;
        n++;
    }

    if (n == 0) return 0.0L;
    // чтобы не было ambiguous sqrt — считаем в long double и используем sqrtl
    return std::sqrt((long double)(se / (long double)n));

}

bool MFModel::fit_with_early_stopping(const std::vector<Rating>& train,
                                      const std::vector<Rating>& val,
                                      const MFParams& p) {
    params_ = p;
    log_.clear();
    if (train.empty()) return false;

    long double sum = 0.0L;
    for (const auto& r : train) sum += (long double)r.rating;
    mu_ = (double)(sum / (long double)train.size());

    build_mappings_from_train(train);
    init_params();

    std::vector<size_t> order(train.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;

    std::mt19937 rng(params_.seed);

    double best_val = 1e9;
    int no_improve = 0;

    std::vector<double> best_bu, best_bi, best_P, best_Q;

    for (int ep = 1; ep <= params_.epochs; ++ep) {
        auto t0 = std::chrono::steady_clock::now();

        sgd_one_epoch(train, order, rng);

        size_t cu_t=0, cm_t=0, cu_v=0, cm_v=0;
        long double train_rmse_ld = rmse_on_set_mapped_ld(train, cu_t, cm_t);
        long double val_rmse_ld   = rmse_on_set_mapped_ld(val,   cu_v, cm_v);

        auto t1 = std::chrono::steady_clock::now();
        double ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        log_.push_back({ep, (double)train_rmse_ld, (double)val_rmse_ld, ms});

        if ((double)val_rmse_ld + 1e-6 < best_val) {
            best_val = (double)val_rmse_ld;
            no_improve = 0;

            best_bu = bu_;
            best_bi = bi_;
            best_P  = P_;
            best_Q  = Q_;
        } else {
            no_improve++;
            if (no_improve >= params_.patience) break;
        }
    }

    if (!best_bu.empty()) {
        bu_ = std::move(best_bu);
        bi_ = std::move(best_bi);
        P_  = std::move(best_P);
        Q_  = std::move(best_Q);
    }

    return true;
}

MFEvalInfo MFModel::evaluate(const std::vector<Rating>& test) const {
    MFEvalInfo info;
    size_t cu=0, cm=0;
    long double rmse_ld = rmse_on_set_mapped_ld(test, cu, cm);
    info.rmse = (double)rmse_ld;
    info.cold_user_cases = cu;
    info.cold_movie_cases = cm;
    return info;
}

bool MFModel::save_training_log_csv(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    out << "epoch,train_rmse,val_rmse,epoch_time_ms\n";
    for (const auto& r : log_) {
        out << r.epoch << "," << r.train_rmse << "," << r.val_rmse << "," << r.epoch_time_ms << "\n";
    }
    return true;
}
