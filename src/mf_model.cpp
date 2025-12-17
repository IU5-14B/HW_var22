#include "mf_model.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>

// -------------------- helpers --------------------

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

    // SVD++
    Y_.assign((size_t)M * (size_t)k, 0.0);
    user_x_.assign((size_t)U * (size_t)k, 0.0);
    user_items_.assign((size_t)U, {});

    std::mt19937 rng(params_.seed);
    std::uniform_real_distribution<double> dist(-0.05, 0.05);

    for (auto& x : P_) x = dist(rng);
    for (auto& x : Q_) x = dist(rng);
    for (auto& x : Y_) x = dist(rng);
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
    // SVD++: r_hat = mu + b_u + b_i + q_i^T (p_u + x_u)
    const int k = params_.k;
    const size_t u_off = (size_t)u_idx * (size_t)k;
    const size_t m_off = (size_t)m_idx * (size_t)k;

    double dot = 0.0;
    for (int f = 0; f < k; ++f) {
        dot += Q_[m_off + (size_t)f] * (P_[u_off + (size_t)f] + user_x_[u_off + (size_t)f]);
    }
    return mu_ + bu_[u_idx] + bi_[m_idx] + dot;
}

double MFModel::predict(int user_id, int movie_id) const {
    auto it_u = u_id_to_idx_.find(user_id);
    auto it_m = m_id_to_idx_.find(movie_id);

    // Если и пользователь, и фильм известны — полноценный SVD++ прогноз
    if (it_u != u_id_to_idx_.end() && it_m != m_id_to_idx_.end()) {
        return predict_mapped(it_u->second, it_m->second);
    }

    // Иначе — базовый прогноз mu + доступные bias'ы
    double pred = mu_;
    if (it_u != u_id_to_idx_.end()) pred += bu_[it_u->second];
    if (it_m != m_id_to_idx_.end()) pred += bi_[it_m->second];
    return pred;
}

// -------------------- SGD (SVD++) --------------------

void MFModel::sgd_one_epoch(const std::vector<Rating>& train,
                            std::vector<size_t>& order,
                            std::mt19937& rng) {
    const int k = params_.k;
    const double lr = params_.lr;
    const double reg = params_.reg;

    std::shuffle(order.begin(), order.end(), rng);

    // Временный буфер для старого q_i (нужно, чтобы обновлять y_j и x_u корректно)
    std::vector<double> q_old((size_t)k);

    for (size_t t = 0; t < order.size(); ++t) {
        const Rating& r = train[order[t]];

        auto it_u = u_id_to_idx_.find(r.user_id);
        auto it_m = m_id_to_idx_.find(r.movie_id);
        if (it_u == u_id_to_idx_.end() || it_m == m_id_to_idx_.end()) continue;

        int u = it_u->second;
        int i = it_m->second;

        const size_t u_off = (size_t)u * (size_t)k;
        const size_t i_off = (size_t)i * (size_t)k;

        // Сохраняем q_i до обновления
        for (int f = 0; f < k; ++f) q_old[(size_t)f] = Q_[i_off + (size_t)f];

        // Предсказание (SVD++)
        double pred = mu_ + bu_[u] + bi_[i];
        for (int f = 0; f < k; ++f) {
            pred += q_old[(size_t)f] * (P_[u_off + (size_t)f] + user_x_[u_off + (size_t)f]);
        }

        double err = (double)r.rating - pred;

        // bias updates
        bu_[u] += lr * (err - reg * bu_[u]);
        bi_[i] += lr * (err - reg * bi_[i]);

        // implicit normalization
        const auto& Nu = user_items_[u];
        const size_t n_u = Nu.size();
        const double inv_sqrt = (n_u > 0) ? (1.0 / std::sqrt((double)n_u)) : 0.0;

        // Update p_u and q_i
        for (int f = 0; f < k; ++f) {
            const double pu_old = P_[u_off + (size_t)f];
            const double qi_old = q_old[(size_t)f];
            const double xu_old = user_x_[u_off + (size_t)f];

            // p_u
            P_[u_off + (size_t)f] = pu_old + lr * (err * qi_old - reg * pu_old);

            // q_i (важно: использует pu_old + x_u_old)
            Q_[i_off + (size_t)f] = Q_[i_off + (size_t)f] + lr * (err * (pu_old + xu_old) - reg * Q_[i_off + (size_t)f]);
        }

        // Update y_j and cached x_u:
        // y_j += lr*(err*q_i_old*inv_sqrt - reg*y_j)
        // x_u += inv_sqrt * sum_j delta_y_j
        if (n_u > 0 && inv_sqrt > 0.0) {
            for (int j : Nu) {
                const size_t j_off = (size_t)j * (size_t)k;
                for (int f = 0; f < k; ++f) {
                    double& yjf = Y_[j_off + (size_t)f];
                    const double y_old = yjf;

                    const double delta_y = lr * (err * q_old[(size_t)f] * inv_sqrt - reg * y_old);
                    yjf = y_old + delta_y;

                    // x_u is normalized sum of y_j, so add delta_y * inv_sqrt
                    user_x_[u_off + (size_t)f] += delta_y * inv_sqrt;
                }
            }
        }
    }
}

// -------------------- RMSE / fit / logs --------------------

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

        // clamp для явных рейтингов (1..5)
        if (pred < 1.0) pred = 1.0;
        else if (pred > 5.0) pred = 5.0;

        long double err = (long double)r.rating - (long double)pred;
        se += err * err;
        n++;
    }

    if (n == 0) return 0.0L;
    return std::sqrt((long double)(se / (long double)n));
}

bool MFModel::fit_with_early_stopping(const std::vector<Rating>& train,
                                      const std::vector<Rating>& val,
                                      const MFParams& p) {
    params_ = p;
    log_.clear();
    if (train.empty()) return false;

    // mu
    long double sum = 0.0L;
    for (const auto& r : train) sum += (long double)r.rating;
    mu_ = (double)(sum / (long double)train.size());

    build_mappings_from_train(train);
    init_params();

    // Build N(u) from TRAIN (не из val/test)
    for (const auto& r : train) {
        auto it_u = u_id_to_idx_.find(r.user_id);
        auto it_m = m_id_to_idx_.find(r.movie_id);
        if (it_u == u_id_to_idx_.end() || it_m == m_id_to_idx_.end()) continue;
        user_items_[(size_t)it_u->second].push_back(it_m->second);
    }

    // Compute initial x_u = inv_sqrt(|N(u)|) * sum_{j in N(u)} y_j
    const int k = params_.k;
    std::fill(user_x_.begin(), user_x_.end(), 0.0);
    for (size_t u = 0; u < user_items_.size(); ++u) {
        const size_t n_u = user_items_[u].size();
        if (n_u == 0) continue;
        const double inv_sqrt = 1.0 / std::sqrt((double)n_u);

        double* x_ptr = &user_x_[u * (size_t)k];
        for (int j : user_items_[u]) {
            const double* y_ptr = &Y_[(size_t)j * (size_t)k];
            for (int f = 0; f < k; ++f) {
                x_ptr[f] += y_ptr[f] * inv_sqrt;
            }
        }
    }

    // order
    std::vector<size_t> order(train.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;

    std::mt19937 rng(params_.seed);

    double best_val = 1e9;
    int no_improve = 0;

    // store best params (IMPORTANT: include Y_ and user_x_)
    std::vector<double> best_bu, best_bi, best_P, best_Q, best_Y, best_X;

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
            best_Y  = Y_;
            best_X  = user_x_;
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
        Y_  = std::move(best_Y);
        user_x_ = std::move(best_X);
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
