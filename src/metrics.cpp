#include "metrics.h"
#include <cmath>
#include <fstream>

namespace metrics {

static double compute_mu(const std::vector<Rating>& train) {
    if (train.empty()) return 0.0;
    long double sum = 0.0;
    for (const auto& r : train) sum += r.rating;
    return static_cast<double>(sum / (long double)train.size());
}

double rmse_global_mean(const std::vector<Rating>& train,
                        const std::vector<Rating>& test,
                        double& out_mu) {
    out_mu = compute_mu(train);
    if (test.empty()) return 0.0;

    long double se = 0.0; // sum of squared errors
    for (const auto& r : test) {
        double e = (double)r.rating - out_mu;
        se += e * e;
    }
    return std::sqrt((double)(se / (long double)test.size()));
}

double rmse_user_mean(const std::vector<Rating>& train,
                      const std::vector<Rating>& test,
                      double mu,
                      size_t& out_cold_users) {
    out_cold_users = 0;
    if (test.empty()) return 0.0;

    // user -> (sum, count)
    std::unordered_map<int, std::pair<long double, int>> acc;
    acc.reserve(train.size() / 20);

    for (const auto& r : train) {
        auto& cell = acc[r.user_id];
        cell.first += r.rating;
        cell.second += 1;
    }

    long double se = 0.0;
    for (const auto& r : test) {
        auto it = acc.find(r.user_id);
        double pred = mu;
        if (it != acc.end() && it->second.second > 0) {
            pred = (double)(it->second.first / (long double)it->second.second);
        } else {
            // холодный старт: пользователь не встречался в train
            out_cold_users += 1;
        }
        double e = (double)r.rating - pred;
        se += e * e;
    }

    return std::sqrt((double)(se / (long double)test.size()));
}

bool save_baseline_csv(const std::string& path,
                       double mu,
                       double rmse_mu,
                       double rmse_user,
                       size_t cold_users,
                       size_t train_size,
                       size_t test_size) {
    std::ofstream out(path);
    if (!out) return false;

    out << "metric,value\n";
    out << "train_size," << train_size << "\n";
    out << "test_size," << test_size << "\n";
    out << "mu_train," << mu << "\n";
    out << "rmse_global_mean," << rmse_mu << "\n";
    out << "rmse_user_mean," << rmse_user << "\n";
    out << "cold_start_user_cases_in_test," << cold_users << "\n";
    return true;
}

}
