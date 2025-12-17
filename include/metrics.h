#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include "data_loader.h"

namespace metrics {

double rmse_global_mean(const std::vector<Rating>& train,
                        const std::vector<Rating>& test,
                        double& out_mu);

double rmse_user_mean(const std::vector<Rating>& train,
                      const std::vector<Rating>& test,
                      double mu,
                      size_t& out_cold_users);

bool save_baseline_csv(const std::string& path,
                       double mu,
                       double rmse_mu,
                       double rmse_user,
                       size_t cold_users,
                       size_t train_size,
                       size_t test_size);

}
