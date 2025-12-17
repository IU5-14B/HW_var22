#include "data_loader.h"
#include "metrics.h"
#include "mf_model.h"

#include <iostream>
#include <filesystem>
#include <random>
#include <algorithm>

static void train_test_split(const std::vector<Rating>& all,
                             std::vector<Rating>& train,
                             std::vector<Rating>& test,
                             double test_ratio,
                             uint32_t seed) {
    train.clear();
    test.clear();
    if (all.empty()) return;

    std::vector<size_t> idx(all.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;

    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    size_t test_size = (size_t)((double)all.size() * test_ratio);
    if (test_size == 0) test_size = 1;
    if (test_size >= all.size()) test_size = all.size() / 5;

    test.reserve(test_size);
    train.reserve(all.size() - test_size);

    for (size_t j = 0; j < idx.size(); ++j) {
        const Rating& r = all[idx[j]];
        if (j < test_size) test.push_back(r);
        else train.push_back(r);
    }
}

int main() {
    const std::string movies_path  = "data/raw/movies.dat";
    const std::string ratings_path = "data/raw/ratings.dat";

    std::filesystem::create_directories("results");

    DataLoader loader;

    if (!loader.load_movies(movies_path)) return 1;
    if (!loader.load_ratings(ratings_path, true, 5)) return 1;

    auto st = loader.compute_stats();

    std::cout << "\n=== Статистика датасета ===\n";
    std::cout << "Количество оценок: " << st.num_ratings << "\n";
    std::cout << "Количество пользователей: " << st.num_users << "\n";
    std::cout << "Количество фильмов с оценками: " << st.num_movies << "\n";
    std::cout << "Средний рейтинг: " << st.mean_rating << "\n";
    std::cout << "Диапазон идентификаторов пользователей: "
              << st.min_user_id << " .. " << st.max_user_id << "\n";
    std::cout << "Диапазон идентификаторов фильмов: "
              << st.min_movie_id << " .. " << st.max_movie_id << "\n";

    loader.save_stats_csv("results/dataset_stats.csv", st);
    loader.save_counts_csv("results/movie_rating_counts.csv",
                           loader.movie_rating_counts(),
                           "movie_id", "ratings_count");
    loader.save_counts_csv("results/user_rating_counts.csv",
                           loader.user_rating_counts(),
                           "user_id", "ratings_count");

    // === ШАГ 2: train/test + baseline RMSE ===
    std::vector<Rating> train, test;
    const double test_ratio = 0.20;
    const uint32_t seed = 42;

    train_test_split(loader.ratings(), train, test, test_ratio, seed);

    std::cout << "\n=== Шаг 2: Разбиение train/test и baseline ===\n";
    std::cout << "Размер train: " << train.size() << "\n";
    std::cout << "Размер test:  " << test.size() << "\n";

    double mu = 0.0;
    double rmse_mu = metrics::rmse_global_mean(train, test, mu);

    size_t cold_cases = 0;
    double rmse_user = metrics::rmse_user_mean(train, test, mu, cold_cases);

    std::cout << "Глобальное среднее по train (mu): " << mu << "\n";
    std::cout << "RMSE baseline (предсказание mu): " << rmse_mu << "\n";
    std::cout << "RMSE baseline (среднее по пользователю): " << rmse_user << "\n";
    std::cout << "Cold Start (случаи в test, где user не был в train): " << cold_cases << "\n";

    metrics::save_baseline_csv("results/baseline_rmse.csv",
                              mu, rmse_mu, rmse_user, cold_cases,
                              train.size(), test.size());

    // === ШАГ 3: Матричная факторизация (SGD) ===
    std::cout << "\n=== Шаг 3: Матричная факторизация (SGD) ===\n";

    MFParams p;
    p.k = 50;
    p.epochs = 80;
    p.lr = 0.007;
    p.reg = 0.07;
    p.seed = 42;

    std::cout << "Параметры: k=" << p.k
              << ", epochs=" << p.epochs
              << ", lr=" << p.lr
              << ", reg=" << p.reg
              << ", seed=" << p.seed << "\n";

    MFModel model;
    if (!model.fit(train, p)) {
        std::cerr << "Ошибка: не удалось обучить MF-модель.\n";
        return 1;
    }

    // Оценка на test + фиксация cold start
    auto eval = model.evaluate(test);

    std::cout << "Mu (средний рейтинг по train): " << model.mu() << "\n";
    std::cout << "Пользователей в train: " << model.num_users_train() << "\n";
    std::cout << "Фильмов в train: " << model.num_movies_train() << "\n";
    std::cout << "RMSE на test (MF): " << eval.rmse << "\n";
    std::cout << "Cold Start по пользователям (в test): " << eval.cold_user_cases << "\n";
    std::cout << "Cold Start по фильмам (в test): " << eval.cold_movie_cases << "\n";

    if (!model.save_training_log_csv("results/mf_training_log.csv")) {
        std::cerr << "Не удалось сохранить results/mf_training_log.csv\n";
    } else {
        std::cout << "Лог обучения сохранён: results/mf_training_log.csv\n";
    }

    std::cout << "\nГотово. Файлы результатов лежат в results/\n";
    return 0;
}
