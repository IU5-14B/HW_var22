#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "data_loader.h"
#include "metrics.h"
#include "mf_model.h"

// -------------------- утилиты --------------------

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

static bool parse_int(const std::string& s, int& out) {
    try {
        size_t p = 0;
        long long v = std::stoll(s, &p);
        if (p != s.size()) return false;
        out = (int)v;
        return true;
    } catch (...) { return false; }
}

static bool parse_u32(const std::string& s, uint32_t& out) {
    try {
        size_t p = 0;
        unsigned long long v = std::stoull(s, &p);
        if (p != s.size()) return false;
        out = (uint32_t)v;
        return true;
    } catch (...) { return false; }
}

static bool parse_double(const std::string& s, double& out) {
    try {
        size_t p = 0;
        double v = std::stod(s, &p);
        if (p != s.size()) return false;
        out = v;
        return true;
    } catch (...) { return false; }
}

// --- кодировки ---
// MovieLens в .dat часто содержит ISO-8859-1 (Latin-1) в названиях.
// Чтобы в терминале и CSV не было "кракозябр", конвертируем в UTF-8.
// Если строка уже валидный UTF-8 — оставляем как есть.
static bool is_valid_utf8(const std::string& s) {
    const unsigned char* p = (const unsigned char*)s.data();
    size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        unsigned char c = p[i];
        if (c <= 0x7F) { i++; continue; }
        if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= n) return false;
            unsigned char c1 = p[i + 1];
            if ((c1 & 0xC0) != 0x80) return false;
            // overlong
            if (c < 0xC2) return false;
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= n) return false;
            unsigned char c1 = p[i + 1], c2 = p[i + 2];
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return false;
            // overlong / surrogate check (простая)
            if (c == 0xE0 && c1 < 0xA0) return false;
            if (c == 0xED && c1 >= 0xA0) return false;
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= n) return false;
            unsigned char c1 = p[i + 1], c2 = p[i + 2], c3 = p[i + 3];
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return false;
            // overlong / > U+10FFFF check (простая)
            if (c == 0xF0 && c1 < 0x90) return false;
            if (c > 0xF4) return false;
            if (c == 0xF4 && c1 > 0x8F) return false;
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

static std::string latin1_to_utf8(const std::string& s) {
    std::string out;
    out.reserve(s.size() * 2);
    for (unsigned char c : s) {
        if (c < 0x80) out.push_back((char)c);
        else {
            out.push_back((char)(0xC0 | (c >> 6)));
            out.push_back((char)(0x80 | (c & 0x3F)));
        }
    }
    return out;
}

static std::string ensure_utf8(const std::string& s) {
    if (is_valid_utf8(s)) return s;
    return latin1_to_utf8(s);
}

struct CmdArgs {
    // пути
    std::string movies_path  = "data/raw/movies.dat";
    std::string ratings_path = "data/raw/ratings.dat";

    // split
    double test_ratio = 0.20;
    uint32_t split_seed = 42;

    // MF params
    MFParams p;

    // long tail
    int tail_threshold = 20; // фильм считается "хвостом", если в train2 <= threshold оценок

    // recommendations
    int recommend_user = -1;
    int topN = 10;
};

static void print_help() {
    std::cout <<
R"(HW_var22 (ML часть) — MovieLens + baseline + MF(SGD) + early stopping + Long Tail + recommendations

Запуск:
  ./build/app [опции]

Опции:
  --movies <path>            путь к movies.dat (по умолчанию data/raw/movies.dat)
  --ratings <path>           путь к ratings.dat (по умолчанию data/raw/ratings.dat)

  --test-ratio <double>      доля test (по умолчанию 0.20)
  --split-seed <u32>         seed для train/test (по умолчанию 42)

  --k <int>                  размерность факторов (по умолчанию 50)
  --epochs <int>             максимум эпох (по умолчанию 60)
  --lr <double>              learning rate (по умолчанию 0.007)
  --reg <double>             регуляризация (по умолчанию 0.05)
  --seed <u32>               seed модели (по умолчанию 42)
  --patience <int>           early stopping patience (по умолчанию 5)

  --tail-threshold <int>     порог "длинного хвоста" по count в train2 (по умолчанию 20)

  --recommend-user <int>     user_id, для которого вывести рекомендации
  --topN <int>               сколько рекомендаций (по умолчанию 10)

  --help                     показать помощь
)";
}

static std::unordered_map<int,int> build_movie_counts(const std::vector<Rating>& data) {
    std::unordered_map<int,int> c;
    c.reserve(8192);
    for (const auto& r : data) c[r.movie_id]++;
    return c;
}

static double rmse_subset_by_movies(const MFModel& model,
                                   const std::vector<Rating>& data,
                                   const std::unordered_map<int,int>& movie_counts_in_train2,
                                   int threshold,
                                   bool take_tail,
                                   size_t& out_n) {
    long double se = 0.0L;
    out_n = 0;

    for (const auto& r : data) {
        auto it = movie_counts_in_train2.find(r.movie_id);
        int cnt = (it == movie_counts_in_train2.end()) ? 0 : it->second;

        bool is_tail = (cnt > 0 && cnt <= threshold); // tail: редко, но встречался в train2
        bool select = take_tail ? is_tail : (!is_tail);

        if (!select) continue;

        double pred = model.predict(r.user_id, r.movie_id);
        // ограничим прогноз в [1;5], чтобы RMSE был адекватным
        if (pred < 1.0) pred = 1.0;
        if (pred > 5.0) pred = 5.0;

        double err = pred - (double)r.rating;
        se += (long double)(err * err);
        out_n++;
    }

    if (out_n == 0) return 0.0;
    return (double)std::sqrt((double)(se / (long double)out_n));
}

static bool save_long_tail_csv(const std::string& path,
                               int threshold,
                               double rmse_head, size_t n_head,
                               double rmse_tail, size_t n_tail) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "tail_threshold,head_n,head_rmse,tail_n,tail_rmse\n";
    out << threshold << ","
        << n_head << "," << std::setprecision(10) << rmse_head << ","
        << n_tail << "," << std::setprecision(10) << rmse_tail << "\n";
    return true;
}

static bool save_recommendations_csv(const std::string& path,
                                    const std::vector<std::tuple<int,std::string,double>>& recs) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "rank,movie_id,title,pred_rating\n";
    for (size_t i = 0; i < recs.size(); ++i) {
        int mid; std::string title; double pred;
        std::tie(mid, title, pred) = recs[i];
        title = ensure_utf8(title); // на всякий случай (если вдруг в recs попало "сырое")
        out << (i + 1) << "," << mid << ",\"" << title << "\"," << std::setprecision(10) << pred << "\n";
    }
    return true;
}

static bool parse_args(int argc, char** argv, CmdArgs& a) {
    // defaults
    a.p.k = 50;
    a.p.epochs = 80;
    a.p.lr = 0.007;
    a.p.reg = 0.05;
    a.p.seed = 42;
    a.p.patience = 5;

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            print_help();
            return false; // сигнал “не запускать”
        }

        auto need_val = [&](std::string& val) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Ошибка: для " << key << " нужно значение.\n";
                return false;
            }
            val = argv[++i];
            return true;
        };

        std::string val;

        if (key == "--movies") {
            if (!need_val(val)) return false;
            a.movies_path = val;
        } else if (key == "--ratings") {
            if (!need_val(val)) return false;
            a.ratings_path = val;
        } else if (key == "--test-ratio") {
            if (!need_val(val)) return false;
            if (!parse_double(val, a.test_ratio) || a.test_ratio <= 0.0 || a.test_ratio >= 0.9) {
                std::cerr << "Ошибка: некорректный --test-ratio\n";
                return false;
            }
        } else if (key == "--split-seed") {
            if (!need_val(val)) return false;
            if (!parse_u32(val, a.split_seed)) {
                std::cerr << "Ошибка: некорректный --split-seed\n";
                return false;
            }
        } else if (key == "--k") {
            if (!need_val(val)) return false;
            if (!parse_int(val, a.p.k) || a.p.k <= 0) return false;
        } else if (key == "--epochs") {
            if (!need_val(val)) return false;
            if (!parse_int(val, a.p.epochs) || a.p.epochs <= 0) return false;
        } else if (key == "--lr") {
            if (!need_val(val)) return false;
            if (!parse_double(val, a.p.lr) || a.p.lr <= 0.0) return false;
        } else if (key == "--reg") {
            if (!need_val(val)) return false;
            if (!parse_double(val, a.p.reg) || a.p.reg < 0.0) return false;
        } else if (key == "--seed") {
            if (!need_val(val)) return false;
            if (!parse_u32(val, a.p.seed)) return false;
        } else if (key == "--patience") {
            if (!need_val(val)) return false;
            if (!parse_int(val, a.p.patience) || a.p.patience < 0) return false;
        } else if (key == "--tail-threshold") {
            if (!need_val(val)) return false;
            if (!parse_int(val, a.tail_threshold) || a.tail_threshold < 1) return false;
        } else if (key == "--recommend-user") {
            if (!need_val(val)) return false;
            if (!parse_int(val, a.recommend_user) || a.recommend_user <= 0) return false;
        } else if (key == "--topN") {
            if (!need_val(val)) return false;
            if (!parse_int(val, a.topN) || a.topN <= 0) return false;
        } else {
            std::cerr << "Неизвестная опция: " << key << "\n";
            std::cerr << "Подсказка: запусти с --help\n";
            return false;
        }
    }
    return true;
}

// -------------------- main --------------------

int main(int argc, char** argv) {
    CmdArgs args;
    if (!parse_args(argc, argv, args)) {
        // если пользователь просил help, parse_args уже всё вывел
        return 0;
    }

    std::filesystem::create_directories("results");

    DataLoader loader;
    if (!loader.load_movies(args.movies_path)) return 1;
    if (!loader.load_ratings(args.ratings_path, true, 5)) return 1;

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
    train_test_split(loader.ratings(), train, test, args.test_ratio, args.split_seed);

    std::cout << "\n=== Шаг 2: Разбиение train/test и baseline ===\n";
    std::cout << "Размер train: " << train.size() << "\n";
    std::cout << "Размер test:  " << test.size() << "\n";

    double mu = 0.0;
    double rmse_mu = metrics::rmse_global_mean(train, test, mu);

    size_t cold_users_baseline = 0;
    double rmse_user = metrics::rmse_user_mean(train, test, mu, cold_users_baseline);

    std::cout << "Глобальное среднее по train (mu): " << std::setprecision(6) << mu << "\n";
    std::cout << "RMSE baseline (предсказание mu): " << std::setprecision(6) << rmse_mu << "\n";
    std::cout << "RMSE baseline (среднее по пользователю): " << std::setprecision(6) << rmse_user << "\n";
    std::cout << "Cold Start (случаи в test, где user не был в train): " << cold_users_baseline << "\n";

    metrics::save_baseline_csv("results/baseline_rmse.csv",
                              mu, rmse_mu, rmse_user, cold_users_baseline,
                              train.size(), test.size());

    // === ШАГ 3: MF (SGD) + early stopping ===
    std::cout << "\n=== Шаг 3: Матричная факторизация (SGD) + early stopping ===\n";

    // train -> train2/val
    std::vector<Rating> train2, val;
    train_test_split(train, train2, val, 0.10, 123); // 10% в val

    std::cout << "Параметры: k=" << args.p.k
              << ", epochs=" << args.p.epochs
              << ", lr=" << args.p.lr
              << ", reg=" << args.p.reg
              << ", seed=" << args.p.seed
              << ", patience=" << args.p.patience << "\n";
    std::cout << "Размер train2: " << train2.size() << ", val: " << val.size() << "\n";

    MFModel model;
    if (!model.fit_with_early_stopping(train2, val, args.p)) {
        std::cerr << "Ошибка: не удалось обучить MF-модель.\n";
        return 1;
    }

    auto eval = model.evaluate(test);

    std::cout << "RMSE на test (MF): " << std::setprecision(6) << eval.rmse << "\n";
    std::cout << "RMSE на test (MF, нормализованный /5): " << std::setprecision(6) << (eval.rmse / 5.0) << "\n";
    std::cout << "Cold Start по пользователям (в test): " << eval.cold_user_cases << "\n";
    std::cout << "Cold Start по фильмам (в test): " << eval.cold_movie_cases << "\n";

    if (!model.save_training_log_csv("results/mf_training_log.csv")) {
        std::cerr << "Не удалось сохранить results/mf_training_log.csv\n";
    } else {
        std::cout << "Лог обучения сохранён: results/mf_training_log.csv\n";
    }

    // === Long Tail (head/tail RMSE) ===
    std::cout << "\n=== Long Tail: оценка качества на 'голове' и 'хвосте' ===\n";
    auto movie_counts_train2 = build_movie_counts(train2);

    size_t n_tail = 0, n_head = 0;
    double rmse_tail = rmse_subset_by_movies(model, test, movie_counts_train2, args.tail_threshold, true, n_tail);
    double rmse_head = rmse_subset_by_movies(model, test, movie_counts_train2, args.tail_threshold, false, n_head);

    std::cout << "Tail threshold (count в train2): <= " << args.tail_threshold << "\n";
    std::cout << "HEAD: n=" << n_head << ", RMSE=" << std::setprecision(6) << rmse_head << "\n";
    std::cout << "TAIL: n=" << n_tail << ", RMSE=" << std::setprecision(6) << rmse_tail << "\n";

    if (save_long_tail_csv("results/long_tail_metrics.csv",
                           args.tail_threshold,
                           rmse_head, n_head,
                           rmse_tail, n_tail)) {
        std::cout << "Long Tail метрики сохранены: results/long_tail_metrics.csv\n";
    } else {
        std::cerr << "Не удалось сохранить results/long_tail_metrics.csv\n";
    }

    // === Recommendations (top-N) ===
    if (args.recommend_user > 0) {
        std::cout << "\n=== Рекомендации (Top-" << args.topN << ") для пользователя "
                  << args.recommend_user << " ===\n";

        // исключаем фильмы, которые пользователь уже оценивал (по всем данным)
        std::unordered_set<int> seen;
        seen.reserve(2048);
        for (const auto& r : loader.ratings()) {
            if (r.user_id == args.recommend_user) {
                seen.insert(r.movie_id);
            }
        }

        // кандидаты: все фильмы из справочника
        std::vector<std::tuple<int,std::string,double>> scored;
        scored.reserve(loader.movie_titles().size());

        for (const auto& kv : loader.movie_titles()) {
            int movie_id = kv.first;
            const std::string& raw_title = kv.second;

            if (seen.find(movie_id) != seen.end()) continue;

            double pred = model.predict(args.recommend_user, movie_id);
            if (pred < 1.0) pred = 1.0;
            if (pred > 5.0) pred = 5.0;

            // сохраняем уже "человеческий" UTF-8 заголовок
            scored.emplace_back(movie_id, ensure_utf8(raw_title), pred);
        }

        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b){
                      return std::get<2>(a) > std::get<2>(b);
                  });

        if ((int)scored.size() > args.topN) scored.resize((size_t)args.topN);

        for (size_t i = 0; i < scored.size(); ++i) {
            int mid; std::string title; double pred;
            std::tie(mid, title, pred) = scored[i];
            std::cout << (i + 1) << ") " << title
                      << " (movie_id=" << mid << ", pred=" << std::setprecision(4) << pred << ")\n";
        }

        std::string out_path = "results/recommendations_user" + std::to_string(args.recommend_user) + ".csv";
        if (save_recommendations_csv(out_path, scored)) {
            std::cout << "Рекомендации сохранены: " << out_path << "\n";
        } else {
            std::cerr << "Не удалось сохранить " << out_path << "\n";
        }
    }

    std::cout << "\nГотово. Файлы результатов лежат в results/\n";
    return 0;
}
