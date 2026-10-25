#pragma once

#include "spark/network.hpp"
#include "spark/environment.hpp"
#include "spark/trainer.hpp"
#include "spark/baselines.hpp"
#include <string>
#include <vector>

namespace spark {

struct ExperimentRecord {
    std::string run_id;
    std::string task;
    uint32_t delay = 1;
    uint64_t seed = 42;
    uint32_t units = 64;
    uint32_t fan_in = 8;
    float alpha = 0.8f;
    float beta = 0.5f;
    uint32_t unroll = 8;
    float learning_rate = 0.03f;
    uint32_t train_steps = 1500;
    size_t train_examples = 0;
    size_t eval_examples = 0;
    float train_loss = 0.0f;
    float validation_loss = 0.0f;
    float test_loss = 0.0f;
    float test_accuracy = 0.0f;
    double training_seconds = 0.0;
    size_t model_bytes = 0;

    std::string to_csv() const;
    static std::string csv_header();
};

struct AggregatedStats {
    std::string config_name;
    uint32_t delay = 0;
    uint32_t units = 0;
    size_t num_runs = 0;
    double mean_accuracy = 0.0;
    double std_accuracy = 0.0;
    double min_accuracy = 0.0;
    double max_accuracy = 0.0;
    double mean_eval_loss = 0.0;
    double mean_train_time = 0.0;
};

class ExperimentRunner {
public:
    static void compare_baselines(uint32_t delay, uint32_t units = 64, uint64_t seed = 42);
    static std::vector<ExperimentRecord> run_capacity_suite(
        const std::vector<uint32_t>& delays,
        const std::vector<uint32_t>& unit_counts,
        size_t num_seeds = 10,
        const std::string& csv_path = ""
    );
    static std::vector<ExperimentRecord> run_delay3_diagnostic_suite(
        const std::string& csv_path = ""
    );
    static void run_inference_ablations(uint32_t delay = 1, uint32_t units = 64, uint64_t seed = 42);

    static AggregatedStats compute_aggregate(const std::vector<ExperimentRecord>& records, const std::string& name);
};

} // namespace spark
