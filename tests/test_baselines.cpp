#include "spark/baselines.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ \
                      << " - " << msg << std::endl; \
            std::exit(1); \
        } \
    } while(0)

void test_oracle_delays() {
    std::cout << "[Test] Running test_oracle_delays..." << std::endl;

    for (uint32_t d : {1, 2, 3, 5, 8}) {
        spark::ShiftRegisterOracle oracle(d);
        spark::TemporalMemoryEnv env(d, 100, 42ULL + d);

        auto res = spark::BaselineEvaluator::evaluate_oracle(oracle, env);
        REQUIRE(res.evaluated_steps == 100 - d, "Warmup steps not properly excluded");
        REQUIRE(res.accuracy == 1.0f, "ShiftRegisterOracle must achieve exactly 100% accuracy");
        REQUIRE(res.loss < 1e-4f, "Oracle loss should be near zero");
    }

    std::cout << "[Test] test_oracle_delays PASSED." << std::endl;
}

void test_baselines_chance_level() {
    std::cout << "[Test] Running test_baselines_chance_level..." << std::endl;

    uint32_t delay = 2;
    spark::TemporalMemoryEnv train_env(delay, 1000, 123ULL);
    spark::TemporalMemoryEnv eval_env(delay, 1000, 456ULL);

    // Constant Predictor
    spark::ConstantPredictor const_pred;
    const_pred.fit(train_env, 1000);
    auto res_const = spark::BaselineEvaluator::evaluate_constant(const_pred, eval_env);

    // Memoryless Predictor
    spark::MemorylessPredictor mem_pred;
    mem_pred.fit(train_env, 2000, 0.05f);
    auto res_mem = spark::BaselineEvaluator::evaluate_memoryless(mem_pred, eval_env);

    // Untrained SPARK
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 64;
    g_cfg.fan_in = 8;
    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 64;
    n_cfg.input_dim = 1;
    n_cfg.output_dim = 1;

    spark::MicroUnitNetwork untrained_net(n_cfg, g_cfg);
    auto res_untrained = spark::BaselineEvaluator::evaluate_untrained_spark(untrained_net, eval_env);

    std::cout << "  Constant Predictor Accuracy   : " << (res_const.accuracy * 100.0f) << "%\n"
              << "  Memoryless Predictor Accuracy : " << (res_mem.accuracy * 100.0f) << "%\n"
              << "  Untrained SPARK Accuracy      : " << (res_untrained.accuracy * 100.0f) << "%\n";

    // For independent IID bits, chance level is ~50%. In 1000 samples, 99% CI is roughly [45%, 55%].
    REQUIRE(res_const.accuracy >= 0.45f && res_const.accuracy <= 0.55f, "Constant predictor deviates from chance");
    REQUIRE(res_mem.accuracy >= 0.45f && res_mem.accuracy <= 0.55f, "Memoryless predictor deviates from chance");
    REQUIRE(res_untrained.accuracy >= 0.45f && res_untrained.accuracy <= 0.55f, "Untrained SPARK deviates from chance");

    std::cout << "[Test] test_baselines_chance_level PASSED." << std::endl;
}

int main() {
    std::cout << "=== Running Baselines & Oracle Tests ===" << std::endl;
    test_oracle_delays();
    test_baselines_chance_level();
    std::cout << "All Baselines & Oracle tests passed successfully!" << std::endl;
    return 0;
}
