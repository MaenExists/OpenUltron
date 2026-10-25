#include "spark/trainer.hpp"
#include "spark/learning.hpp"
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

void test_delayed_target_alignment_and_warmup() {
    std::cout << "[Regression] Testing delayed target alignment and warmup exclusion..." << std::endl;

    uint32_t delay = 3;
    uint32_t seq_len = 50;
    spark::TemporalMemoryEnv env(delay, seq_len, 777ULL);

    std::vector<float> observations;
    std::vector<spark::Target> targets;

    while (!env.is_done()) {
        observations.push_back(env.get_observation().values[0]);
        targets.push_back(env.get_target());
        env.step();
    }

    REQUIRE(observations.size() == seq_len, "Observation sequence length mismatch");
    REQUIRE(targets.size() == seq_len, "Target sequence length mismatch");

    // Warmup steps (t < delay) must be marked invalid
    for (uint32_t t = 0; t < delay; ++t) {
        REQUIRE(!targets[t].valid, "Warmup target marked valid");
    }

    // From t = delay onward, target must match observation from t - delay
    for (uint32_t t = delay; t < seq_len; ++t) {
        REQUIRE(targets[t].valid, "Valid target marked invalid");
        REQUIRE(targets[t].values[0] == observations[t - delay], "Delayed target misaligned");
    }

    std::cout << "[Regression] Delayed target alignment PASSED." << std::endl;
}

void test_generator_isolation() {
    std::cout << "[Regression] Testing train/test generator isolation..." << std::endl;

    spark::TemporalMemoryEnv train_env(2, 100, 1000ULL);
    spark::TemporalMemoryEnv eval_env(2, 100, 2000ULL);

    // Sequences generated from distinct seeds must not be identical
    REQUIRE(train_env.sequence() != eval_env.sequence(), "Train and eval sequences are identical");

    std::cout << "[Regression] Generator isolation PASSED." << std::endl;
}

void test_no_parameter_change_during_eval() {
    std::cout << "[Regression] Testing evaluation parameter immutability..." << std::endl;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 32;
    n_cfg.input_dim = 1;
    n_cfg.output_dim = 1;
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 32;
    g_cfg.fan_in = 4;

    spark::MicroUnitNetwork net(n_cfg, g_cfg);
    spark::TemporalMemoryEnv train_env(1, 100, 10ULL);
    spark::TemporalMemoryEnv eval_env(1, 100, 20ULL);

    auto saved_weights = net.graph().weights;
    auto saved_U = net.input_weights();
    auto saved_V = net.output_weights();
    auto saved_biases = net.biases();
    auto saved_c = net.output_biases();

    spark::Trainer trainer(net, train_env, &eval_env);
    trainer.evaluate(eval_env, 100);

    REQUIRE(net.graph().weights == saved_weights, "Recurrent weights modified during evaluation");
    REQUIRE(net.input_weights() == saved_U, "Input weights modified during evaluation");
    REQUIRE(net.output_weights() == saved_V, "Output weights modified during evaluation");
    REQUIRE(net.biases() == saved_biases, "Biases modified during evaluation");
    REQUIRE(net.output_biases() == saved_c, "Output biases modified during evaluation");

    std::cout << "[Regression] Evaluation parameter immutability PASSED." << std::endl;
}

void test_hidden_state_reset() {
    std::cout << "[Regression] Testing hidden-state reset..." << std::endl;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 32;
    n_cfg.input_dim = 1;
    n_cfg.output_dim = 1;
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 32;
    g_cfg.fan_in = 4;

    spark::MicroUnitNetwork net(n_cfg, g_cfg);
    std::vector<float> logits, probs;
    net.forward_step({1.0f}, logits, probs);

    bool has_nonzero = false;
    for (float h : net.state()) {
        if (std::abs(h) > 1e-5f) has_nonzero = true;
    }
    REQUIRE(has_nonzero, "Network produced all zero state after step");

    net.reset_state();
    for (float h : net.state()) {
        REQUIRE(h == 0.0f, "Network state not zeroed after reset_state");
    }
    for (float h : net.prev_state()) {
        REQUIRE(h == 0.0f, "Network prev_state not zeroed after reset_state");
    }

    std::cout << "[Regression] Hidden-state reset PASSED." << std::endl;
}

int main() {
    std::cout << "=== Running Scientific Regression & Audit Tests ===" << std::endl;
    test_delayed_target_alignment_and_warmup();
    test_generator_isolation();
    test_no_parameter_change_during_eval();
    test_hidden_state_reset();
    std::cout << "All Scientific Regression tests passed successfully!" << std::endl;
    return 0;
}
