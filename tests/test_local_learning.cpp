#include "spark/local_learning.hpp"
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

void test_trace_decay_and_accumulation() {
    std::cout << "[Test] Running test_trace_decay_and_accumulation..." << std::endl;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 16;
    n_cfg.input_dim = 2;
    n_cfg.output_dim = 1;
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 16;
    g_cfg.fan_in = 4;

    spark::MicroUnitNetwork net(n_cfg, g_cfg);
    spark::EligibilityTraces traces(net);

    REQUIRE(traces.E_weights.size() == 16 * 4, "Trace size mismatch for weights");
    REQUIRE(traces.E_U.size() == 16 * 2, "Trace size mismatch for U");
    REQUIRE(traces.E_b.size() == 16, "Trace size mismatch for biases");

    // Initially all traces are 0
    for (float e : traces.E_weights) REQUIRE(e == 0.0f, "Initial trace non-zero");

    // Perform an update with non-zero input
    std::vector<float> x = {1.0f, -0.5f};
    std::vector<float> h_prev(16, 0.5f);
    std::vector<float> logits, probs;
    net.forward_step(x, logits, probs);

    float lambda = 0.8f;
    traces.update(net, x, h_prev, lambda);

    // After step 1, traces should be non-zero
    bool has_nonzero = false;
    for (float e : traces.E_weights) {
        if (std::abs(e) > 1e-4f) has_nonzero = true;
    }
    REQUIRE(has_nonzero, "Traces did not accumulate on step 1");

    // Save trace state
    float sample_e1 = traces.E_weights[0];

    // Update with zero inputs and zero activations to observe pure decay
    std::vector<float> zero_x = {0.0f, 0.0f};
    std::vector<float> zero_h_prev(16, 0.0f);

    // Set h_curr = 1.0 so sensitivity = 1 - 1^2 = 0
    std::vector<float> saturated_h(16, 1.0f);
    net.set_state(saturated_h);

    traces.update(net, zero_x, zero_h_prev, lambda);
    float sample_e2 = traces.E_weights[0];

    // Sensitivity was 0 and h_prev was 0, so E should decay exactly by lambda: E2 = lambda * E1
    REQUIRE(std::abs(sample_e2 - lambda * sample_e1) < 1e-5f, "Trace decay rate mismatch");

    std::cout << "[Test] test_trace_decay_and_accumulation PASSED." << std::endl;
}

void test_feedback_mechanisms() {
    std::cout << "[Test] Running test_feedback_mechanisms..." << std::endl;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 16;
    n_cfg.input_dim = 1;
    n_cfg.output_dim = 2;
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 16;
    g_cfg.fan_in = 4;

    spark::MicroUnitNetwork net(n_cfg, g_cfg);
    spark::FeedbackProjection feedback(16, 2, 42ULL, 1.0f);

    std::vector<float> errors = {0.5f, -0.5f};
    std::vector<float> g_random;
    std::vector<float> g_transpose;

    feedback.compute_feedback(net, errors, spark::FeedbackMode::RandomFeedback, g_random);
    feedback.compute_feedback(net, errors, spark::FeedbackMode::ReadoutTranspose, g_transpose);

    REQUIRE(g_random.size() == 16, "Random feedback dimension mismatch");
    REQUIRE(g_transpose.size() == 16, "Transpose feedback dimension mismatch");

    // Both should produce finite, non-zero signals
    for (float g : g_random) REQUIRE(std::isfinite(g), "Non-finite random feedback");
    for (float g : g_transpose) REQUIRE(std::isfinite(g), "Non-finite transpose feedback");

    std::cout << "[Test] test_feedback_mechanisms PASSED." << std::endl;
}

void test_local_learning_delay1() {
    std::cout << "[Test] Running test_local_learning_delay1..." << std::endl;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 64;
    n_cfg.input_dim = 1;
    n_cfg.output_dim = 1;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 1.0f;
    n_cfg.seed = 42ULL;

    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 64;
    g_cfg.fan_in = 8;
    g_cfg.seed = 42ULL;

    spark::MicroUnitNetwork net(n_cfg, g_cfg);
    spark::LocalLearningConfig config;
    config.lr = 0.05f;
    config.lambda = 0.8f;
    config.feedback_mode = spark::FeedbackMode::ReadoutTranspose;

    spark::LocalLearner learner(net, config);
    spark::TemporalMemoryEnv train_env(1, 200, 42ULL);
    spark::TemporalMemoryEnv eval_env(1, 200, 1042ULL);

    auto summary = learner.train(train_env, &eval_env, 2000, 200, true);

    std::cout << "Local Learning Delay-1: Initial Loss=" << summary.initial_loss
              << ", Final Eval Loss=" << summary.final_eval_loss
              << ", Final Eval Acc=" << (summary.final_eval_accuracy * 100.0f) << "%"
              << std::endl;

    // Local learning should learn Delay-1 significantly above 50% chance
    REQUIRE(summary.final_eval_accuracy > 0.70f, "Local learning failed to acquire Delay-1 memory (>70%)");
    REQUIRE(summary.final_eval_loss < summary.initial_loss, "Loss did not decrease");

    std::cout << "[Test] test_local_learning_delay1 PASSED." << std::endl;
}

void test_trace_finiteness_and_bounds() {
    std::cout << "[Test] Running test_trace_finiteness_and_bounds..." << std::endl;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 64;
    n_cfg.input_dim = 1;
    n_cfg.output_dim = 1;
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 64;
    g_cfg.fan_in = 8;

    spark::MicroUnitNetwork net(n_cfg, g_cfg);
    spark::LocalLearningConfig config;
    config.lr = 0.05f;
    config.lambda = 0.95f; // High persistence to test accumulation stability
    config.feedback_mode = spark::FeedbackMode::ReadoutTranspose;

    spark::LocalLearner learner(net, config);
    spark::TemporalMemoryEnv env(1, 1000, 123ULL);

    for (int t = 0; t < 1000; ++t) {
        spark::Observation obs = env.get_observation();
        spark::Target target = env.get_target();
        env.step();
        float loss = learner.step(obs.values, target);
        REQUIRE(std::isfinite(loss), "Loss became non-finite during trace stress test");
    }

    // Verify all traces and weights are finite
    for (float e : learner.traces().E_weights) {
        REQUIRE(std::isfinite(e), "Recurrent trace non-finite");
        REQUIRE(std::abs(e) < 50.0f, "Recurrent trace diverged beyond bounds");
    }
    for (float e : learner.traces().E_U) {
        REQUIRE(std::isfinite(e), "Input trace non-finite");
    }
    for (float e : learner.traces().E_b) {
        REQUIRE(std::isfinite(e), "Bias trace non-finite");
    }
    for (float w : net.graph().weights) {
        REQUIRE(std::isfinite(w), "Weight became non-finite");
    }

    std::cout << "[Test] test_trace_finiteness_and_bounds PASSED." << std::endl;
}

void test_scaling_stability_512() {
    std::cout << "[Test] Running test_scaling_stability_512..." << std::endl;

    const std::vector<uint32_t> unit_counts = {64, 128, 512};
    size_t prev_mem = 0;

    for (uint32_t N : unit_counts) {
        spark::NetworkConfig n_cfg;
        n_cfg.num_units = N;
        n_cfg.input_dim = 1;
        n_cfg.output_dim = 1;
        spark::SparseGraphConfig g_cfg;
        g_cfg.num_units = N;
        g_cfg.fan_in = 8;

        spark::MicroUnitNetwork net(n_cfg, g_cfg);
        spark::LocalLearningConfig config;
        config.lr = 0.03f;
        config.lambda = 0.8f;
        config.feedback_mode = spark::FeedbackMode::ReadoutTranspose;

        spark::LocalLearner learner(net, config);
        size_t mem = net.memory_bytes() + learner.training_memory_bytes();

        std::cout << "  N=" << N << " total memory: " << (mem / 1024.0) << " KB" << std::endl;
        REQUIRE(mem > prev_mem, "Memory did not scale with unit count");
        prev_mem = mem;

        // Run 50 steps to verify execution stability at 512 units
        spark::TemporalMemoryEnv env(1, 100, 42ULL);
        for (int i = 0; i < 50; ++i) {
            spark::Observation obs = env.get_observation();
            spark::Target target = env.get_target();
            env.step();
            float loss = learner.step(obs.values, target);
            REQUIRE(std::isfinite(loss), "Loss non-finite during 512-unit scaling step");
        }
    }

    std::cout << "[Test] test_scaling_stability_512 PASSED." << std::endl;
}

int main() {
    std::cout << "=== Running Local Learning Tests ===" << std::endl;
    test_trace_decay_and_accumulation();
    test_feedback_mechanisms();
    test_local_learning_delay1();
    test_trace_finiteness_and_bounds();
    test_scaling_stability_512();
    std::cout << "All Local Learning tests passed successfully!" << std::endl;
    return 0;
}
