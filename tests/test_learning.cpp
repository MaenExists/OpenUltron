#include "spark/trainer.hpp"
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

void test_learning_convergence_and_reproducibility() {
    std::cout << "[Test] Running test_learning_convergence_and_reproducibility..." << std::endl;

    spark::NetworkConfig net_cfg;
    net_cfg.num_units = 32;
    net_cfg.input_dim = 1;
    net_cfg.output_dim = 1;
    net_cfg.alpha = 0.8f;
    net_cfg.beta = 0.5f;
    net_cfg.seed = 42ULL;

    spark::SparseGraphConfig graph_cfg;
    graph_cfg.num_units = 32;
    graph_cfg.fan_in = 4;
    graph_cfg.seed = 42ULL;

    spark::MicroUnitNetwork net1(net_cfg, graph_cfg);
    spark::TemporalMemoryEnv train_env1(1, 100, 100ULL);
    spark::TemporalMemoryEnv eval_env1(1, 100, 200ULL);

    spark::Trainer trainer1(net1, train_env1, &eval_env1, 0.05f, 1.0f);
    auto summary1 = trainer1.train(300, 8, 100, false);

    std::cout << "Initial Loss: " << summary1.initial_loss
              << ", Final Train Loss: " << summary1.final_train_loss
              << ", Final Eval Acc: " << (summary1.final_eval_accuracy * 100.0f) << "%"
              << std::endl;

    REQUIRE(summary1.final_train_loss < summary1.initial_loss, "Training failed to reduce loss");
    REQUIRE(summary1.final_eval_accuracy > 0.70f, "Evaluation accuracy should exceed random baseline 50%");

    // Reproducibility test: identical seed produces identical results
    spark::MicroUnitNetwork net2(net_cfg, graph_cfg);
    spark::TemporalMemoryEnv train_env2(1, 100, 100ULL);
    spark::TemporalMemoryEnv eval_env2(1, 100, 200ULL);

    spark::Trainer trainer2(net2, train_env2, &eval_env2, 0.05f, 1.0f);
    auto summary2 = trainer2.train(300, 8, 100, true);

    REQUIRE(std::abs(summary1.final_train_loss - summary2.final_train_loss) < 1e-5f,
            "Reproducibility test failed: losses do not match");
    REQUIRE(std::abs(summary1.final_eval_accuracy - summary2.final_eval_accuracy) < 1e-5f,
            "Reproducibility test failed: eval accuracies do not match");

    std::cout << "[Test] test_learning_convergence_and_reproducibility PASSED." << std::endl;
}

int main() {
    std::cout << "=== Running Learning Verification Tests ===" << std::endl;
    test_learning_convergence_and_reproducibility();
    std::cout << "All Learning Verification tests passed successfully!" << std::endl;
    return 0;
}
