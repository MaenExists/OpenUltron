#include "spark/learning.hpp"
#include <iostream>
#include <cstdlib>

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ \
                      << " - " << msg << std::endl; \
            std::exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "=== Running Exact Gradient Verification (Finite Differences) ===" << std::endl;

    spark::NetworkConfig net_cfg;
    net_cfg.num_units = 8;
    net_cfg.input_dim = 2;
    net_cfg.output_dim = 1;
    net_cfg.alpha = 0.7f;
    net_cfg.beta = 0.6f;
    net_cfg.seed = 42ULL;

    spark::SparseGraphConfig graph_cfg;
    graph_cfg.num_units = 8;
    graph_cfg.fan_in = 3;
    graph_cfg.seed = 42ULL;

    spark::MicroUnitNetwork net(net_cfg, graph_cfg);

    // Create a 4-step observation and target sequence
    std::vector<spark::Observation> obs = {
        spark::Observation{{0.5f, -0.3f}},
        spark::Observation{{-0.8f, 0.2f}},
        spark::Observation{{0.1f, 0.9f}},
        spark::Observation{{-0.4f, -0.6f}}
    };

    std::vector<spark::Target> targets = {
        spark::Target{{1.0f}, true},
        spark::Target{{0.0f}, true},
        spark::Target{{1.0f}, true},
        spark::Target{{0.0f}, true}
    };

    std::cout << "Testing analytical vs numerical gradients across all parameter groups..." << std::endl;
    // Uses double-precision simulation for finite differences to achieve high precision
    auto result = spark::verify_gradients_finite_difference(net, obs, targets, 1e-6f, 1e-3);

    std::cout << "  Status             : " << (result.passed ? "PASSED" : "FAILED") << "\n"
              << "  Max Relative Error : " << result.max_relative_error << "\n"
              << "  Worst Parameter    : " << result.param_name << "[" << result.param_index << "]\n"
              << "  Analytical Value   : " << result.analytical_val << "\n"
              << "  Numerical Value    : " << result.numerical_val << "\n"
              << "  Message            : " << result.message << std::endl;

    REQUIRE(result.passed, result.message);
    REQUIRE(result.max_relative_error < 1e-3, "Max relative error exceeded tolerance 1e-3");

    std::cout << "\nExact Gradient Verification PASSED successfully!" << std::endl;
    return 0;
}
