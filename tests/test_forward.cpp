#include "spark/network.hpp"
#include "spark/metrics.hpp"
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

void test_forward_dimensions_and_determinism() {
    std::cout << "[Test] Running test_forward_dimensions_and_determinism..." << std::endl;
    spark::NetworkConfig net_cfg;
    net_cfg.num_units = 128;
    net_cfg.input_dim = 4;
    net_cfg.output_dim = 2;
    net_cfg.alpha = 0.8f;
    net_cfg.beta = 0.5f;
    net_cfg.seed = 42ULL;

    spark::SparseGraphConfig graph_cfg;
    graph_cfg.num_units = 128;
    graph_cfg.fan_in = 8;
    graph_cfg.seed = 42ULL;

    spark::MicroUnitNetwork net1(net_cfg, graph_cfg);
    spark::MicroUnitNetwork net2(net_cfg, graph_cfg);

    std::string err;
    REQUIRE(net1.validate(err), err);
    REQUIRE(net2.validate(err), err);

    std::vector<float> input = {0.5f, -0.2f, 0.8f, -0.1f};
    std::vector<float> logits1, probs1;
    std::vector<float> logits2, probs2;

    net1.forward_step(input, logits1, probs1, spark::ReadoutMode::Sigmoid);
    net2.forward_step(input, logits2, probs2, spark::ReadoutMode::Sigmoid);

    REQUIRE(logits1.size() == 2, "Logits size mismatch");
    REQUIRE(probs1.size() == 2, "Probs size mismatch");
    REQUIRE(logits1 == logits2, "Deterministic logits mismatch");
    REQUIRE(probs1 == probs2, "Deterministic probs mismatch");
    REQUIRE(net1.state() == net2.state(), "Deterministic state mismatch");

    // Check activation bounds and finiteness
    auto stats = spark::compute_activation_stats(net1.state());
    REQUIRE(stats.non_finite_count == 0, "Non-finite activation detected");
    REQUIRE(stats.min_val >= -1.0f && stats.max_val <= 1.0f, "Activation outside [-1, 1]");

    std::cout << "[Test] test_forward_dimensions_and_determinism PASSED." << std::endl;
}

void test_double_buffering_and_history() {
    std::cout << "[Test] Running test_double_buffering_and_history..." << std::endl;
    spark::NetworkConfig net_cfg;
    net_cfg.num_units = 64;
    net_cfg.input_dim = 2;
    net_cfg.output_dim = 1;
    net_cfg.seed = 101ULL;

    spark::SparseGraphConfig graph_cfg;
    graph_cfg.num_units = 64;
    graph_cfg.fan_in = 4;
    graph_cfg.seed = 101ULL;

    spark::MicroUnitNetwork net(net_cfg, graph_cfg);

    std::vector<float> input1 = {1.0f, 0.0f};
    std::vector<float> logits, probs;

    // Initially states are all 0
    for (float h : net.state()) {
        REQUIRE(h == 0.0f, "Initial state should be 0");
    }

    net.forward_step(input1, logits, probs);
    std::vector<float> state_t1 = net.state();

    std::vector<float> input2 = {0.0f, 1.0f};
    net.forward_step(input2, logits, probs);
    std::vector<float> state_t2 = net.state();
    std::vector<float> prev_t2 = net.prev_state();

    // In synchronous double buffering, prev_state at end of t2 should reflect the state
    // used to transition forward, which is state_t2 (ready for t3).
    REQUIRE(prev_t2 == state_t2, "prev_state mismatch with current state at tick completion");
    REQUIRE(state_t1 != state_t2, "State did not transition across ticks");

    std::cout << "[Test] test_double_buffering_and_history PASSED." << std::endl;
}

void test_input_and_weight_sensitivity() {
    std::cout << "[Test] Running test_input_and_weight_sensitivity..." << std::endl;
    spark::NetworkConfig net_cfg;
    net_cfg.num_units = 64;
    net_cfg.input_dim = 2;
    net_cfg.output_dim = 1;
    net_cfg.seed = 777ULL;

    spark::SparseGraphConfig graph_cfg;
    graph_cfg.num_units = 64;
    graph_cfg.fan_in = 4;
    graph_cfg.seed = 777ULL;

    spark::MicroUnitNetwork netA(net_cfg, graph_cfg);
    spark::MicroUnitNetwork netB(net_cfg, graph_cfg);

    std::vector<float> inA = {1.0f, 0.5f};
    std::vector<float> inB = {-1.0f, -0.5f};
    std::vector<float> logitsA, probsA, logitsB, probsB;

    netA.forward_step(inA, logitsA, probsA);
    netB.forward_step(inB, logitsB, probsB);

    // Verify sensitivity to distinct inputs
    REQUIRE(netA.state() != netB.state(), "Network insensitive to input differences");
    REQUIRE(logitsA != logitsB, "Readout insensitive to input differences");

    // Now test weight perturbation sensitivity
    spark::MicroUnitNetwork netC(net_cfg, graph_cfg);
    spark::MicroUnitNetwork netD(net_cfg, graph_cfg);

    // Perturb one recurrent weight in netD
    REQUIRE(!netD.graph().weights.empty(), "Graph weights empty");
    netD.graph().weights[0] += 1.5f;

    std::vector<float> logitsC, probsC, logitsD, probsD;
    // Step 1: activates hidden units from zero initial state
    netC.forward_step(inA, logitsC, probsC);
    netD.forward_step(inA, logitsD, probsD);

    // Step 2: recurrent connections propagate non-zero activations from step 1
    netC.forward_step(inA, logitsC, probsC);
    netD.forward_step(inA, logitsD, probsD);

    REQUIRE(netC.state() != netD.state(), "Network insensitive to weight perturbation");
    REQUIRE(logitsC != logitsD, "Readout insensitive to weight perturbation");

    std::cout << "[Test] test_input_and_weight_sensitivity PASSED." << std::endl;
}

void test_numerical_stability() {
    std::cout << "[Test] Running test_numerical_stability..." << std::endl;
    // Sigmoid extremes
    float sig_pos = spark::MicroUnitNetwork::stable_sigmoid(100.0f);
    float sig_neg = spark::MicroUnitNetwork::stable_sigmoid(-100.0f);
    float sig_zero = spark::MicroUnitNetwork::stable_sigmoid(0.0f);

    REQUIRE(sig_pos > 0.999999f && sig_pos <= 1.0f, "Sigmoid extreme positive failed");
    REQUIRE(sig_neg >= 0.0f && sig_neg < 1e-6f, "Sigmoid extreme negative failed");
    REQUIRE(std::abs(sig_zero - 0.5f) < 1e-6, "Sigmoid zero failed");

    // Softmax extremes
    std::vector<float> logits = {1000.0f, 1002.0f, 999.0f};
    std::vector<float> probs;
    spark::MicroUnitNetwork::stable_softmax(logits, probs);

    float sum = 0.0f;
    for (float p : probs) {
        REQUIRE(std::isfinite(p), "Non-finite softmax probability");
        REQUIRE(p >= 0.0f && p <= 1.0f, "Softmax probability outside [0, 1]");
        sum += p;
    }
    REQUIRE(std::abs(sum - 1.0f) < 1e-5, "Softmax does not sum to 1.0");
    REQUIRE(probs[1] > probs[0], "Softmax monotonic order violated");
    REQUIRE(probs[0] > probs[2], "Softmax monotonic order violated");

    std::cout << "[Test] test_numerical_stability PASSED." << std::endl;
}

int main() {
    std::cout << "=== Running MicroUnitNetwork Forward Tests ===" << std::endl;
    test_forward_dimensions_and_determinism();
    test_double_buffering_and_history();
    test_input_and_weight_sensitivity();
    test_numerical_stability();
    std::cout << "All MicroUnitNetwork tests passed successfully!" << std::endl;
    return 0;
}

