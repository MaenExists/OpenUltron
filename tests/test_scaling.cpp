#include "spark/network.hpp"
#include "spark/graph.hpp"
#include "spark/config.hpp"
#include "spark/random.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cassert>

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ \
                      << " - " << msg << std::endl; \
            std::exit(1); \
        } \
    } while(0)

namespace {

// Test 1: Config bounds and safe allocation capacity limits
void test_capacity_limits() {
    std::cout << "[Test 1] Running test_capacity_limits..." << std::endl;
    std::string err;

    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 0;
    REQUIRE(!g_cfg.validate(err), "SparseGraphConfig should reject num_units = 0");

    g_cfg.num_units = spark::SparseGraphConfig::MAX_UNITS + 1;
    REQUIRE(!g_cfg.validate(err), "SparseGraphConfig should reject num_units > MAX_UNITS");

    g_cfg.num_units = 2048;
    REQUIRE(g_cfg.validate(err), "SparseGraphConfig should accept num_units = 2048");

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 0;
    REQUIRE(!n_cfg.validate(err), "NetworkConfig should reject num_units = 0");

    n_cfg.num_units = spark::NetworkConfig::MAX_UNITS + 1;
    REQUIRE(!n_cfg.validate(err), "NetworkConfig should reject num_units > MAX_UNITS");

    n_cfg.num_units = 2048;
    REQUIRE(n_cfg.validate(err), "NetworkConfig should accept num_units = 2048");

    std::cout << "  Passed: Safe capacity limits verified (1 <= N <= 65536)." << std::endl;
}

// Test 2: Scaling graph bounds and invariant checks across N in {128, 256, 512, 1024, 2048}
void test_graph_scaling() {
    std::cout << "[Test 2] Running test_graph_scaling..." << std::endl;
    const std::vector<uint32_t> test_sizes = {128, 256, 512, 1024, 2048};
    const uint32_t K = 8;

    for (uint32_t N : test_sizes) {
        spark::SparseGraphConfig g_cfg;
        g_cfg.num_units = N;
        g_cfg.fan_in = K;
        g_cfg.seed = 42ULL + N;
        g_cfg.allow_self_edges = false;

        std::string err;
        REQUIRE(g_cfg.validate(err), "Config validation failed for N=" + std::to_string(N));

        spark::SparseGraph graph = spark::SparseGraph::generate(g_cfg);
        REQUIRE(graph.validate(err), "Graph structure validation failed for N=" + std::to_string(N) + ": " + err);
        REQUIRE(graph.num_units() == N, "Graph unit count mismatch");
        REQUIRE(graph.total_edges() == static_cast<size_t>(N) * K, "Graph edge count mismatch");
        REQUIRE(graph.offsets.size() == N + 1, "Offsets size mismatch");
        REQUIRE(graph.sources.size() == static_cast<size_t>(N) * K, "Sources size mismatch");
        REQUIRE(graph.weights.size() == static_cast<size_t>(N) * K, "Weights size mismatch");

        // Verify each unit has exactly K in-bounds distinct incoming sources
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t start = graph.offsets[i];
            uint32_t end = graph.offsets[i + 1];
            REQUIRE(end - start == K, "Fan-in invariant violated for unit " + std::to_string(i));
            for (uint32_t e = start; e < end; ++e) {
                uint32_t src = graph.sources[e];
                REQUIRE(src < N, "Source out of bounds");
                REQUIRE(src != i, "Self-edge detected when disabled");
                if (e > start) {
                    REQUIRE(graph.sources[e] > graph.sources[e - 1], "Sources must be strictly sorted and unique");
                }
            }
        }
    }
    std::cout << "  Passed: Graph topology invariants verified for N=128..2048." << std::endl;
}

// Test 3: Parameter counts, memory footprints, and forward step stability
void test_network_scaling_and_stability() {
    std::cout << "[Test 3] Running test_network_scaling_and_stability..." << std::endl;
    const std::vector<uint32_t> test_sizes = {128, 256, 512, 1024, 2048};
    const uint32_t Din = 55;
    const uint32_t Dout = 25;
    const uint32_t K = 8;

    for (uint32_t N : test_sizes) {
        spark::NetworkConfig n_cfg;
        n_cfg.num_units = N;
        n_cfg.input_dim = Din;
        n_cfg.output_dim = Dout;
        n_cfg.alpha = 0.8f;
        n_cfg.beta = 0.5f;
        n_cfg.seed = 100ULL + N;

        spark::SparseGraphConfig g_cfg;
        g_cfg.num_units = N;
        g_cfg.fan_in = K;
        g_cfg.seed = 200ULL + N;

        spark::MicroUnitNetwork net(n_cfg, g_cfg);
        std::string err;
        REQUIRE(net.validate(err), "Network validation failed for N=" + std::to_string(N) + ": " + err);

        // Theoretical parameter count: N * K + N * Din + N + Dout * N + Dout = N * (K + Din + Dout + 1) + Dout
        size_t expected_params = static_cast<size_t>(N) * (K + Din + Dout + 1) + Dout;
        REQUIRE(net.parameter_count() == expected_params, "Parameter count mismatch for N=" + std::to_string(N));

        size_t mem_bytes = net.memory_bytes();
        std::cout << "  N=" << N << ": Params=" << net.parameter_count()
                  << ", Memory=" << (mem_bytes / 1024.0f) << " KB" << std::endl;

        // Run 50 forward steps with dummy inputs and check bounds
        std::vector<float> input(Din, 0.1f);
        std::vector<float> logits, probs;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < 50; ++t) {
            net.forward_step(input, logits, probs, spark::ReadoutMode::Softmax);
            REQUIRE(logits.size() == Dout, "Logits size mismatch");
            REQUIRE(probs.size() == Dout, "Probs size mismatch");

            // Probability distribution sum check
            float sum_p = 0.0f;
            for (float p : probs) {
                REQUIRE(std::isfinite(p), "Non-finite probability");
                REQUIRE(p >= 0.0f && p <= 1.0f, "Probability out of [0, 1] range");
                sum_p += p;
            }
            REQUIRE(std::abs(sum_p - 1.0f) < 1e-4f, "Softmax probabilities must sum to 1.0");

            // Hidden states check: tanh must be strictly in [-1, 1]
            for (float h : net.state()) {
                REQUIRE(std::isfinite(h), "Non-finite hidden state");
                REQUIRE(std::abs(h) <= 1.0f, "Hidden state exceeded tanh range [-1, 1]");
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 50.0;
        std::cout << "    Forward step latency: " << latency_us << " us/step" << std::endl;
    }
    std::cout << "  Passed: Network scaling, numerical stability, and latency verified." << std::endl;
}

// Test 4: Serialization and bitwise deserialization round-trip across all sizes
void test_checkpoint_serialization_all_sizes() {
    std::cout << "[Test 4] Running test_checkpoint_serialization_all_sizes..." << std::endl;
    const std::vector<uint32_t> test_sizes = {128, 256, 512, 1024, 2048};
    const uint32_t Din = 55;
    const uint32_t Dout = 25;
    const uint32_t K = 8;

    std::filesystem::create_directories("scratch/scaling_test");

    for (uint32_t N : test_sizes) {
        spark::NetworkConfig n_cfg;
        n_cfg.num_units = N;
        n_cfg.input_dim = Din;
        n_cfg.output_dim = Dout;
        n_cfg.seed = 300ULL + N;

        spark::SparseGraphConfig g_cfg;
        g_cfg.num_units = N;
        g_cfg.fan_in = K;
        g_cfg.seed = 400ULL + N;

        spark::MicroUnitNetwork original_net(n_cfg, g_cfg);
        std::string filepath = "scratch/scaling_test/ckpt_N" + std::to_string(N) + ".spark";

        REQUIRE(original_net.save_checkpoint(filepath), "Failed to save checkpoint for N=" + std::to_string(N));

        spark::MicroUnitNetwork restored_net(n_cfg, g_cfg); // Will be overwritten by load_checkpoint
        REQUIRE(spark::MicroUnitNetwork::load_checkpoint(filepath, restored_net),
                "Failed to load checkpoint for N=" + std::to_string(N));

        // Bitwise parameter comparison
        REQUIRE(restored_net.config().num_units == N, "Restored unit count mismatch");
        REQUIRE(restored_net.parameter_count() == original_net.parameter_count(), "Restored parameter count mismatch");
        REQUIRE(restored_net.input_weights() == original_net.input_weights(), "U weights bitwise mismatch");
        REQUIRE(restored_net.output_weights() == original_net.output_weights(), "V weights bitwise mismatch");
        REQUIRE(restored_net.biases() == original_net.biases(), "Biases bitwise mismatch");
        REQUIRE(restored_net.output_biases() == original_net.output_biases(), "Output biases bitwise mismatch");
        REQUIRE(restored_net.graph().weights == original_net.graph().weights, "Graph weights bitwise mismatch");
        REQUIRE(restored_net.graph().sources == original_net.graph().sources, "Graph sources bitwise mismatch");
        REQUIRE(restored_net.graph().offsets == original_net.graph().offsets, "Graph offsets bitwise mismatch");

        // Forward step comparison
        std::vector<float> input(Din, 0.42f);
        std::vector<float> orig_logits, orig_probs, rest_logits, rest_probs;
        original_net.forward_step(input, orig_logits, orig_probs, spark::ReadoutMode::Softmax);
        restored_net.forward_step(input, rest_logits, rest_probs, spark::ReadoutMode::Softmax);

        for (size_t i = 0; i < Dout; ++i) {
            REQUIRE(orig_logits[i] == rest_logits[i], "Logit mismatch after restoration");
            REQUIRE(orig_probs[i] == rest_probs[i], "Prob mismatch after restoration");
        }
    }
    std::filesystem::remove_all("scratch/scaling_test");
    std::cout << "  Passed: Checkpoint serialization and bitwise restoration verified for N=128..2048." << std::endl;
}

} // namespace

int main() {
    std::cout << "=== Running Milestone 6A.1 Network Scaling Verification Suite ===" << std::endl;
    test_capacity_limits();
    test_graph_scaling();
    test_network_scaling_and_stability();
    test_checkpoint_serialization_all_sizes();
    std::cout << "=== All Milestone 6A.1 Scaling Tests PASSED Successfully! ===" << std::endl;
    return 0;
}
