#include "spark/graph.hpp"
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

void test_random_graph() {
    std::cout << "[Test] Running test_random_graph..." << std::endl;
    spark::SparseGraphConfig config;
    config.num_units = 128;
    config.fan_in = 8;
    config.topology = spark::TopologyType::RandomSparse;
    config.seed = 12345ULL;

    spark::SparseGraph graph(config);
    std::string err;
    bool valid = graph.validate(err);
    REQUIRE(valid, err);

    REQUIRE(graph.num_units() == 128, "Incorrect unit count");
    REQUIRE(graph.total_edges() == 128 * 8, "Incorrect edge count");

    auto stats = graph.compute_stats();
    REQUIRE(stats.min_fan_in == 8, "Incorrect min fan-in");
    REQUIRE(stats.max_fan_in == 8, "Incorrect max fan-in");
    REQUIRE(std::abs(stats.avg_fan_in - 8.0) < 1e-6, "Incorrect avg fan-in");
    REQUIRE(stats.memory_bytes > 0, "Memory bytes must be > 0");

    // Test deterministic initialization
    spark::SparseGraph graph2(config);
    REQUIRE(graph.sources == graph2.sources, "Sources mismatch between seeded instances");
    REQUIRE(graph.weights == graph2.weights, "Weights mismatch between seeded instances");
    std::cout << "[Test] test_random_graph PASSED." << std::endl;
}

void test_clustered_graph() {
    std::cout << "[Test] Running test_clustered_graph..." << std::endl;
    spark::SparseGraphConfig config;
    config.num_units = 256;
    config.fan_in = 16;
    config.topology = spark::TopologyType::ClusteredSparse;
    config.num_clusters = 4;
    config.intra_cluster_prob = 0.85f;
    config.seed = 999ULL;

    spark::SparseGraph graph(config);
    std::string err;
    REQUIRE(graph.validate(err), err);
    REQUIRE(graph.num_units() == 256, "Incorrect unit count");
    REQUIRE(graph.total_edges() == 256 * 16, "Incorrect edge count");

    // Verify intra-cluster proportion is approximately as configured
    auto cluster_of = [](uint32_t u, uint32_t N, uint32_t C) -> uint32_t {
        return (static_cast<uint64_t>(u) * C) / N;
    };

    size_t intra_count = 0;
    for (uint32_t i = 0; i < graph.num_units(); ++i) {
        uint32_t my_c = cluster_of(i, 256, 4);
        uint32_t start = graph.offsets[i];
        uint32_t end = graph.offsets[i + 1];
        for (uint32_t e = start; e < end; ++e) {
            if (cluster_of(graph.sources[e], 256, 4) == my_c) {
                intra_count++;
            }
        }
    }

    double observed_intra_prob = static_cast<double>(intra_count) / graph.total_edges();
    std::cout << "Observed intra-cluster edge proportion: " << observed_intra_prob << std::endl;
    REQUIRE(observed_intra_prob > 0.70 && observed_intra_prob < 0.95, "Intra-cluster proportion out of expected bounds");

    std::cout << "[Test] test_clustered_graph PASSED." << std::endl;
}

int main() {
    std::cout << "=== Running SparseGraph Tests ===" << std::endl;
    test_random_graph();
    test_clustered_graph();
    std::cout << "All SparseGraph tests passed successfully!" << std::endl;
    return 0;
}

