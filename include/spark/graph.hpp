#pragma once

#include "spark/config.hpp"
#include "spark/random.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

namespace spark {

struct GraphStats {
    uint32_t num_units = 0;
    size_t total_edges = 0;
    uint32_t min_fan_in = 0;
    uint32_t max_fan_in = 0;
    double avg_fan_in = 0.0;
    float min_weight = 0.0f;
    float max_weight = 0.0f;
    float mean_weight = 0.0f;
    float std_weight = 0.0f;
    size_t memory_bytes = 0;
};

class SparseGraph {
public:
    // CSR data structures
    std::vector<uint32_t> offsets; // size num_units + 1
    std::vector<uint32_t> sources; // size total_edges
    std::vector<float> weights;    // size total_edges

    SparseGraph() = default;
    explicit SparseGraph(const SparseGraphConfig& config);

    // Initialization methods
    static SparseGraph generate(const SparseGraphConfig& config);
    static SparseGraph generate_random(const SparseGraphConfig& config, Random& rng);
    static SparseGraph generate_clustered(const SparseGraphConfig& config, Random& rng);

    // Accessors
    const SparseGraphConfig& config() const { return config_; }
    void set_config(const SparseGraphConfig& c) { config_ = c; }

    uint32_t num_units() const {
        return offsets.empty() ? 0 : static_cast<uint32_t>(offsets.size() - 1);
    }

    size_t total_edges() const {
        return sources.size();
    }

    uint32_t fan_in(uint32_t unit_idx) const {
        return offsets[unit_idx + 1] - offsets[unit_idx];
    }

    // Validation
    bool validate(std::string& err_msg) const;

    // Statistics and profiling
    GraphStats compute_stats() const;
    size_t memory_bytes() const;

private:
    SparseGraphConfig config_;
};

} // namespace spark
