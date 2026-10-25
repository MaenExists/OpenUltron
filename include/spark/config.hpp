#pragma once

#include <cstdint>
#include <string>
#include <sstream>

namespace spark {

enum class TopologyType {
    RandomSparse,
    ClusteredSparse,
    AdaptiveSparse
};

inline const char* topology_name(TopologyType type) {
    switch (type) {
        case TopologyType::RandomSparse: return "random_sparse";
        case TopologyType::ClusteredSparse: return "clustered_sparse";
        case TopologyType::AdaptiveSparse: return "adaptive_sparse";
        default: return "unknown";
    }
}

struct SparseGraphConfig {
    uint32_t num_units = 128;
    uint32_t fan_in = 8;
    TopologyType topology = TopologyType::RandomSparse;
    uint32_t num_clusters = 4;
    float intra_cluster_prob = 0.8f;
    uint64_t seed = 42ULL;
    float weight_init_std = 0.5f;
    bool allow_self_edges = false;

    bool validate(std::string& err_msg) const {
        if (num_units == 0) {
            err_msg = "num_units must be > 0";
            return false;
        }
        if (fan_in == 0) {
            err_msg = "fan_in must be > 0";
            return false;
        }
        if (!allow_self_edges && fan_in >= num_units) {
            err_msg = "fan_in must be < num_units when self-edges are disabled";
            return false;
        }
        if (allow_self_edges && fan_in > num_units) {
            err_msg = "fan_in must be <= num_units";
            return false;
        }
        if (topology == TopologyType::ClusteredSparse) {
            if (num_clusters == 0 || num_clusters > num_units) {
                err_msg = "num_clusters must be between 1 and num_units";
                return false;
            }
            if (intra_cluster_prob < 0.0f || intra_cluster_prob > 1.0f) {
                err_msg = "intra_cluster_prob must be between 0.0 and 1.0";
                return false;
            }
        }
        if (weight_init_std <= 0.0f) {
            err_msg = "weight_init_std must be > 0";
            return false;
        }
        return true;
    }
};

struct NetworkConfig {
    uint32_t num_units = 128;
    uint32_t input_dim = 4;
    uint32_t output_dim = 1;
    float alpha = 0.8f;     // Recurrent state retention coefficient
    float beta = 0.5f;      // Recurrent input scaling coefficient
    uint32_t microsteps = 1; // Recurrent internal refinement microsteps
    uint64_t seed = 42ULL;

    bool validate(std::string& err_msg) const {
        if (num_units == 0) {
            err_msg = "num_units must be > 0";
            return false;
        }
        if (input_dim == 0) {
            err_msg = "input_dim must be > 0";
            return false;
        }
        if (output_dim == 0) {
            err_msg = "output_dim must be > 0";
            return false;
        }
        if (alpha < 0.0f || alpha > 1.0f) {
            err_msg = "alpha must be in range [0.0, 1.0]";
            return false;
        }
        if (beta < 0.0f) {
            err_msg = "beta must be >= 0.0";
            return false;
        }
        if (microsteps == 0) {
            err_msg = "microsteps must be >= 1";
            return false;
        }
        return true;
    }
};

} // namespace spark
