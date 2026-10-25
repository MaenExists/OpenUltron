#include "spark/graph.hpp"
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <stdexcept>

namespace spark {

SparseGraph::SparseGraph(const SparseGraphConfig& config)
    : config_(config) {
    *this = generate(config);
}

SparseGraph SparseGraph::generate(const SparseGraphConfig& config) {
    std::string err;
    if (!config.validate(err)) {
        throw std::invalid_argument("Invalid SparseGraphConfig: " + err);
    }

    Random rng(config.seed);
    if (config.topology == TopologyType::ClusteredSparse) {
        return generate_clustered(config, rng);
    } else {
        return generate_random(config, rng);
    }
}

SparseGraph SparseGraph::generate_random(const SparseGraphConfig& config, Random& rng) {
    SparseGraph graph;
    graph.config_ = config;
    const uint32_t N = config.num_units;
    const uint32_t K = config.fan_in;

    graph.offsets.resize(N + 1, 0);
    graph.sources.reserve(static_cast<size_t>(N) * K);
    graph.weights.reserve(static_cast<size_t>(N) * K);

    std::vector<uint32_t> pool;
    pool.reserve(N);

    for (uint32_t i = 0; i < N; ++i) {
        pool.clear();
        for (uint32_t j = 0; j < N; ++j) {
            if (config.allow_self_edges || j != i) {
                pool.push_back(j);
            }
        }

        // Partial Fisher-Yates shuffle to pick K distinct sources
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t pick_idx = rng.uniform_uint32(k, static_cast<uint32_t>(pool.size() - 1));
            std::swap(pool[k], pool[pick_idx]);
        }

        // Sort chosen sources for deterministic cache-friendly traversal
        std::sort(pool.begin(), pool.begin() + K);

        for (uint32_t k = 0; k < K; ++k) {
            graph.sources.push_back(pool[k]);
            graph.weights.push_back(rng.normal_float(0.0f, config.weight_init_std));
        }

        graph.offsets[i + 1] = static_cast<uint32_t>(graph.sources.size());
    }

    return graph;
}

SparseGraph SparseGraph::generate_clustered(const SparseGraphConfig& config, Random& rng) {
    SparseGraph graph;
    graph.config_ = config;
    const uint32_t N = config.num_units;
    const uint32_t K = config.fan_in;
    const uint32_t C = config.num_clusters;

    graph.offsets.resize(N + 1, 0);
    graph.sources.reserve(static_cast<size_t>(N) * K);
    graph.weights.reserve(static_cast<size_t>(N) * K);

    auto cluster_of = [N, C](uint32_t u) -> uint32_t {
        return (static_cast<uint64_t>(u) * C) / N;
    };

    std::vector<std::vector<uint32_t>> cluster_members(C);
    for (uint32_t u = 0; u < N; ++u) {
        cluster_members[cluster_of(u)].push_back(u);
    }

    std::vector<uint32_t> chosen;
    chosen.reserve(K);
    std::unordered_set<uint32_t> chosen_set;

    for (uint32_t i = 0; i < N; ++i) {
        chosen.clear();
        chosen_set.clear();
        uint32_t my_cluster = cluster_of(i);

        // Separate candidates into intra and inter
        std::vector<uint32_t> intra_pool;
        for (uint32_t u : cluster_members[my_cluster]) {
            if (config.allow_self_edges || u != i) {
                intra_pool.push_back(u);
            }
        }

        std::vector<uint32_t> inter_pool;
        for (uint32_t c = 0; c < C; ++c) {
            if (c == my_cluster) continue;
            for (uint32_t u : cluster_members[c]) {
                inter_pool.push_back(u);
            }
        }

        while (chosen.size() < K) {
            bool pick_intra = (rng.uniform_float(0.0f, 1.0f) < config.intra_cluster_prob);
            if (pick_intra && !intra_pool.empty()) {
                uint32_t idx = rng.uniform_uint32(0, static_cast<uint32_t>(intra_pool.size() - 1));
                uint32_t candidate = intra_pool[idx];
                if (chosen_set.insert(candidate).second) {
                    chosen.push_back(candidate);
                }
            } else if (!inter_pool.empty()) {
                uint32_t idx = rng.uniform_uint32(0, static_cast<uint32_t>(inter_pool.size() - 1));
                uint32_t candidate = inter_pool[idx];
                if (chosen_set.insert(candidate).second) {
                    chosen.push_back(candidate);
                }
            } else if (!intra_pool.empty()) {
                uint32_t idx = rng.uniform_uint32(0, static_cast<uint32_t>(intra_pool.size() - 1));
                uint32_t candidate = intra_pool[idx];
                if (chosen_set.insert(candidate).second) {
                    chosen.push_back(candidate);
                }
            } else {
                break;
            }
        }

        // Sort chosen sources
        std::sort(chosen.begin(), chosen.end());

        for (uint32_t src : chosen) {
            graph.sources.push_back(src);
            graph.weights.push_back(rng.normal_float(0.0f, config.weight_init_std));
        }

        graph.offsets[i + 1] = static_cast<uint32_t>(graph.sources.size());
    }

    return graph;
}

bool SparseGraph::validate(std::string& err_msg) const {
    if (offsets.empty()) {
        err_msg = "Offsets array is empty";
        return false;
    }
    const uint32_t N = num_units();
    if (offsets[0] != 0) {
        err_msg = "offsets[0] must be 0";
        return false;
    }
    if (offsets.back() != sources.size()) {
        err_msg = "offsets.back() does not match sources.size()";
        return false;
    }
    if (sources.size() != weights.size()) {
        err_msg = "sources.size() does not match weights.size()";
        return false;
    }

    for (uint32_t i = 0; i < N; ++i) {
        uint32_t start = offsets[i];
        uint32_t end = offsets[i + 1];
        if (start > end) {
            err_msg = "Monotonicity violated in offsets array";
            return false;
        }

        for (uint32_t e = start; e < end; ++e) {
            uint32_t src = sources[e];
            if (src >= N) {
                err_msg = "Source index " + std::to_string(src) + " out of bounds for unit count " + std::to_string(N);
                return false;
            }
            if (!config_.allow_self_edges && src == i) {
                err_msg = "Self-edge detected at unit " + std::to_string(i);
                return false;
            }
            if (e > start && sources[e] == sources[e - 1]) {
                err_msg = "Duplicate edge from " + std::to_string(src) + " to " + std::to_string(i);
                return false;
            }
            if (std::isnan(weights[e]) || std::isinf(weights[e])) {
                err_msg = "Non-finite weight detected at edge " + std::to_string(e);
                return false;
            }
        }
    }

    return true;
}

GraphStats SparseGraph::compute_stats() const {
    GraphStats stats;
    stats.num_units = num_units();
    stats.total_edges = total_edges();
    stats.memory_bytes = memory_bytes();

    if (stats.num_units == 0) return stats;

    stats.min_fan_in = UINT32_MAX;
    stats.max_fan_in = 0;
    double fan_in_sum = 0.0;

    for (uint32_t i = 0; i < stats.num_units; ++i) {
        uint32_t deg = fan_in(i);
        if (deg < stats.min_fan_in) stats.min_fan_in = deg;
        if (deg > stats.max_fan_in) stats.max_fan_in = deg;
        fan_in_sum += deg;
    }
    stats.avg_fan_in = fan_in_sum / stats.num_units;

    if (stats.total_edges > 0) {
        stats.min_weight = weights[0];
        stats.max_weight = weights[0];
        double weight_sum = 0.0;
        double weight_sq_sum = 0.0;

        for (float w : weights) {
            if (w < stats.min_weight) stats.min_weight = w;
            if (w > stats.max_weight) stats.max_weight = w;
            weight_sum += w;
            weight_sq_sum += static_cast<double>(w) * w;
        }

        stats.mean_weight = static_cast<float>(weight_sum / stats.total_edges);
        double variance = (weight_sq_sum / stats.total_edges) - (static_cast<double>(stats.mean_weight) * stats.mean_weight);
        stats.std_weight = static_cast<float>(std::sqrt(std::max(0.0, variance)));
    }

    return stats;
}

size_t SparseGraph::memory_bytes() const {
    return (offsets.capacity() * sizeof(uint32_t)) +
           (sources.capacity() * sizeof(uint32_t)) +
           (weights.capacity() * sizeof(float)) +
           sizeof(SparseGraph);
}

} // namespace spark
