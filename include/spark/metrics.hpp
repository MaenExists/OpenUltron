#pragma once

#include <chrono>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace spark {

class Timer {
public:
    Timer() {
        reset();
    }

    void reset() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_seconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }

    double elapsed_ms() const {
        return elapsed_seconds() * 1000.0;
    }

    double elapsed_us() const {
        return elapsed_seconds() * 1000000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

struct MemoryUsage {
    size_t resident_bytes = 0;
    size_t virtual_bytes = 0;
    size_t peak_resident_bytes = 0;
};

MemoryUsage get_current_memory_usage();

struct ActivationStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean_val = 0.0f;
    float std_val = 0.0f;
    float sparsity = 0.0f; // Fraction of activations with |h| < 0.05
    size_t non_finite_count = 0;
};

ActivationStats compute_activation_stats(const std::vector<float>& activations);

} // namespace spark
