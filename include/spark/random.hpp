#pragma once

#include <cstdint>
#include <random>
#include <cmath>

namespace spark {

class Random {
public:
    explicit Random(uint64_t seed = 42ULL)
        : seed_(seed), rng_(seed) {}

    void reseed(uint64_t seed) {
        seed_ = seed;
        rng_.seed(seed);
    }

    uint64_t get_seed() const {
        return seed_;
    }

    float uniform_float(float low = 0.0f, float high = 1.0f) {
        std::uniform_real_distribution<float> dist(low, high);
        return dist(rng_);
    }

    float normal_float(float mean = 0.0f, float stddev = 1.0f) {
        std::normal_distribution<float> dist(mean, stddev);
        return dist(rng_);
    }

    int uniform_int(int low, int high) {
        std::uniform_int_distribution<int> dist(low, high);
        return dist(rng_);
    }

    uint32_t uniform_uint32(uint32_t low, uint32_t high) {
        std::uniform_int_distribution<uint32_t> dist(low, high);
        return dist(rng_);
    }

    float xavier(size_t fan_in, size_t fan_out) {
        float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
        return uniform_float(-limit, limit);
    }

private:
    uint64_t seed_;
    std::mt19937_64 rng_;
};

} // namespace spark
