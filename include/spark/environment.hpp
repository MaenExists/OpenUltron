#pragma once

#include "spark/random.hpp"
#include <vector>
#include <cstdint>
#include <string>

namespace spark {

struct Observation {
    std::vector<float> values;
};

struct Target {
    std::vector<float> values;
    bool valid = true; // Indicates whether supervision should apply at this step
};

class Environment {
public:
    virtual ~Environment() = default;

    virtual void reset() = 0;
    virtual Observation get_observation() const = 0;
    virtual Target get_target() const = 0;
    virtual void step() = 0;
    virtual bool is_done() const = 0;

    virtual uint32_t input_dim() const = 0;
    virtual uint32_t output_dim() const = 0;
    virtual uint32_t current_step() const = 0;
};

class TemporalMemoryEnv : public Environment {
public:
    TemporalMemoryEnv(uint32_t delay, uint32_t sequence_length, uint64_t seed = 42ULL);

    void reset() override;
    Observation get_observation() const override;
    Target get_target() const override;
    void step() override;
    bool is_done() const override;

    uint32_t input_dim() const override { return 1; }
    uint32_t output_dim() const override { return 1; }
    uint32_t current_step() const override { return step_idx_; }

    uint32_t delay() const { return delay_; }
    uint32_t sequence_length() const { return sequence_length_; }
    const std::vector<float>& sequence() const { return sequence_; }

private:
    void generate_sequence();

    uint32_t delay_;
    uint32_t sequence_length_;
    uint64_t seed_;
    Random rng_;
    uint32_t step_idx_ = 0;
    std::vector<float> sequence_;
};

} // namespace spark
