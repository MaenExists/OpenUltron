#include "spark/environment.hpp"
#include <stdexcept>

namespace spark {

TemporalMemoryEnv::TemporalMemoryEnv(uint32_t delay, uint32_t sequence_length, uint64_t seed)
    : delay_(delay), sequence_length_(sequence_length), seed_(seed), rng_(seed) {
    if (delay_ == 0) {
        throw std::invalid_argument("delay must be >= 1 for temporal memory");
    }
    if (sequence_length_ <= delay_) {
        throw std::invalid_argument("sequence_length must be strictly greater than delay");
    }
    reset();
}

void TemporalMemoryEnv::reset() {
    step_idx_ = 0;
    generate_sequence();
}

void TemporalMemoryEnv::reset_to_initial_seed() {
    rng_.reseed(seed_);
    step_idx_ = 0;
    generate_sequence();
}

void TemporalMemoryEnv::generate_sequence() {
    sequence_.resize(sequence_length_);
    for (uint32_t i = 0; i < sequence_length_; ++i) {
        // Uniform random bit 0.0 or 1.0
        sequence_[i] = (rng_.uniform_float(0.0f, 1.0f) >= 0.5f) ? 1.0f : 0.0f;
    }
}

Observation TemporalMemoryEnv::get_observation() const {
    if (step_idx_ < sequence_length_) {
        return Observation{{sequence_[step_idx_]}};
    }
    return Observation{{0.0f}};
}

Target TemporalMemoryEnv::get_target() const {
    if (step_idx_ < delay_) {
        // Target is from before the start of the sequence, mark as invalid
        return Target{{0.0f}, false};
    }
    if (step_idx_ < sequence_length_) {
        return Target{{sequence_[step_idx_ - delay_]}, true};
    }
    return Target{{0.0f}, false};
}

void TemporalMemoryEnv::step() {
    if (step_idx_ < sequence_length_) {
        step_idx_++;
    }
}

bool TemporalMemoryEnv::is_done() const {
    return step_idx_ >= sequence_length_;
}

} // namespace spark
