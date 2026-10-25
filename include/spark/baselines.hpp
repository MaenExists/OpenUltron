#pragma once

#include "spark/environment.hpp"
#include "spark/network.hpp"
#include <vector>
#include <deque>
#include <cstdint>
#include <string>

namespace spark {

struct BaselineResult {
    std::string name;
    float accuracy = 0.0f;
    float loss = 0.0f;
    size_t evaluated_steps = 0;
};

// 1. Shift-Register Oracle: Ground-truth reference retaining exactly D past bits
class ShiftRegisterOracle {
public:
    explicit ShiftRegisterOracle(uint32_t delay) : delay_(delay) {}

    void reset() {
        buffer_.clear();
    }

    // Process input bit and return predicted delayed bit
    // Returns pair<prediction, valid>
    std::pair<float, bool> step(float input_bit) {
        buffer_.push_back(input_bit);
        if (buffer_.size() <= delay_) {
            return {0.0f, false}; // Warmup
        }
        float delayed_bit = buffer_.front();
        buffer_.pop_front();
        return {delayed_bit, true};
    }

    uint32_t delay() const { return delay_; }

private:
    uint32_t delay_;
    std::deque<float> buffer_;
};

// 2. Constant Predictor: Always predicts the majority class observed in training
class ConstantPredictor {
public:
    explicit ConstantPredictor(float constant_prob = 0.5f) : constant_prob_(constant_prob) {}

    void fit(Environment& train_env, uint32_t max_steps = 1000);
    float predict() const { return constant_prob_; }

private:
    float constant_prob_ = 0.5f;
};

// 3. Memoryless Predictor: Predicts target using only the current observation via logistic regression
class MemorylessPredictor {
public:
    MemorylessPredictor() = default;

    void fit(Environment& train_env, uint32_t max_steps = 2000, float lr = 0.05f);
    float predict(float current_obs) const;

    float weight() const { return w_; }
    float bias() const { return b_; }

private:
    float w_ = 0.0f;
    float b_ = 0.0f;
};

// Baseline evaluation harness
class BaselineEvaluator {
public:
    static BaselineResult evaluate_oracle(ShiftRegisterOracle& oracle, Environment& env, uint32_t max_steps = 500);
    static BaselineResult evaluate_constant(const ConstantPredictor& pred, Environment& env, uint32_t max_steps = 500);
    static BaselineResult evaluate_memoryless(const MemorylessPredictor& pred, Environment& env, uint32_t max_steps = 500);
    static BaselineResult evaluate_untrained_spark(MicroUnitNetwork& net, Environment& env, uint32_t max_steps = 500);
    static BaselineResult evaluate_trained_spark(MicroUnitNetwork& net, Environment& env, uint32_t max_steps = 500);
};

} // namespace spark
