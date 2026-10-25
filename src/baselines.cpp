#include "spark/baselines.hpp"
#include "spark/learning.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace spark {

void ConstantPredictor::fit(Environment& train_env, uint32_t max_steps) {
    train_env.reset();
    size_t count1 = 0;
    size_t count0 = 0;

    for (uint32_t s = 0; s < max_steps && !train_env.is_done(); ++s) {
        Target target = train_env.get_target();
        train_env.step();
        if (target.valid && !target.values.empty()) {
            if (target.values[0] >= 0.5f) count1++;
            else count0++;
        }
    }

    constant_prob_ = (count1 >= count0) ? 1.0f : 0.0f;
}

void MemorylessPredictor::fit(Environment& train_env, uint32_t max_steps, float lr) {
    train_env.reset();
    w_ = 0.0f;
    b_ = 0.0f;

    for (uint32_t s = 0; s < max_steps; ++s) {
        if (train_env.is_done()) train_env.reset();

        Observation obs = train_env.get_observation();
        Target target = train_env.get_target();
        train_env.step();

        if (target.valid && !obs.values.empty() && !target.values.empty()) {
            float x = obs.values[0];
            float y = target.values[0];
            float z = w_ * x + b_;
            float p = MicroUnitNetwork::stable_sigmoid(z);
            float grad = p - y;

            w_ -= lr * grad * x;
            b_ -= lr * grad;
        }
    }
}

float MemorylessPredictor::predict(float current_obs) const {
    float z = w_ * current_obs + b_;
    return MicroUnitNetwork::stable_sigmoid(z);
}

BaselineResult BaselineEvaluator::evaluate_oracle(ShiftRegisterOracle& oracle, Environment& env, uint32_t max_steps) {
    BaselineResult res;
    res.name = "Shift-Register Oracle";
    oracle.reset();
    env.reset();

    size_t valid_count = 0;
    size_t correct_count = 0;
    double total_loss = 0.0;

    for (uint32_t s = 0; s < max_steps && !env.is_done(); ++s) {
        Observation obs = env.get_observation();
        Target target = env.get_target();

        auto oracle_out = oracle.step(obs.values[0]);
        env.step();

        if (target.valid) {
            valid_count++;
            float pred_bit = oracle_out.first;
            if (std::abs(pred_bit - target.values[0]) < 1e-3f) {
                correct_count++;
            }
            float z = (pred_bit >= 0.5f) ? 15.0f : -15.0f; // High-confidence logits
            total_loss += TBPTTTrainer::bce_loss(z, target.values[0]);
        }
    }

    res.evaluated_steps = valid_count;
    res.correct_steps = correct_count;
    if (valid_count > 0) {
        res.accuracy = static_cast<float>(correct_count) / static_cast<float>(valid_count);
        res.loss = static_cast<float>(total_loss / valid_count);
    }
    return res;
}

BaselineResult BaselineEvaluator::evaluate_constant(const ConstantPredictor& pred, Environment& env, uint32_t max_steps) {
    BaselineResult res;
    res.name = "Constant Predictor (Majority)";
    env.reset();

    size_t valid_count = 0;
    size_t correct_count = 0;
    double total_loss = 0.0;
    float prob = pred.predict();
    float pred_bit = (prob >= 0.5f) ? 1.0f : 0.0f;
    // For loss, bounded prob between [1e-5, 1 - 1e-5]
    float p_clamped = std::clamp(prob, 1e-5f, 1.0f - 1e-5f);
    float z = std::log(p_clamped / (1.0f - p_clamped));

    for (uint32_t s = 0; s < max_steps && !env.is_done(); ++s) {
        Target target = env.get_target();
        env.step();

        if (target.valid) {
            valid_count++;
            if (std::abs(pred_bit - target.values[0]) < 1e-3f) {
                correct_count++;
            }
            total_loss += TBPTTTrainer::bce_loss(z, target.values[0]);
        }
    }

    res.evaluated_steps = valid_count;
    res.correct_steps = correct_count;
    if (valid_count > 0) {
        res.accuracy = static_cast<float>(correct_count) / static_cast<float>(valid_count);
        res.loss = static_cast<float>(total_loss / valid_count);
    }
    return res;
}

BaselineResult BaselineEvaluator::evaluate_memoryless(const MemorylessPredictor& pred, Environment& env, uint32_t max_steps) {
    BaselineResult res;
    res.name = "Memoryless Predictor";
    env.reset();

    size_t valid_count = 0;
    size_t correct_count = 0;
    double total_loss = 0.0;

    for (uint32_t s = 0; s < max_steps && !env.is_done(); ++s) {
        Observation obs = env.get_observation();
        Target target = env.get_target();
        env.step();

        if (target.valid) {
            valid_count++;
            float prob = pred.predict(obs.values[0]);
            float pred_bit = (prob >= 0.5f) ? 1.0f : 0.0f;
            if (std::abs(pred_bit - target.values[0]) < 1e-3f) {
                correct_count++;
            }
            float p_clamped = std::clamp(prob, 1e-5f, 1.0f - 1e-5f);
            float z = std::log(p_clamped / (1.0f - p_clamped));
            total_loss += TBPTTTrainer::bce_loss(z, target.values[0]);
        }
    }

    res.evaluated_steps = valid_count;
    res.correct_steps = correct_count;
    if (valid_count > 0) {
        res.accuracy = static_cast<float>(correct_count) / static_cast<float>(valid_count);
        res.loss = static_cast<float>(total_loss / valid_count);
    }
    return res;
}

BaselineResult BaselineEvaluator::evaluate_untrained_spark(MicroUnitNetwork& net, Environment& env, uint32_t max_steps) {
    BaselineResult res;
    res.name = "Untrained SPARK";
    env.reset();
    net.reset_state();

    size_t valid_count = 0;
    size_t correct_count = 0;
    double total_loss = 0.0;
    std::vector<float> logits, probs;

    for (uint32_t s = 0; s < max_steps && !env.is_done(); ++s) {
        Observation obs = env.get_observation();
        Target target = env.get_target();

        net.forward_step(obs.values, logits, probs, ReadoutMode::Sigmoid);
        env.step();

        if (target.valid) {
            valid_count++;
            float pred_bit = (probs[0] >= 0.5f) ? 1.0f : 0.0f;
            if (std::abs(pred_bit - target.values[0]) < 1e-3f) {
                correct_count++;
            }
            total_loss += TBPTTTrainer::bce_loss(logits[0], target.values[0]);
        }
    }

    res.evaluated_steps = valid_count;
    res.correct_steps = correct_count;
    if (valid_count > 0) {
        res.accuracy = static_cast<float>(correct_count) / static_cast<float>(valid_count);
        res.loss = static_cast<float>(total_loss / valid_count);
    }
    return res;
}

BaselineResult BaselineEvaluator::evaluate_trained_spark(MicroUnitNetwork& net, Environment& env, uint32_t max_steps) {
    BaselineResult res;
    res.name = "Trained SPARK (TBPTT)";
    env.reset();
    net.reset_state();

    size_t valid_count = 0;
    size_t correct_count = 0;
    double total_loss = 0.0;
    std::vector<float> logits, probs;

    for (uint32_t s = 0; s < max_steps && !env.is_done(); ++s) {
        Observation obs = env.get_observation();
        Target target = env.get_target();

        net.forward_step(obs.values, logits, probs, ReadoutMode::Sigmoid);
        env.step();

        if (target.valid) {
            valid_count++;
            float pred_bit = (probs[0] >= 0.5f) ? 1.0f : 0.0f;
            if (std::abs(pred_bit - target.values[0]) < 1e-3f) {
                correct_count++;
            }
            total_loss += TBPTTTrainer::bce_loss(logits[0], target.values[0]);
        }
    }

    res.evaluated_steps = valid_count;
    res.correct_steps = correct_count;
    if (valid_count > 0) {
        res.accuracy = static_cast<float>(correct_count) / static_cast<float>(valid_count);
        res.loss = static_cast<float>(total_loss / valid_count);
    }
    return res;
}

} // namespace spark
