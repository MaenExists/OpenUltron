#pragma once

#include "spark/network.hpp"
#include "spark/environment.hpp"
#include "spark/learning.hpp"
#include <cstdint>
#include <vector>
#include <string>

namespace spark {

struct EvalMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
    size_t correct_steps = 0;
    size_t evaluated_steps = 0;
};

struct TrainHistoryEntry {
    uint32_t step = 0;
    float train_loss = 0.0f;
    float grad_norm = 0.0f;
    float eval_loss = 0.0f;
    float eval_accuracy = 0.0f;
    double elapsed_sec = 0.0;
};

struct TrainSummary {
    uint32_t steps_completed = 0;
    float initial_loss = 0.0f;
    float final_train_loss = 0.0f;
    float final_eval_loss = 0.0f;
    float final_eval_accuracy = 0.0f;
    size_t final_eval_correct = 0;
    size_t final_eval_total = 0;
    double total_seconds = 0.0;
    std::vector<TrainHistoryEntry> history;
};

class Trainer {
public:
    Trainer(
        MicroUnitNetwork& net,
        Environment& train_env,
        Environment* eval_env = nullptr,
        float learning_rate = 0.02f,
        float max_grad_norm = 1.0f
    );

    TrainSummary train(
        uint32_t total_steps,
        uint32_t unroll_steps = 8,
        uint32_t eval_interval = 200,
        bool quiet = false
    );

    EvalMetrics evaluate(Environment& eval_env, uint32_t max_eval_steps = 500);

    LossType loss_type() const { return loss_type_; }
    void set_loss_type(LossType type) { loss_type_ = type; }

    void set_train_recurrent(bool enabled) { train_recurrent_ = enabled; }
    void set_train_input(bool enabled) { train_input_ = enabled; }
    void set_train_biases(bool enabled) { train_biases_ = enabled; }
    void set_train_readout(bool enabled) { train_readout_ = enabled; }
    void set_readout_only(bool readout_only) {
        train_recurrent_ = !readout_only;
        train_input_ = !readout_only;
        train_biases_ = !readout_only;
        train_readout_ = true;
    }

private:
    MicroUnitNetwork& net_;
    Environment& train_env_;
    Environment* eval_env_;
    float learning_rate_;
    float max_grad_norm_;
    LossType loss_type_ = LossType::BinaryCrossEntropy;
    bool train_recurrent_ = true;
    bool train_input_ = true;
    bool train_biases_ = true;
    bool train_readout_ = true;
};

} // namespace spark
