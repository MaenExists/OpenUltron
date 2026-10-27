#pragma once

#include "spark/network.hpp"
#include "spark/environment.hpp"
#include "spark/trainer.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

namespace spark {

enum class FeedbackMode {
    RandomFeedback,        // Mode B: Fixed random matrix B
    ReadoutTranspose       // Mode C: Readout transpose V^T
};

inline const char* feedback_mode_name(FeedbackMode mode) {
    switch (mode) {
        case FeedbackMode::RandomFeedback: return "random_feedback";
        case FeedbackMode::ReadoutTranspose: return "readout_transpose";
        default: return "unknown";
    }
}

enum class LearningAlgorithmMode {
    ExactTBPTT,                  // Mode A: Exact TBPTT reference
    LocalRandomFeedback,         // Mode B: Local learning with random feedback
    LocalReadoutTranspose,       // Mode C: Local learning with readout-transpose feedback
    ReadoutOnly                  // Mode D: Frozen recurrent graph, train readout only
};

inline const char* learning_algorithm_name(LearningAlgorithmMode mode) {
    switch (mode) {
        case LearningAlgorithmMode::ExactTBPTT: return "exact_tbptt";
        case LearningAlgorithmMode::LocalRandomFeedback: return "local_random_feedback";
        case LearningAlgorithmMode::LocalReadoutTranspose: return "local_readout_transpose";
        case LearningAlgorithmMode::ReadoutOnly: return "readout_only";
        default: return "unknown";
    }
}

// Local eligibility traces aligned with network CSR structures
struct EligibilityTraces {
    std::vector<float> E_weights; // Aligned with graph.weights [total_edges]
    std::vector<float> E_U;       // Aligned with input_weights [num_units x input_dim]
    std::vector<float> E_b;       // Aligned with biases [num_units]

    explicit EligibilityTraces(const MicroUnitNetwork& net);

    void reset();
    void update(
        const MicroUnitNetwork& net,
        const std::vector<float>& x,
        const std::vector<float>& h_prev,
        float lambda
    );

    size_t memory_bytes() const;
};

// Feedback projection matrix
class FeedbackProjection {
public:
    FeedbackProjection() = default;
    FeedbackProjection(uint32_t num_units, uint32_t output_dim, uint64_t seed, float scale = 1.0f);

    void compute_feedback(
        const MicroUnitNetwork& net,
        const std::vector<float>& output_errors,
        FeedbackMode mode,
        std::vector<float>& g_out
    ) const;

    size_t memory_bytes() const;
    const std::vector<float>& B() const { return B_; }

private:
    uint32_t num_units_ = 0;
    uint32_t output_dim_ = 0;
    std::vector<float> B_; // [num_units x output_dim]
};

struct LocalLearningConfig {
    float lr = 0.03f;
    float lambda = 0.8f;                     // Eligibility trace decay
    FeedbackMode feedback_mode = FeedbackMode::RandomFeedback;
    LossType loss_type = LossType::BinaryCrossEntropy;
    float feedback_scale = 1.0f;
    float max_update_norm = 1.0f;
    float weight_decay = 0.0f;
    bool train_recurrent = true;
    bool train_input = true;
    bool train_readout = true;
    uint64_t feedback_seed = 42ULL;
};

class LocalLearner {
public:
    LocalLearner(MicroUnitNetwork& net, const LocalLearningConfig& config);

    void reset();

    // Performs one online step:
    // 1. Forward recurrent step
    // 2. Update eligibility traces
    // 3. Compute output error and feedback g_i
    // 4. Update local weights
    // Returns step loss (if target valid, else 0.0f)
    float step(const std::vector<float>& x, const Target& target);

    // Supervised online training loop
    TrainSummary train(
        Environment& train_env,
        Environment* eval_env,
        uint32_t total_steps,
        uint32_t eval_interval = 200,
        bool quiet = false
    );

    MicroUnitNetwork& network() { return net_; }
    const LocalLearningConfig& config() const { return config_; }
    const EligibilityTraces& traces() const { return traces_; }
    const FeedbackProjection& feedback() const { return feedback_; }

    size_t training_memory_bytes() const;

private:
    MicroUnitNetwork& net_;
    LocalLearningConfig config_;
    EligibilityTraces traces_;
    FeedbackProjection feedback_;

    std::vector<float> g_feedback_;
    std::vector<float> output_errors_;
    std::vector<float> logits_;
    std::vector<float> probs_;
};

} // namespace spark
