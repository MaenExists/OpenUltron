#pragma once

#include "spark/network.hpp"
#include "spark/environment.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>

namespace spark {

enum class LossType {
    BinaryCrossEntropy,
    CategoricalCrossEntropy
};

struct Gradients {
    std::vector<float> d_weights; // CSR recurrent weights [total_edges]
    std::vector<float> d_U;       // Input projection [num_units x input_dim]
    std::vector<float> d_biases;  // Unit biases [num_units]
    std::vector<float> d_V;       // Output projection [output_dim x num_units]
    std::vector<float> d_c;       // Output biases [output_dim]

    explicit Gradients(const MicroUnitNetwork& net);

    void zero();
    double compute_global_norm() const;
    void clip_by_global_norm(float max_norm);
};

struct StepRecord {
    std::vector<float> x;      // Input observation
    std::vector<float> h_prev; // State at t-1
    std::vector<float> a;      // Pre-activation at t
    std::vector<float> h;      // State at t
    std::vector<float> z;      // Output logits
    std::vector<float> p;      // Output probabilities
    Target target;             // Supervision target
};

class TBPTTTrainer {
public:
    TBPTTTrainer(MicroUnitNetwork& net, float learning_rate = 0.01f, float max_grad_norm = 1.0f);

    // Forward unroll for window of T steps
    // Records activations in step_records
    float forward_unroll(
        const std::vector<Observation>& observations,
        const std::vector<Target>& targets,
        std::vector<StepRecord>& step_records
    );

    // Backward pass: computes gradients accumulated over step_records
    void backward_unroll(
        const std::vector<StepRecord>& step_records,
        Gradients& grads
    );

    // Numerically stable BCE loss and gradient computation for single logit
    static float bce_loss(float z, float y);
    static float bce_grad(float z, float y);

    // Numerically stable Categorical Cross Entropy loss for logits and one-hot target
    static float categorical_ce_loss(const std::vector<float>& logits, const std::vector<float>& target_one_hot);

    LossType loss_type() const { return loss_type_; }
    void set_loss_type(LossType type) { loss_type_ = type; }

    MicroUnitNetwork& network() { return net_; }
    float learning_rate() const { return lr_; }
    void set_learning_rate(float lr) { lr_ = lr; }
    float max_grad_norm() const { return max_grad_norm_; }
    void set_max_grad_norm(float norm) { max_grad_norm_ = norm; }

private:
    MicroUnitNetwork& net_;
    float lr_;
    float max_grad_norm_;
    LossType loss_type_ = LossType::BinaryCrossEntropy;
};

class SGDOptimizer {
public:
    SGDOptimizer(MicroUnitNetwork& net, float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);

    void step(const Gradients& grads);
    void zero_momentum();

    float lr() const { return lr_; }
    void set_lr(float lr) { lr_ = lr; }
    float momentum() const { return momentum_; }
    void set_momentum(float m) { momentum_ = m; }

    void set_train_recurrent(bool enabled) { train_recurrent_ = enabled; }
    void set_train_input(bool enabled) { train_input_ = enabled; }
    void set_train_biases(bool enabled) { train_biases_ = enabled; }
    void set_train_readout(bool enabled) { train_readout_ = enabled; }

private:
    MicroUnitNetwork& net_;
    float lr_;
    float momentum_;
    float weight_decay_;

    bool train_recurrent_ = true;
    bool train_input_ = true;
    bool train_biases_ = true;
    bool train_readout_ = true;

    std::vector<float> v_weights_;
    std::vector<float> v_U_;
    std::vector<float> v_biases_;
    std::vector<float> v_V_;
    std::vector<float> v_c_;
};

// Numerical finite difference gradient verification
struct GradientCheckResult {
    bool passed = false;
    double max_relative_error = 0.0;
    std::string param_name;
    size_t param_index = 0;
    float analytical_val = 0.0f;
    float numerical_val = 0.0f;
    std::string message;
};

GradientCheckResult verify_gradients_finite_difference(
    MicroUnitNetwork& net,
    const std::vector<Observation>& observations,
    const std::vector<Target>& targets,
    float epsilon = 1e-4f,
    double tolerance = 1e-3
);

GradientCheckResult verify_categorical_gradients_finite_difference(
    MicroUnitNetwork& net,
    const std::vector<Observation>& observations,
    const std::vector<Target>& targets,
    float epsilon = 1e-4f,
    double tolerance = 1e-3
);

} // namespace spark
