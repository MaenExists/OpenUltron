#include "spark/learning.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

namespace spark {

Gradients::Gradients(const MicroUnitNetwork& net) {
    d_weights.assign(net.graph().total_edges(), 0.0f);
    d_U.assign(net.input_weights().size(), 0.0f);
    d_biases.assign(net.biases().size(), 0.0f);
    d_V.assign(net.output_weights().size(), 0.0f);
    d_c.assign(net.output_biases().size(), 0.0f);
}

void Gradients::zero() {
    std::fill(d_weights.begin(), d_weights.end(), 0.0f);
    std::fill(d_U.begin(), d_U.end(), 0.0f);
    std::fill(d_biases.begin(), d_biases.end(), 0.0f);
    std::fill(d_V.begin(), d_V.end(), 0.0f);
    std::fill(d_c.begin(), d_c.end(), 0.0f);
}

double Gradients::compute_global_norm() const {
    double sum_sq = 0.0;
    for (float g : d_weights) sum_sq += static_cast<double>(g) * g;
    for (float g : d_U) sum_sq += static_cast<double>(g) * g;
    for (float g : d_biases) sum_sq += static_cast<double>(g) * g;
    for (float g : d_V) sum_sq += static_cast<double>(g) * g;
    for (float g : d_c) sum_sq += static_cast<double>(g) * g;
    return std::sqrt(sum_sq);
}

void Gradients::clip_by_global_norm(float max_norm) {
    if (max_norm <= 0.0f) return;
    double norm = compute_global_norm();
    if (norm > static_cast<double>(max_norm) && norm > 1e-12) {
        float scale = static_cast<float>(max_norm / norm);
        for (float& g : d_weights) g *= scale;
        for (float& g : d_U) g *= scale;
        for (float& g : d_biases) g *= scale;
        for (float& g : d_V) g *= scale;
        for (float& g : d_c) g *= scale;
    }
}

TBPTTTrainer::TBPTTTrainer(MicroUnitNetwork& net, float learning_rate, float max_grad_norm)
    : net_(net), lr_(learning_rate), max_grad_norm_(max_grad_norm) {}

float TBPTTTrainer::bce_loss(float z, float y) {
    // Numerically stable BCE: max(z, 0) - z*y + log(1 + exp(-|z|))
    float max_val = std::max(z, 0.0f);
    return max_val - z * y + std::log(1.0f + std::exp(-std::abs(z)));
}

float TBPTTTrainer::bce_grad(float z, float y) {
    // dL/dz = sigma(z) - y
    return MicroUnitNetwork::stable_sigmoid(z) - y;
}

float TBPTTTrainer::forward_unroll(
    const std::vector<Observation>& observations,
    const std::vector<Target>& targets,
    std::vector<StepRecord>& step_records
) {
    step_records.clear();
    step_records.reserve(observations.size());

    double total_loss = 0.0;
    size_t valid_elements = 0;

    std::vector<float> logits;
    std::vector<float> probs;

    for (size_t t = 0; t < observations.size(); ++t) {
        StepRecord rec;
        rec.x = observations[t].values;
        rec.h_prev = net_.state();
        rec.target = targets[t];

        net_.forward_step(rec.x, logits, probs, ReadoutMode::Sigmoid);

        rec.a = net_.pre_activations();
        rec.h = net_.state();
        rec.z = logits;
        rec.p = probs;

        if (rec.target.valid) {
            for (size_t k = 0; k < logits.size(); ++k) {
                total_loss += bce_loss(logits[k], rec.target.values[k]);
                valid_elements++;
            }
        }

        step_records.push_back(std::move(rec));
    }

    if (valid_elements == 0) return 0.0f;
    return static_cast<float>(total_loss / valid_elements);
}

void TBPTTTrainer::backward_unroll(
    const std::vector<StepRecord>& step_records,
    Gradients& grads
) {
    if (step_records.empty()) return;

    const uint32_t N = net_.config().num_units;
    const uint32_t Din = net_.config().input_dim;
    const uint32_t Dout = net_.config().output_dim;
    const float alpha = net_.config().alpha;
    const float beta = net_.config().beta;

    const auto& offsets = net_.graph().offsets;
    const auto& sources = net_.graph().sources;
    const auto& weights = net_.graph().weights;
    const auto& V = net_.output_weights();

    // Count valid supervised target elements across the unroll window
    size_t valid_elements = 0;
    for (const auto& rec : step_records) {
        if (rec.target.valid) {
            valid_elements += Dout;
        }
    }
    float norm_factor = (valid_elements > 0) ? (1.0f / static_cast<float>(valid_elements)) : 1.0f;

    std::vector<float> d_h_next(N, 0.0f);
    std::vector<float> d_h_prev(N, 0.0f);
    std::vector<float> d_z(Dout, 0.0f);

    // Backward pass in reverse chronological order
    for (int t = static_cast<int>(step_records.size()) - 1; t >= 0; --t) {
        const auto& rec = step_records[t];

        // 1. Output readout gradients dL/dz_k(t)
        if (rec.target.valid) {
            for (uint32_t k = 0; k < Dout; ++k) {
                d_z[k] = (rec.p[k] - rec.target.values[k]) * norm_factor;
                grads.d_c[k] += d_z[k];

                size_t v_row = static_cast<size_t>(k) * N;
                for (uint32_t i = 0; i < N; ++i) {
                    grads.d_V[v_row + i] += d_z[k] * rec.h[i];
                }
            }
        } else {
            std::fill(d_z.begin(), d_z.end(), 0.0f);
        }

        // 2. Error propagated into hidden state h_i(t)
        // d_h[i] = direct_readout_gradient + future_recurrent_gradient
        std::fill(d_h_prev.begin(), d_h_prev.end(), 0.0f);

        for (uint32_t i = 0; i < N; ++i) {
            float d_h_i = d_h_next[i];
            if (rec.target.valid) {
                for (uint32_t k = 0; k < Dout; ++k) {
                    d_h_i += d_z[k] * V[static_cast<size_t>(k) * N + i];
                }
            }

            // Backprop through activation function: h = tanh(a) => dh/da = 1 - h^2
            float d_a_i = d_h_i * (1.0f - rec.h[i] * rec.h[i]);

            // Unit bias gradient
            grads.d_biases[i] += d_a_i;

            // Input projection gradient
            size_t u_row = static_cast<size_t>(i) * Din;
            for (uint32_t d = 0; d < Din; ++d) {
                grads.d_U[u_row + d] += d_a_i * rec.x[d];
            }

            // Recurrent self-retention contribution to h_prev[i]
            d_h_prev[i] += alpha * d_a_i;

            // Recurrent incoming connections contribution
            uint32_t start = offsets[i];
            uint32_t end = offsets[i + 1];
            uint32_t Ki = end - start;
            if (Ki > 0) {
                float scale = (beta / std::sqrt(static_cast<float>(Ki)));
                float scaled_d_a = scale * d_a_i;

                for (uint32_t e = start; e < end; ++e) {
                    uint32_t src = sources[e];
                    grads.d_weights[e] += scaled_d_a * rec.h_prev[src];
                    d_h_prev[src] += scaled_d_a * weights[e];
                }
            }
        }

        d_h_next = d_h_prev;
    }
}

SGDOptimizer::SGDOptimizer(MicroUnitNetwork& net, float lr, float momentum, float weight_decay)
    : net_(net), lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {
    v_weights_.assign(net_.graph().total_edges(), 0.0f);
    v_U_.assign(net_.input_weights().size(), 0.0f);
    v_biases_.assign(net_.biases().size(), 0.0f);
    v_V_.assign(net_.output_weights().size(), 0.0f);
    v_c_.assign(net_.output_biases().size(), 0.0f);
}

void SGDOptimizer::zero_momentum() {
    std::fill(v_weights_.begin(), v_weights_.end(), 0.0f);
    std::fill(v_U_.begin(), v_U_.end(), 0.0f);
    std::fill(v_biases_.begin(), v_biases_.end(), 0.0f);
    std::fill(v_V_.begin(), v_V_.end(), 0.0f);
    std::fill(v_c_.begin(), v_c_.end(), 0.0f);
}

void SGDOptimizer::step(const Gradients& grads) {
    auto update_vector = [this](std::vector<float>& params, const std::vector<float>& grads_vec, std::vector<float>& vel) {
        for (size_t i = 0; i < params.size(); ++i) {
            float g = grads_vec[i];
            if (weight_decay_ > 0.0f) {
                g += weight_decay_ * params[i];
            }
            if (momentum_ > 0.0f) {
                vel[i] = momentum_ * vel[i] + lr_ * g;
                params[i] -= vel[i];
            } else {
                params[i] -= lr_ * g;
            }
        }
    };

    update_vector(net_.graph().weights, grads.d_weights, v_weights_);
    update_vector(net_.input_weights(), grads.d_U, v_U_);
    update_vector(net_.biases(), grads.d_biases, v_biases_);
    update_vector(net_.output_weights(), grads.d_V, v_V_);
    update_vector(net_.output_biases(), grads.d_c, v_c_);
}

GradientCheckResult verify_gradients_finite_difference(
    MicroUnitNetwork& net,
    const std::vector<Observation>& observations,
    const std::vector<Target>& targets,
    float epsilon,
    double tolerance
) {
    GradientCheckResult result;
    result.passed = true;
    result.max_relative_error = 0.0;

    // 1. Analytical backward pass
    TBPTTTrainer trainer(net);
    std::vector<StepRecord> records;

    net.reset_state();
    std::vector<float> initial_state = net.state();

    float base_loss = trainer.forward_unroll(observations, targets, records);
    (void)base_loss;

    Gradients analytical_grads(net);
    trainer.backward_unroll(records, analytical_grads);

    // Double-precision forward simulation for high-precision finite difference checking
    auto compute_loss_double = [&](
        const std::string& target_group,
        size_t target_idx,
        double delta
    ) -> double {
        const uint32_t N = net.config().num_units;
        const uint32_t Din = net.config().input_dim;
        const uint32_t Dout = net.config().output_dim;
        const double alpha = static_cast<double>(net.config().alpha);
        const double beta = static_cast<double>(net.config().beta);

        const auto& offsets = net.graph().offsets;
        const auto& sources = net.graph().sources;
        const auto& weights = net.graph().weights;
        const auto& U = net.input_weights();
        const auto& biases = net.biases();
        const auto& V = net.output_weights();
        const auto& c = net.output_biases();

        std::vector<double> h_prev(N, 0.0);
        std::vector<double> h_curr(N, 0.0);

        double total_loss = 0.0;
        size_t valid_elements = 0;

        for (size_t t = 0; t < observations.size(); ++t) {
            const auto& obs_vec = observations[t].values;

            for (uint32_t i = 0; i < N; ++i) {
                double rec_sum = 0.0;
                uint32_t start = offsets[i];
                uint32_t end = offsets[i + 1];
                uint32_t Ki = end - start;

                for (uint32_t e = start; e < end; ++e) {
                    uint32_t src = sources[e];
                    double w = static_cast<double>(weights[e]);
                    if (target_group == "recurrent_weights" && e == target_idx) {
                        w += delta;
                    }
                    rec_sum += w * h_prev[src];
                }

                double rec_term = 0.0;
                if (Ki > 0) {
                    rec_term = (beta / std::sqrt(static_cast<double>(Ki))) * rec_sum;
                }

                double in_term = 0.0;
                size_t u_row = static_cast<size_t>(i) * Din;
                for (uint32_t d = 0; d < Din; ++d) {
                    double u_val = static_cast<double>(U[u_row + d]);
                    if (target_group == "input_weights" && (u_row + d) == target_idx) {
                        u_val += delta;
                    }
                    in_term += u_val * static_cast<double>(obs_vec[d]);
                }

                double b_val = static_cast<double>(biases[i]);
                if (target_group == "unit_biases" && i == target_idx) {
                    b_val += delta;
                }

                double a_i = alpha * h_prev[i] + rec_term + in_term + b_val;
                h_curr[i] = std::tanh(a_i);
            }

            h_prev = h_curr;

            if (targets[t].valid) {
                for (uint32_t k = 0; k < Dout; ++k) {
                    double c_val = static_cast<double>(c[k]);
                    if (target_group == "output_biases" && k == target_idx) {
                        c_val += delta;
                    }

                    double z_k = c_val;
                    size_t v_row = static_cast<size_t>(k) * N;
                    for (uint32_t i = 0; i < N; ++i) {
                        double v_val = static_cast<double>(V[v_row + i]);
                        if (target_group == "output_weights" && (v_row + i) == target_idx) {
                            v_val += delta;
                        }
                        z_k += v_val * h_curr[i];
                    }

                    double y_val = static_cast<double>(targets[t].values[k]);
                    double max_val = std::max(z_k, 0.0);
                    double bce = max_val - z_k * y_val + std::log(1.0 + std::exp(-std::abs(z_k)));
                    total_loss += bce;
                    valid_elements++;
                }
            }
        }

        return (valid_elements > 0) ? (total_loss / valid_elements) : 0.0;
    };

    auto check_params = [&](
        const std::string& name,
        size_t param_count,
        const std::vector<float>& ana_grads
    ) -> bool {
        double eps = (epsilon > 0.0f) ? static_cast<double>(epsilon) : 1e-6; // High precision step
        for (size_t idx = 0; idx < param_count; ++idx) {
            double loss_plus = compute_loss_double(name, idx, +eps);
            double loss_minus = compute_loss_double(name, idx, -eps);

            double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
            double ana_grad = static_cast<double>(ana_grads[idx]);

            double denom = std::max({std::abs(ana_grad), std::abs(num_grad), 1e-4});
            double rel_err = std::abs(ana_grad - num_grad) / denom;

            if (rel_err > result.max_relative_error) {
                result.max_relative_error = rel_err;
                result.param_name = name;
                result.param_index = idx;
                result.analytical_val = static_cast<float>(ana_grad);
                result.numerical_val = static_cast<float>(num_grad);
            }

            if (rel_err > tolerance) {
                result.passed = false;
                result.message = "Gradient mismatch in " + name + " at index " + std::to_string(idx) +
                                 ": analytical=" + std::to_string(ana_grad) +
                                 ", numerical=" + std::to_string(num_grad) +
                                 ", rel_error=" + std::to_string(rel_err);
                return false;
            }
        }
        return true;
    };

    if (!check_params("output_biases", net.output_biases().size(), analytical_grads.d_c)) return result;
    if (!check_params("output_weights", net.output_weights().size(), analytical_grads.d_V)) return result;
    if (!check_params("unit_biases", net.biases().size(), analytical_grads.d_biases)) return result;
    if (!check_params("input_weights", net.input_weights().size(), analytical_grads.d_U)) return result;
    if (!check_params("recurrent_weights", net.graph().weights.size(), analytical_grads.d_weights)) return result;

    result.message = "All parameters verified successfully within tolerance";
    return result;
}

} // namespace spark
