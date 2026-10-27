#include "spark/local_learning.hpp"
#include "spark/metrics.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace spark {

EligibilityTraces::EligibilityTraces(const MicroUnitNetwork& net) {
    E_weights.assign(net.graph().total_edges(), 0.0f);
    E_U.assign(net.input_weights().size(), 0.0f);
    E_b.assign(net.biases().size(), 0.0f);
}

void EligibilityTraces::reset() {
    std::fill(E_weights.begin(), E_weights.end(), 0.0f);
    std::fill(E_U.begin(), E_U.end(), 0.0f);
    std::fill(E_b.begin(), E_b.end(), 0.0f);
}

void EligibilityTraces::update(
    const MicroUnitNetwork& net,
    const std::vector<float>& x,
    const std::vector<float>& h_prev,
    float lambda
) {
    const uint32_t N = net.config().num_units;
    const uint32_t Din = net.config().input_dim;
    const auto& offsets = net.graph().offsets;
    const auto& sources = net.graph().sources;
    const auto& h_curr = net.state();

    for (uint32_t i = 0; i < N; ++i) {
        float h_i = h_curr[i];
        float sensitivity = 1.0f - h_i * h_i; // dh_i / da_i

        // Bias trace
        E_b[i] = lambda * E_b[i] + sensitivity;

        // Input projection trace
        size_t u_row = static_cast<size_t>(i) * Din;
        for (uint32_t d = 0; d < Din; ++d) {
            E_U[u_row + d] = lambda * E_U[u_row + d] + sensitivity * x[d];
        }

        // Recurrent connection traces
        uint32_t start = offsets[i];
        uint32_t end = offsets[i + 1];
        for (uint32_t e = start; e < end; ++e) {
            uint32_t src = sources[e];
            E_weights[e] = lambda * E_weights[e] + sensitivity * h_prev[src];
        }
    }
}

size_t EligibilityTraces::memory_bytes() const {
    return (E_weights.capacity() + E_U.capacity() + E_b.capacity()) * sizeof(float) + sizeof(EligibilityTraces);
}

FeedbackProjection::FeedbackProjection(
    uint32_t num_units,
    uint32_t output_dim,
    uint64_t seed,
    float scale
) : num_units_(num_units), output_dim_(output_dim) {
    B_.assign(static_cast<size_t>(num_units) * output_dim, 0.0f);
    Random rng(seed);

    float limit = scale * std::sqrt(6.0f / static_cast<float>(num_units + output_dim));
    for (float& b : B_) {
        b = rng.uniform_float(-limit, limit);
    }
}

void FeedbackProjection::compute_feedback(
    const MicroUnitNetwork& net,
    const std::vector<float>& output_errors,
    FeedbackMode mode,
    std::vector<float>& g_out
) const {
    const uint32_t N = net.config().num_units;
    const uint32_t Dout = net.config().output_dim;
    g_out.assign(N, 0.0f);

    if (mode == FeedbackMode::RandomFeedback) {
        for (uint32_t i = 0; i < N; ++i) {
            float sum = 0.0f;
            size_t b_row = static_cast<size_t>(i) * Dout;
            for (uint32_t k = 0; k < Dout; ++k) {
                sum += B_[b_row + k] * output_errors[k];
            }
            g_out[i] = sum;
        }
    } else { // ReadoutTranspose V^T
        const auto& V = net.output_weights();
        for (uint32_t i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < Dout; ++k) {
                sum += V[static_cast<size_t>(k) * N + i] * output_errors[k];
            }
            g_out[i] = sum;
        }
    }
}

size_t FeedbackProjection::memory_bytes() const {
    return B_.capacity() * sizeof(float) + sizeof(FeedbackProjection);
}

LocalLearner::LocalLearner(MicroUnitNetwork& net, const LocalLearningConfig& config)
    : net_(net),
      config_(config),
      traces_(net),
      feedback_(net.config().num_units, net.config().output_dim, config.feedback_seed, config.feedback_scale) {
    g_feedback_.assign(net_.config().num_units, 0.0f);
    output_errors_.assign(net_.config().output_dim, 0.0f);
}

void LocalLearner::reset() {
    net_.reset_state();
    traces_.reset();
}

float LocalLearner::step(const std::vector<float>& x, const Target& target) {
    const uint32_t N = net_.config().num_units;
    const uint32_t Din = net_.config().input_dim;
    const uint32_t Dout = net_.config().output_dim;
    const float beta = net_.config().beta;
    const float eta = config_.lr;

    // 1. Capture state prior to step
    std::vector<float> h_prev = net_.state();

    // 2. Forward recurrent step
    ReadoutMode mode = (config_.loss_type == LossType::CategoricalCrossEntropy)
        ? ReadoutMode::Softmax
        : ReadoutMode::Sigmoid;
    net_.forward_step(x, logits_, probs_, mode);

    // 3. Update local eligibility traces
    traces_.update(net_, x, h_prev, config_.lambda);

    if (!target.valid) {
        return 0.0f; // Warmup tick: no error signal, traces still accumulate activity
    }

    // 4. Compute output error e_k = p_k - y_k and loss
    float loss = 0.0f;
    if (config_.loss_type == LossType::CategoricalCrossEntropy) {
        for (uint32_t k = 0; k < Dout; ++k) {
            output_errors_[k] = probs_[k] - target.values[k];
            if (target.values[k] > 0.5f) {
                loss -= std::log(std::max(probs_[k], 1e-7f));
            }
        }
    } else {
        for (uint32_t k = 0; k < Dout; ++k) {
            output_errors_[k] = probs_[k] - target.values[k];
            loss += TBPTTTrainer::bce_loss(logits_[k], target.values[k]);
        }
        loss /= static_cast<float>(Dout);
    }

    // 5. Compute feedback signal g_i
    feedback_.compute_feedback(net_, output_errors_, config_.feedback_mode, g_feedback_);

    // 6. Update Readout parameters (exact direct gradient)
    if (config_.train_readout) {
        auto& V = net_.output_weights();
        auto& c = net_.output_biases();
        const auto& h_curr = net_.state();

        for (uint32_t k = 0; k < Dout; ++k) {
            float err_k = output_errors_[k];
            c[k] -= eta * err_k;

            size_t v_row = static_cast<size_t>(k) * N;
            for (uint32_t i = 0; i < N; ++i) {
                float delta_V = eta * (err_k * h_curr[i] + config_.weight_decay * V[v_row + i]);
                V[v_row + i] -= delta_V;
            }
        }
    }

    // 7. Update Unit Biases
    if (config_.train_input) {
        auto& biases = net_.biases();
        for (uint32_t i = 0; i < N; ++i) {
            float delta_b = eta * g_feedback_[i] * traces_.E_b[i];
            biases[i] -= delta_b;
        }

        // 8. Update Input Projection U
        auto& U = net_.input_weights();
        for (uint32_t i = 0; i < N; ++i) {
            size_t u_row = static_cast<size_t>(i) * Din;
            for (uint32_t d = 0; d < Din; ++d) {
                float delta_u = eta * (g_feedback_[i] * traces_.E_U[u_row + d] + config_.weight_decay * U[u_row + d]);
                U[u_row + d] -= delta_u;
            }
        }
    }

    // 9. Update Recurrent Synaptic Weights W
    if (config_.train_recurrent) {
        auto& weights = net_.graph().weights;
        const auto& offsets = net_.graph().offsets;

        for (uint32_t i = 0; i < N; ++i) {
            uint32_t start = offsets[i];
            uint32_t end = offsets[i + 1];
            uint32_t Ki = end - start;
            if (Ki == 0) continue;

            float scale = beta / std::sqrt(static_cast<float>(Ki));
            float g_i_scaled = g_feedback_[i] * scale;

            for (uint32_t e = start; e < end; ++e) {
                float delta_w = eta * (g_i_scaled * traces_.E_weights[e] + config_.weight_decay * weights[e]);
                weights[e] -= delta_w;
            }
        }
    }

    return loss;
}

TrainSummary LocalLearner::train(
    Environment& train_env,
    Environment* eval_env,
    uint32_t total_steps,
    uint32_t eval_interval,
    bool quiet
) {
    TrainSummary summary;
    Timer timer;
    reset();
    train_env.reset();

    double running_loss = 0.0;
    size_t valid_count = 0;

    if (!quiet) {
        std::cout << std::string(75, '-') << "\n"
                  << std::setw(8) << "Step"
                  << std::setw(14) << "Train Loss"
                  << std::setw(14) << "Eval Loss"
                  << std::setw(16) << "Eval Acc"
                  << std::setw(12) << "Elapsed"
                  << "\n" << std::string(75, '-') << std::endl;
    }

    Trainer eval_runner(net_, train_env, eval_env);
    eval_runner.set_loss_type(config_.loss_type);

    for (uint32_t s = 1; s <= total_steps; ++s) {
        if (train_env.is_done()) {
            train_env.reset();
            reset(); // Reset state and traces at sequence/episode boundary
        }

        Observation obs = train_env.get_observation();
        Target target = train_env.get_target();
        train_env.step();

        float loss = step(obs.values, target);
        if (target.valid) {
            running_loss += loss;
            valid_count++;
            if (summary.initial_loss == 0.0f) {
                summary.initial_loss = loss;
            }
        }

        if ((eval_interval > 0 && s % eval_interval == 0) || s == total_steps) {
            float avg_train_loss = (valid_count > 0) ? static_cast<float>(running_loss / valid_count) : 0.0f;
            float ev_loss = 0.0f;
            float ev_acc = 0.0f;

            if (eval_env != nullptr) {
                // Save and restore traces around evaluation
                auto saved_traces = traces_;
                EvalMetrics ev = eval_runner.evaluate(*eval_env);
                ev_loss = ev.loss;
                ev_acc = ev.accuracy;
                traces_ = saved_traces;
            }

            TrainHistoryEntry entry;
            entry.step = s;
            entry.train_loss = avg_train_loss;
            entry.eval_loss = ev_loss;
            entry.eval_accuracy = ev_acc;
            entry.elapsed_sec = timer.elapsed_seconds();
            summary.history.push_back(entry);

            if (!quiet) {
                std::cout << std::setw(8) << s
                          << std::setw(14) << std::fixed << std::setprecision(5) << avg_train_loss
                          << std::setw(14) << std::setprecision(5) << ev_loss
                          << std::setw(15) << std::setprecision(2) << (ev_acc * 100.0f) << "%"
                          << std::setw(11) << std::setprecision(1) << entry.elapsed_sec << "s"
                          << std::endl;
            }

            running_loss = 0.0;
            valid_count = 0;
        }
    }

    if (!quiet) {
        std::cout << std::string(75, '-') << std::endl;
    }

    summary.steps_completed = total_steps;
    summary.total_seconds = timer.elapsed_seconds();
    if (!summary.history.empty()) {
        summary.final_train_loss = summary.history.back().train_loss;
        summary.final_eval_loss = summary.history.back().eval_loss;
        summary.final_eval_accuracy = summary.history.back().eval_accuracy;
        if (eval_env != nullptr) {
            EvalMetrics final_ev = eval_runner.evaluate(*eval_env);
            summary.final_eval_correct = final_ev.correct_steps;
            summary.final_eval_total = final_ev.evaluated_steps;
        }
    }

    return summary;
}

size_t LocalLearner::training_memory_bytes() const {
    return traces_.memory_bytes() +
           feedback_.memory_bytes() +
           (g_feedback_.capacity() + output_errors_.capacity() + logits_.capacity() + probs_.capacity()) * sizeof(float) +
           sizeof(LocalLearner);
}

} // namespace spark
