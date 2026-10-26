#include "spark/trainer.hpp"
#include "spark/metrics.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace spark {

Trainer::Trainer(
    MicroUnitNetwork& net,
    Environment& train_env,
    Environment* eval_env,
    float learning_rate,
    float max_grad_norm
) : net_(net),
    train_env_(train_env),
    eval_env_(eval_env),
    learning_rate_(learning_rate),
    max_grad_norm_(max_grad_norm) {}

EvalMetrics Trainer::evaluate(Environment& eval_env, uint32_t max_eval_steps) {
    EvalMetrics metrics;
    eval_env.reset();

    // Preserve network hidden state
    std::vector<float> saved_state = net_.state();
    net_.reset_state();

    std::vector<float> logits, probs;
    double total_loss = 0.0;
    size_t valid_count = 0;
    size_t correct_steps = 0;

    ReadoutMode r_mode = (loss_type_ == LossType::CategoricalCrossEntropy) ? ReadoutMode::Softmax : ReadoutMode::Sigmoid;

    for (uint32_t s = 0; s < max_eval_steps && !eval_env.is_done(); ++s) {
        Observation obs = eval_env.get_observation();
        Target target = eval_env.get_target();

        net_.forward_step(obs.values, logits, probs, r_mode);
        eval_env.step();

        if (target.valid) {
            valid_count++;
            if (loss_type_ == LossType::CategoricalCrossEntropy) {
                total_loss += TBPTTTrainer::categorical_ce_loss(logits, target.values);
                uint32_t pred_class = static_cast<uint32_t>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
                uint32_t true_class = static_cast<uint32_t>(std::distance(target.values.begin(), std::max_element(target.values.begin(), target.values.end())));
                if (pred_class == true_class) {
                    correct_steps++;
                }
            } else {
                for (size_t k = 0; k < logits.size(); ++k) {
                    total_loss += TBPTTTrainer::bce_loss(logits[k], target.values[k]);
                    float pred_bit = (probs[k] >= 0.5f) ? 1.0f : 0.0f;
                    if (std::abs(pred_bit - target.values[k]) < 1e-3f) {
                        correct_steps++;
                    }
                }
            }
        }
    }

    // Restore network state
    net_.set_state(saved_state);

    metrics.evaluated_steps = valid_count;
    metrics.correct_steps = correct_steps;
    if (valid_count > 0) {
        metrics.loss = static_cast<float>(total_loss / valid_count);
        metrics.accuracy = static_cast<float>(correct_steps) / static_cast<float>(valid_count);
    }

    return metrics;
}

TrainSummary Trainer::train(
    uint32_t total_steps,
    uint32_t unroll_steps,
    uint32_t eval_interval,
    bool quiet
) {
    TrainSummary summary;
    TBPTTTrainer tbptt(net_, learning_rate_, max_grad_norm_);
    tbptt.set_loss_type(loss_type_);
    SGDOptimizer optimizer(net_, learning_rate_, 0.9f, 0.0f); // Momentum 0.9 for faster convergence
    optimizer.set_train_recurrent(train_recurrent_);
    optimizer.set_train_input(train_input_);
    optimizer.set_train_biases(train_biases_);
    optimizer.set_train_readout(train_readout_);
    Gradients grads(net_);

    std::vector<Observation> window_obs;
    std::vector<Target> window_targets;
    std::vector<StepRecord> step_records;
    window_obs.reserve(unroll_steps);
    window_targets.reserve(unroll_steps);

    Timer timer;
    double running_loss = 0.0;
    size_t window_count = 0;

    train_env_.reset();
    net_.reset_state();

    if (!quiet) {
        std::cout << std::string(75, '-') << "\n"
                  << std::setw(8) << "Step"
                  << std::setw(14) << "Train Loss"
                  << std::setw(12) << "Grad Norm"
                  << std::setw(14) << "Eval Loss"
                  << std::setw(14) << "Eval Acc"
                  << std::setw(12) << "Elapsed"
                  << "\n" << std::string(75, '-') << std::endl;
    }

    for (uint32_t s = 1; s <= total_steps; ++s) {
        window_obs.clear();
        window_targets.clear();

        for (uint32_t t = 0; t < unroll_steps; ++t) {
            if (train_env_.is_done()) {
                train_env_.reset();
                net_.reset_state(); // Reset recurrent hidden state at episode boundary
            }
            window_obs.push_back(train_env_.get_observation());
            window_targets.push_back(train_env_.get_target());
            train_env_.step();
        }

        // Forward unroll
        float loss = tbptt.forward_unroll(window_obs, window_targets, step_records);
        running_loss += loss;
        window_count++;

        if (s == 1) {
            summary.initial_loss = loss;
        }

        // Backward unroll
        grads.zero();
        tbptt.backward_unroll(step_records, grads);

        double grad_norm = grads.compute_global_norm();
        grads.clip_by_global_norm(max_grad_norm_);

        // Optimizer step
        optimizer.step(grads);

        // Evaluation
        if ((eval_interval > 0 && s % eval_interval == 0) || s == total_steps) {
            float avg_train_loss = static_cast<float>(running_loss / window_count);
            float ev_loss = 0.0f;
            float ev_acc = 0.0f;

            if (eval_env_ != nullptr) {
                EvalMetrics ev = evaluate(*eval_env_);
                ev_loss = ev.loss;
                ev_acc = ev.accuracy;
            }

            TrainHistoryEntry entry;
            entry.step = s;
            entry.train_loss = avg_train_loss;
            entry.grad_norm = static_cast<float>(grad_norm);
            entry.eval_loss = ev_loss;
            entry.eval_accuracy = ev_acc;
            entry.elapsed_sec = timer.elapsed_seconds();
            summary.history.push_back(entry);

            if (!quiet) {
                std::cout << std::setw(8) << s
                          << std::setw(14) << std::fixed << std::setprecision(5) << avg_train_loss
                          << std::setw(12) << std::setprecision(4) << grad_norm
                          << std::setw(14) << std::setprecision(5) << ev_loss
                          << std::setw(13) << std::setprecision(2) << (ev_acc * 100.0f) << "%"
                          << std::setw(11) << std::setprecision(1) << entry.elapsed_sec << "s"
                          << std::endl;
            }

            running_loss = 0.0;
            window_count = 0;
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
        if (eval_env_ != nullptr) {
            EvalMetrics final_ev = evaluate(*eval_env_);
            summary.final_eval_correct = final_ev.correct_steps;
            summary.final_eval_total = final_ev.evaluated_steps;
        }
    }

    return summary;
}

} // namespace spark
