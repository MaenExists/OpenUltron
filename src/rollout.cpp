#include "spark/rollout.hpp"
#include "spark/metrics.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace spark {

uint32_t RolloutEngine::compute_manhattan_distance(uint32_t pos1, uint32_t pos2, uint32_t width) {
    if (pos1 >= 25 || pos2 >= 25) return 10; // Defensive out-of-bounds penalty
    int r1 = static_cast<int>(pos1 / width);
    int c1 = static_cast<int>(pos1 % width);
    int r2 = static_cast<int>(pos2 / width);
    int c2 = static_cast<int>(pos2 % width);
    return static_cast<uint32_t>(std::abs(r1 - r2) + std::abs(c1 - c2));
}

std::vector<Action> RolloutEngine::generate_action_sequence(uint32_t horizon, Random& rng) {
    std::vector<Action> actions(horizon);
    for (uint32_t i = 0; i < horizon; ++i) {
        actions[i] = static_cast<Action>(rng.next_u64() % GridWorldEnv::NUM_ACTIONS);
    }
    return actions;
}

RolloutResult RolloutEngine::run_single_rollout(
    MicroUnitNetwork& net,
    const GridWorldLayout& layout,
    uint32_t start_pos,
    const std::vector<Action>& actions,
    RolloutMode mode
) {
    RolloutResult res;
    res.initial_position = start_pos;
    res.entire_trajectory_correct = true;
    res.steps.reserve(actions.size());

    net.reset_state();

    uint32_t current_actual_pos = start_pos;
    uint32_t current_input_pos = start_pos;

    std::vector<float> x(GridWorldEnv::FULL_INPUT_DIM, 0.0f);
    std::vector<float> logits;
    std::vector<float> probs;

    for (uint32_t t = 0; t < actions.size(); ++t) {
        Action a = actions[t];

        // 1. Advance ground-truth simulator to obtain actual target
        auto actual_trans = GridWorldEnv::simulate_transition(
            current_actual_pos,
            a,
            layout.obstacles,
            layout.width,
            layout.height
        );
        uint32_t actual_next_pos = actual_trans.to_pos;

        // 2. Encode model input vector (Full observable layout)
        std::fill(x.begin(), x.end(), 0.0f);

        // Position one-hot (current_input_pos)
        if (current_input_pos < GridWorldEnv::NUM_CELLS) {
            x[current_input_pos] = 1.0f;
        }

        // Static obstacle map
        for (uint32_t i = 0; i < GridWorldEnv::NUM_CELLS; ++i) {
            if (i < layout.obstacles.size() && layout.obstacles[i] != 0) {
                x[GridWorldEnv::NUM_CELLS + i] = 1.0f;
            }
        }

        // Action one-hot
        uint32_t a_idx = static_cast<uint32_t>(a);
        if (a_idx < GridWorldEnv::NUM_ACTIONS) {
            x[2 * GridWorldEnv::NUM_CELLS + a_idx] = 1.0f;
        }

        // 3. Model forward step (preserves recurrent state h_t)
        net.forward_step(x, logits, probs, ReadoutMode::Softmax);

        // 4. Argmax prediction
        uint32_t pred_pos = static_cast<uint32_t>(
            std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
        );

        // 5. Evaluate prediction vs actual simulator state
        bool correct = (pred_pos == actual_next_pos);
        if (!correct) {
            res.entire_trajectory_correct = false;
        }

        float target_p = (actual_next_pos < probs.size()) ? probs[actual_next_pos] : 0.0f;
        float ce = -std::log(std::max(target_p, 1e-12f));
        uint32_t dist = compute_manhattan_distance(pred_pos, actual_next_pos, layout.width);

        res.total_manhattan_error += dist;
        res.total_cross_entropy += ce;

        RolloutStep step;
        step.step_index = t;
        step.predicted_position = pred_pos;
        step.actual_position = actual_next_pos;
        step.target_probability = target_p;
        step.cross_entropy = ce;
        step.correct = correct;
        step.manhattan_error = dist;
        res.steps.push_back(step);

        res.final_predicted_position = pred_pos;
        res.final_actual_position = actual_next_pos;

        // 6. Advance state for next step
        current_actual_pos = actual_next_pos;

        if (mode == RolloutMode::TeacherForced) {
            // Teacher-Forced Mode: provide the ground-truth state as the next input
            current_input_pos = actual_next_pos;
        } else {
            // Autoregressive Open-Loop Mode: feed model's own predicted state
            current_input_pos = pred_pos;
        }
    }

    return res;
}

RolloutResult RolloutEngine::run_oracle_rollout(
    const GridWorldLayout& layout,
    uint32_t start_pos,
    const std::vector<Action>& actions
) {
    RolloutResult res;
    res.initial_position = start_pos;
    res.entire_trajectory_correct = true;
    res.steps.reserve(actions.size());

    uint32_t current_pos = start_pos;

    for (uint32_t t = 0; t < actions.size(); ++t) {
        Action a = actions[t];
        auto trans = GridWorldEnv::simulate_transition(
            current_pos,
            a,
            layout.obstacles,
            layout.width,
            layout.height
        );
        uint32_t next_pos = trans.to_pos;

        RolloutStep step;
        step.step_index = t;
        step.predicted_position = next_pos;
        step.actual_position = next_pos;
        step.target_probability = 1.0f;
        step.cross_entropy = 0.0f;
        step.correct = true;
        step.manhattan_error = 0;

        res.steps.push_back(step);
        current_pos = next_pos;
    }

    res.final_predicted_position = current_pos;
    res.final_actual_position = current_pos;
    return res;
}

RolloutMetrics RolloutEngine::evaluate_rollouts(
    MicroUnitNetwork& net,
    const std::vector<GridWorldLayout>& layouts,
    uint32_t horizon,
    uint32_t num_episodes,
    RolloutMode mode,
    uint64_t seed
) {
    RolloutMetrics metrics;
    metrics.model_name = "SPARK";
    metrics.mode = mode;
    metrics.horizon = horizon;
    metrics.total_episodes = num_episodes;

    metrics.step_total.assign(horizon, 0);
    metrics.step_correct.assign(horizon, 0);
    metrics.step_manhattan.assign(horizon, 0);
    metrics.step_cross_entropy.assign(horizon, 0.0);

    if (layouts.empty()) {
        return metrics;
    }

    Random rng(seed);
    Timer timer;

    for (uint32_t ep = 0; ep < num_episodes; ++ep) {
        const auto& layout = layouts[ep % layouts.size()];

        // Find free cells
        std::vector<uint32_t> free_cells;
        for (uint32_t i = 0; i < layout.obstacles.size(); ++i) {
            if (layout.obstacles[i] == 0) {
                free_cells.push_back(i);
            }
        }
        if (free_cells.empty()) continue;

        uint32_t start_pos = free_cells[rng.next_u64() % free_cells.size()];
        auto actions = generate_action_sequence(horizon, rng);

        RolloutResult res = run_single_rollout(net, layout, start_pos, actions, mode);

        metrics.total_steps += res.steps.size();
        metrics.total_manhattan_error += res.total_manhattan_error;
        metrics.total_cross_entropy += res.total_cross_entropy;

        if (res.entire_trajectory_correct) {
            metrics.complete_trajectories_correct++;
        }
        if (res.final_predicted_position == res.final_actual_position) {
            metrics.correct_final_positions++;
        }

        for (uint32_t t = 0; t < res.steps.size(); ++t) {
            const auto& st = res.steps[t];
            metrics.step_total[t]++;
            metrics.step_manhattan[t] += st.manhattan_error;
            metrics.step_cross_entropy[t] += st.cross_entropy;
            if (st.correct) {
                metrics.correct_steps++;
                metrics.step_correct[t]++;
            }
        }
    }

    metrics.inference_seconds = timer.elapsed_seconds();
    return metrics;
}

RolloutMetrics RolloutEngine::evaluate_oracle(
    const std::vector<GridWorldLayout>& layouts,
    uint32_t horizon,
    uint32_t num_episodes,
    uint64_t seed
) {
    RolloutMetrics metrics;
    metrics.model_name = "GroundTruth_Oracle";
    metrics.mode = RolloutMode::TeacherForced;
    metrics.horizon = horizon;
    metrics.total_episodes = num_episodes;

    metrics.step_total.assign(horizon, 0);
    metrics.step_correct.assign(horizon, 0);
    metrics.step_manhattan.assign(horizon, 0);
    metrics.step_cross_entropy.assign(horizon, 0.0);

    if (layouts.empty()) return metrics;

    Random rng(seed);
    Timer timer;

    for (uint32_t ep = 0; ep < num_episodes; ++ep) {
        const auto& layout = layouts[ep % layouts.size()];
        std::vector<uint32_t> free_cells;
        for (uint32_t i = 0; i < layout.obstacles.size(); ++i) {
            if (layout.obstacles[i] == 0) free_cells.push_back(i);
        }
        if (free_cells.empty()) continue;

        uint32_t start_pos = free_cells[rng.next_u64() % free_cells.size()];
        auto actions = generate_action_sequence(horizon, rng);

        RolloutResult res = run_oracle_rollout(layout, start_pos, actions);

        metrics.total_steps += res.steps.size();
        if (res.entire_trajectory_correct) metrics.complete_trajectories_correct++;
        if (res.final_predicted_position == res.final_actual_position) metrics.correct_final_positions++;

        for (uint32_t t = 0; t < res.steps.size(); ++t) {
            metrics.step_total[t]++;
            metrics.correct_steps++;
            metrics.step_correct[t]++;
        }
    }

    metrics.inference_seconds = timer.elapsed_seconds();
    return metrics;
}

void RolloutMetrics::print_summary() const {
    std::cout << std::string(90, '-') << "\n"
              << "Rollout Summary: " << model_name << " (" << rollout_mode_to_string(mode)
              << ", Horizon: " << horizon << ", Episodes: " << total_episodes << ")\n"
              << std::string(90, '-') << "\n"
              << "  Overall Step Accuracy        : " << std::fixed << std::setprecision(2) << (overall_step_accuracy() * 100.0f) << "%\n"
              << "  Complete Trajectory Accuracy : " << (complete_trajectory_accuracy() * 100.0f) << "%\n"
              << "  Final-Position Accuracy      : " << (final_position_accuracy() * 100.0f) << "%\n"
              << "  Mean Manhattan Error         : " << std::setprecision(3) << mean_manhattan_error() << " cells\n"
              << "  Mean Cross-Entropy Loss      : " << std::setprecision(4) << mean_cross_entropy() << "\n"
              << "  Total Inference Time         : " << std::setprecision(3) << inference_seconds << "s\n"
              << "Step-by-Step Breakdown:\n";
    for (size_t s = 0; s < step_total.size(); ++s) {
        std::cout << "  Step " << std::setw(2) << (s + 1) << ": "
                  << "Acc = " << std::setw(6) << std::setprecision(2) << (step_accuracy(s) * 100.0f) << "% | "
                  << "Manhattan = " << std::setprecision(2) << step_manhattan_error(s) << " | "
                  << "CE = " << std::setprecision(4) << step_cross_entropy_val(s) << "\n";
    }
    std::cout << std::string(90, '-') << std::endl;
}

std::string RolloutMetrics::csv_header() {
    return "model_name,mode,horizon,episodes,total_steps,overall_step_acc,complete_traj_acc,final_pos_acc,mean_manhattan,mean_ce,inference_sec";
}

std::string RolloutMetrics::to_csv() const {
    std::ostringstream ss;
    ss << model_name << ","
       << rollout_mode_to_string(mode) << ","
       << horizon << ","
       << total_episodes << ","
       << total_steps << ","
       << std::fixed << std::setprecision(4) << overall_step_accuracy() << ","
       << complete_trajectory_accuracy() << ","
       << final_position_accuracy() << ","
       << std::setprecision(3) << mean_manhattan_error() << ","
       << std::setprecision(4) << mean_cross_entropy() << ","
       << std::setprecision(3) << inference_seconds;
    return ss.str();
}

} // namespace spark
