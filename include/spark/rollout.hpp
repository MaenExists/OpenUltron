#pragma once

#include "spark/network.hpp"
#include "spark/gridworld.hpp"
#include "spark/random.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace spark {

enum class RolloutMode {
    TeacherForced,
    OpenLoop
};

inline const char* rollout_mode_to_string(RolloutMode mode) {
    switch (mode) {
        case RolloutMode::TeacherForced: return "teacher-forced";
        case RolloutMode::OpenLoop: return "open-loop";
    }
    return "unknown";
}

enum class RolloutAblation {
    None,
    StateReset
};

inline const char* rollout_ablation_to_string(RolloutAblation abl) {
    switch (abl) {
        case RolloutAblation::None: return "none";
        case RolloutAblation::StateReset: return "state-reset";
    }
    return "unknown";
}

enum class RolloutBaselineType {
    Oracle,
    CopyCurrent,
    NaiveMove,
    RandomPredictor
};

inline const char* rollout_baseline_to_string(RolloutBaselineType b) {
    switch (b) {
        case RolloutBaselineType::Oracle: return "oracle";
        case RolloutBaselineType::CopyCurrent: return "copy-current";
        case RolloutBaselineType::NaiveMove: return "naive-move";
        case RolloutBaselineType::RandomPredictor: return "random-predictor";
    }
    return "unknown";
}

struct RolloutStep {
    uint32_t step_index = 0;
    uint32_t predicted_position = 0;
    uint32_t actual_position = 0;
    float target_probability = 0.0f;
    float cross_entropy = 0.0f;
    bool correct = false;
    uint32_t manhattan_error = 0;
};

struct RolloutResult {
    std::vector<RolloutStep> steps;
    uint32_t initial_position = 0;
    uint32_t final_predicted_position = 0;
    uint32_t final_actual_position = 0;
    bool entire_trajectory_correct = false;
    uint32_t total_manhattan_error = 0;
    double total_cross_entropy = 0.0;
};

struct RolloutMetrics {
    std::string model_name;
    RolloutMode mode = RolloutMode::TeacherForced;
    std::string ablation = "none";
    uint64_t seed = 42ULL;
    uint32_t horizon = 1;
    size_t total_episodes = 0;
    size_t total_steps = 0;
    size_t correct_steps = 0;
    size_t correct_final_positions = 0;
    size_t complete_trajectories_correct = 0;
    uint64_t total_manhattan_error = 0;
    double total_cross_entropy = 0.0;
    double inference_seconds = 0.0;

    std::vector<size_t> step_total;
    std::vector<size_t> step_correct;
    std::vector<uint64_t> step_manhattan;
    std::vector<double> step_cross_entropy;

    float overall_step_accuracy() const {
        return total_steps > 0 ? static_cast<float>(correct_steps) / total_steps : 0.0f;
    }
    float final_position_accuracy() const {
        return total_episodes > 0 ? static_cast<float>(correct_final_positions) / total_episodes : 0.0f;
    }
    float complete_trajectory_accuracy() const {
        return total_episodes > 0 ? static_cast<float>(complete_trajectories_correct) / total_episodes : 0.0f;
    }
    float mean_manhattan_error() const {
        return total_steps > 0 ? static_cast<float>(total_manhattan_error) / total_steps : 0.0f;
    }
    float mean_cross_entropy() const {
        return total_steps > 0 ? static_cast<float>(total_cross_entropy / total_steps) : 0.0f;
    }
    float step_accuracy(size_t s) const {
        return (s < step_total.size() && step_total[s] > 0) ? static_cast<float>(step_correct[s]) / step_total[s] : 0.0f;
    }
    float step_manhattan_error(size_t s) const {
        return (s < step_total.size() && step_total[s] > 0) ? static_cast<float>(step_manhattan[s]) / step_total[s] : 0.0f;
    }
    float step_cross_entropy_val(size_t s) const {
        return (s < step_total.size() && step_total[s] > 0) ? static_cast<float>(step_cross_entropy[s] / step_total[s]) : 0.0f;
    }

    void print_summary() const;
    std::string to_csv() const;
    static std::string csv_header();
};

class RolloutEngine {
public:
    static uint32_t compute_manhattan_distance(uint32_t pos1, uint32_t pos2, uint32_t width = 5);

    static RolloutResult run_single_rollout(
        MicroUnitNetwork& net,
        const GridWorldLayout& layout,
        uint32_t start_pos,
        const std::vector<Action>& actions,
        RolloutMode mode,
        RolloutAblation ablation = RolloutAblation::None
    );

    static RolloutResult run_oracle_rollout(
        const GridWorldLayout& layout,
        uint32_t start_pos,
        const std::vector<Action>& actions
    );

    static RolloutResult run_baseline_rollout(
        const GridWorldLayout& layout,
        uint32_t start_pos,
        const std::vector<Action>& actions,
        RolloutBaselineType baseline,
        RolloutMode mode,
        Random* rng = nullptr
    );

    static RolloutMetrics evaluate_rollouts(
        MicroUnitNetwork& net,
        const std::vector<GridWorldLayout>& layouts,
        uint32_t horizon,
        uint32_t num_episodes,
        RolloutMode mode,
        uint64_t seed = 42ULL,
        RolloutAblation ablation = RolloutAblation::None
    );

    static RolloutMetrics evaluate_oracle(
        const std::vector<GridWorldLayout>& layouts,
        uint32_t horizon,
        uint32_t num_episodes,
        uint64_t seed = 42ULL
    );

    static RolloutMetrics evaluate_baseline(
        const std::vector<GridWorldLayout>& layouts,
        uint32_t horizon,
        uint32_t num_episodes,
        RolloutBaselineType baseline,
        RolloutMode mode,
        uint64_t seed = 42ULL
    );

    static std::vector<Action> generate_action_sequence(
        uint32_t horizon,
        Random& rng
    );
};

} // namespace spark
