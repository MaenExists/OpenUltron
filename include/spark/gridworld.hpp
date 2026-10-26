#pragma once

#include "spark/environment.hpp"
#include "spark/random.hpp"
#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <unordered_set>

namespace spark {

class MicroUnitNetwork;

enum class Action : uint8_t {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
    Stay = 4
};

inline const char* action_name(Action a) {
    switch (a) {
        case Action::Up: return "Up";
        case Action::Down: return "Down";
        case Action::Left: return "Left";
        case Action::Right: return "Right";
        case Action::Stay: return "Stay";
        default: return "Unknown";
    }
}

enum class GridObservationMode {
    FullyObservable,     // 55 features: pos (25) + obstacles (25) + action (5)
    PartiallyObservable  // 56 features: pos (25) + pos_mask (1) + obstacles (25) + action (5)
};

struct GridWorldLayout {
    uint32_t width = 5;
    uint32_t height = 5;
    std::vector<uint8_t> obstacles; // 25 elements: 1 if obstacle, 0 if free
    uint64_t layout_id = 0;

    bool operator==(const GridWorldLayout& other) const {
        return width == other.width && height == other.height && obstacles == other.obstacles;
    }

    std::string to_string() const;
    uint32_t obstacle_count() const;
};

// Transition result diagnostics
struct TransitionResult {
    uint32_t from_pos = 0;
    Action action = Action::Stay;
    uint32_t to_pos = 0;
    bool blocked_by_boundary = false;
    bool blocked_by_obstacle = false;
    bool moved = false;
};

// Layout dataset splits with strict uniqueness guarantees
struct LayoutSplits {
    std::vector<GridWorldLayout> train_layouts;
    std::vector<GridWorldLayout> val_layouts;
    std::vector<GridWorldLayout> test_layouts;

    size_t total_unique() const {
        return train_layouts.size() + val_layouts.size() + test_layouts.size();
    }
};

// Comprehensive evaluation diagnostics for world model transitions
struct WorldModelMetrics {
    std::string model_name = "SPARK";
    size_t total_transitions = 0;
    size_t correct_transitions = 0;
    double total_loss = 0.0;

    size_t successful_moves_total = 0;
    size_t successful_moves_correct = 0;

    size_t blocked_moves_total = 0;
    size_t blocked_moves_correct = 0;

    size_t boundary_blocks_total = 0;
    size_t boundary_blocks_correct = 0;

    size_t obstacle_blocks_total = 0;
    size_t obstacle_blocks_correct = 0;

    double inference_seconds = 0.0;

    float overall_accuracy() const {
        return (total_transitions > 0) ? static_cast<float>(correct_transitions) / total_transitions : 0.0f;
    }
    float mean_loss() const {
        return (total_transitions > 0) ? static_cast<float>(total_loss / total_transitions) : 0.0f;
    }
    float successful_move_accuracy() const {
        return (successful_moves_total > 0) ? static_cast<float>(successful_moves_correct) / successful_moves_total : 0.0f;
    }
    float blocked_move_accuracy() const {
        return (blocked_moves_total > 0) ? static_cast<float>(blocked_moves_correct) / blocked_moves_total : 0.0f;
    }
    float boundary_block_accuracy() const {
        return (boundary_blocks_total > 0) ? static_cast<float>(boundary_blocks_correct) / boundary_blocks_total : 0.0f;
    }
    float obstacle_block_accuracy() const {
        return (obstacle_blocks_total > 0) ? static_cast<float>(obstacle_blocks_correct) / obstacle_blocks_total : 0.0f;
    }

    void print_summary() const;
};

struct Milestone5bRecord {
    std::string experiment_id;
    uint64_t seed = 42;
    std::string task = "gridworld-transition";
    std::string trainer = "exact_tbptt";
    uint32_t units = 128;
    uint32_t fan_in = 8;
    size_t training_examples = 0;
    size_t test_examples = 0;
    float obstacle_density = 0.2f;
    std::string visibility_mode = "fully_observable";
    uint32_t rollout_horizon = 1;
    float train_loss = 0.0f;
    float validation_loss = 0.0f;
    float test_loss = 0.0f;
    float next_state_accuracy = 0.0f;
    float successful_move_acc = 0.0f;
    float blocked_move_acc = 0.0f;
    float boundary_collision_acc = 0.0f;
    float obstacle_collision_acc = 0.0f;
    double training_seconds = 0.0;
    double inference_seconds = 0.0;
    size_t model_bytes = 0;

    std::string to_csv() const;
    static std::string csv_header();
};

// Partial Observability Step Diagnostics
struct PartialObsStepMetrics {
    uint32_t step_index = 0; // 0-indexed step in episode (0 = initial observed step)
    size_t total_evals = 0;
    size_t correct_evals = 0;
    double total_loss = 0.0;

    float accuracy() const {
        return (total_evals > 0) ? static_cast<float>(correct_evals) / total_evals : 0.0f;
    }
    float mean_loss() const {
        return (total_evals > 0) ? static_cast<float>(total_loss / total_evals) : 0.0f;
    }
};

// Partial Observability Full Metrics
struct PartialObsMetrics {
    std::string model_name = "SPARK_PartiallyObservable";
    uint32_t horizon = 16;
    size_t total_episodes = 0;
    size_t total_transitions = 0;
    size_t correct_transitions = 0;
    double total_loss = 0.0;
    double inference_seconds = 0.0;

    std::vector<PartialObsStepMetrics> step_metrics;

    float overall_accuracy() const {
        return (total_transitions > 0) ? static_cast<float>(correct_transitions) / total_transitions : 0.0f;
    }
    float mean_loss() const {
        return (total_transitions > 0) ? static_cast<float>(total_loss / total_transitions) : 0.0f;
    }
    float accuracy_at_step(uint32_t step) const {
        if (step < step_metrics.size() && step_metrics[step].total_evals > 0) {
            return step_metrics[step].accuracy();
        }
        return 0.0f;
    }

    void print_summary() const;
};

enum class PartialObsAblation {
    None,
    ResetStateEachStep,
    ZeroRecurrentWeights
};

struct Milestone5cRecord {
    std::string experiment_id;
    uint64_t seed = 42;
    std::string model_type; // "trained_spark", "reset_hidden", "zero_weights", "frozen_recurrent", "readout_only", "oracle", "random"
    uint32_t units = 128;
    uint32_t fan_in = 8;
    uint32_t horizon = 16;
    size_t evaluated_episodes = 0;
    float overall_accuracy = 0.0f;
    float step1_acc = 0.0f; // step 0 (observed) -> predicting s_1
    float step2_acc = 0.0f; // step 1 (hidden) -> predicting s_2
    float step4_acc = 0.0f; // step 3 (hidden) -> predicting s_4
    float step8_acc = 0.0f; // step 7 (hidden) -> predicting s_8
    float step16_acc = 0.0f; // step 15 (hidden) -> predicting s_16
    float mean_loss = 0.0f;
    double elapsed_seconds = 0.0;

    std::string to_csv() const;
    static std::string csv_header();
};

class GridWorldEnv : public Environment {
public:
    static constexpr uint32_t DEFAULT_WIDTH = 5;
    static constexpr uint32_t DEFAULT_HEIGHT = 5;
    static constexpr uint32_t NUM_CELLS = DEFAULT_WIDTH * DEFAULT_HEIGHT; // 25
    static constexpr uint32_t NUM_ACTIONS = 5;
    static constexpr uint32_t FULL_INPUT_DIM = NUM_CELLS + NUM_CELLS + NUM_ACTIONS; // 55
    static constexpr uint32_t PARTIAL_INPUT_DIM = NUM_CELLS + 1 + NUM_CELLS + NUM_ACTIONS; // 56
    static constexpr uint32_t OUTPUT_DIM = NUM_CELLS; // 25

    GridWorldEnv(
        uint32_t episode_length = 50,
        uint64_t seed = 42ULL,
        GridObservationMode obs_mode = GridObservationMode::FullyObservable
    );

    GridWorldEnv(
        const GridWorldLayout& layout,
        uint32_t episode_length = 50,
        uint64_t seed = 42ULL,
        GridObservationMode obs_mode = GridObservationMode::FullyObservable
    );

    GridWorldEnv(
        const std::vector<GridWorldLayout>& layouts,
        uint32_t episode_length = 50,
        uint64_t seed = 42ULL,
        GridObservationMode obs_mode = GridObservationMode::FullyObservable
    );

    // Environment interface
    void reset() override;
    Observation get_observation() const override;
    Target get_target() const override;
    void step() override;
    bool is_done() const override;

    uint32_t input_dim() const override {
        return (obs_mode_ == GridObservationMode::FullyObservable) ? FULL_INPUT_DIM : PARTIAL_INPUT_DIM;
    }
    uint32_t output_dim() const override { return OUTPUT_DIM; }
    uint32_t current_step() const override { return step_idx_; }

    // Grid-World specific operations
    void load_layout(const GridWorldLayout& layout);
    void set_layouts(const std::vector<GridWorldLayout>& layouts);
    const std::vector<GridWorldLayout>& layouts() const { return layouts_; }

    void set_agent_position(uint32_t pos);
    uint32_t agent_position() const { return agent_pos_; }
    uint32_t next_position() const { return next_pos_; }
    Action current_action() const { return current_action_; }
    void set_action(Action a);
    const GridWorldLayout& layout() const { return layout_; }
    GridObservationMode observation_mode() const { return obs_mode_; }
    void set_observation_mode(GridObservationMode mode) { obs_mode_ = mode; }
    void set_position_observed(bool observed) { pos_observed_ = observed; }
    bool is_position_observed() const { return pos_observed_; }
    void set_episode_length(uint32_t len) { episode_length_ = len; }
    uint32_t episode_length() const { return episode_length_; }

    const TransitionResult& last_transition() const { return last_transition_; }

    // Pure ground-truth simulator rule
    static TransitionResult simulate_transition(
        uint32_t pos,
        Action action,
        const std::vector<uint8_t>& obstacles,
        uint32_t width = DEFAULT_WIDTH,
        uint32_t height = DEFAULT_HEIGHT
    );

    // Static layout generators with uniqueness guarantees across splits
    static GridWorldLayout generate_random_layout(
        uint64_t seed,
        uint32_t min_obstacles = 3,
        uint32_t max_obstacles = 7,
        uint32_t width = DEFAULT_WIDTH,
        uint32_t height = DEFAULT_HEIGHT
    );

    static LayoutSplits generate_split_layouts(
        size_t num_train,
        size_t num_val,
        size_t num_test,
        uint32_t min_obstacles = 3,
        uint32_t max_obstacles = 7,
        uint64_t master_seed = 1000ULL
    );

private:
    void generate_default_layout();
    void pick_random_action();
    void place_agent_randomly();
    void compute_next_transition();

    uint32_t episode_length_;
    uint64_t seed_;
    Random rng_;
    GridObservationMode obs_mode_;

    std::vector<GridWorldLayout> layouts_;
    size_t current_layout_idx_ = 0;

    GridWorldLayout layout_;
    uint32_t agent_pos_ = 0;
    Action current_action_ = Action::Stay;
    uint32_t next_pos_ = 0;
    TransitionResult last_transition_;

    uint32_t step_idx_ = 0;
    bool pos_observed_ = true;
};

// Evaluation functions
WorldModelMetrics evaluate_world_model(
    MicroUnitNetwork& net,
    GridWorldEnv& env,
    uint32_t total_transitions
);

struct BaselineSuiteResults {
    WorldModelMetrics random_predictor;
    WorldModelMetrics copy_current_predictor;
    WorldModelMetrics naive_movement_predictor;
    WorldModelMetrics ground_truth_oracle;
};

BaselineSuiteResults evaluate_all_baselines(
    GridWorldEnv& env,
    uint32_t total_transitions,
    uint64_t seed = 42ULL
);

// Milestone 5C: Partial Observability Evaluators
PartialObsMetrics evaluate_partial_observability(
    MicroUnitNetwork& net,
    GridWorldEnv& env,
    uint32_t num_episodes,
    uint32_t horizon = 16,
    PartialObsAblation ablation = PartialObsAblation::None
);

PartialObsMetrics evaluate_oracle_partial_obs(
    GridWorldEnv& env,
    uint32_t num_episodes,
    uint32_t horizon = 16
);

PartialObsMetrics evaluate_random_partial_obs(
    GridWorldEnv& env,
    uint32_t num_episodes,
    uint32_t horizon = 16,
    uint64_t seed = 42ULL
);

// Milestone 5D.0: Comprehensive One-Step Diagnostic Breakdown
struct OneStepDiagnosticMetrics {
    std::string model_name;
    size_t total_transitions = 0;
    size_t correct_transitions = 0;
    double total_loss = 0.0;

    // Movement outcome breakdown (partitions 100% of transitions)
    size_t succ_moves_total = 0, succ_moves_correct = 0;
    size_t stay_total = 0, stay_correct = 0;
    size_t bound_blocks_total = 0, bound_blocks_correct = 0;
    size_t obs_blocks_total = 0, obs_blocks_correct = 0;

    // Action breakdown: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
    size_t action_total[5] = {0, 0, 0, 0, 0};
    size_t action_correct[5] = {0, 0, 0, 0, 0};

    // Obstacle density breakdown: 0: Low (<=4), 1: Med (5-6), 2: High (>=7)
    size_t density_total[3] = {0, 0, 0};
    size_t density_correct[3] = {0, 0, 0};

    float overall_acc() const { return total_transitions > 0 ? static_cast<float>(correct_transitions) / total_transitions : 0.0f; }
    float mean_loss() const { return total_transitions > 0 ? static_cast<float>(total_loss / total_transitions) : 0.0f; }
    float succ_acc() const { return succ_moves_total > 0 ? static_cast<float>(succ_moves_correct) / succ_moves_total : 0.0f; }
    float stay_acc() const { return stay_total > 0 ? static_cast<float>(stay_correct) / stay_total : 0.0f; }
    float bound_acc() const { return bound_blocks_total > 0 ? static_cast<float>(bound_blocks_correct) / bound_blocks_total : 0.0f; }
    float obs_acc() const { return obs_blocks_total > 0 ? static_cast<float>(obs_blocks_correct) / obs_blocks_total : 0.0f; }
    float action_acc(int a) const { return (a >= 0 && a < 5 && action_total[a] > 0) ? static_cast<float>(action_correct[a]) / action_total[a] : 0.0f; }
    float density_acc(int d) const { return (d >= 0 && d < 3 && density_total[d] > 0) ? static_cast<float>(density_correct[d]) / density_total[d] : 0.0f; }

    void print_report() const;
};

OneStepDiagnosticMetrics evaluate_onestep_diagnostic(
    MicroUnitNetwork& net,
    GridWorldEnv& env,
    uint32_t total_transitions
);

struct BaselineOneStepDiagnostics {
    OneStepDiagnosticMetrics random_predictor;
    OneStepDiagnosticMetrics copy_current_predictor;
    OneStepDiagnosticMetrics naive_movement_predictor;
    OneStepDiagnosticMetrics ground_truth_oracle;
};

BaselineOneStepDiagnostics evaluate_baseline_diagnostics(
    GridWorldEnv& env,
    uint32_t total_transitions,
    uint64_t seed = 42ULL
);

} // namespace spark
