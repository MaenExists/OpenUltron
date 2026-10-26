#pragma once

#include "spark/environment.hpp"
#include "spark/random.hpp"
#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <unordered_set>

namespace spark {

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

    GridWorldLayout layout_;
    uint32_t agent_pos_ = 0;
    Action current_action_ = Action::Stay;
    uint32_t next_pos_ = 0;
    TransitionResult last_transition_;

    uint32_t step_idx_ = 0;
    bool pos_observed_ = true;
};

} // namespace spark
