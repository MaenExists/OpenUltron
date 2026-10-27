#pragma once

#include "spark/network.hpp"
#include "spark/gridworld.hpp"
#include "spark/random.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace spark {

// ============================================================================
// 1. Ground-Truth BFS Reference Oracle (Validation & Reachability Only)
// ============================================================================
class BFSSolver {
public:
    struct BFSResult {
        bool reachable = false;
        uint32_t shortest_path_length = 0;
        std::vector<Action> shortest_path;
    };

    // Computes ground-truth shortest path in grid layout respecting static obstacles.
    // NOTE: Used ONLY to verify reachability and compute reference shortest-path lengths.
    // The actual learned-model planning system must never call this during search.
    static BFSResult find_shortest_path(
        const GridWorldLayout& layout,
        uint32_t start_pos,
        uint32_t goal_pos
    );

    static bool is_reachable(
        const GridWorldLayout& layout,
        uint32_t start_pos,
        uint32_t goal_pos
    ) {
        return find_shortest_path(layout, start_pos, goal_pos).reachable;
    }
};

// ============================================================================
// 2. Planning Episode Representation & Generator
// ============================================================================
struct PlanningEpisode {
    uint32_t start_pos = 0;
    uint32_t goal_pos = 0;
    GridWorldLayout layout;
    uint32_t max_steps = 16;
    uint64_t layout_id = 0;
    uint64_t seed = 42ULL;
    uint32_t shortest_path_length = 0;
    std::vector<Action> reference_shortest_path;

    std::string to_string() const;
};

class PlanningEpisodeGenerator {
public:
    // Generates a single reachable start-goal pair on the specified layout.
    // Guarantees: start_pos and goal_pos are free cells, reachable via BFS,
    // and shortest path is >= min_path_length and <= max_steps.
    static PlanningEpisode generate_reachable_episode(
        const GridWorldLayout& layout,
        uint64_t seed,
        uint32_t max_steps = 16,
        uint32_t min_path_length = 1
    );

    // Generates a deterministic dataset of planning episodes across a set of layouts.
    static std::vector<PlanningEpisode> generate_dataset(
        const std::vector<GridWorldLayout>& layouts,
        size_t episodes_per_layout,
        uint64_t base_seed = 42ULL,
        uint32_t max_steps = 16,
        uint32_t min_path_length = 1
    );
};

// ============================================================================
// 3. Trajectory Scoring & Planning Objective
// ============================================================================
// Objective: J = R_goal - lambda_d * D - lambda_s * L + early_goal_bonus
struct PlanningScoreConfig {
    float goal_reward = 100.0f;
    float lambda_dist = 2.0f;
    float lambda_step = 0.5f;
    float early_goal_bonus_per_step = 1.0f; // Bonus for reaching goal earlier
};

class PlanningScorer {
public:
    static float compute_score(
        uint32_t predicted_pos,
        uint32_t goal_pos,
        uint32_t depth,
        uint32_t max_steps,
        bool goal_reached,
        const PlanningScoreConfig& cfg,
        uint32_t grid_width = 5
    );
};

// ============================================================================
// 4. Lightweight Planning Node Representation
// ============================================================================
struct PlanningNode {
    uint32_t predicted_position = 0;
    std::vector<float> recurrent_state; // Snapshot of h(t) (N=128 floats)
    std::vector<Action> actions_from_root;
    int32_t parent_idx = -1;
    Action action_from_parent = Action::Stay;
    float score = 0.0f;
    uint32_t depth = 0;
    bool goal_reached = false;
    uint32_t manhattan_to_goal = 0;
    float target_probability = 1.0f;

    // Deterministic tie-breaking comparator:
    // Returns true if 'this' is strictly better than 'other'.
    bool is_better_than(const PlanningNode& other) const;
};

// ============================================================================
// 5. Search Configurations and Result Representation
// ============================================================================
struct BeamSearchConfig {
    uint32_t beam_width = 4; // K: 2, 4, 8
    uint32_t max_depth = 6;  // H: 2, 4, 6
    PlanningScoreConfig score_cfg;
    bool return_first_goal = false;
};

struct PlanningSearchResult {
    PlanningNode best_node;
    std::vector<Action> best_plan;
    bool goal_reached = false;
    uint32_t predicted_final_position = 0;
    size_t nodes_expanded = 0;
    size_t model_evaluations = 0;
    std::vector<PlanningNode> completed_goal_candidates;
    double search_runtime_ms = 0.0;
    uint32_t depth_reached = 0;
    uint32_t beam_width = 0;
    uint32_t max_depth = 0;
};

// ============================================================================
// 6. Planning Engine & Recurrent-State Branching Foundation
// ============================================================================
class PlanningEngine {
public:
    static uint32_t compute_manhattan_distance(uint32_t pos1, uint32_t pos2, uint32_t width = 5);

    // Root initialization: Cold start (h(0) = 0 vector)
    static PlanningNode create_cold_start_root(
        uint32_t start_pos,
        uint32_t goal_pos,
        uint32_t num_units = 128,
        uint32_t max_steps = 16,
        const PlanningScoreConfig& score_cfg = PlanningScoreConfig()
    );

    // Root initialization: Warmed-up (active recurrent state from observation history)
    static PlanningNode create_warmed_up_root(
        uint32_t current_pos,
        uint32_t goal_pos,
        const std::vector<float>& current_h,
        uint32_t max_steps = 16,
        const PlanningScoreConfig& score_cfg = PlanningScoreConfig()
    );

    // Encode input vector x (55 features: pos one-hot, static obstacles, action one-hot)
    static void encode_model_input(
        uint32_t current_pos,
        const GridWorldLayout& layout,
        Action action,
        std::vector<float>& out_x
    );

    // Branch expansion: queries learned world model without mutating it or environment.
    // Creates an independent child PlanningNode with its own branched recurrent state.
    static PlanningNode expand_branch(
        const MicroUnitNetwork& net,
        const GridWorldLayout& layout,
        const PlanningNode& parent,
        Action action,
        uint32_t goal_pos,
        uint32_t max_steps = 16,
        const PlanningScoreConfig& score_cfg = PlanningScoreConfig(),
        int32_t parent_index = -1
    );

    // Expand all 5 candidate actions from parent node
    static std::vector<PlanningNode> expand_all_actions(
        const MicroUnitNetwork& net,
        const GridWorldLayout& layout,
        const PlanningNode& parent,
        uint32_t goal_pos,
        uint32_t max_steps = 16,
        const PlanningScoreConfig& score_cfg = PlanningScoreConfig(),
        int32_t parent_index = -1
    );

    // Bounded Beam Search Planner (Milestone 5E.2)
    static PlanningSearchResult beam_search(
        const MicroUnitNetwork& net,
        const GridWorldLayout& layout,
        const PlanningNode& root,
        uint32_t goal_pos,
        const BeamSearchConfig& config
    );

    // Strategy A: One-step greedy lookahead (evaluates immediate 5 actions from root)
    static PlanningSearchResult greedy_search(
        const MicroUnitNetwork& net,
        const GridWorldLayout& layout,
        const PlanningNode& root,
        uint32_t goal_pos,
        const PlanningScoreConfig& score_cfg = PlanningScoreConfig()
    );

    // Strategy C: Shallow exhaustive search (depth 1 to 3)
    static PlanningSearchResult exhaustive_search(
        const MicroUnitNetwork& net,
        const GridWorldLayout& layout,
        const PlanningNode& root,
        uint32_t goal_pos,
        uint32_t depth = 3,
        const PlanningScoreConfig& score_cfg = PlanningScoreConfig()
    );
};

} // namespace spark
