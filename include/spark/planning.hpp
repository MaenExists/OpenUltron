#pragma once

#include "spark/network.hpp"
#include "spark/gridworld.hpp"
#include "spark/random.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <functional>
#include <cmath>
#include <map>

namespace spark {

// Type definition for stateless transition predictor
using PredictorFn = std::function<void(
    const std::vector<float>& h_in,
    const std::vector<float>& x,
    std::vector<float>& h_out,
    std::vector<float>& logits,
    std::vector<float>& probs
)>;

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
    std::string episode_id;
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

    // Generates a completely fresh, unseen confirmatory dataset (>= 200 episodes)
    // with 0 layout overlap with train and val sets, deterministic generation,
    // and stable episode identifiers.
    static std::vector<PlanningEpisode> generate_fresh_confirmatory_dataset(
        const std::vector<GridWorldLayout>& fresh_test_layouts,
        size_t episodes_per_layout = 6,
        uint64_t base_seed = 5000500ULL,
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

    bool validate(std::string& err) const {
        if (!std::isfinite(goal_reward) || goal_reward < 0.0f) {
            err = "goal_reward must be finite and non-negative";
            return false;
        }
        if (!std::isfinite(lambda_dist) || lambda_dist < 0.0f) {
            err = "lambda_dist must be finite and non-negative";
            return false;
        }
        if (!std::isfinite(lambda_step) || lambda_step < 0.0f) {
            err = "lambda_step must be finite and non-negative";
            return false;
        }
        if (!std::isfinite(early_goal_bonus_per_step) || early_goal_bonus_per_step < 0.0f) {
            err = "early_goal_bonus_per_step must be finite and non-negative";
            return false;
        }
        return true;
    }
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
    std::vector<uint32_t> predicted_trajectory; // [s_0, s_1, ..., s_L]
    int32_t parent_idx = -1;
    Action action_from_parent = Action::Stay;
    float score = 0.0f;
    uint32_t depth = 0;
    bool goal_reached = false;
    uint32_t manhattan_to_goal = 0;
    float target_probability = 1.0f;
    float cum_log_confidence = 0.0f;
    float mean_step_confidence = 1.0f;

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
    std::vector<uint32_t> predicted_trajectory;
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
// 5B. Real Environment Execution & Scientific Evaluation Diagnostics
// ============================================================================
enum class FailureCategory {
    None,                     // Plan succeeded in real environment
    ObstacleHallucination,    // Model predicted passing through an obstacle cell
    IncorrectPosition,        // Model predicted position different from actual outcome
    DepthInsufficient,        // Goal requires more steps than planning horizon
    PrunedValidRoute,         // Valid route existed within depth but was pruned by beam
    FalseGoalPrediction       // Model predicted goal reached, but actual environment did not reach goal
};

inline const char* failure_category_name(FailureCategory cat) {
    switch (cat) {
        case FailureCategory::None: return "None (Success)";
        case FailureCategory::ObstacleHallucination: return "ObstacleHallucination";
        case FailureCategory::IncorrectPosition: return "IncorrectPosition";
        case FailureCategory::DepthInsufficient: return "DepthInsufficient";
        case FailureCategory::PrunedValidRoute: return "PrunedValidRoute";
        case FailureCategory::FalseGoalPrediction: return "FalseGoalPrediction";
        default: return "Unknown";
    }
}

enum class EpisodeGroup {
    GroupA_Direct,            // Shortest path == Manhattan distance
    GroupB_Detour,            // Shortest path > Manhattan distance
    GroupC_Misleading,        // First step increases Manhattan distance
    GroupD_Longer             // Shortest path > Search Horizon
};

inline const char* episode_group_name(EpisodeGroup group) {
    switch (group) {
        case EpisodeGroup::GroupA_Direct: return "Group A (Direct)";
        case EpisodeGroup::GroupB_Detour: return "Group B (Detour)";
        case EpisodeGroup::GroupC_Misleading: return "Group C (Misleading)";
        case EpisodeGroup::GroupD_Longer: return "Group D (Longer)";
        default: return "Unknown";
    }
}

struct PlanExecutionResult {
    PlanningEpisode episode;
    EpisodeGroup group = EpisodeGroup::GroupA_Direct;
    std::vector<Action> executed_plan;
    std::vector<uint32_t> predicted_trajectory;
    std::vector<uint32_t> actual_trajectory;
    bool predicted_goal_reached = false;
    bool actual_goal_reached = false;
    bool predicted_actual_agreement = false;
    int32_t first_divergence_step = -1; // -1 if no divergence
    float path_efficiency = 0.0f;       // L_executed / L_oracle if actual_goal_reached, else 0.0f
    FailureCategory failure_category = FailureCategory::None;
    size_t nodes_expanded = 0;
    size_t model_evaluations = 0;
    double search_runtime_ms = 0.0;
    float mean_step_confidence = 1.0f;
    float final_score = 0.0f;
    bool cold_start = true;
    bool is_closed_loop = false;
    uint32_t replan_count = 0;
    uint32_t transition_errors = 0;
    size_t peak_memory_bytes = 0;
    bool open_loop_incomplete = false;
};

// ============================================================================
// 5B. Paired Evaluation Data Structures (Milestone 5E.3 Step 2A)
// ============================================================================
enum class PairedOutcome {
    BothSucceed,
    BothFail,
    OpenOnlySucceed,
    ClosedOnlySucceed
};

inline const char* paired_outcome_name(PairedOutcome po) {
    switch (po) {
        case PairedOutcome::BothSucceed: return "BothSucceed";
        case PairedOutcome::BothFail: return "BothFail";
        case PairedOutcome::OpenOnlySucceed: return "OpenOnlySucceed";
        case PairedOutcome::ClosedOnlySucceed: return "ClosedOnlySucceed";
        default: return "Unknown";
    }
}

struct PairedEpisodeResult {
    std::string episode_id;
    uint64_t layout_id = 0;
    uint32_t start_pos = 0;
    uint32_t goal_pos = 0;
    uint32_t oracle_path_length = 0;
    EpisodeGroup group = EpisodeGroup::GroupA_Direct;
    bool is_eligible = false; // oracle_path_length <= common action budget (H)

    PlanExecutionResult open_loop_res;
    PlanExecutionResult closed_loop_res;

    PairedOutcome paired_outcome = PairedOutcome::BothFail;
    bool closed_loop_recovered_from_error = false;
};

struct PairedBenchmarkSummary {
    std::string configuration_name;
    uint32_t beam_width = 4;
    uint32_t horizon = 4; // Action budget H
    bool cold_start = true;

    size_t total_episodes = 0;
    size_t eligible_episodes = 0; // oracle_path_length <= H
    size_t longer_episodes = 0;   // oracle_path_length > H

    // Paired counts on eligible episodes:
    size_t open_loop_successes = 0;
    size_t closed_loop_successes = 0;
    size_t both_succeed = 0;
    size_t both_fail = 0;
    size_t open_only_succeed = 0;
    size_t closed_only_succeed = 0;

    double open_loop_success_rate = 0.0;
    double closed_loop_success_rate = 0.0;
    double delta_percentage_points = 0.0; // closed - open

    // Bootstrap 95% Confidence Interval for delta (percentage points)
    double delta_ci_lower = 0.0;
    double delta_ci_upper = 0.0;

    // Layout-clustered bootstrap 95% Confidence Interval for delta
    double clustered_ci_lower = 0.0;
    double clustered_ci_upper = 0.0;

    // Efficiency & latency
    double open_loop_mean_efficiency = 0.0;
    double closed_loop_mean_efficiency = 0.0;
    double open_loop_mean_runtime_ms = 0.0;
    double closed_loop_mean_runtime_ms = 0.0;
    size_t open_loop_model_evaluations = 0;
    size_t closed_loop_model_evaluations = 0;
    size_t open_loop_peak_memory_bytes = 0;
    size_t closed_loop_peak_memory_bytes = 0;

    // Transition prediction & recovery
    size_t closed_loop_total_replans = 0;
    size_t closed_loop_transition_errors = 0;
    size_t closed_loop_recovery_opportunities = 0; // episodes with transition_errors > 0
    size_t closed_loop_recoveries = 0;             // reached goal despite transition_errors > 0
    double recovery_rate = 0.0;

    // Group breakdown (eligible episodes):
    std::map<EpisodeGroup, size_t> group_eligible_totals;
    std::map<EpisodeGroup, size_t> group_open_success;
    std::map<EpisodeGroup, size_t> group_closed_success;

    // Layout-clustered stats
    size_t total_unique_layouts = 0;

    void print_summary(const std::string& title = "Paired Evaluation Summary") const;
};

struct PlanningEvaluationSummary {
    size_t total_episodes = 0;
    size_t predicted_goals_reached = 0;
    size_t actual_goals_reached = 0;
    size_t predicted_and_actual_success = 0;
    size_t solvable_episodes = 0;       // Episodes where L_oracle <= H
    size_t solvable_actual_success = 0; // Actual success on solvable episodes
    size_t unsolvable_episodes = 0;     // Episodes where L_oracle > H
    size_t unsolvable_actual_success = 0;

    double mean_path_efficiency = 0.0; // Over successful episodes
    size_t efficiency_count = 0;

    double mean_divergence_step = 0.0;
    size_t divergence_count = 0;

    size_t total_nodes_expanded = 0;
    size_t total_model_evaluations = 0;
    double total_runtime_ms = 0.0;

    std::map<FailureCategory, size_t> failure_counts;
    std::map<EpisodeGroup, size_t> group_totals;
    std::map<EpisodeGroup, size_t> group_actual_success;

    void print_summary(const std::string& title = "Planning Evaluation Summary") const;
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

    // Branch expansion using custom/mock predictor function
    static PlanningNode expand_branch_custom(
        const PredictorFn& predictor,
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

    // Expand all 5 candidate actions using custom predictor
    static std::vector<PlanningNode> expand_all_actions_custom(
        const PredictorFn& predictor,
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

    // Bounded Beam Search Planner using custom/mock predictor
    static PlanningSearchResult beam_search_custom(
        const PredictorFn& predictor,
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

    // Shallow exhaustive search using custom predictor
    static PlanningSearchResult exhaustive_search_custom(
        const PredictorFn& predictor,
        const GridWorldLayout& layout,
        const PlanningNode& root,
        uint32_t goal_pos,
        uint32_t depth = 3,
        const PlanningScoreConfig& score_cfg = PlanningScoreConfig()
    );

    // Classify episode into Group A, B, C, D
    static EpisodeGroup classify_episode(const PlanningEpisode& episode, uint32_t max_depth);

    // Open-loop plan execution in real simulator environment
    static PlanExecutionResult execute_plan_open_loop(
        const PlanningEpisode& episode,
        const PlanningSearchResult& search_res,
        bool cold_start = true,
        uint32_t max_executed_steps = 16
    );

    // Closed-loop receding-horizon plan execution in real simulator environment
    static PlanExecutionResult execute_plan_closed_loop(
        const MicroUnitNetwork& net,
        const PlanningEpisode& episode,
        const BeamSearchConfig& config,
        uint32_t max_executed_steps = 16,
        bool cold_start = true
    );

    // Closed-loop receding-horizon plan execution using custom/mock predictor
    static PlanExecutionResult execute_plan_closed_loop_custom(
        const PredictorFn& predictor,
        const PlanningEpisode& episode,
        const std::vector<float>& initial_recurrent_state,
        const BeamSearchConfig& config,
        uint32_t max_executed_steps = 16,
        bool cold_start = true
    );
};

// ============================================================================
// 7. Scientific Planning Evaluator & Benchmark Suite
// ============================================================================
class PlanningEvaluator {
public:
    // Evaluate a specific search strategy across an episode dataset
    static PlanningEvaluationSummary evaluate_strategy(
        const MicroUnitNetwork& net,
        const std::vector<PlanningEpisode>& episodes,
        uint32_t beam_width, // 1 for greedy (depth 1), K for beam search
        uint32_t max_depth,
        bool cold_start = true,
        const PlanningScoreConfig& score_cfg = PlanningScoreConfig()
    );

    // Evaluate ground-truth BFS oracle on the dataset
    static PlanningEvaluationSummary evaluate_oracle(
        const std::vector<PlanningEpisode>& episodes,
        uint32_t max_depth
    );

    // Complete benchmark suite comparing Greedy, Beam (K=2,4,8), and Oracle
    // across horizons H=2, 4, 6 and Cold-start vs Warm-state
    static void run_milestone5e2d_benchmark(
        const std::string& checkpoint_path,
        const std::string& csv_path = "experiments/results/phase5/milestone5e2d_planning_eval.csv"
    );

    // Evaluates paired comparison between open-loop and closed-loop on an episode dataset
    // with exact same action budget = horizon H (Milestone 5E.3 Step 2A)
    static PairedBenchmarkSummary evaluate_paired_comparison(
        const MicroUnitNetwork& net,
        const std::vector<PlanningEpisode>& episodes,
        uint32_t beam_width,
        uint32_t horizon, // action budget H
        bool cold_start = true,
        const PlanningScoreConfig& score_cfg = PlanningScoreConfig(),
        std::vector<PairedEpisodeResult>* out_paired_episodes = nullptr
    );

    // Bootstrap 95% confidence interval computation for delta (percentage points)
    static std::pair<double, double> compute_paired_bootstrap_ci(
        const std::vector<PairedEpisodeResult>& eligible_results,
        size_t num_bootstraps = 2000,
        uint64_t seed = 4242ULL
    );

    // Layout-clustered bootstrap 95% confidence interval computation for delta
    static std::pair<double, double> compute_layout_clustered_bootstrap_ci(
        const std::vector<PairedEpisodeResult>& eligible_results,
        size_t num_bootstraps = 2000,
        uint64_t seed = 4242ULL
    );
};

} // namespace spark
