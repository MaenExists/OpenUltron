#include "spark/planning.hpp"
#include "spark/metrics.hpp"
#include <queue>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace spark {

// ============================================================================
// 1. Ground-Truth BFS Reference Oracle Implementation
// ============================================================================
BFSSolver::BFSResult BFSSolver::find_shortest_path(
    const GridWorldLayout& layout,
    uint32_t start_pos,
    uint32_t goal_pos
) {
    BFSResult result;
    const uint32_t num_cells = layout.width * layout.height;

    if (start_pos >= num_cells || goal_pos >= num_cells) {
        return result;
    }
    if (start_pos < layout.obstacles.size() && layout.obstacles[start_pos] != 0) {
        return result; // Start inside obstacle
    }
    if (goal_pos < layout.obstacles.size() && layout.obstacles[goal_pos] != 0) {
        return result; // Goal inside obstacle
    }

    if (start_pos == goal_pos) {
        result.reachable = true;
        result.shortest_path_length = 0;
        return result;
    }

    std::vector<bool> visited(num_cells, false);
    std::vector<int32_t> parent(num_cells, -1);
    std::vector<Action> parent_action(num_cells, Action::Stay);
    std::queue<uint32_t> q;

    visited[start_pos] = true;
    q.push(start_pos);

    // Standard directional moves for pathfinding
    const std::vector<Action> search_actions = {
        Action::Up,
        Action::Down,
        Action::Left,
        Action::Right
    };

    bool found = false;

    while (!q.empty() && !found) {
        uint32_t curr = q.front();
        q.pop();

        int r = static_cast<int>(curr / layout.width);
        int c = static_cast<int>(curr % layout.width);

        for (Action a : search_actions) {
            int nr = r;
            int nc = c;

            switch (a) {
                case Action::Up:    nr--; break;
                case Action::Down:  nr++; break;
                case Action::Left:  nc--; break;
                case Action::Right: nc++; break;
                default: break;
            }

            // Check boundary
            if (nr < 0 || nr >= static_cast<int>(layout.height) ||
                nc < 0 || nc >= static_cast<int>(layout.width)) {
                continue;
            }

            uint32_t next_pos = static_cast<uint32_t>(nr * layout.width + nc);

            // Check obstacle
            if (next_pos < layout.obstacles.size() && layout.obstacles[next_pos] != 0) {
                continue;
            }

            if (!visited[next_pos]) {
                visited[next_pos] = true;
                parent[next_pos] = static_cast<int32_t>(curr);
                parent_action[next_pos] = a;
                q.push(next_pos);

                if (next_pos == goal_pos) {
                    found = true;
                    break;
                }
            }
        }
    }

    if (!found) {
        return result; // Unreachable
    }

    // Reconstruct path
    std::vector<Action> path;
    int32_t curr = static_cast<int32_t>(goal_pos);
    while (curr != static_cast<int32_t>(start_pos) && curr >= 0) {
        path.push_back(parent_action[curr]);
        curr = parent[curr];
    }

    std::reverse(path.begin(), path.end());
    result.reachable = true;
    result.shortest_path_length = static_cast<uint32_t>(path.size());
    result.shortest_path = std::move(path);
    return result;
}

// ============================================================================
// 2. Planning Episode Generator Implementation
// ============================================================================
std::string PlanningEpisode::to_string() const {
    std::ostringstream ss;
    ss << "Episode(seed=" << seed << ", layout_id=" << layout_id
       << ", start=(" << (start_pos / layout.width) << "," << (start_pos % layout.width) << ")"
       << ", goal=(" << (goal_pos / layout.width) << "," << (goal_pos % layout.width) << ")"
       << ", shortest_path=" << shortest_path_length << ")";
    return ss.str();
}

PlanningEpisode PlanningEpisodeGenerator::generate_reachable_episode(
    const GridWorldLayout& layout,
    uint64_t seed,
    uint32_t max_steps,
    uint32_t min_path_length
) {
    std::vector<uint32_t> free_cells;
    for (uint32_t i = 0; i < layout.obstacles.size(); ++i) {
        if (layout.obstacles[i] == 0) {
            free_cells.push_back(i);
        }
    }

    if (free_cells.size() < 2) {
        throw std::runtime_error("Layout does not have at least 2 free cells for start/goal generation");
    }

    Random rng(seed);
    const size_t num_free = free_cells.size();

    // 1. Try randomized candidate pairs
    for (uint32_t attempt = 0; attempt < 200; ++attempt) {
        size_t idx1 = rng.next_u64() % num_free;
        size_t idx2 = rng.next_u64() % num_free;
        if (idx1 == idx2) continue;

        uint32_t start_pos = free_cells[idx1];
        uint32_t goal_pos = free_cells[idx2];

        auto bfs = BFSSolver::find_shortest_path(layout, start_pos, goal_pos);
        if (bfs.reachable && bfs.shortest_path_length >= min_path_length && bfs.shortest_path_length <= max_steps) {
            PlanningEpisode ep;
            ep.start_pos = start_pos;
            ep.goal_pos = goal_pos;
            ep.layout = layout;
            ep.max_steps = max_steps;
            ep.layout_id = layout.layout_id;
            ep.seed = seed;
            ep.shortest_path_length = bfs.shortest_path_length;
            ep.reference_shortest_path = std::move(bfs.shortest_path);
            return ep;
        }
    }

    // 2. Systematic search fallback if random sampling didn't satisfy length constraints
    for (size_t i = 0; i < num_free; ++i) {
        for (size_t j = 0; j < num_free; ++j) {
            if (i == j) continue;
            uint32_t start_pos = free_cells[i];
            uint32_t goal_pos = free_cells[j];
            auto bfs = BFSSolver::find_shortest_path(layout, start_pos, goal_pos);
            if (bfs.reachable && bfs.shortest_path_length >= min_path_length && bfs.shortest_path_length <= max_steps) {
                PlanningEpisode ep;
                ep.start_pos = start_pos;
                ep.goal_pos = goal_pos;
                ep.layout = layout;
                ep.max_steps = max_steps;
                ep.layout_id = layout.layout_id;
                ep.seed = seed;
                ep.shortest_path_length = bfs.shortest_path_length;
                ep.reference_shortest_path = std::move(bfs.shortest_path);
                return ep;
            }
        }
    }

    throw std::runtime_error("Could not find any reachable start/goal pair satisfying path length constraints");
}

std::vector<PlanningEpisode> PlanningEpisodeGenerator::generate_dataset(
    const std::vector<GridWorldLayout>& layouts,
    size_t episodes_per_layout,
    uint64_t base_seed,
    uint32_t max_steps,
    uint32_t min_path_length
) {
    std::vector<PlanningEpisode> dataset;
    dataset.reserve(layouts.size() * episodes_per_layout);

    for (size_t l_idx = 0; l_idx < layouts.size(); ++l_idx) {
        const auto& layout = layouts[l_idx];
        for (size_t ep_idx = 0; ep_idx < episodes_per_layout; ++ep_idx) {
            uint64_t ep_seed = base_seed + l_idx * 1000003ULL + ep_idx * 7919ULL;
            auto ep = generate_reachable_episode(layout, ep_seed, max_steps, min_path_length);
            dataset.push_back(std::move(ep));
        }
    }

    return dataset;
}

// ============================================================================
// 3. Trajectory Scoring Implementation
// ============================================================================
float PlanningScorer::compute_score(
    uint32_t predicted_pos,
    uint32_t goal_pos,
    uint32_t depth,
    uint32_t max_steps,
    bool goal_reached,
    const PlanningScoreConfig& cfg,
    uint32_t grid_width
) {
    std::string err;
    if (!cfg.validate(err)) {
        throw std::invalid_argument("Invalid PlanningScoreConfig: " + err);
    }

    uint32_t D = PlanningEngine::compute_manhattan_distance(predicted_pos, goal_pos, grid_width);
    float score = -cfg.lambda_dist * static_cast<float>(D) - cfg.lambda_step * static_cast<float>(depth);

    if (goal_reached) {
        score += cfg.goal_reward;
        if (max_steps > depth) {
            score += cfg.early_goal_bonus_per_step * static_cast<float>(max_steps - depth);
        }
    }

    if (!std::isfinite(score)) {
        throw std::runtime_error("PlanningScorer computed non-finite score");
    }

    return score;
}

// ============================================================================
// 4. Planning Node Implementation
// ============================================================================
bool PlanningNode::is_better_than(const PlanningNode& other) const {
    // 1. Primary: Score comparison (strict scalar comparison for Strict Weak Ordering)
    if (score != other.score) {
        return score > other.score;
    }

    // 2. Tie-break: Goal reached preference
    if (goal_reached != other.goal_reached) {
        return goal_reached; // true > false
    }

    // 3. Tie-break: Manhattan distance to goal (closer is better)
    if (manhattan_to_goal != other.manhattan_to_goal) {
        return manhattan_to_goal < other.manhattan_to_goal;
    }

    // 4. Tie-break: Candidate path length (shorter depth is better)
    if (depth != other.depth) {
        return depth < other.depth;
    }

    // 5. Tie-break: Cumulative log confidence (higher is better)
    if (cum_log_confidence != other.cum_log_confidence) {
        return cum_log_confidence > other.cum_log_confidence;
    }

    // 6. Tie-break: Target probability (higher is better)
    if (target_probability != other.target_probability) {
        return target_probability > other.target_probability;
    }

    // 7. Strict deterministic tie-break: Lexicographical action sequence
    const size_t min_len = std::min(actions_from_root.size(), other.actions_from_root.size());
    for (size_t i = 0; i < min_len; ++i) {
        auto a1 = static_cast<uint8_t>(actions_from_root[i]);
        auto a2 = static_cast<uint8_t>(other.actions_from_root[i]);
        if (a1 != a2) {
            return a1 < a2;
        }
    }

    if (actions_from_root.size() != other.actions_from_root.size()) {
        return actions_from_root.size() < other.actions_from_root.size();
    }

    // Nodes are completely identical under this ordering
    return false;
}

// ============================================================================
// 5. Planning Engine & Recurrent Branching Implementation
// ============================================================================
uint32_t PlanningEngine::compute_manhattan_distance(uint32_t pos1, uint32_t pos2, uint32_t width) {
    if (pos1 >= 25 || pos2 >= 25) return 10;
    int r1 = static_cast<int>(pos1 / width);
    int c1 = static_cast<int>(pos1 % width);
    int r2 = static_cast<int>(pos2 / width);
    int c2 = static_cast<int>(pos2 % width);
    return static_cast<uint32_t>(std::abs(r1 - r2) + std::abs(c1 - c2));
}

PlanningNode PlanningEngine::create_cold_start_root(
    uint32_t start_pos,
    uint32_t goal_pos,
    uint32_t num_units,
    uint32_t max_steps,
    const PlanningScoreConfig& score_cfg
) {
    PlanningNode root;
    root.predicted_position = start_pos;
    root.recurrent_state.assign(num_units, 0.0f); // Cold start: zero initial hidden state
    root.actions_from_root.clear();
    root.predicted_trajectory = {start_pos};
    root.parent_idx = -1;
    root.action_from_parent = Action::Stay;
    root.depth = 0;
    root.goal_reached = (start_pos == goal_pos);
    root.manhattan_to_goal = compute_manhattan_distance(start_pos, goal_pos, 5);
    root.target_probability = 1.0f;
    root.cum_log_confidence = 0.0f;
    root.mean_step_confidence = 1.0f;
    root.score = PlanningScorer::compute_score(
        start_pos, goal_pos, 0, max_steps, root.goal_reached, score_cfg, 5
    );
    return root;
}

PlanningNode PlanningEngine::create_warmed_up_root(
    uint32_t current_pos,
    uint32_t goal_pos,
    const std::vector<float>& current_h,
    uint32_t max_steps,
    const PlanningScoreConfig& score_cfg
) {
    PlanningNode root;
    root.predicted_position = current_pos;
    root.recurrent_state = current_h; // Warmed-up: preserves historical observation context
    root.actions_from_root.clear();
    root.predicted_trajectory = {current_pos};
    root.parent_idx = -1;
    root.action_from_parent = Action::Stay;
    root.depth = 0;
    root.goal_reached = (current_pos == goal_pos);
    root.manhattan_to_goal = compute_manhattan_distance(current_pos, goal_pos, 5);
    root.target_probability = 1.0f;
    root.cum_log_confidence = 0.0f;
    root.mean_step_confidence = 1.0f;
    root.score = PlanningScorer::compute_score(
        current_pos, goal_pos, 0, max_steps, root.goal_reached, score_cfg, 5
    );
    return root;
}

void PlanningEngine::encode_model_input(
    uint32_t current_pos,
    const GridWorldLayout& layout,
    Action action,
    std::vector<float>& out_x
) {
    out_x.assign(GridWorldEnv::FULL_INPUT_DIM, 0.0f);

    // 1. Agent Position One-Hot (25 features)
    if (current_pos < GridWorldEnv::NUM_CELLS) {
        out_x[current_pos] = 1.0f;
    }

    // 2. Static Obstacle Map (25 features)
    for (uint32_t i = 0; i < GridWorldEnv::NUM_CELLS; ++i) {
        if (i < layout.obstacles.size() && layout.obstacles[i] != 0) {
            out_x[GridWorldEnv::NUM_CELLS + i] = 1.0f;
        }
    }

    // 3. Candidate Action One-Hot (5 features)
    uint32_t a_idx = static_cast<uint32_t>(action);
    if (a_idx < GridWorldEnv::NUM_ACTIONS) {
        out_x[2 * GridWorldEnv::NUM_CELLS + a_idx] = 1.0f;
    }
}

PlanningNode PlanningEngine::expand_branch_custom(
    const PredictorFn& predictor,
    const GridWorldLayout& layout,
    const PlanningNode& parent,
    Action action,
    uint32_t goal_pos,
    uint32_t max_steps,
    const PlanningScoreConfig& score_cfg,
    int32_t parent_index
) {
    PlanningNode child;
    child.depth = parent.depth + 1;
    child.parent_idx = parent_index;
    child.action_from_parent = action;
    child.actions_from_root = parent.actions_from_root;
    child.actions_from_root.push_back(action);
    child.predicted_trajectory = parent.predicted_trajectory;

    // Encode input strictly from parent's predicted position and layout
    std::vector<float> x;
    encode_model_input(parent.predicted_position, layout, action, x);

    // Branching forward step: advances from parent's recurrent state
    std::vector<float> child_state;
    std::vector<float> logits;
    std::vector<float> probs;

    predictor(
        parent.recurrent_state,
        x,
        child_state,
        logits,
        probs
    );

    // Defensive check
    if (probs.empty()) {
        probs.assign(GridWorldEnv::NUM_CELLS, 1.0f / static_cast<float>(GridWorldEnv::NUM_CELLS));
    }

    // Argmax predicted next position
    uint32_t pred_pos = static_cast<uint32_t>(
        std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
    );

    child.predicted_position = pred_pos;
    child.predicted_trajectory.push_back(pred_pos);
    child.recurrent_state = std::move(child_state);
    child.goal_reached = (pred_pos == goal_pos);
    child.manhattan_to_goal = compute_manhattan_distance(pred_pos, goal_pos, layout.width);

    float p_t = (pred_pos < probs.size()) ? probs[pred_pos] : 0.0f;
    child.target_probability = p_t;

    // Numerical safety clamp to avoid log(0)
    const float eps = 1e-7f;
    float step_log_p = std::log(std::max(p_t, eps));
    child.cum_log_confidence = parent.cum_log_confidence + step_log_p;
    child.mean_step_confidence = std::exp(child.cum_log_confidence / static_cast<float>(child.depth));

    child.score = PlanningScorer::compute_score(
        pred_pos,
        goal_pos,
        child.depth,
        max_steps,
        child.goal_reached,
        score_cfg,
        layout.width
    );

    return child;
}

PlanningNode PlanningEngine::expand_branch(
    const MicroUnitNetwork& net,
    const GridWorldLayout& layout,
    const PlanningNode& parent,
    Action action,
    uint32_t goal_pos,
    uint32_t max_steps,
    const PlanningScoreConfig& score_cfg,
    int32_t parent_index
) {
    PredictorFn predictor = [&net](
        const std::vector<float>& h_in,
        const std::vector<float>& x,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        net.forward_step_branching(
            h_in,
            x,
            h_out,
            logits,
            probs,
            ReadoutMode::Softmax
        );
    };

    return expand_branch_custom(
        predictor,
        layout,
        parent,
        action,
        goal_pos,
        max_steps,
        score_cfg,
        parent_index
    );
}

std::vector<PlanningNode> PlanningEngine::expand_all_actions(
    const MicroUnitNetwork& net,
    const GridWorldLayout& layout,
    const PlanningNode& parent,
    uint32_t goal_pos,
    uint32_t max_steps,
    const PlanningScoreConfig& score_cfg,
    int32_t parent_index
) {
    PredictorFn predictor = [&net](
        const std::vector<float>& h_in,
        const std::vector<float>& x,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        net.forward_step_branching(
            h_in,
            x,
            h_out,
            logits,
            probs,
            ReadoutMode::Softmax
        );
    };

    return expand_all_actions_custom(
        predictor,
        layout,
        parent,
        goal_pos,
        max_steps,
        score_cfg,
        parent_index
    );
}

std::vector<PlanningNode> PlanningEngine::expand_all_actions_custom(
    const PredictorFn& predictor,
    const GridWorldLayout& layout,
    const PlanningNode& parent,
    uint32_t goal_pos,
    uint32_t max_steps,
    const PlanningScoreConfig& score_cfg,
    int32_t parent_index
) {
    const std::vector<Action> candidate_actions = {
        Action::Up,
        Action::Down,
        Action::Left,
        Action::Right,
        Action::Stay
    };

    std::vector<PlanningNode> children;
    children.reserve(candidate_actions.size());

    for (Action a : candidate_actions) {
        children.push_back(expand_branch_custom(
            predictor, layout, parent, a, goal_pos, max_steps, score_cfg, parent_index
        ));
    }

    return children;
}

PlanningSearchResult PlanningEngine::beam_search(
    const MicroUnitNetwork& net,
    const GridWorldLayout& layout,
    const PlanningNode& root,
    uint32_t goal_pos,
    const BeamSearchConfig& config
) {
    PredictorFn predictor = [&net](
        const std::vector<float>& h_in,
        const std::vector<float>& x,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        net.forward_step_branching(
            h_in,
            x,
            h_out,
            logits,
            probs,
            ReadoutMode::Softmax
        );
    };

    return beam_search_custom(
        predictor,
        layout,
        root,
        goal_pos,
        config
    );
}

PlanningSearchResult PlanningEngine::beam_search_custom(
    const PredictorFn& predictor,
    const GridWorldLayout& layout,
    const PlanningNode& root,
    uint32_t goal_pos,
    const BeamSearchConfig& config
) {
    Timer timer;
    PlanningSearchResult result;
    result.beam_width = config.beam_width;
    result.max_depth = config.max_depth;

    if (config.beam_width == 0 || config.max_depth == 0) {
        result.best_node = root;
        result.best_plan = root.actions_from_root;
        result.predicted_trajectory = root.predicted_trajectory;
        result.goal_reached = root.goal_reached;
        result.predicted_final_position = root.predicted_position;
        result.search_runtime_ms = timer.elapsed_ms();
        return result;
    }

    // Check if root already at goal
    if (root.predicted_position == goal_pos) {
        result.best_node = root;
        result.best_node.goal_reached = true;
        result.best_plan = root.actions_from_root;
        result.predicted_trajectory = root.predicted_trajectory;
        result.goal_reached = true;
        result.predicted_final_position = root.predicted_position;
        result.completed_goal_candidates.push_back(result.best_node);
        result.search_runtime_ms = timer.elapsed_ms();
        return result;
    }

    const std::vector<Action> candidate_actions = {
        Action::Up,
        Action::Down,
        Action::Left,
        Action::Right,
        Action::Stay
    };

    std::vector<PlanningNode> current_beam;
    current_beam.push_back(root);

    // Track best overall non-root candidate across search (fallback if no goal reached)
    bool has_best_candidate = false;
    PlanningNode best_overall_candidate;

    for (uint32_t d = 1; d <= config.max_depth; ++d) {
        result.depth_reached = d;
        std::vector<PlanningNode> next_candidates;
        next_candidates.reserve(current_beam.size() * candidate_actions.size());

        for (const auto& parent : current_beam) {
            // Do not expand candidates that have already reached the goal
            if (parent.goal_reached) {
                continue;
            }

            result.nodes_expanded++;

            // Expand all 5 hypothetical actions
            for (Action a : candidate_actions) {
                PlanningNode child = expand_branch_custom(
                    predictor,
                    layout,
                    parent,
                    a,
                    goal_pos,
                    config.max_depth,
                    config.score_cfg
                );
                result.model_evaluations++;

                // Track best non-root candidate for fallback
                if (!has_best_candidate || child.is_better_than(best_overall_candidate)) {
                    best_overall_candidate = child;
                    has_best_candidate = true;
                }

                if (child.goal_reached) {
                    result.completed_goal_candidates.push_back(child);
                } else {
                    next_candidates.push_back(std::move(child));
                }
            }
        }

        // Early stopping if goal reached and return_first_goal requested
        if (config.return_first_goal && !result.completed_goal_candidates.empty()) {
            break;
        }

        // If no unfinished candidates remain (all reached goal or dead ends)
        if (next_candidates.empty()) {
            break;
        }

        // Rank remaining unfinished candidates deterministically
        std::sort(next_candidates.begin(), next_candidates.end(),
            [](const PlanningNode& a, const PlanningNode& b) {
                return a.is_better_than(b);
            }
        );

        // Keep the strongest K unfinished candidates
        if (next_candidates.size() > config.beam_width) {
            next_candidates.resize(config.beam_width);
        }

        current_beam = std::move(next_candidates);
    }

    // Determine final best plan:
    // Strict separation: completed goal candidates take absolute priority over unfinished candidates
    if (!result.completed_goal_candidates.empty()) {
        std::sort(result.completed_goal_candidates.begin(), result.completed_goal_candidates.end(),
            [](const PlanningNode& a, const PlanningNode& b) {
                return a.is_better_than(b);
            }
        );
        result.best_node = result.completed_goal_candidates[0];
        result.best_plan = result.best_node.actions_from_root;
        result.goal_reached = true;
        result.predicted_final_position = result.best_node.predicted_position;
    } else if (has_best_candidate) {
        result.best_node = best_overall_candidate;
        result.best_plan = best_overall_candidate.actions_from_root;
        result.goal_reached = false;
        result.predicted_final_position = best_overall_candidate.predicted_position;
    } else {
        result.best_node = root;
        result.best_plan = root.actions_from_root;
        result.goal_reached = false;
        result.predicted_final_position = root.predicted_position;
    }

    result.predicted_trajectory = result.best_node.predicted_trajectory;
    result.search_runtime_ms = timer.elapsed_ms();
    return result;
}

PlanningSearchResult PlanningEngine::greedy_search(
    const MicroUnitNetwork& net,
    const GridWorldLayout& layout,
    const PlanningNode& root,
    uint32_t goal_pos,
    const PlanningScoreConfig& score_cfg
) {
    BeamSearchConfig cfg;
    cfg.beam_width = 1;
    cfg.max_depth = 1;
    cfg.score_cfg = score_cfg;
    return beam_search(net, layout, root, goal_pos, cfg);
}

PlanningSearchResult PlanningEngine::exhaustive_search(
    const MicroUnitNetwork& net,
    const GridWorldLayout& layout,
    const PlanningNode& root,
    uint32_t goal_pos,
    uint32_t depth,
    const PlanningScoreConfig& score_cfg
) {
    if (depth > 4) {
        throw std::invalid_argument("Exhaustive search depth must be <= 4 to prevent exponential blowup");
    }
    BeamSearchConfig cfg;
    // 5^depth upper bound ensures unpruned breadth
    cfg.beam_width = static_cast<uint32_t>(std::pow(5, depth));
    cfg.max_depth = depth;
    cfg.score_cfg = score_cfg;
    return beam_search(net, layout, root, goal_pos, cfg);
}

PlanningSearchResult PlanningEngine::exhaustive_search_custom(
    const PredictorFn& predictor,
    const GridWorldLayout& layout,
    const PlanningNode& root,
    uint32_t goal_pos,
    uint32_t depth,
    const PlanningScoreConfig& score_cfg
) {
    if (depth > 4) {
        throw std::invalid_argument("Exhaustive search depth must be <= 4 to prevent exponential blowup");
    }
    BeamSearchConfig cfg;
    cfg.beam_width = static_cast<uint32_t>(std::pow(5, depth));
    cfg.max_depth = depth;
    cfg.score_cfg = score_cfg;
    return beam_search_custom(predictor, layout, root, goal_pos, cfg);
}

EpisodeGroup PlanningEngine::classify_episode(const PlanningEpisode& episode, uint32_t max_depth) {
    uint32_t D0 = compute_manhattan_distance(episode.start_pos, episode.goal_pos, episode.layout.width);
    if (episode.shortest_path_length > max_depth) {
        return EpisodeGroup::GroupD_Longer;
    }
    if (episode.shortest_path_length == D0) {
        return EpisodeGroup::GroupA_Direct;
    }
    if (!episode.reference_shortest_path.empty()) {
        Action a0 = episode.reference_shortest_path[0];
        auto tr = GridWorldEnv::simulate_transition(
            episode.start_pos, a0, episode.layout.obstacles, episode.layout.width, episode.layout.height
        );
        uint32_t D1 = compute_manhattan_distance(tr.to_pos, episode.goal_pos, episode.layout.width);
        if (D1 > D0) {
            return EpisodeGroup::GroupC_Misleading;
        }
    }
    return EpisodeGroup::GroupB_Detour;
}

PlanExecutionResult PlanningEngine::execute_plan_open_loop(
    const PlanningEpisode& episode,
    const PlanningSearchResult& search_res,
    bool cold_start
) {
    PlanExecutionResult res;
    res.episode = episode;
    res.executed_plan = search_res.best_plan;
    res.predicted_trajectory = search_res.predicted_trajectory;
    res.predicted_goal_reached = search_res.goal_reached;
    res.nodes_expanded = search_res.nodes_expanded;
    res.model_evaluations = search_res.model_evaluations;
    res.search_runtime_ms = search_res.search_runtime_ms;
    res.mean_step_confidence = search_res.best_node.mean_step_confidence;
    res.final_score = search_res.best_node.score;
    res.cold_start = cold_start;
    res.group = classify_episode(episode, search_res.max_depth);

    res.actual_trajectory.push_back(episode.start_pos);
    uint32_t curr_pos = episode.start_pos;
    int32_t first_div = -1;
    bool had_discrepancy = false;
    bool hit_obstacle_hallucination = false;

    for (size_t t = 0; t < search_res.best_plan.size(); ++t) {
        Action a = search_res.best_plan[t];
        auto trans = GridWorldEnv::simulate_transition(
            curr_pos, a, episode.layout.obstacles, episode.layout.width, episode.layout.height
        );
        curr_pos = trans.to_pos;
        res.actual_trajectory.push_back(curr_pos);

        uint32_t pred_t = (t + 1 < search_res.predicted_trajectory.size())
            ? search_res.predicted_trajectory[t + 1]
            : search_res.predicted_final_position;

        if (first_div == -1 && pred_t != curr_pos) {
            first_div = static_cast<int32_t>(t + 1);
        }
        if (pred_t < episode.layout.obstacles.size() && episode.layout.obstacles[pred_t] != 0) {
            hit_obstacle_hallucination = true;
        }
        if (pred_t != curr_pos) {
            had_discrepancy = true;
        }
    }

    res.first_divergence_step = first_div;
    res.actual_goal_reached = (curr_pos == episode.goal_pos);
    res.predicted_actual_agreement = (res.predicted_goal_reached == res.actual_goal_reached);

    if (res.actual_goal_reached && episode.shortest_path_length > 0) {
        res.path_efficiency = static_cast<float>(search_res.best_plan.size()) / static_cast<float>(episode.shortest_path_length);
    } else {
        res.path_efficiency = 0.0f;
    }

    // Failure categorization
    if (res.actual_goal_reached) {
        res.failure_category = FailureCategory::None;
    } else if (episode.shortest_path_length > search_res.max_depth) {
        res.failure_category = FailureCategory::DepthInsufficient;
    } else if (res.predicted_goal_reached && !res.actual_goal_reached) {
        if (hit_obstacle_hallucination) {
            res.failure_category = FailureCategory::ObstacleHallucination;
        } else {
            res.failure_category = FailureCategory::FalseGoalPrediction;
        }
    } else if (hit_obstacle_hallucination) {
        res.failure_category = FailureCategory::ObstacleHallucination;
    } else if (had_discrepancy) {
        res.failure_category = FailureCategory::IncorrectPosition;
    } else {
        res.failure_category = FailureCategory::PrunedValidRoute;
    }

    return res;
}

PlanExecutionResult PlanningEngine::execute_plan_closed_loop_custom(
    const PredictorFn& predictor,
    const PlanningEpisode& episode,
    const std::vector<float>& initial_recurrent_state,
    const BeamSearchConfig& config,
    uint32_t max_executed_steps,
    bool cold_start
) {
    Timer total_timer;
    PlanExecutionResult res;
    res.episode = episode;
    res.cold_start = cold_start;
    res.is_closed_loop = true;
    res.group = classify_episode(episode, config.max_depth);
    res.actual_trajectory.push_back(episode.start_pos);
    res.predicted_trajectory.push_back(episode.start_pos);

    uint32_t curr_pos = episode.start_pos;
    std::vector<float> h_live = initial_recurrent_state;

    // Peak memory estimate (root + beam candidates + expanded children)
    size_t node_size = sizeof(PlanningNode) + (h_live.size() * sizeof(float));
    res.peak_memory_bytes = (config.beam_width * 5 + 1) * node_size;

    while (curr_pos != episode.goal_pos && res.executed_plan.size() < max_executed_steps) {
        // 1. Obtain current permitted observation (curr_pos, episode.layout)
        // 2. Initialize search root from current observation & live recurrent state
        PlanningNode root;
        root.predicted_position = curr_pos;
        root.recurrent_state = h_live; // Independent copy: search cannot mutate h_live
        root.predicted_trajectory.push_back(curr_pos);
        root.depth = 0;
        root.goal_reached = (curr_pos == episode.goal_pos);
        root.manhattan_to_goal = compute_manhattan_distance(curr_pos, episode.goal_pos, episode.layout.width);
        root.score = PlanningScorer::compute_score(
            curr_pos, episode.goal_pos, 0, config.max_depth, root.goal_reached, config.score_cfg, episode.layout.width
        );

        // 3. Execute bounded beam search using world model predictor
        auto search_res = beam_search_custom(predictor, episode.layout, root, episode.goal_pos, config);
        res.nodes_expanded += search_res.nodes_expanded;
        res.model_evaluations += search_res.model_evaluations;
        res.replan_count++;

        if (search_res.best_plan.empty()) {
            // No candidate action returned
            break;
        }

        // 4. Select the first action of the best predicted plan
        Action action_to_execute = search_res.best_plan[0];

        // 5. Execute that action in the actual simulator
        auto trans = GridWorldEnv::simulate_transition(
            curr_pos, action_to_execute, episode.layout.obstacles, episode.layout.width, episode.layout.height
        );
        uint32_t real_next_pos = trans.to_pos;

        // 6 & 7. Advance live recurrent state using the same temporal convention as training
        std::vector<float> x_step;
        encode_model_input(curr_pos, episode.layout, action_to_execute, x_step);

        std::vector<float> h_next, logits, probs;
        predictor(h_live, x_step, h_next, logits, probs);

        uint32_t pred_step = static_cast<uint32_t>(
            std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
        );

        if (pred_step != real_next_pos) {
            res.transition_errors++;
            if (res.first_divergence_step == -1) {
                res.first_divergence_step = static_cast<int32_t>(res.executed_plan.size() + 1);
            }
        }

        h_live = std::move(h_next);
        curr_pos = real_next_pos;

        res.executed_plan.push_back(action_to_execute);
        res.actual_trajectory.push_back(curr_pos);
        res.predicted_trajectory.push_back(pred_step);
    }

    res.search_runtime_ms = total_timer.elapsed_ms();
    res.actual_goal_reached = (curr_pos == episode.goal_pos);
    res.predicted_goal_reached = res.actual_goal_reached;
    res.predicted_actual_agreement = true;

    if (res.actual_goal_reached && episode.shortest_path_length > 0) {
        res.path_efficiency = static_cast<float>(res.executed_plan.size()) / static_cast<float>(episode.shortest_path_length);
    } else {
        res.path_efficiency = 0.0f;
    }

    // Failure categorization
    if (res.actual_goal_reached) {
        res.failure_category = FailureCategory::None;
    } else if (episode.shortest_path_length > max_executed_steps) {
        res.failure_category = FailureCategory::DepthInsufficient;
    } else if (res.transition_errors > 0) {
        res.failure_category = FailureCategory::IncorrectPosition;
    } else {
        res.failure_category = FailureCategory::PrunedValidRoute;
    }

    return res;
}

PlanExecutionResult PlanningEngine::execute_plan_closed_loop(
    const MicroUnitNetwork& net,
    const PlanningEpisode& episode,
    const BeamSearchConfig& config,
    uint32_t max_executed_steps,
    bool cold_start
) {
    PredictorFn predictor = [&net](
        const std::vector<float>& h_in,
        const std::vector<float>& x,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        net.forward_step_branching(
            h_in,
            x,
            h_out,
            logits,
            probs,
            ReadoutMode::Softmax
        );
    };

    std::vector<float> initial_h;
    if (cold_start) {
        initial_h.assign(net.config().num_units, 0.0f);
    } else {
        // Legitimate warm state: observe initial state s_0 with Action::Stay to compute h(1)
        std::vector<float> x_prior;
        encode_model_input(episode.start_pos, episode.layout, Action::Stay, x_prior);
        std::vector<float> logits, probs;
        net.forward_step_branching(
            std::vector<float>(net.config().num_units, 0.0f),
            x_prior,
            initial_h,
            logits,
            probs,
            ReadoutMode::Softmax
        );
    }

    return execute_plan_closed_loop_custom(
        predictor,
        episode,
        initial_h,
        config,
        max_executed_steps,
        cold_start
    );
}

// ============================================================================
// 6. Scientific Planning Evaluator Implementation
// ============================================================================
void PlanningEvaluationSummary::print_summary(const std::string& title) const {
    std::cout << "\n=======================================================\n"
              << " " << title << "\n"
              << "=======================================================\n";
    std::cout << " Total Episodes               : " << total_episodes << "\n";
    std::cout << " Solvable (L_oracle <= H)     : " << solvable_episodes
              << " (" << std::fixed << std::setprecision(1)
              << (total_episodes > 0 ? (100.0 * solvable_episodes / total_episodes) : 0.0) << "%)\n";
    std::cout << " Actual Goal Success Rate     : " << actual_goals_reached << "/" << total_episodes
              << " (" << std::setprecision(2)
              << (total_episodes > 0 ? (100.0 * actual_goals_reached / total_episodes) : 0.0) << "%)\n";
    std::cout << " Solvable Actual Success Rate : " << solvable_actual_success << "/" << solvable_episodes
              << " (" << std::setprecision(2)
              << (solvable_episodes > 0 ? (100.0 * solvable_actual_success / solvable_episodes) : 0.0) << "%)\n";
    std::cout << " Predicted Goal Success Rate  : " << predicted_goals_reached << "/" << total_episodes
              << " (" << std::setprecision(2)
              << (total_episodes > 0 ? (100.0 * predicted_goals_reached / total_episodes) : 0.0) << "%)\n";
    std::cout << " Predicted/Actual Agreement   : " << predicted_and_actual_success << "/" << total_episodes
              << " (" << std::setprecision(2)
              << (total_episodes > 0 ? (100.0 * predicted_and_actual_success / total_episodes) : 0.0) << "%)\n";
    std::cout << " Mean Path Efficiency (succ)  : " << std::setprecision(3) << mean_path_efficiency << "\n";
    std::cout << " Mean Divergence Step (div)   : " << std::setprecision(2) << mean_divergence_step << "\n";
    std::cout << " Total Nodes Expanded         : " << total_nodes_expanded << "\n";
    std::cout << " Total Model Evaluations      : " << total_model_evaluations << "\n";
    std::cout << " Total Runtime                : " << std::setprecision(2) << total_runtime_ms << " ms\n";

    std::cout << "\n--- Failure Mode Categorization ---\n";
    const std::vector<FailureCategory> cats = {
        FailureCategory::None,
        FailureCategory::ObstacleHallucination,
        FailureCategory::IncorrectPosition,
        FailureCategory::DepthInsufficient,
        FailureCategory::PrunedValidRoute,
        FailureCategory::FalseGoalPrediction
    };
    for (auto c : cats) {
        size_t count = failure_counts.count(c) ? failure_counts.at(c) : 0;
        double pct = total_episodes > 0 ? (100.0 * count / total_episodes) : 0.0;
        std::cout << "  " << std::left << std::setw(26) << failure_category_name(c)
                  << ": " << std::right << std::setw(4) << count
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }

    std::cout << "\n--- Episode Group Breakdown ---\n";
    const std::vector<EpisodeGroup> grps = {
        EpisodeGroup::GroupA_Direct,
        EpisodeGroup::GroupB_Detour,
        EpisodeGroup::GroupC_Misleading,
        EpisodeGroup::GroupD_Longer
    };
    for (auto g : grps) {
        size_t total_g = group_totals.count(g) ? group_totals.at(g) : 0;
        size_t succ_g = group_actual_success.count(g) ? group_actual_success.at(g) : 0;
        double pct = total_g > 0 ? (100.0 * succ_g / total_g) : 0.0;
        std::cout << "  " << std::left << std::setw(24) << episode_group_name(g)
                  << ": " << std::right << std::setw(3) << succ_g << "/" << std::setw(3) << total_g
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }
    std::cout << "=======================================================\n";
}

PlanningEvaluationSummary PlanningEvaluator::evaluate_strategy(
    const MicroUnitNetwork& net,
    const std::vector<PlanningEpisode>& episodes,
    uint32_t beam_width,
    uint32_t max_depth,
    bool cold_start,
    const PlanningScoreConfig& score_cfg
) {
    PlanningEvaluationSummary summary;
    summary.total_episodes = episodes.size();

    BeamSearchConfig cfg;
    cfg.beam_width = beam_width;
    cfg.max_depth = max_depth;
    cfg.score_cfg = score_cfg;
    cfg.return_first_goal = false;

    double sum_eff = 0.0;
    double sum_div = 0.0;

    for (const auto& ep : episodes) {
        PlanningSearchResult s_res;
        if (cold_start) {
            auto root = PlanningEngine::create_cold_start_root(ep.start_pos, ep.goal_pos, 128, max_depth, score_cfg);
            if (beam_width == 1 && max_depth == 1) {
                s_res = PlanningEngine::greedy_search(net, ep.layout, root, ep.goal_pos, score_cfg);
            } else {
                s_res = PlanningEngine::beam_search(net, ep.layout, root, ep.goal_pos, cfg);
            }
        } else {
            // Legitimate warm state: observe initial state s_0 with Action::Stay to compute h(1)
            std::vector<float> x_prior;
            PlanningEngine::encode_model_input(ep.start_pos, ep.layout, Action::Stay, x_prior);
            std::vector<float> h_out, logits, probs;
            net.forward_step_branching(std::vector<float>(128, 0.0f), x_prior, h_out, logits, probs, ReadoutMode::Softmax);

            auto root = PlanningEngine::create_warmed_up_root(ep.start_pos, ep.goal_pos, h_out, max_depth, score_cfg);
            if (beam_width == 1 && max_depth == 1) {
                s_res = PlanningEngine::greedy_search(net, ep.layout, root, ep.goal_pos, score_cfg);
            } else {
                s_res = PlanningEngine::beam_search(net, ep.layout, root, ep.goal_pos, cfg);
            }
        }

        auto exec_res = PlanningEngine::execute_plan_open_loop(ep, s_res, cold_start);

        summary.total_nodes_expanded += exec_res.nodes_expanded;
        summary.total_model_evaluations += exec_res.model_evaluations;
        summary.total_runtime_ms += exec_res.search_runtime_ms;

        if (exec_res.predicted_goal_reached) summary.predicted_goals_reached++;
        if (exec_res.actual_goal_reached) summary.actual_goals_reached++;
        if (exec_res.predicted_actual_agreement) summary.predicted_and_actual_success++;

        bool is_solvable = (ep.shortest_path_length <= max_depth);
        if (is_solvable) {
            summary.solvable_episodes++;
            if (exec_res.actual_goal_reached) summary.solvable_actual_success++;
        } else {
            summary.unsolvable_episodes++;
            if (exec_res.actual_goal_reached) summary.unsolvable_actual_success++;
        }

        if (exec_res.actual_goal_reached && exec_res.path_efficiency > 0.0f) {
            sum_eff += exec_res.path_efficiency;
            summary.efficiency_count++;
        }

        if (exec_res.first_divergence_step > 0) {
            sum_div += exec_res.first_divergence_step;
            summary.divergence_count++;
        }

        summary.failure_counts[exec_res.failure_category]++;
        summary.group_totals[exec_res.group]++;
        if (exec_res.actual_goal_reached) {
            summary.group_actual_success[exec_res.group]++;
        }
    }

    if (summary.efficiency_count > 0) {
        summary.mean_path_efficiency = sum_eff / summary.efficiency_count;
    }
    if (summary.divergence_count > 0) {
        summary.mean_divergence_step = sum_div / summary.divergence_count;
    }

    return summary;
}

PlanningEvaluationSummary PlanningEvaluator::evaluate_oracle(
    const std::vector<PlanningEpisode>& episodes,
    uint32_t max_depth
) {
    PlanningEvaluationSummary summary;
    summary.total_episodes = episodes.size();

    double sum_eff = 0.0;
    for (const auto& ep : episodes) {
        auto bfs = BFSSolver::find_shortest_path(ep.layout, ep.start_pos, ep.goal_pos);
        bool solvable = (bfs.reachable && bfs.shortest_path_length <= max_depth);
        auto grp = PlanningEngine::classify_episode(ep, max_depth);
        summary.group_totals[grp]++;

        if (solvable) {
            summary.solvable_episodes++;
            summary.solvable_actual_success++;
            summary.actual_goals_reached++;
            summary.predicted_goals_reached++;
            summary.predicted_and_actual_success++;
            sum_eff += 1.0;
            summary.efficiency_count++;
            summary.group_actual_success[grp]++;
            summary.failure_counts[FailureCategory::None]++;
        } else {
            summary.unsolvable_episodes++;
            summary.failure_counts[FailureCategory::DepthInsufficient]++;
        }
    }

    if (summary.efficiency_count > 0) {
        summary.mean_path_efficiency = sum_eff / summary.efficiency_count;
    }
    return summary;
}

void PlanningEvaluator::run_milestone5e2d_benchmark(
    const std::string& checkpoint_path,
    const std::string& csv_path
) {
    std::cout << "\n=======================================================\n"
              << " SPARK Milestone 5E.2D — Scientific Planning Benchmark\n"
              << "=======================================================\n"
              << " Checkpoint: " << checkpoint_path << "\n";

    spark::SparseGraphConfig g_cfg;
    spark::NetworkConfig n_cfg;
    spark::MicroUnitNetwork net(n_cfg, g_cfg);

    if (!MicroUnitNetwork::load_checkpoint(checkpoint_path, net)) {
        throw std::runtime_error("Could not load checkpoint from " + checkpoint_path);
    }

    // Splits: 100 train, 20 val, 20 test
    auto splits = GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);

    // 1. Validation set (Smoke test: 20 episodes on validation layouts)
    std::cout << " Generating 20 validation episodes on validation layouts..." << std::endl;
    auto val_episodes = PlanningEpisodeGenerator::generate_dataset(splits.val_layouts, 1, 42ULL, 6, 1);

    // 2. Held-out test set (40 episodes on unseen test layouts)
    std::cout << " Generating 40 test episodes on unseen held-out test layouts..." << std::endl;
    auto test_episodes = PlanningEpisodeGenerator::generate_dataset(splits.test_layouts, 2, 8888ULL, 6, 1);

    // CSV Header
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        throw std::runtime_error("Could not open CSV file for writing: " + csv_path);
    }
    csv << "Split,Strategy,BeamWidth,MaxDepth,Condition,TotalEpisodes,SolvableEpisodes,"
        << "SolvableActualSuccessRate,OverallActualSuccessRate,PredictedSuccessRate,"
        << "AgreementRate,MeanPathEfficiency,MeanDivergenceStep,MeanRuntimeMs,NodesExpanded,"
        << "ObstacleHallucinations,IncorrectPositions,DepthInsufficient,PrunedValidRoutes,FalseGoalPredictions\n";

    struct ConfigToEval {
        std::string name;
        uint32_t beam_width;
        uint32_t max_depth;
        bool is_oracle;
    };

    const std::vector<ConfigToEval> configs = {
        {"GreedyLookahead", 1, 1, false},
        {"Beam_K2_H2", 2, 2, false},
        {"Beam_K2_H4", 2, 4, false},
        {"Beam_K2_H6", 2, 6, false},
        {"Beam_K4_H2", 4, 2, false},
        {"Beam_K4_H4", 4, 4, false},
        {"Beam_K4_H6", 4, 6, false},
        {"Beam_K8_H2", 8, 2, false},
        {"Beam_K8_H4", 8, 4, false},
        {"Beam_K8_H6", 8, 6, false},
        {"Oracle_H2", 0, 2, true},
        {"Oracle_H4", 0, 4, true},
        {"Oracle_H6", 0, 6, true}
    };

    auto run_and_log = [&](const std::string& split_name, const std::vector<PlanningEpisode>& eps) {
        for (const auto& c : configs) {
            std::vector<bool> conditions = {true}; // Cold start
            if (!c.is_oracle) conditions.push_back(false); // Warm state

            for (bool cold : conditions) {
                std::string cond_name = c.is_oracle ? "N/A" : (cold ? "ColdStart" : "WarmedUp");
                PlanningEvaluationSummary sum;
                if (c.is_oracle) {
                    sum = evaluate_oracle(eps, c.max_depth);
                } else {
                    sum = evaluate_strategy(net, eps, c.beam_width, c.max_depth, cold);
                }

                double solv_succ_rate = sum.solvable_episodes > 0 ? (100.0 * sum.solvable_actual_success / sum.solvable_episodes) : 0.0;
                double actual_succ_rate = sum.total_episodes > 0 ? (100.0 * sum.actual_goals_reached / sum.total_episodes) : 0.0;
                double pred_succ_rate = sum.total_episodes > 0 ? (100.0 * sum.predicted_goals_reached / sum.total_episodes) : 0.0;
                double agree_rate = sum.total_episodes > 0 ? (100.0 * sum.predicted_and_actual_success / sum.total_episodes) : 0.0;

                csv << split_name << "," << c.name << "," << c.beam_width << "," << c.max_depth << ","
                    << cond_name << "," << sum.total_episodes << "," << sum.solvable_episodes << ","
                    << std::fixed << std::setprecision(2) << solv_succ_rate << ","
                    << actual_succ_rate << "," << pred_succ_rate << "," << agree_rate << ","
                    << std::setprecision(3) << sum.mean_path_efficiency << ","
                    << std::setprecision(2) << sum.mean_divergence_step << ","
                    << (sum.total_episodes > 0 ? sum.total_runtime_ms / sum.total_episodes : 0.0) << ","
                    << sum.total_nodes_expanded << ","
                    << sum.failure_counts[FailureCategory::ObstacleHallucination] << ","
                    << sum.failure_counts[FailureCategory::IncorrectPosition] << ","
                    << sum.failure_counts[FailureCategory::DepthInsufficient] << ","
                    << sum.failure_counts[FailureCategory::PrunedValidRoute] << ","
                    << sum.failure_counts[FailureCategory::FalseGoalPrediction] << "\n";

                std::cout << " [" << split_name << "] " << std::left << std::setw(18) << c.name
                          << " | " << std::setw(10) << cond_name
                          << " | Solvable: " << std::right << std::setw(5) << std::fixed << std::setprecision(1) << solv_succ_rate << "%"
                          << " | Overall: " << std::setw(5) << std::fixed << std::setprecision(1) << actual_succ_rate << "%"
                          << " | Pred: " << std::setw(5) << std::fixed << std::setprecision(1) << pred_succ_rate << "%"
                          << " | Eff: " << std::setprecision(2) << sum.mean_path_efficiency
                          << " | Runtime: " << std::setprecision(1) << sum.total_runtime_ms << " ms"
                          << std::endl;
            }
        }
    };

    std::cout << "\n--- Running Smoke Evaluation on Validation Layouts ---" << std::endl;
    run_and_log("Validation", val_episodes);

    std::cout << "\n--- Running Held-Out Evaluation on Test Layouts ---" << std::endl;
    run_and_log("HeldOutTest", test_episodes);

    csv.close();
    std::cout << "\nResults exported successfully to: " << csv_path << std::endl;
}

} // namespace spark
