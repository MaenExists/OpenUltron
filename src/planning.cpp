#include "spark/planning.hpp"
#include "spark/metrics.hpp"
#include <queue>
#include <algorithm>
#include <cmath>
#include <sstream>
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
    uint32_t D = PlanningEngine::compute_manhattan_distance(predicted_pos, goal_pos, grid_width);
    float score = -cfg.lambda_dist * static_cast<float>(D) - cfg.lambda_step * static_cast<float>(depth);

    if (goal_reached) {
        score += cfg.goal_reward;
        if (max_steps > depth) {
            score += cfg.early_goal_bonus_per_step * static_cast<float>(max_steps - depth);
        }
    }

    return score;
}

// ============================================================================
// 4. Planning Node Implementation
// ============================================================================
bool PlanningNode::is_better_than(const PlanningNode& other) const {
    // 1. Primary: Score comparison (with numerical tolerance)
    const float score_diff = score - other.score;
    if (std::abs(score_diff) > 1e-4f) {
        return score_diff > 0.0f;
    }

    // 2. Tie-break: Goal reached preference
    if (goal_reached != other.goal_reached) {
        return goal_reached;
    }

    // 3. Tie-break: Manhattan distance to goal (closer is better)
    if (manhattan_to_goal != other.manhattan_to_goal) {
        return manhattan_to_goal < other.manhattan_to_goal;
    }

    // 4. Tie-break: Candidate path length (shorter depth is better)
    if (depth != other.depth) {
        return depth < other.depth;
    }

    // 5. Tie-break: Model prediction confidence
    if (std::abs(target_probability - other.target_probability) > 1e-4f) {
        return target_probability > other.target_probability;
    }

    // 6. Strict deterministic tie-break: Lexicographical action sequence
    size_t min_len = std::min(actions_from_root.size(), other.actions_from_root.size());
    for (size_t i = 0; i < min_len; ++i) {
        auto a1 = static_cast<uint8_t>(actions_from_root[i]);
        auto a2 = static_cast<uint8_t>(other.actions_from_root[i]);
        if (a1 != a2) {
            return a1 < a2;
        }
    }

    return actions_from_root.size() < other.actions_from_root.size();
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
    root.parent_idx = -1;
    root.action_from_parent = Action::Stay;
    root.depth = 0;
    root.goal_reached = (start_pos == goal_pos);
    root.manhattan_to_goal = compute_manhattan_distance(start_pos, goal_pos, 5);
    root.target_probability = 1.0f;
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
    root.parent_idx = -1;
    root.action_from_parent = Action::Stay;
    root.depth = 0;
    root.goal_reached = (current_pos == goal_pos);
    root.manhattan_to_goal = compute_manhattan_distance(current_pos, goal_pos, 5);
    root.target_probability = 1.0f;
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
    PlanningNode child;
    child.depth = parent.depth + 1;
    child.parent_idx = parent_index;
    child.action_from_parent = action;
    child.actions_from_root = parent.actions_from_root;
    child.actions_from_root.push_back(action);

    // Encode input strictly from parent's predicted position and layout
    std::vector<float> x;
    encode_model_input(parent.predicted_position, layout, action, x);

    // Branching forward step: advances from parent's recurrent state without mutating net
    std::vector<float> child_state;
    std::vector<float> logits;
    std::vector<float> probs;

    net.forward_step_branching(
        parent.recurrent_state,
        x,
        child_state,
        logits,
        probs,
        ReadoutMode::Softmax
    );

    // Argmax predicted next position
    uint32_t pred_pos = static_cast<uint32_t>(
        std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
    );

    child.predicted_position = pred_pos;
    child.recurrent_state = std::move(child_state);
    child.goal_reached = (pred_pos == goal_pos);
    child.manhattan_to_goal = compute_manhattan_distance(pred_pos, goal_pos, layout.width);
    child.target_probability = (pred_pos < probs.size()) ? probs[pred_pos] : 0.0f;
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

std::vector<PlanningNode> PlanningEngine::expand_all_actions(
    const MicroUnitNetwork& net,
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
        children.push_back(expand_branch(
            net, layout, parent, a, goal_pos, max_steps, score_cfg, parent_index
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
    Timer timer;
    PlanningSearchResult result;
    result.beam_width = config.beam_width;
    result.max_depth = config.max_depth;

    if (config.beam_width == 0 || config.max_depth == 0) {
        result.best_node = root;
        result.best_plan = root.actions_from_root;
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

    // Track best overall non-root candidate across search
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
                PlanningNode child = expand_branch(
                    net,
                    layout,
                    parent,
                    a,
                    goal_pos,
                    config.max_depth,
                    config.score_cfg
                );
                result.model_evaluations++;

                // Track best non-root candidate
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

    // Determine final best plan
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

} // namespace spark
