#include "spark/gridworld.hpp"
#include "spark/network.hpp"
#include "spark/learning.hpp"
#include "spark/metrics.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace spark {

std::string GridWorldLayout::to_string() const {
    std::ostringstream ss;
    for (uint32_t r = 0; r < height; ++r) {
        for (uint32_t c = 0; c < width; ++c) {
            uint32_t idx = r * width + c;
            ss << (obstacles[idx] ? '#' : '.');
        }
        ss << "\n";
    }
    return ss.str();
}

uint32_t GridWorldLayout::obstacle_count() const {
    uint32_t cnt = 0;
    for (uint8_t obs : obstacles) {
        if (obs) cnt++;
    }
    return cnt;
}

void WorldModelMetrics::print_summary() const {
    std::cout << std::left << std::setw(30) << model_name
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(12) << (overall_accuracy() * 100.0f) << "%"
              << std::setw(12) << (successful_move_accuracy() * 100.0f) << "%"
              << std::setw(12) << (blocked_move_accuracy() * 100.0f) << "%"
              << std::setw(12) << (boundary_block_accuracy() * 100.0f) << "%"
              << std::setw(12) << (obstacle_block_accuracy() * 100.0f) << "%"
              << std::setprecision(4) << std::setw(12) << mean_loss()
              << std::endl;
}

std::string Milestone5bRecord::csv_header() {
    return "experiment_id,seed,task,trainer,units,fan_in,training_examples,test_examples,obstacle_density,visibility_mode,rollout_horizon,train_loss,validation_loss,test_loss,next_state_accuracy,successful_move_acc,blocked_move_acc,boundary_collision_acc,obstacle_collision_acc,training_seconds,inference_seconds,model_bytes";
}

std::string Milestone5bRecord::to_csv() const {
    std::ostringstream ss;
    ss << experiment_id << ","
       << seed << ","
       << task << ","
       << trainer << ","
       << units << ","
       << fan_in << ","
       << training_examples << ","
       << test_examples << ","
       << std::fixed << std::setprecision(3) << obstacle_density << ","
       << visibility_mode << ","
       << rollout_horizon << ","
       << std::setprecision(5) << train_loss << ","
       << validation_loss << ","
       << test_loss << ","
       << std::setprecision(4) << next_state_accuracy << ","
       << successful_move_acc << ","
       << blocked_move_acc << ","
       << boundary_collision_acc << ","
       << obstacle_collision_acc << ","
       << std::setprecision(3) << training_seconds << ","
       << inference_seconds << ","
       << model_bytes;
    return ss.str();
}

TransitionResult GridWorldEnv::simulate_transition(
    uint32_t pos,
    Action action,
    const std::vector<uint8_t>& obstacles,
    uint32_t width,
    uint32_t height
) {
    TransitionResult res;
    res.from_pos = pos;
    res.action = action;
    res.to_pos = pos;
    res.blocked_by_boundary = false;
    res.blocked_by_obstacle = false;
    res.moved = false;

    if (pos >= width * height) {
        return res; // Defensive out-of-bounds guard
    }

    uint32_t r = pos / width;
    uint32_t c = pos % width;
    int target_r = static_cast<int>(r);
    int target_c = static_cast<int>(c);

    switch (action) {
        case Action::Up:
            target_r -= 1;
            break;
        case Action::Down:
            target_r += 1;
            break;
        case Action::Left:
            target_c -= 1;
            break;
        case Action::Right:
            target_c += 1;
            break;
        case Action::Stay:
            break;
    }

    // 1. Check boundary collision
    if (target_r < 0 || target_r >= static_cast<int>(height) ||
        target_c < 0 || target_c >= static_cast<int>(width)) {
        res.blocked_by_boundary = true;
        res.to_pos = pos;
        return res;
    }

    uint32_t cand_pos = static_cast<uint32_t>(target_r) * width + static_cast<uint32_t>(target_c);

    // 2. Check obstacle collision
    if (cand_pos < obstacles.size() && obstacles[cand_pos] != 0) {
        res.blocked_by_obstacle = true;
        res.to_pos = pos;
        return res;
    }

    // 3. Successful movement
    res.to_pos = cand_pos;
    res.moved = (cand_pos != pos);
    return res;
}

GridWorldLayout GridWorldEnv::generate_random_layout(
    uint64_t seed,
    uint32_t min_obstacles,
    uint32_t max_obstacles,
    uint32_t width,
    uint32_t height
) {
    GridWorldLayout layout;
    layout.width = width;
    layout.height = height;
    layout.layout_id = seed;
    layout.obstacles.assign(width * height, 0);

    Random rng(seed);
    uint32_t total_cells = width * height;
    if (max_obstacles >= total_cells - 2) {
        max_obstacles = total_cells - 2;
    }
    if (min_obstacles > max_obstacles) {
        min_obstacles = max_obstacles;
    }

    uint32_t num_obs = min_obstacles + static_cast<uint32_t>(rng.next_u64() % (max_obstacles - min_obstacles + 1));

    std::vector<uint32_t> indices(total_cells);
    std::iota(indices.begin(), indices.end(), 0);

    // Fisher-Yates shuffle
    for (uint32_t i = total_cells - 1; i > 0; --i) {
        uint32_t j = static_cast<uint32_t>(rng.next_u64() % (i + 1));
        std::swap(indices[i], indices[j]);
    }

    for (uint32_t i = 0; i < num_obs; ++i) {
        layout.obstacles[indices[i]] = 1;
    }

    return layout;
}

LayoutSplits GridWorldEnv::generate_split_layouts(
    size_t num_train,
    size_t num_val,
    size_t num_test,
    uint32_t min_obstacles,
    uint32_t max_obstacles,
    uint64_t master_seed
) {
    LayoutSplits splits;
    Random master_rng(master_seed);

    auto layout_to_key = [](const GridWorldLayout& l) -> std::string {
        std::string s;
        s.reserve(l.obstacles.size());
        for (uint8_t b : l.obstacles) s.push_back(b ? '1' : '0');
        return s;
    };

    std::unordered_set<std::string> seen_layouts;

    auto populate_split = [&](std::vector<GridWorldLayout>& target_list, size_t count, uint32_t split_tag) {
        target_list.reserve(count);
        size_t attempts = 0;
        const size_t max_attempts = count * 2000;

        while (target_list.size() < count && attempts < max_attempts) {
            attempts++;
            uint64_t cand_seed = master_rng.next_u64();
            GridWorldLayout cand = generate_random_layout(cand_seed, min_obstacles, max_obstacles);
            std::string key = layout_to_key(cand);

            if (seen_layouts.find(key) == seen_layouts.end()) {
                seen_layouts.insert(key);
                cand.layout_id = static_cast<uint64_t>(split_tag) * 1000000ULL + target_list.size();
                target_list.push_back(cand);
            }
        }

        if (target_list.size() < count) {
            throw std::runtime_error("Failed to generate enough unique layouts for split!");
        }
    };

    populate_split(splits.train_layouts, num_train, 1);
    populate_split(splits.val_layouts, num_val, 2);
    populate_split(splits.test_layouts, num_test, 3);

    return splits;
}

GridWorldEnv::GridWorldEnv(
    uint32_t episode_length,
    uint64_t seed,
    GridObservationMode obs_mode
) : episode_length_(episode_length),
    seed_(seed),
    rng_(seed),
    obs_mode_(obs_mode) {
    generate_default_layout();
    reset();
}

GridWorldEnv::GridWorldEnv(
    const GridWorldLayout& layout,
    uint32_t episode_length,
    uint64_t seed,
    GridObservationMode obs_mode
) : episode_length_(episode_length),
    seed_(seed),
    rng_(seed),
    obs_mode_(obs_mode),
    layout_(layout) {
    reset();
}

GridWorldEnv::GridWorldEnv(
    const std::vector<GridWorldLayout>& layouts,
    uint32_t episode_length,
    uint64_t seed,
    GridObservationMode obs_mode
) : episode_length_(episode_length),
    seed_(seed),
    rng_(seed),
    obs_mode_(obs_mode),
    layouts_(layouts) {
    if (layouts_.empty()) {
        generate_default_layout();
    } else {
        layout_ = layouts_[0];
    }
    reset();
}

void GridWorldEnv::generate_default_layout() {
    layout_ = generate_random_layout(seed_, 3, 6, DEFAULT_WIDTH, DEFAULT_HEIGHT);
}

void GridWorldEnv::load_layout(const GridWorldLayout& layout) {
    layout_ = layout;
    layouts_.clear();
    reset();
}

void GridWorldEnv::set_layouts(const std::vector<GridWorldLayout>& layouts) {
    layouts_ = layouts;
    current_layout_idx_ = 0;
    if (!layouts_.empty()) {
        layout_ = layouts_[0];
    }
    reset();
}

void GridWorldEnv::place_agent_randomly() {
    std::vector<uint32_t> free_cells;
    for (uint32_t i = 0; i < layout_.obstacles.size(); ++i) {
        if (layout_.obstacles[i] == 0) {
            free_cells.push_back(i);
        }
    }

    if (free_cells.empty()) {
        throw std::runtime_error("Cannot place agent: no free cells in grid world!");
    }

    uint32_t idx = static_cast<uint32_t>(rng_.next_u64() % free_cells.size());
    agent_pos_ = free_cells[idx];
}

void GridWorldEnv::set_agent_position(uint32_t pos) {
    if (pos >= NUM_CELLS) {
        throw std::out_of_range("Agent position out of grid bounds");
    }
    if (pos < layout_.obstacles.size() && layout_.obstacles[pos] != 0) {
        throw std::invalid_argument("Cannot place agent inside an obstacle cell");
    }
    agent_pos_ = pos;
    compute_next_transition();
}

void GridWorldEnv::pick_random_action() {
    uint32_t a_idx = static_cast<uint32_t>(rng_.next_u64() % NUM_ACTIONS);
    current_action_ = static_cast<Action>(a_idx);
}

void GridWorldEnv::set_action(Action a) {
    current_action_ = a;
    compute_next_transition();
}

void GridWorldEnv::compute_next_transition() {
    last_transition_ = simulate_transition(
        agent_pos_,
        current_action_,
        layout_.obstacles,
        layout_.width,
        layout_.height
    );
    next_pos_ = last_transition_.to_pos;
}

void GridWorldEnv::reset() {
    step_idx_ = 0;
    pos_observed_ = true;

    // If multiple layouts are loaded, cycle through them uniformly
    if (!layouts_.empty()) {
        layout_ = layouts_[current_layout_idx_ % layouts_.size()];
        current_layout_idx_++;
    }

    place_agent_randomly();
    pick_random_action();
    compute_next_transition();
}

Observation GridWorldEnv::get_observation() const {
    Observation obs;
    obs.values.assign(input_dim(), 0.0f);

    if (obs_mode_ == GridObservationMode::FullyObservable) {
        // [0..24]: Agent position one-hot
        if (agent_pos_ < NUM_CELLS) {
            obs.values[agent_pos_] = 1.0f;
        }

        // [25..49]: Obstacle map binary
        for (uint32_t i = 0; i < NUM_CELLS; ++i) {
            if (i < layout_.obstacles.size() && layout_.obstacles[i] != 0) {
                obs.values[NUM_CELLS + i] = 1.0f;
            }
        }

        // [50..54]: Selected action one-hot
        uint32_t a_idx = static_cast<uint32_t>(current_action_);
        if (a_idx < NUM_ACTIONS) {
            obs.values[2 * NUM_CELLS + a_idx] = 1.0f;
        }
    } else { // PartiallyObservable (56 dimensions)
        // [0..24]: Agent position one-hot (only if pos_observed_)
        if (pos_observed_ && agent_pos_ < NUM_CELLS) {
            obs.values[agent_pos_] = 1.0f;
        }

        // [25]: Position observation mask flag
        obs.values[NUM_CELLS] = pos_observed_ ? 1.0f : 0.0f;

        // [26..50]: Obstacle map binary
        for (uint32_t i = 0; i < NUM_CELLS; ++i) {
            if (i < layout_.obstacles.size() && layout_.obstacles[i] != 0) {
                obs.values[NUM_CELLS + 1 + i] = 1.0f;
            }
        }

        // [51..55]: Selected action one-hot
        uint32_t a_idx = static_cast<uint32_t>(current_action_);
        if (a_idx < NUM_ACTIONS) {
            obs.values[2 * NUM_CELLS + 1 + a_idx] = 1.0f;
        }
    }

    return obs;
}

Target GridWorldEnv::get_target() const {
    Target target;
    target.values.assign(OUTPUT_DIM, 0.0f);
    target.valid = true;

    if (next_pos_ < OUTPUT_DIM) {
        target.values[next_pos_] = 1.0f;
    }

    return target;
}

void GridWorldEnv::step() {
    if (is_done()) return;

    agent_pos_ = next_pos_;
    step_idx_++;

    if (obs_mode_ == GridObservationMode::PartiallyObservable) {
        pos_observed_ = false; // By default in partial observability, position is only revealed at step 0
    }

    if (!is_done()) {
        pick_random_action();
        compute_next_transition();
    }
}

bool GridWorldEnv::is_done() const {
    return step_idx_ >= episode_length_;
}

WorldModelMetrics evaluate_world_model(
    MicroUnitNetwork& net,
    GridWorldEnv& env,
    uint32_t total_transitions
) {
    WorldModelMetrics metrics;
    metrics.model_name = "SPARK_TBPTT";
    metrics.total_transitions = total_transitions;

    Timer timer;
    std::vector<float> logits;
    std::vector<float> probs;

    env.reset();
    net.reset_state();

    for (uint32_t step = 0; step < total_transitions; ++step) {
        if (env.is_done()) {
            env.reset();
            net.reset_state();
        }

        Observation obs = env.get_observation();
        Target target = env.get_target();
        const auto& trans = env.last_transition();

        net.forward_step(obs.values, logits, probs, ReadoutMode::Softmax);

        // Compute loss
        float loss = TBPTTTrainer::categorical_ce_loss(logits, target.values);
        metrics.total_loss += loss;

        // Prediction via argmax
        uint32_t pred = static_cast<uint32_t>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
        uint32_t truth = trans.to_pos;

        bool correct = (pred == truth);
        if (correct) metrics.correct_transitions++;

        if (trans.moved) {
            metrics.successful_moves_total++;
            if (correct) metrics.successful_moves_correct++;
        } else {
            metrics.blocked_moves_total++;
            if (correct) metrics.blocked_moves_correct++;

            if (trans.blocked_by_boundary) {
                metrics.boundary_blocks_total++;
                if (correct) metrics.boundary_blocks_correct++;
            }
            if (trans.blocked_by_obstacle) {
                metrics.obstacle_blocks_total++;
                if (correct) metrics.obstacle_blocks_correct++;
            }
        }

        env.step();
    }

    metrics.inference_seconds = timer.elapsed_seconds();
    return metrics;
}

BaselineSuiteResults evaluate_all_baselines(
    GridWorldEnv& env,
    uint32_t total_transitions,
    uint64_t seed
) {
    BaselineSuiteResults results;
    results.random_predictor.model_name = "1. Random Position";
    results.random_predictor.total_transitions = total_transitions;

    results.copy_current_predictor.model_name = "2. Copy Current Position";
    results.copy_current_predictor.total_transitions = total_transitions;

    results.naive_movement_predictor.model_name = "3. Naive Move (Ignore Obs)";
    results.naive_movement_predictor.total_transitions = total_transitions;

    results.ground_truth_oracle.model_name = "4. Ground Truth Oracle";
    results.ground_truth_oracle.total_transitions = total_transitions;

    Random rng(seed);
    std::vector<uint8_t> empty_grid(GridWorldEnv::NUM_CELLS, 0);

    env.reset();

    for (uint32_t step = 0; step < total_transitions; ++step) {
        if (env.is_done()) {
            env.reset();
        }

        const auto& trans = env.last_transition();
        uint32_t truth = trans.to_pos;
        uint32_t current_pos = env.agent_position();
        Action a = env.current_action();

        // 1. Random position baseline
        uint32_t rand_pred = static_cast<uint32_t>(rng.next_u64() % GridWorldEnv::NUM_CELLS);
        results.random_predictor.total_loss += 3.2188758; // -log(1/25)
        bool rand_correct = (rand_pred == truth);
        if (rand_correct) results.random_predictor.correct_transitions++;
        if (trans.moved) {
            results.random_predictor.successful_moves_total++;
            if (rand_correct) results.random_predictor.successful_moves_correct++;
        } else {
            results.random_predictor.blocked_moves_total++;
            if (rand_correct) results.random_predictor.blocked_moves_correct++;
            if (trans.blocked_by_boundary) {
                results.random_predictor.boundary_blocks_total++;
                if (rand_correct) results.random_predictor.boundary_blocks_correct++;
            }
            if (trans.blocked_by_obstacle) {
                results.random_predictor.obstacle_blocks_total++;
                if (rand_correct) results.random_predictor.obstacle_blocks_correct++;
            }
        }

        // 2. Copy current position baseline
        uint32_t copy_pred = current_pos;
        bool copy_correct = (copy_pred == truth);
        results.copy_current_predictor.total_loss += (copy_correct ? 0.0 : 15.0);
        if (copy_correct) results.copy_current_predictor.correct_transitions++;
        if (trans.moved) {
            results.copy_current_predictor.successful_moves_total++;
            if (copy_correct) results.copy_current_predictor.successful_moves_correct++;
        } else {
            results.copy_current_predictor.blocked_moves_total++;
            if (copy_correct) results.copy_current_predictor.blocked_moves_correct++;
            if (trans.blocked_by_boundary) {
                results.copy_current_predictor.boundary_blocks_total++;
                if (copy_correct) results.copy_current_predictor.boundary_blocks_correct++;
            }
            if (trans.blocked_by_obstacle) {
                results.copy_current_predictor.obstacle_blocks_total++;
                if (copy_correct) results.copy_current_predictor.obstacle_blocks_correct++;
            }
        }

        // 3. Naive movement baseline (handles boundary, ignores obstacles)
        auto naive_trans = GridWorldEnv::simulate_transition(current_pos, a, empty_grid);
        uint32_t naive_pred = naive_trans.to_pos;
        bool naive_correct = (naive_pred == truth);
        results.naive_movement_predictor.total_loss += (naive_correct ? 0.0 : 15.0);
        if (naive_correct) results.naive_movement_predictor.correct_transitions++;
        if (trans.moved) {
            results.naive_movement_predictor.successful_moves_total++;
            if (naive_correct) results.naive_movement_predictor.successful_moves_correct++;
        } else {
            results.naive_movement_predictor.blocked_moves_total++;
            if (naive_correct) results.naive_movement_predictor.blocked_moves_correct++;
            if (trans.blocked_by_boundary) {
                results.naive_movement_predictor.boundary_blocks_total++;
                if (naive_correct) results.naive_movement_predictor.boundary_blocks_correct++;
            }
            if (trans.blocked_by_obstacle) {
                results.naive_movement_predictor.obstacle_blocks_total++;
                if (naive_correct) results.naive_movement_predictor.obstacle_blocks_correct++;
            }
        }

        // 4. Ground truth oracle
        results.ground_truth_oracle.total_loss += 0.0;
        results.ground_truth_oracle.correct_transitions++;
        if (trans.moved) {
            results.ground_truth_oracle.successful_moves_total++;
            results.ground_truth_oracle.successful_moves_correct++;
        } else {
            results.ground_truth_oracle.blocked_moves_total++;
            results.ground_truth_oracle.blocked_moves_correct++;
            if (trans.blocked_by_boundary) {
                results.ground_truth_oracle.boundary_blocks_total++;
                results.ground_truth_oracle.boundary_blocks_correct++;
            }
            if (trans.blocked_by_obstacle) {
                results.ground_truth_oracle.obstacle_blocks_total++;
                results.ground_truth_oracle.obstacle_blocks_correct++;
            }
        }

        env.step();
    }

    return results;
}

} // namespace spark
