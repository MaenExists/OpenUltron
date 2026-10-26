#include "spark/gridworld.hpp"
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

void GridWorldEnv::generate_default_layout() {
    layout_ = generate_random_layout(seed_, 3, 6, DEFAULT_WIDTH, DEFAULT_HEIGHT);
}

void GridWorldEnv::load_layout(const GridWorldLayout& layout) {
    layout_ = layout;
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

} // namespace spark
