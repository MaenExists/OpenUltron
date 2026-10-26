#include "spark/gridworld.hpp"
#include "spark/learning.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <unordered_set>

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ \
                      << " - " << msg << std::endl; \
            std::exit(1); \
        } \
    } while(0)

void test_deterministic_transitions() {
    std::cout << "[Test] Running test_deterministic_transitions..." << std::endl;

    std::vector<uint8_t> empty_obstacles(25, 0);
    uint32_t center = 12; // (2, 2)

    // Up: (2,2) -> (1,2) [idx 7]
    auto res_up = spark::GridWorldEnv::simulate_transition(center, spark::Action::Up, empty_obstacles);
    REQUIRE(res_up.to_pos == 7, "Up transition failed");
    REQUIRE(res_up.moved, "Up should move");
    REQUIRE(!res_up.blocked_by_boundary, "Up should not be blocked");
    REQUIRE(!res_up.blocked_by_obstacle, "Up should not hit obstacle");

    // Down: (2,2) -> (3,2) [idx 17]
    auto res_down = spark::GridWorldEnv::simulate_transition(center, spark::Action::Down, empty_obstacles);
    REQUIRE(res_down.to_pos == 17, "Down transition failed");
    REQUIRE(res_down.moved, "Down should move");

    // Left: (2,2) -> (2,1) [idx 11]
    auto res_left = spark::GridWorldEnv::simulate_transition(center, spark::Action::Left, empty_obstacles);
    REQUIRE(res_left.to_pos == 11, "Left transition failed");
    REQUIRE(res_left.moved, "Left should move");

    // Right: (2,2) -> (2,3) [idx 13]
    auto res_right = spark::GridWorldEnv::simulate_transition(center, spark::Action::Right, empty_obstacles);
    REQUIRE(res_right.to_pos == 13, "Right transition failed");
    REQUIRE(res_right.moved, "Right should move");

    // Stay: (2,2) -> (2,2) [idx 12]
    auto res_stay = spark::GridWorldEnv::simulate_transition(center, spark::Action::Stay, empty_obstacles);
    REQUIRE(res_stay.to_pos == 12, "Stay transition failed");
    REQUIRE(!res_stay.moved, "Stay should not move");

    std::cout << "[Test] test_deterministic_transitions PASSED." << std::endl;
}

void test_boundary_collisions() {
    std::cout << "[Test] Running test_boundary_collisions..." << std::endl;

    std::vector<uint8_t> empty_obstacles(25, 0);

    // Top boundary (row 0, col 2 = idx 2) -> Up
    auto res_top = spark::GridWorldEnv::simulate_transition(2, spark::Action::Up, empty_obstacles);
    REQUIRE(res_top.to_pos == 2, "Top boundary collision should remain at pos 2");
    REQUIRE(res_top.blocked_by_boundary, "Top collision must report blocked_by_boundary");
    REQUIRE(!res_top.moved, "Top collision must not move");

    // Bottom boundary (row 4, col 2 = idx 22) -> Down
    auto res_bot = spark::GridWorldEnv::simulate_transition(22, spark::Action::Down, empty_obstacles);
    REQUIRE(res_bot.to_pos == 22, "Bottom boundary collision should remain at pos 22");
    REQUIRE(res_bot.blocked_by_boundary, "Bottom collision must report blocked_by_boundary");

    // Left boundary (row 2, col 0 = idx 10) -> Left
    auto res_l = spark::GridWorldEnv::simulate_transition(10, spark::Action::Left, empty_obstacles);
    REQUIRE(res_l.to_pos == 10, "Left boundary collision should remain at pos 10");
    REQUIRE(res_l.blocked_by_boundary, "Left collision must report blocked_by_boundary");

    // Right boundary (row 2, col 4 = idx 14) -> Right
    auto res_r = spark::GridWorldEnv::simulate_transition(14, spark::Action::Right, empty_obstacles);
    REQUIRE(res_r.to_pos == 14, "Right boundary collision should remain at pos 14");
    REQUIRE(res_r.blocked_by_boundary, "Right collision must report blocked_by_boundary");

    // Top-Left corner (0, 0)
    auto res_tl_up = spark::GridWorldEnv::simulate_transition(0, spark::Action::Up, empty_obstacles);
    auto res_tl_left = spark::GridWorldEnv::simulate_transition(0, spark::Action::Left, empty_obstacles);
    REQUIRE(res_tl_up.to_pos == 0 && res_tl_up.blocked_by_boundary, "Top-left Up failed");
    REQUIRE(res_tl_left.to_pos == 0 && res_tl_left.blocked_by_boundary, "Top-left Left failed");

    // Bottom-Right corner (4, 4 = idx 24)
    auto res_br_down = spark::GridWorldEnv::simulate_transition(24, spark::Action::Down, empty_obstacles);
    auto res_br_right = spark::GridWorldEnv::simulate_transition(24, spark::Action::Right, empty_obstacles);
    REQUIRE(res_br_down.to_pos == 24 && res_br_down.blocked_by_boundary, "Bottom-right Down failed");
    REQUIRE(res_br_right.to_pos == 24 && res_br_right.blocked_by_boundary, "Bottom-right Right failed");

    std::cout << "[Test] test_boundary_collisions PASSED." << std::endl;
}

void test_obstacle_collisions() {
    std::cout << "[Test] Running test_obstacle_collisions..." << std::endl;

    std::vector<uint8_t> obstacles(25, 0);
    obstacles[7] = 1; // Obstacle directly above (2, 2) [idx 12]

    // Up into obstacle at idx 7
    auto res_obs = spark::GridWorldEnv::simulate_transition(12, spark::Action::Up, obstacles);
    REQUIRE(res_obs.to_pos == 12, "Obstacle collision must keep agent at original position");
    REQUIRE(res_obs.blocked_by_obstacle, "Obstacle collision must report blocked_by_obstacle");
    REQUIRE(!res_obs.blocked_by_boundary, "Should not report boundary block for obstacle");
    REQUIRE(!res_obs.moved, "Obstacle collision must not move");

    // Left into empty cell (2, 1 = idx 11)
    auto res_free = spark::GridWorldEnv::simulate_transition(12, spark::Action::Left, obstacles);
    REQUIRE(res_free.to_pos == 11, "Moving into empty cell must succeed");
    REQUIRE(res_free.moved, "Moving into empty cell must report moved");
    REQUIRE(!res_free.blocked_by_obstacle, "Empty cell move must not report obstacle block");

    std::cout << "[Test] test_obstacle_collisions PASSED." << std::endl;
}

void test_agent_placement_and_validation() {
    std::cout << "[Test] Running test_agent_placement_and_validation..." << std::endl;

    spark::GridWorldLayout layout;
    layout.obstacles.assign(25, 0);
    layout.obstacles[0] = 1;
    layout.obstacles[5] = 1;
    layout.obstacles[10] = 1;

    spark::GridWorldEnv env(layout, 50, 42ULL);

    // Test 100 resets to ensure agent is never placed on an obstacle
    for (int i = 0; i < 100; ++i) {
        env.reset();
        uint32_t pos = env.agent_position();
        REQUIRE(pos < 25, "Agent position out of bounds");
        REQUIRE(layout.obstacles[pos] == 0, "Agent placed on obstacle cell");
    }

    // Attempting to manually place agent on an obstacle cell must throw invalid_argument
    bool caught_invalid = false;
    try {
        env.set_agent_position(0); // idx 0 is an obstacle
    } catch (const std::invalid_argument&) {
        caught_invalid = true;
    }
    REQUIRE(caught_invalid, "set_agent_position into obstacle must throw invalid_argument");

    // Out of bounds position must throw out_of_range
    bool caught_oor = false;
    try {
        env.set_agent_position(25);
    } catch (const std::out_of_range&) {
        caught_oor = true;
    }
    REQUIRE(caught_oor, "set_agent_position out of bounds must throw out_of_range");

    std::cout << "[Test] test_agent_placement_and_validation PASSED." << std::endl;
}

void test_reproducible_generation() {
    std::cout << "[Test] Running test_reproducible_generation..." << std::endl;

    spark::GridWorldEnv env1(30, 12345ULL);
    spark::GridWorldEnv env2(30, 12345ULL);

    // Initial layouts must be identical
    REQUIRE(env1.layout() == env2.layout(), "Environments with same seed must have identical layouts");
    REQUIRE(env1.agent_position() == env2.agent_position(), "Initial positions must match");
    REQUIRE(env1.current_action() == env2.current_action(), "Initial actions must match");

    // Stepping through episodes must yield identical observations and targets
    for (int t = 0; t < 25; ++t) {
        spark::Observation obs1 = env1.get_observation();
        spark::Observation obs2 = env2.get_observation();
        REQUIRE(obs1.values == obs2.values, "Observations diverged across steps");

        spark::Target tgt1 = env1.get_target();
        spark::Target tgt2 = env2.get_target();
        REQUIRE(tgt1.values == tgt2.values, "Targets diverged across steps");

        env1.step();
        env2.step();
    }

    // Different seed must produce different layout
    spark::GridWorldEnv env3(30, 99999ULL);
    REQUIRE(!(env1.layout() == env3.layout()), "Different seeds must generate distinct layouts");

    std::cout << "[Test] test_reproducible_generation PASSED." << std::endl;
}

void test_layout_splits_and_data_integrity() {
    std::cout << "[Test] Running test_layout_splits_and_data_integrity..." << std::endl;

    size_t num_train = 40;
    size_t num_val = 10;
    size_t num_test = 10;

    auto splits = spark::GridWorldEnv::generate_split_layouts(num_train, num_val, num_test, 3, 7, 777ULL);

    REQUIRE(splits.train_layouts.size() == num_train, "Train layout count mismatch");
    REQUIRE(splits.val_layouts.size() == num_val, "Val layout count mismatch");
    REQUIRE(splits.test_layouts.size() == num_test, "Test layout count mismatch");
    REQUIRE(splits.total_unique() == num_train + num_val + num_test, "Total layout count mismatch");

    auto to_key = [](const spark::GridWorldLayout& l) -> std::string {
        std::string s;
        for (uint8_t b : l.obstacles) s.push_back(b ? '1' : '0');
        return s;
    };

    std::unordered_set<std::string> train_keys;
    for (const auto& l : splits.train_layouts) {
        std::string key = to_key(l);
        REQUIRE(train_keys.find(key) == train_keys.end(), "Duplicate layout found inside train set");
        train_keys.insert(key);
    }

    std::unordered_set<std::string> val_keys;
    for (const auto& l : splits.val_layouts) {
        std::string key = to_key(l);
        REQUIRE(train_keys.find(key) == train_keys.end(), "Data leakage: Val layout exists in train set");
        REQUIRE(val_keys.find(key) == val_keys.end(), "Duplicate layout found inside val set");
        val_keys.insert(key);
    }

    std::unordered_set<std::string> test_keys;
    for (const auto& l : splits.test_layouts) {
        std::string key = to_key(l);
        REQUIRE(train_keys.find(key) == train_keys.end(), "Data leakage: Test layout exists in train set");
        REQUIRE(val_keys.find(key) == val_keys.end(), "Data leakage: Test layout exists in val set");
        REQUIRE(test_keys.find(key) == test_keys.end(), "Duplicate layout found inside test set");
        test_keys.insert(key);
    }

    std::cout << "  Generated " << splits.train_layouts.size() << " train, "
              << splits.val_layouts.size() << " val, and "
              << splits.test_layouts.size() << " test strictly unique layouts (0 overlap)."
              << std::endl;

    std::cout << "[Test] test_layout_splits_and_data_integrity PASSED." << std::endl;
}

void test_input_and_target_encoding() {
    std::cout << "[Test] Running test_input_and_target_encoding..." << std::endl;

    // 1. Fully Observable mode
    spark::GridWorldEnv full_env(20, 42ULL, spark::GridObservationMode::FullyObservable);
    REQUIRE(full_env.input_dim() == 55, "Full input dim must be 55");
    REQUIRE(full_env.output_dim() == 25, "Output dim must be 25");

    spark::Observation full_obs = full_env.get_observation();
    REQUIRE(full_obs.values.size() == 55, "Observation vector size mismatch");

    // Agent position one-hot: exactly one 1.0f in [0..24]
    uint32_t pos = full_env.agent_position();
    for (uint32_t i = 0; i < 25; ++i) {
        float expected = (i == pos) ? 1.0f : 0.0f;
        REQUIRE(full_obs.values[i] == expected, "Agent position one-hot encoding mismatch");
    }

    // Obstacle map binary in [25..49]
    for (uint32_t i = 0; i < 25; ++i) {
        float expected = full_env.layout().obstacles[i] ? 1.0f : 0.0f;
        REQUIRE(full_obs.values[25 + i] == expected, "Obstacle map encoding mismatch");
    }

    // Action one-hot in [50..54]
    uint32_t action_idx = static_cast<uint32_t>(full_env.current_action());
    for (uint32_t i = 0; i < 5; ++i) {
        float expected = (i == action_idx) ? 1.0f : 0.0f;
        REQUIRE(full_obs.values[50 + i] == expected, "Action one-hot encoding mismatch");
    }

    // Target one-hot
    spark::Target full_tgt = full_env.get_target();
    REQUIRE(full_tgt.values.size() == 25, "Target vector size mismatch");
    REQUIRE(full_tgt.valid, "Target must be marked valid");
    uint32_t next_p = full_env.next_position();
    for (uint32_t i = 0; i < 25; ++i) {
        float expected = (i == next_p) ? 1.0f : 0.0f;
        REQUIRE(full_tgt.values[i] == expected, "Target one-hot encoding mismatch");
    }

    // 2. Partially Observable mode
    spark::GridWorldEnv partial_env(20, 42ULL, spark::GridObservationMode::PartiallyObservable);
    REQUIRE(partial_env.input_dim() == 56, "Partial input dim must be 56");

    // At step 0: position is observed
    spark::Observation p_obs0 = partial_env.get_observation();
    REQUIRE(p_obs0.values[25] == 1.0f, "Position mask at step 0 must be 1.0f");
    REQUIRE(p_obs0.values[partial_env.agent_position()] == 1.0f, "Position must be encoded at step 0");

    // Step to step 1: position becomes withheld
    partial_env.step();
    spark::Observation p_obs1 = partial_env.get_observation();
    REQUIRE(p_obs1.values[25] == 0.0f, "Position mask after step 0 must be 0.0f");
    for (uint32_t i = 0; i < 25; ++i) {
        REQUIRE(p_obs1.values[i] == 0.0f, "Withheld position must be masked to zero");
    }

    std::cout << "[Test] test_input_and_target_encoding PASSED." << std::endl;
}

void test_softmax_and_categorical_loss() {
    std::cout << "[Test] Running test_softmax_and_categorical_loss..." << std::endl;

    // Uniform logits -> equal probabilities
    std::vector<float> uniform_logits(25, 0.0f);
    std::vector<float> probs;
    spark::MicroUnitNetwork::stable_softmax(uniform_logits, probs);
    REQUIRE(probs.size() == 25, "Softmax output size mismatch");
    for (float p : probs) {
        REQUIRE(std::abs(p - 0.04f) < 1e-5f, "Uniform softmax mismatch");
    }

    // High logit for target -> low loss
    std::vector<float> peaked_logits(25, 0.0f);
    peaked_logits[5] = 20.0f; // Target class 5 strongly predicted
    std::vector<float> target_one_hot(25, 0.0f);
    target_one_hot[5] = 1.0f;

    float low_loss = spark::TBPTTTrainer::categorical_ce_loss(peaked_logits, target_one_hot);
    REQUIRE(low_loss < 0.001f, "Confident correct prediction should have near-zero loss");

    // Low logit for target -> high loss
    peaked_logits[5] = -20.0f;
    float high_loss = spark::TBPTTTrainer::categorical_ce_loss(peaked_logits, target_one_hot);
    REQUIRE(high_loss > 15.0f, "Incorrect confident prediction should have high loss");

    // Numerical stability under extreme values
    std::vector<float> extreme_logits(25, -500.0f);
    extreme_logits[3] = 500.0f;
    target_one_hot.assign(25, 0.0f);
    target_one_hot[3] = 1.0f;
    float stable_loss = spark::TBPTTTrainer::categorical_ce_loss(extreme_logits, target_one_hot);
    REQUIRE(std::isfinite(stable_loss), "Loss must remain finite under extreme logits");
    REQUIRE(stable_loss >= 0.0f, "Categorical loss must be non-negative");

    std::cout << "[Test] test_softmax_and_categorical_loss PASSED." << std::endl;
}

void test_categorical_gradient_finite_difference() {
    std::cout << "[Test] Running test_categorical_gradient_finite_difference..." << std::endl;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 16;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 0.5f;
    n_cfg.seed = 42ULL;

    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 16;
    g_cfg.fan_in = 4;
    g_cfg.seed = 42ULL;

    spark::MicroUnitNetwork net(n_cfg, g_cfg);

    // Create a 4-step sequence from GridWorldEnv
    spark::GridWorldEnv env(10, 42ULL);
    std::vector<spark::Observation> observations;
    std::vector<spark::Target> targets;

    for (int t = 0; t < 4; ++t) {
        observations.push_back(env.get_observation());
        targets.push_back(env.get_target());
        env.step();
    }

    auto result = spark::verify_categorical_gradients_finite_difference(net, observations, targets, 1e-6f, 1e-4);

    std::cout << "  Categorical Gradient Check: Max Relative Error = "
              << result.max_relative_error << " on parameter "
              << result.param_name << "[" << result.param_index << "]"
              << std::endl;

    REQUIRE(result.passed, result.message.c_str());
    REQUIRE(result.max_relative_error < 1e-4, "Categorical gradient relative error exceeds tolerance 1e-4");

    std::cout << "[Test] test_categorical_gradient_finite_difference PASSED." << std::endl;
}

void test_env_lifecycle_and_reset() {
    std::cout << "[Test] Running test_env_lifecycle_and_reset..." << std::endl;

    const uint32_t ep_len = 15;
    spark::GridWorldEnv env(ep_len, 777ULL);

    REQUIRE(!env.is_done(), "Environment should not be done at step 0");
    REQUIRE(env.current_step() == 0, "Current step should be 0");

    for (uint32_t s = 0; s < ep_len; ++s) {
        REQUIRE(!env.is_done(), "Environment marked done prematurely");
        REQUIRE(env.current_step() == s, "Step counter mismatch");
        env.step();
    }

    REQUIRE(env.is_done(), "Environment should be done after episode length");
    REQUIRE(env.current_step() == ep_len, "Step counter should equal episode length");

    // Resetting environment
    env.reset();
    REQUIRE(!env.is_done(), "Environment should not be done after reset");
    REQUIRE(env.current_step() == 0, "Step counter should be reset to 0");

    std::cout << "[Test] test_env_lifecycle_and_reset PASSED." << std::endl;
}

int main() {
    std::cout << "=== Running SPARK Milestone 5A (Grid-World & Categorical Foundation) Tests ===" << std::endl;

    test_deterministic_transitions();
    test_boundary_collisions();
    test_obstacle_collisions();
    test_agent_placement_and_validation();
    test_reproducible_generation();
    test_layout_splits_and_data_integrity();
    test_input_and_target_encoding();
    test_softmax_and_categorical_loss();
    test_categorical_gradient_finite_difference();
    test_env_lifecycle_and_reset();

    std::cout << "All Milestone 5A tests passed successfully!" << std::endl;
    return 0;
}
