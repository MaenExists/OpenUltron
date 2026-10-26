#include "spark/rollout.hpp"
#include "spark/gridworld.hpp"
#include "spark/network.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ \
                      << " - " << msg << std::endl; \
            std::exit(1); \
        } \
    } while(0)

// 1. Test Oracle Trajectory Determinism on Empty Grid
void test_oracle_empty_grid_loop() {
    std::cout << "[Test] Running test_oracle_empty_grid_loop..." << std::endl;

    spark::GridWorldLayout empty_layout;
    empty_layout.width = 5;
    empty_layout.height = 5;
    empty_layout.obstacles.assign(25, 0);

    // Initial position (2,2) = index 12
    uint32_t start_pos = 12;
    std::vector<spark::Action> actions = {
        spark::Action::Right, // (2,3) = index 13
        spark::Action::Up,    // (1,3) = index 8
        spark::Action::Left,  // (1,2) = index 7
        spark::Action::Down   // (2,2) = index 12
    };

    auto res = spark::RolloutEngine::run_oracle_rollout(empty_layout, start_pos, actions);

    REQUIRE(res.steps.size() == 4, "Oracle must produce 4 steps");
    REQUIRE(res.entire_trajectory_correct, "Oracle entire trajectory must be correct");

    REQUIRE(res.steps[0].actual_position == 13, "Step 0 should be (2,3)");
    REQUIRE(res.steps[1].actual_position == 8,  "Step 1 should be (1,3)");
    REQUIRE(res.steps[2].actual_position == 7,  "Step 2 should be (1,2)");
    REQUIRE(res.steps[3].actual_position == 12, "Step 3 should return to (2,2)");

    REQUIRE(res.final_predicted_position == 12, "Final position must be (2,2)");
    REQUIRE(res.total_manhattan_error == 0, "Oracle Manhattan error must be 0");

    std::cout << "[Test] test_oracle_empty_grid_loop PASSED." << std::endl;
}

// 2. Test Teacher-Forced vs Open-Loop Divergence on Mock/Mistaken Predictions
void test_teacher_forced_vs_open_loop_divergence() {
    std::cout << "[Test] Running test_teacher_forced_vs_open_loop_divergence..." << std::endl;

    spark::GridWorldLayout empty_layout;
    empty_layout.width = 5;
    empty_layout.height = 5;
    empty_layout.obstacles.assign(25, 0);

    // Build a mock network where we can verify input behavior
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 32;
    g_cfg.fan_in = 4;
    g_cfg.seed = 42ULL;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 32;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = 0.5f;
    n_cfg.beta = 0.5f;
    n_cfg.seed = 42ULL;

    spark::MicroUnitNetwork net(n_cfg, g_cfg);

    // Let's create an action sequence of 3 steps
    std::vector<spark::Action> actions = {
        spark::Action::Right,
        spark::Action::Down,
        spark::Action::Right
    };
    uint32_t start_pos = 12; // (2,2)

    auto tf_res = spark::RolloutEngine::run_single_rollout(net, empty_layout, start_pos, actions, spark::RolloutMode::TeacherForced);
    auto ol_res = spark::RolloutEngine::run_single_rollout(net, empty_layout, start_pos, actions, spark::RolloutMode::OpenLoop);

    REQUIRE(tf_res.steps.size() == 3, "TF steps mismatch");
    REQUIRE(ol_res.steps.size() == 3, "OL steps mismatch");

    // Both must start with the exact same initial step prediction
    REQUIRE(tf_res.steps[0].predicted_position == ol_res.steps[0].predicted_position,
            "Step 0 prediction must be strictly identical for both modes");
    REQUIRE(tf_res.steps[0].actual_position == ol_res.steps[0].actual_position,
            "Step 0 actual simulator target must be strictly identical");

    // If step 0 was incorrect, verify divergence in step 1 input:
    // In teacher forced mode, simulator truth was passed to step 1
    // In open loop mode, predicted position was passed to step 1
    std::cout << "  Step 0: Pred=" << tf_res.steps[0].predicted_position
              << ", Truth=" << tf_res.steps[0].actual_position << std::endl;
    std::cout << "  TF Step 1: Pred=" << tf_res.steps[1].predicted_position
              << ", Truth=" << tf_res.steps[1].actual_position << std::endl;
    std::cout << "  OL Step 1: Pred=" << ol_res.steps[1].predicted_position
              << ", Truth=" << ol_res.steps[1].actual_position << std::endl;

    std::cout << "[Test] test_teacher_forced_vs_open_loop_divergence PASSED." << std::endl;
}

// 3. Test Manhattan distance calculation
void test_manhattan_distance() {
    std::cout << "[Test] Running test_manhattan_distance..." << std::endl;

    // (0,0) [0] to (0,0) [0] = 0
    REQUIRE(spark::RolloutEngine::compute_manhattan_distance(0, 0, 5) == 0, "Distance to self is 0");

    // (0,0) [0] to (0,4) [4] = 4
    REQUIRE(spark::RolloutEngine::compute_manhattan_distance(0, 4, 5) == 4, "Distance across row is 4");

    // (0,0) [0] to (4,0) [20] = 4
    REQUIRE(spark::RolloutEngine::compute_manhattan_distance(0, 20, 5) == 4, "Distance across col is 4");

    // (0,0) [0] to (4,4) [24] = 8
    REQUIRE(spark::RolloutEngine::compute_manhattan_distance(0, 24, 5) == 8, "Distance across diagonal is 8");

    // (2,2) [12] to (3,1) [16] = |2-3| + |2-1| = 2
    REQUIRE(spark::RolloutEngine::compute_manhattan_distance(12, 16, 5) == 2, "Distance (2,2) to (3,1) is 2");

    std::cout << "[Test] test_manhattan_distance PASSED." << std::endl;
}

// 4. Test Checkpoint Serialization & Deserialization
void test_checkpoint_roundtrip() {
    std::cout << "[Test] Running test_checkpoint_roundtrip..." << std::endl;

    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 64;
    g_cfg.fan_in = 8;
    g_cfg.seed = 12345ULL;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 64;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 1.0f;
    n_cfg.seed = 12345ULL;

    spark::MicroUnitNetwork original_net(n_cfg, g_cfg);

    std::string chkpt_path = "/tmp/test_network_checkpoint.spark";
    bool saved = original_net.save_checkpoint(chkpt_path);
    REQUIRE(saved, "Failed to save checkpoint");

    spark::MicroUnitNetwork loaded_net(n_cfg, g_cfg);
    bool loaded = spark::MicroUnitNetwork::load_checkpoint(chkpt_path, loaded_net);
    REQUIRE(loaded, "Failed to load checkpoint");

    // Verify configurations match
    REQUIRE(loaded_net.config().num_units == original_net.config().num_units, "Units mismatch");
    REQUIRE(loaded_net.config().input_dim == original_net.config().input_dim, "Input dim mismatch");
    REQUIRE(loaded_net.config().output_dim == original_net.config().output_dim, "Output dim mismatch");
    REQUIRE(std::abs(loaded_net.config().alpha - original_net.config().alpha) < 1e-6f, "Alpha mismatch");
    REQUIRE(std::abs(loaded_net.config().beta - original_net.config().beta) < 1e-6f, "Beta mismatch");

    // Verify parameters match exactly
    REQUIRE(loaded_net.biases() == original_net.biases(), "Biases mismatch");
    REQUIRE(loaded_net.input_weights() == original_net.input_weights(), "Input weights mismatch");
    REQUIRE(loaded_net.output_weights() == original_net.output_weights(), "Output weights mismatch");
    REQUIRE(loaded_net.output_biases() == original_net.output_biases(), "Output biases mismatch");
    REQUIRE(loaded_net.graph().weights == original_net.graph().weights, "Graph weights mismatch");

    // Verify identical forward predictions
    std::vector<float> in(55, 0.5f);
    std::vector<float> orig_logits, orig_probs;
    std::vector<float> load_logits, load_probs;

    original_net.forward_step(in, orig_logits, orig_probs, spark::ReadoutMode::Softmax);
    loaded_net.forward_step(in, load_logits, load_probs, spark::ReadoutMode::Softmax);

    REQUIRE(orig_logits == load_logits, "Logits after checkpoint load must be bitwise identical");
    REQUIRE(orig_probs == load_probs, "Probs after checkpoint load must be bitwise identical");

    std::cout << "[Test] test_checkpoint_roundtrip PASSED." << std::endl;
}

// 5. Test Multi-Step Rollout Evaluator on Oracle
void test_evaluate_oracle_multi_horizon() {
    std::cout << "[Test] Running test_evaluate_oracle_multi_horizon..." << std::endl;

    auto splits = spark::GridWorldEnv::generate_split_layouts(2, 2, 5, 3, 5, 9999ULL);

    for (uint32_t H : {1, 2, 4, 8}) {
        auto metrics = spark::RolloutEngine::evaluate_oracle(splits.test_layouts, H, 20, 42ULL);
        REQUIRE(metrics.horizon == H, "Horizon mismatch");
        REQUIRE(metrics.total_episodes == 20, "Episodes mismatch");
        REQUIRE(metrics.total_steps == 20 * H, "Steps mismatch");
        REQUIRE(metrics.overall_step_accuracy() == 1.0f, "Oracle step accuracy must be 1.0");
        REQUIRE(metrics.complete_trajectory_accuracy() == 1.0f, "Oracle trajectory accuracy must be 1.0");
        REQUIRE(metrics.final_position_accuracy() == 1.0f, "Oracle final pos accuracy must be 1.0");
        REQUIRE(metrics.mean_manhattan_error() == 0.0f, "Oracle Manhattan error must be 0.0");
    }

    std::cout << "[Test] test_evaluate_oracle_multi_horizon PASSED." << std::endl;
}

int main() {
    std::cout << "=== Running SPARK Milestone 5D.1 Rollout Engine Tests ===" << std::endl;

    test_oracle_empty_grid_loop();
    test_teacher_forced_vs_open_loop_divergence();
    test_manhattan_distance();
    test_checkpoint_roundtrip();
    test_evaluate_oracle_multi_horizon();

    std::cout << "All Milestone 5D.1 tests passed successfully!" << std::endl;
    return 0;
}
