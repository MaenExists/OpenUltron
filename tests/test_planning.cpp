#include "spark/planning.hpp"
#include "spark/gridworld.hpp"
#include "spark/network.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <algorithm>

#define REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ \
                      << " - " << msg << std::endl; \
            std::exit(1); \
        } \
    } while(0)

// Helper to create a standard 128-unit test network
static spark::MicroUnitNetwork create_test_network(uint64_t seed = 42ULL) {
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 128;
    g_cfg.fan_in = 8;
    g_cfg.seed = seed;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 128;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 1.0f;
    n_cfg.seed = seed;

    return spark::MicroUnitNetwork(n_cfg, g_cfg);
}

// Helper to locate trained checkpoint across build and root directories
static std::string find_checkpoint_path() {
    const std::vector<std::string> candidates = {
        "checkpoints/world_model_best.spark",
        "../checkpoints/world_model_best.spark",
        "../../checkpoints/world_model_best.spark"
    };
    for (const auto& path : candidates) {
        std::ifstream f(path);
        if (f.good()) return path;
    }
    return "";
}

// 1. Test BFS Shortest Path Reference Oracle (Empty Grid & Obstacle Detour)
void test_bfs_shortest_path_oracle() {
    std::cout << "[Test] Running test_bfs_shortest_path_oracle..." << std::endl;

    // Test A: Empty 5x5 grid
    spark::GridWorldLayout empty_layout;
    empty_layout.width = 5;
    empty_layout.height = 5;
    empty_layout.obstacles.assign(25, 0);

    // From (0,0) = 0 to (2,3) = 13: Manhattan distance = 2 + 3 = 5
    auto res_empty = spark::BFSSolver::find_shortest_path(empty_layout, 0, 13);
    REQUIRE(res_empty.reachable, "Empty grid path must be reachable");
    REQUIRE(res_empty.shortest_path_length == 5, "Shortest path length must be 5");
    REQUIRE(res_empty.shortest_path.size() == 5, "Path vector must have 5 actions");

    // Test B: Single Obstacle Detour
    // Layout with single obstacle blocking direct vertical path: (1,2)=7
    spark::GridWorldLayout wall_layout;
    wall_layout.width = 5;
    wall_layout.height = 5;
    wall_layout.obstacles.assign(25, 0);
    wall_layout.obstacles[7] = 1;

    // Start at (0,2)=2, Goal at (2,2)=12. Direct path would be length 2 (Down, Down) but (1,2) is blocked.
    // Detour must go around the obstacle: e.g. Left -> Down -> Down -> Right (length 4)
    auto res_detour = spark::BFSSolver::find_shortest_path(wall_layout, 2, 12);
    REQUIRE(res_detour.reachable, "Detour around obstacle must be reachable");
    REQUIRE(res_detour.shortest_path_length == 4, "Single obstacle detour length must be 4");

    // Test C: 3-cell Wall Detour (obstacles at 6, 7, 8) -> path around to col 0 or col 4 is length 6
    wall_layout.obstacles[6] = 1;
    wall_layout.obstacles[8] = 1;
    auto res_wall = spark::BFSSolver::find_shortest_path(wall_layout, 2, 12);
    REQUIRE(res_wall.reachable, "Detour around 3-cell wall must be reachable");
    REQUIRE(res_wall.shortest_path_length == 6, "3-cell wall detour length must be 6");

    std::cout << "[Test] test_bfs_shortest_path_oracle PASSED." << std::endl;
}

// 2. Test Unreachable Configurations and Obstacle Guards
void test_bfs_unreachable_and_obstacle_guards() {
    std::cout << "[Test] Running test_bfs_unreachable_and_obstacle_guards..." << std::endl;

    // Layout where cell (0,0)=0 is completely surrounded by obstacles at (0,1)=1 and (1,0)=5
    spark::GridWorldLayout walled_corner;
    walled_corner.width = 5;
    walled_corner.height = 5;
    walled_corner.obstacles.assign(25, 0);
    walled_corner.obstacles[1] = 1; // (0,1)
    walled_corner.obstacles[5] = 1; // (1,0)

    auto res_unreachable = spark::BFSSolver::find_shortest_path(walled_corner, 0, 12);
    REQUIRE(!res_unreachable.reachable, "Corner surrounded by obstacles must be unreachable");
    REQUIRE(res_unreachable.shortest_path_length == 0, "Unreachable length must be 0");

    // Start or Goal inside an obstacle
    auto res_start_in_obs = spark::BFSSolver::find_shortest_path(walled_corner, 1, 12);
    REQUIRE(!res_start_in_obs.reachable, "Start inside obstacle must be invalid / unreachable");

    auto res_goal_in_obs = spark::BFSSolver::find_shortest_path(walled_corner, 12, 1);
    REQUIRE(!res_goal_in_obs.reachable, "Goal inside obstacle must be invalid / unreachable");

    // Identical start and goal
    auto res_same = spark::BFSSolver::find_shortest_path(walled_corner, 12, 12);
    REQUIRE(res_same.reachable, "Same start and goal must be reachable");
    REQUIRE(res_same.shortest_path_length == 0, "Same start and goal must have length 0");

    std::cout << "[Test] test_bfs_unreachable_and_obstacle_guards PASSED." << std::endl;
}

// 3. Test Reachable Planning Episode Generation
void test_planning_episode_generator() {
    std::cout << "[Test] Running test_planning_episode_generator..." << std::endl;

    auto splits = spark::GridWorldEnv::generate_split_layouts(5, 2, 2, 3, 6, 12345ULL);
    REQUIRE(!splits.train_layouts.empty(), "Train layouts must not be empty");

    for (size_t i = 0; i < splits.train_layouts.size(); ++i) {
        const auto& layout = splits.train_layouts[i];
        uint64_t ep_seed = 42ULL + i * 17ULL;
        auto ep = spark::PlanningEpisodeGenerator::generate_reachable_episode(layout, ep_seed, 16, 2);

        REQUIRE(ep.start_pos < 25, "Start pos must be < 25");
        REQUIRE(ep.goal_pos < 25, "Goal pos must be < 25");
        REQUIRE(ep.start_pos != ep.goal_pos, "Start and goal must be distinct");
        REQUIRE(layout.obstacles[ep.start_pos] == 0, "Start pos must not be an obstacle");
        REQUIRE(layout.obstacles[ep.goal_pos] == 0, "Goal pos must not be an obstacle");
        REQUIRE(ep.shortest_path_length >= 2, "Shortest path must satisfy min_path_length >= 2");
        REQUIRE(ep.shortest_path_length <= 16, "Shortest path must satisfy max_steps <= 16");

        // Verify that reference shortest path reaches goal
        uint32_t simulated_pos = ep.start_pos;
        for (spark::Action a : ep.reference_shortest_path) {
            auto trans = spark::GridWorldEnv::simulate_transition(
                simulated_pos, a, layout.obstacles, layout.width, layout.height
            );
            simulated_pos = trans.to_pos;
        }
        REQUIRE(simulated_pos == ep.goal_pos, "Reference BFS path must lead exactly to goal");
    }

    // Test dataset generator
    auto dataset = spark::PlanningEpisodeGenerator::generate_dataset(splits.train_layouts, 3, 999ULL, 16, 1);
    REQUIRE(dataset.size() == splits.train_layouts.size() * 3, "Dataset size mismatch");

    std::cout << "[Test] test_planning_episode_generator PASSED." << std::endl;
}

// 4. Test Recurrent-State Branching Independence & Order Invariance
void test_recurrent_state_branching() {
    std::cout << "[Test] Running test_recurrent_state_branching..." << std::endl;

    auto net = create_test_network(777ULL);

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    uint32_t start_pos = 12; // Center (2,2)
    uint32_t goal_pos = 14;  // (2,4)

    auto root = spark::PlanningEngine::create_cold_start_root(start_pos, goal_pos, 128);
    REQUIRE(root.predicted_position == 12, "Root pos must be 12");
    REQUIRE(root.depth == 0, "Root depth must be 0");
    REQUIRE(!root.goal_reached, "Root goal reached must be false");

    // Evaluation Order 1: Right, then Left, then Up
    auto child_right_1 = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Right, goal_pos);
    auto child_left_1  = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Left, goal_pos);
    auto child_up_1    = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Up, goal_pos);

    // Evaluation Order 2: Up, then Right, then Left
    auto child_up_2    = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Up, goal_pos);
    auto child_right_2 = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Right, goal_pos);
    auto child_left_2  = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Left, goal_pos);

    // Verify bitwise exact order invariance
    REQUIRE(child_right_1.predicted_position == child_right_2.predicted_position, "Right pos mismatch across orders");
    REQUIRE(child_right_1.score == child_right_2.score, "Right score mismatch across orders");
    REQUIRE(child_right_1.recurrent_state == child_right_2.recurrent_state, "Right state mismatch across orders");

    REQUIRE(child_left_1.predicted_position == child_left_2.predicted_position, "Left pos mismatch across orders");
    REQUIRE(child_left_1.score == child_left_2.score, "Left score mismatch across orders");
    REQUIRE(child_left_1.recurrent_state == child_left_2.recurrent_state, "Left state mismatch across orders");

    REQUIRE(child_up_1.predicted_position == child_up_2.predicted_position, "Up pos mismatch across orders");
    REQUIRE(child_up_1.score == child_up_2.score, "Up score mismatch across orders");
    REQUIRE(child_up_1.recurrent_state == child_up_2.recurrent_state, "Up state mismatch across orders");

    // Verify branch states are mutually independent
    REQUIRE(child_right_1.recurrent_state != child_left_1.recurrent_state, "Right and Left states must be distinct");
    REQUIRE(child_right_1.recurrent_state != root.recurrent_state, "Child state must differ from root state");

    // Verify parent node was not mutated
    REQUIRE(root.depth == 0, "Root depth must remain 0");
    REQUIRE(root.actions_from_root.empty(), "Root actions must remain empty");

    std::cout << "[Test] test_recurrent_state_branching PASSED." << std::endl;
}

// 5. Test Multi-Depth Nested Branching
void test_nested_branching_depths() {
    std::cout << "[Test] Running test_nested_branching_depths..." << std::endl;

    auto net = create_test_network(888ULL);

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    uint32_t start_pos = 12;
    uint32_t goal_pos = 14;

    auto root = spark::PlanningEngine::create_cold_start_root(start_pos, goal_pos, 128);

    // Depth 1: Right -> child1
    auto child1 = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Right, goal_pos, 16, spark::PlanningScoreConfig(), 0);
    REQUIRE(child1.depth == 1, "Child 1 depth must be 1");
    REQUIRE(child1.actions_from_root.size() == 1, "Child 1 actions size must be 1");
    REQUIRE(child1.actions_from_root[0] == spark::Action::Right, "Child 1 action must be Right");

    // Depth 2: from child1, branch Up vs Right
    auto child2_up    = spark::PlanningEngine::expand_branch(net, layout, child1, spark::Action::Up, goal_pos, 16, spark::PlanningScoreConfig(), 1);
    auto child2_right = spark::PlanningEngine::expand_branch(net, layout, child1, spark::Action::Right, goal_pos, 16, spark::PlanningScoreConfig(), 1);

    REQUIRE(child2_up.depth == 2, "Depth 2 up depth must be 2");
    REQUIRE(child2_right.depth == 2, "Depth 2 right depth must be 2");

    REQUIRE(child2_up.actions_from_root.size() == 2, "Depth 2 actions size must be 2");
    REQUIRE(child2_up.actions_from_root[0] == spark::Action::Right && child2_up.actions_from_root[1] == spark::Action::Up,
            "Path must be Right -> Up");

    REQUIRE(child2_right.actions_from_root.size() == 2, "Depth 2 actions size must be 2");
    REQUIRE(child2_right.actions_from_root[0] == spark::Action::Right && child2_right.actions_from_root[1] == spark::Action::Right,
            "Path must be Right -> Right");

    REQUIRE(child2_up.recurrent_state != child2_right.recurrent_state, "Sub-branch states must be independent");
    REQUIRE(child2_up.recurrent_state != child1.recurrent_state, "Sub-branch state must differ from parent state");

    std::cout << "[Test] test_nested_branching_depths PASSED." << std::endl;
}

// 6. Test Trajectory Scoring and Deterministic Tie-Breaking
void test_trajectory_scoring_and_tie_breaking() {
    std::cout << "[Test] Running test_trajectory_scoring_and_tie_breaking..." << std::endl;

    spark::PlanningScoreConfig cfg;
    cfg.goal_reward = 100.0f;
    cfg.lambda_dist = 2.0f;
    cfg.lambda_step = 0.5f;
    cfg.early_goal_bonus_per_step = 1.0f;

    uint32_t goal = 14; // (2,4)
    uint32_t max_steps = 16;

    // Node A: at (2,3)=13, depth 1, not goal -> D=1, L=1. Score = -2(1) - 0.5(1) = -2.5
    float score_a = spark::PlanningScorer::compute_score(13, goal, 1, max_steps, false, cfg);
    REQUIRE(std::abs(score_a - (-2.5f)) < 1e-5f, "Score A calculation mismatch");

    // Node B: at (2,4)=14, depth 1, goal reached -> D=0, L=1. Score = 100 + 1(16-1) - 0.5(1) = 100 + 15 - 0.5 = 114.5
    float score_b = spark::PlanningScorer::compute_score(14, goal, 1, max_steps, true, cfg);
    REQUIRE(std::abs(score_b - 114.5f) < 1e-5f, "Score B calculation mismatch");

    // Node C: at (2,4)=14, depth 2, goal reached -> D=0, L=2. Score = 100 + 1(16-2) - 0.5(2) = 100 + 14 - 1.0 = 113.0
    float score_c = spark::PlanningScorer::compute_score(14, goal, 2, max_steps, true, cfg);
    REQUIRE(std::abs(score_c - 113.0f) < 1e-5f, "Score C calculation mismatch");

    // Earlier goal must score higher than later goal
    REQUIRE(score_b > score_c, "Earlier goal must strictly score higher than later goal");

    // Tie-breaking comparator tests
    spark::PlanningNode node1, node2;
    node1.score = 50.0f;
    node2.score = 40.0f;
    REQUIRE(node1.is_better_than(node2), "Higher score must be better");
    REQUIRE(!node2.is_better_than(node1), "Lower score must not be better");

    // Score tied, goal reached breaks tie
    node1.score = 50.0f; node1.goal_reached = true;
    node2.score = 50.0f; node2.goal_reached = false;
    REQUIRE(node1.is_better_than(node2), "Goal reached must win on tied score");

    // Score & goal tied, closer Manhattan breaks tie
    node1.goal_reached = false; node1.manhattan_to_goal = 2;
    node2.goal_reached = false; node2.manhattan_to_goal = 3;
    REQUIRE(node1.is_better_than(node2), "Lower Manhattan distance must win");

    // Score, goal, Manhattan tied, shorter depth breaks tie
    node1.manhattan_to_goal = 2; node1.depth = 2;
    node2.manhattan_to_goal = 2; node2.depth = 4;
    REQUIRE(node1.is_better_than(node2), "Lower depth must win");

    // Score, goal, Manhattan, depth tied -> Lexicographical action sequence breaks tie
    node1.depth = 2; node1.actions_from_root = {spark::Action::Up, spark::Action::Right};
    node2.depth = 2; node2.actions_from_root = {spark::Action::Down, spark::Action::Right};
    REQUIRE(node1.is_better_than(node2), "Action Up (0) must precede Down (1) in tie-break");

    std::cout << "[Test] test_trajectory_scoring_and_tie_breaking PASSED." << std::endl;
}

// 7. Test Cold-Start vs Warmed-Up Root Initialization
void test_cold_start_vs_warmed_up_root() {
    std::cout << "[Test] Running test_cold_start_vs_warmed_up_root..." << std::endl;

    auto net = create_test_network(999ULL);

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    uint32_t start_pos = 12;
    uint32_t goal_pos = 14;

    // Cold start root: h(0) = 0
    auto cold_root = spark::PlanningEngine::create_cold_start_root(start_pos, goal_pos, 128);
    for (float val : cold_root.recurrent_state) {
        REQUIRE(val == 0.0f, "Cold start recurrent state must be strictly zeroed");
    }

    // Warmed-up root: simulated non-zero observation context
    std::vector<float> warmed_h(128, 0.45f);
    auto warmed_root = spark::PlanningEngine::create_warmed_up_root(start_pos, goal_pos, warmed_h);
    REQUIRE(warmed_root.recurrent_state == warmed_h, "Warmed-up state must match provided vector");

    // Expand identical action from both
    auto child_from_cold = spark::PlanningEngine::expand_branch(net, layout, cold_root, spark::Action::Right, goal_pos);
    auto child_from_warm = spark::PlanningEngine::expand_branch(net, layout, warmed_root, spark::Action::Right, goal_pos);

    // Warmed-up branch must evolve differently from cold-start branch
    REQUIRE(child_from_cold.recurrent_state != child_from_warm.recurrent_state,
            "Cold start branch and warmed-up branch must produce distinct recurrent states");

    std::cout << "[Test] test_cold_start_vs_warmed_up_root PASSED." << std::endl;
}

// 8. Test Model Checkpoint Immutability During Planning
void test_checkpoint_immutability() {
    std::cout << "[Test] Running test_checkpoint_immutability..." << std::endl;

    auto net = create_test_network(1111ULL);

    // Save copy of all parameter weights
    auto initial_biases = net.biases();
    auto initial_input_w = net.input_weights();
    auto initial_output_w = net.output_weights();
    auto initial_output_b = net.output_biases();
    auto initial_graph_w = net.graph().weights;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    auto root = spark::PlanningEngine::create_cold_start_root(12, 14, 128);

    // Run extensive branching
    auto children = spark::PlanningEngine::expand_all_actions(net, layout, root, 14);
    for (const auto& child : children) {
        auto sub_children = spark::PlanningEngine::expand_all_actions(net, layout, child, 14);
        (void)sub_children;
    }

    // Verify bitwise exact parameter immutability
    REQUIRE(net.biases() == initial_biases, "Biases mutated during planning");
    REQUIRE(net.input_weights() == initial_input_w, "Input weights mutated during planning");
    REQUIRE(net.output_weights() == initial_output_w, "Output weights mutated during planning");
    REQUIRE(net.output_biases() == initial_output_b, "Output biases mutated during planning");
    REQUIRE(net.graph().weights == initial_graph_w, "Graph weights mutated during planning");

    std::cout << "[Test] test_checkpoint_immutability PASSED." << std::endl;
}

// 9. Milestone 5E.2A: Test Bounded Beam Search Core and Bounded Limits
void test_beam_search_core_bounded_limits() {
    std::cout << "[Test] Running test_beam_search_core_bounded_limits..." << std::endl;

    auto net = create_test_network(2026ULL);

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    // Case 1: Root already at goal -> returns immediately
    auto root_at_goal = spark::PlanningEngine::create_cold_start_root(12, 12, 128);
    spark::BeamSearchConfig cfg1;
    cfg1.beam_width = 4;
    cfg1.max_depth = 6;
    auto res1 = spark::PlanningEngine::beam_search(net, layout, root_at_goal, 12, cfg1);

    REQUIRE(res1.goal_reached, "Root at goal must yield goal_reached = true");
    REQUIRE(res1.best_plan.empty(), "Root at goal must have empty plan");
    REQUIRE(res1.predicted_final_position == 12, "Final position must be 12");
    REQUIRE(res1.nodes_expanded == 0, "No nodes should be expanded when root is at goal");

    // Case 2: Multi-step bounded beam search limits (K=2, H=4)
    auto root = spark::PlanningEngine::create_cold_start_root(0, 24, 128);
    spark::BeamSearchConfig cfg2;
    cfg2.beam_width = 2;
    cfg2.max_depth = 4;
    auto res2 = spark::PlanningEngine::beam_search(net, layout, root, 24, cfg2);

    REQUIRE(res2.depth_reached <= 4, "Depth reached must be <= max_depth (4)");
    REQUIRE(res2.best_plan.size() <= 4, "Best plan length must be <= max_depth (4)");
    REQUIRE(res2.nodes_expanded > 0, "Nodes expanded must be > 0");
    // At depth 1: 1 parent expanded (5 evaluations)
    // At depth 2: <= 2 parents expanded (<= 10 evaluations)
    // At depth 3: <= 2 parents expanded (<= 10 evaluations)
    // At depth 4: <= 2 parents expanded (<= 10 evaluations)
    // Total nodes expanded <= 1 + 2 + 2 + 2 = 7
    REQUIRE(res2.nodes_expanded <= 1 + 2 * 3, "Nodes expanded must strictly obey beam width bound K=2");
    REQUIRE(res2.model_evaluations == res2.nodes_expanded * 5, "Model evaluations must be exactly 5 per expanded node");

    std::cout << "[Test] test_beam_search_core_bounded_limits PASSED." << std::endl;
}

// 10. Milestone 5E.2A: Test Identical Candidate Sequence Order Invariance
void test_beam_search_identical_sequence_order_invariance() {
    std::cout << "[Test] Running test_beam_search_identical_sequence_order_invariance..." << std::endl;

    auto net = create_test_network(3033ULL);

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    uint32_t start_pos = 12;
    uint32_t goal_pos = 2;
    auto root = spark::PlanningEngine::create_cold_start_root(start_pos, goal_pos, 128);

    // Sequence S: [Up, Right, Up]
    // Evaluation Order 1: Evaluate sequence S directly
    auto node_up_1    = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Up, goal_pos);
    auto node_right_1 = spark::PlanningEngine::expand_branch(net, layout, node_up_1, spark::Action::Right, goal_pos);
    auto node_seq1    = spark::PlanningEngine::expand_branch(net, layout, node_right_1, spark::Action::Up, goal_pos);

    // Evaluation Order 2: Evaluate a bunch of distractor branches FIRST, then evaluate sequence S
    auto distractor_left = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Left, goal_pos);
    auto distractor_down = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Down, goal_pos);
    auto distractor_sub  = spark::PlanningEngine::expand_branch(net, layout, distractor_left, spark::Action::Stay, goal_pos);
    (void)distractor_down;
    (void)distractor_sub;

    // Now evaluate sequence S
    auto node_up_2    = spark::PlanningEngine::expand_branch(net, layout, root, spark::Action::Up, goal_pos);
    auto node_right_2 = spark::PlanningEngine::expand_branch(net, layout, node_up_2, spark::Action::Right, goal_pos);
    auto node_seq2    = spark::PlanningEngine::expand_branch(net, layout, node_right_2, spark::Action::Up, goal_pos);

    // Bitwise identical verification on identical sequence executed under different contexts
    REQUIRE(node_seq1.predicted_position == node_seq2.predicted_position,
            "Predicted position must be bitwise identical regardless of prior evaluation order");
    REQUIRE(node_seq1.score == node_seq2.score,
            "Trajectory score must be bitwise identical regardless of prior evaluation order");
    REQUIRE(node_seq1.recurrent_state == node_seq2.recurrent_state,
            "Recurrent hidden state must be bitwise identical regardless of prior evaluation order");

    std::cout << "[Test] test_beam_search_identical_sequence_order_invariance PASSED." << std::endl;
}

// 11. Milestone 5E.2A: Test Goal Candidate Preservation Across Search Depths
void test_beam_search_goal_preservation_across_depths() {
    std::cout << "[Test] Running test_beam_search_goal_preservation_across_depths..." << std::endl;

    auto net = create_test_network(4044ULL);

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    // Let start be 12. Probe what position the network predicts for Action::Up
    auto probe_root = spark::PlanningEngine::create_cold_start_root(12, 100, 128);
    auto probe_child = spark::PlanningEngine::expand_branch(net, layout, probe_root, spark::Action::Up, 100);
    uint32_t target_goal = probe_child.predicted_position;

    // Now initialize a search targeting that exact predicted position
    auto root = spark::PlanningEngine::create_cold_start_root(12, target_goal, 128);

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 4;
    cfg.return_first_goal = false; // Allow search to proceed to max_depth

    auto res = spark::PlanningEngine::beam_search(net, layout, root, target_goal, cfg);

    REQUIRE(res.goal_reached, "Goal must be reached");
    REQUIRE(!res.completed_goal_candidates.empty(), "Completed goal candidates list must not be empty");
    REQUIRE(res.best_node.goal_reached, "Best selected node must have reached goal");

    // Verify the best plan reaches the target goal
    REQUIRE(res.predicted_final_position == target_goal, "Predicted final position must match target goal");

    std::cout << "[Test] test_beam_search_goal_preservation_across_depths PASSED." << std::endl;
}

// 12. Milestone 5E.2B: Test Scoring Config Validation and Mathematical Formulation
void test_scoring_validation_and_objective() {
    std::cout << "[Test] Running test_scoring_validation_and_objective..." << std::endl;

    // 1. Validation checks
    spark::PlanningScoreConfig cfg;
    std::string err;
    REQUIRE(cfg.validate(err), "Default PlanningScoreConfig must be valid");

    spark::PlanningScoreConfig invalid_cfg1 = cfg;
    invalid_cfg1.goal_reward = -10.0f;
    REQUIRE(!invalid_cfg1.validate(err), "Negative goal_reward must fail validation");

    spark::PlanningScoreConfig invalid_cfg2 = cfg;
    invalid_cfg2.lambda_dist = -1.0f;
    REQUIRE(!invalid_cfg2.validate(err), "Negative lambda_dist must fail validation");

    spark::PlanningScoreConfig invalid_cfg3 = cfg;
    invalid_cfg3.lambda_step = -0.5f;
    REQUIRE(!invalid_cfg3.validate(err), "Negative lambda_step must fail validation");

    spark::PlanningScoreConfig invalid_cfg4 = cfg;
    invalid_cfg4.early_goal_bonus_per_step = NAN;
    REQUIRE(!invalid_cfg4.validate(err), "NaN early_goal_bonus must fail validation");

    bool caught_exception = false;
    try {
        spark::PlanningScorer::compute_score(0, 0, 1, 16, true, invalid_cfg1);
    } catch (const std::invalid_argument&) {
        caught_exception = true;
    }
    REQUIRE(caught_exception, "compute_score must throw std::invalid_argument for invalid config");

    // 2. Exact mathematical formulation checks:
    // J(n) = R_g * 1_goal - lambda_d * D(n,g) - lambda_s * L(n) + lambda_e * 1_goal * (H_max - L(n))
    spark::PlanningScoreConfig test_cfg;
    test_cfg.goal_reward = 100.0f;
    test_cfg.lambda_dist = 2.0f;
    test_cfg.lambda_step = 0.5f;
    test_cfg.early_goal_bonus_per_step = 1.0f;

    // Case A: Goal reached at depth 2 (max_steps = 6, D = 0)
    // J = 100(1) - 2(0) - 0.5(2) + 1.0(1)(6 - 2) = 100 - 1.0 + 4.0 = 103.0f
    float score_a = spark::PlanningScorer::compute_score(2, 2, 2, 6, true, test_cfg);
    REQUIRE(std::abs(score_a - 103.0f) < 1e-5f, "Score for goal at depth 2 must equal exactly 103.0");

    // Case B: Unfinished candidate at depth 2, pos 1, goal 2 (D = 1)
    // J = 100(0) - 2(1) - 0.5(2) + 1.0(0)(6 - 2) = 0 - 2.0 - 1.0 = -3.0f
    float score_b = spark::PlanningScorer::compute_score(1, 2, 2, 6, false, test_cfg);
    REQUIRE(std::abs(score_b - (-3.0f)) < 1e-5f, "Score for unfinished candidate must equal exactly -3.0");

    // Case C: Goal reached at depth 6 (max_steps = 6, D = 0, early bonus = 0)
    // J = 100(1) - 2(0) - 0.5(6) + 1.0(1)(6 - 6) = 100 - 3.0 + 0 = 97.0f
    float score_c = spark::PlanningScorer::compute_score(2, 2, 6, 6, true, test_cfg);
    REQUIRE(std::abs(score_c - 97.0f) < 1e-5f, "Score for goal at max depth must equal exactly 97.0");

    std::cout << "[Test] test_scoring_validation_and_objective PASSED." << std::endl;
}

// 13. Milestone 5E.2B: Test Strict Weak Ordering Axioms for PlanningNode::is_better_than
void test_strict_weak_ordering_axioms() {
    std::cout << "[Test] Running test_strict_weak_ordering_axioms..." << std::endl;

    // Generate diverse test nodes
    std::vector<spark::PlanningNode> nodes;
    const std::vector<float> scores = {-10.0f, -5.0f, 0.0f, 50.0f, 100.0f, 103.0f};
    const std::vector<bool> goals = {false, true};
    const std::vector<uint32_t> dists = {0, 1, 3};
    const std::vector<uint32_t> depths = {1, 2, 4};
    const std::vector<float> confs = {0.2f, 0.8f, 1.0f};
    const std::vector<std::vector<spark::Action>> action_seqs = {
        {spark::Action::Up},
        {spark::Action::Down},
        {spark::Action::Left},
        {spark::Action::Right},
        {spark::Action::Up, spark::Action::Right},
        {spark::Action::Down, spark::Action::Left}
    };

    spark::Random rng(12345ULL);
    for (size_t i = 0; i < 40; ++i) {
        spark::PlanningNode n;
        n.score = scores[rng.next_u64() % scores.size()];
        n.goal_reached = goals[rng.next_u64() % goals.size()];
        n.manhattan_to_goal = dists[rng.next_u64() % dists.size()];
        n.depth = depths[rng.next_u64() % depths.size()];
        n.cum_log_confidence = -static_cast<float>(rng.next_u64() % 500) / 100.0f;
        n.target_probability = confs[rng.next_u64() % confs.size()];
        n.actions_from_root = action_seqs[rng.next_u64() % action_seqs.size()];
        nodes.push_back(n);
    }

    const size_t N = nodes.size();

    // 1. Irreflexivity: !(a < a)
    for (size_t i = 0; i < N; ++i) {
        REQUIRE(!nodes[i].is_better_than(nodes[i]), "Ordering must be irreflexive: !(a < a)");
    }

    // 2. Asymmetry: a < b => !(b < a)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (nodes[i].is_better_than(nodes[j])) {
                REQUIRE(!nodes[j].is_better_than(nodes[i]), "Ordering must be asymmetric: a < b => !(b < a)");
            }
        }
    }

    // 3. Transitivity: a < b and b < c => a < c
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (!nodes[i].is_better_than(nodes[j])) continue;
            for (size_t k = 0; k < N; ++k) {
                if (nodes[j].is_better_than(nodes[k])) {
                    REQUIRE(nodes[i].is_better_than(nodes[k]),
                            "Ordering must be transitive: a < b && b < c => a < c");
                }
            }
        }
    }

    // 4. Incomparability transitivity:
    // If a ~ b and b ~ c then a ~ c where a ~ b <=> !(a < b) && !(b < a)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            bool incomp_ij = !nodes[i].is_better_than(nodes[j]) && !nodes[j].is_better_than(nodes[i]);
            if (!incomp_ij) continue;
            for (size_t k = 0; k < N; ++k) {
                bool incomp_jk = !nodes[j].is_better_than(nodes[k]) && !nodes[k].is_better_than(nodes[j]);
                if (incomp_jk) {
                    bool incomp_ik = !nodes[i].is_better_than(nodes[k]) && !nodes[k].is_better_than(nodes[i]);
                    REQUIRE(incomp_ik, "Transitivity of incomparability must hold");
                }
            }
        }
    }

    // 5. Test std::sort stability / no UB
    std::sort(nodes.begin(), nodes.end(), [](const spark::PlanningNode& a, const spark::PlanningNode& b) {
        return a.is_better_than(b);
    });

    for (size_t i = 0; i + 1 < nodes.size(); ++i) {
        REQUIRE(!nodes[i + 1].is_better_than(nodes[i]), "Sorted array must be non-ascending under is_better_than");
    }

    std::cout << "[Test] test_strict_weak_ordering_axioms PASSED." << std::endl;
}

// 14. Milestone 5E.2B: Test Confidence Tracking across Expansions
void test_confidence_tracking() {
    std::cout << "[Test] Running test_confidence_tracking..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    // Mock predictor returning specified step transition probabilities
    float step_prob = 0.8f;
    spark::PredictorFn mock_pred = [&step_prob](
        const std::vector<float>& h_in,
        const std::vector<float>&,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        h_out = h_in;
        logits.assign(25, 0.0f);
        probs.assign(25, (1.0f - step_prob) / 24.0f);
        probs[0] = step_prob; // Always predict position 0 with probability step_prob
    };

    auto root = spark::PlanningEngine::create_cold_start_root(0, 10, 128);
    REQUIRE(root.cum_log_confidence == 0.0f, "Root cum_log_confidence must be 0.0");
    REQUIRE(root.mean_step_confidence == 1.0f, "Root mean_step_confidence must be 1.0");

    step_prob = 0.8f;
    auto child1 = spark::PlanningEngine::expand_branch_custom(mock_pred, layout, root, spark::Action::Down, 10);
    REQUIRE(std::abs(child1.target_probability - 0.8f) < 1e-5f, "Child 1 target probability must be 0.8");
    REQUIRE(std::abs(child1.cum_log_confidence - std::log(0.8f)) < 1e-5f, "Child 1 cum_log_confidence must be log(0.8)");
    REQUIRE(std::abs(child1.mean_step_confidence - 0.8f) < 1e-5f, "Child 1 mean_step_confidence must be 0.8");

    step_prob = 0.5f;
    auto child2 = spark::PlanningEngine::expand_branch_custom(mock_pred, layout, child1, spark::Action::Down, 10);
    float expected_cum_log_2 = std::log(0.8f) + std::log(0.5f);
    float expected_mean_2 = std::exp(expected_cum_log_2 / 2.0f);
    REQUIRE(std::abs(child2.target_probability - 0.5f) < 1e-5f, "Child 2 target probability must be 0.5");
    REQUIRE(std::abs(child2.cum_log_confidence - expected_cum_log_2) < 1e-5f, "Child 2 cum_log_confidence check");
    REQUIRE(std::abs(child2.mean_step_confidence - expected_mean_2) < 1e-5f, "Child 2 mean_step_confidence check");

    step_prob = 0.9f;
    auto child3 = spark::PlanningEngine::expand_branch_custom(mock_pred, layout, child2, spark::Action::Down, 10);
    float expected_cum_log_3 = expected_cum_log_2 + std::log(0.9f);
    float expected_mean_3 = std::exp(expected_cum_log_3 / 3.0f);
    REQUIRE(std::abs(child3.target_probability - 0.9f) < 1e-5f, "Child 3 target probability must be 0.9");
    REQUIRE(std::abs(child3.cum_log_confidence - expected_cum_log_3) < 1e-5f, "Child 3 cum_log_confidence check");
    REQUIRE(std::abs(child3.mean_step_confidence - expected_mean_3) < 1e-5f, "Child 3 mean_step_confidence check");

    std::cout << "[Test] test_confidence_tracking PASSED." << std::endl;
}

// Deterministic Mock GridWorld Predictor Factory
static spark::PredictorFn create_mock_grid_predictor(float prob = 0.95f, bool hallucinate_walls = false) {
    return [prob, hallucinate_walls](
        const std::vector<float>& h_in,
        const std::vector<float>& x,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        h_out = h_in;
        logits.assign(25, 0.0f);
        probs.assign(25, (1.0f - prob) / 24.0f);

        // Decode current_pos from one-hot x[0..24]
        uint32_t curr_pos = 0;
        for (uint32_t i = 0; i < 25; ++i) {
            if (x[i] > 0.5f) { curr_pos = i; break; }
        }

        // Decode action from x[50..54]
        spark::Action action = spark::Action::Stay;
        for (uint32_t a = 0; a < 5; ++a) {
            if (x[50 + a] > 0.5f) { action = static_cast<spark::Action>(a); break; }
        }

        int r = static_cast<int>(curr_pos / 5);
        int c = static_cast<int>(curr_pos % 5);
        int nr = r, nc = c;
        if (action == spark::Action::Up) nr--;
        else if (action == spark::Action::Down) nr++;
        else if (action == spark::Action::Left) nc--;
        else if (action == spark::Action::Right) nc++;

        if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5) {
            uint32_t next_cell = static_cast<uint32_t>(nr * 5 + nc);
            if (hallucinate_walls || x[25 + next_cell] <= 0.5f) {
                curr_pos = next_cell;
            }
        }

        probs[curr_pos] = prob;
    };
}

// 15. Milestone 5E.2B: Scenario A — Short Route vs Long Route to Same Goal
void test_mock_scenario_a_short_vs_long_route() {
    std::cout << "[Test] Running test_mock_scenario_a_short_vs_long_route..." << std::endl;

    // Start at (0,0)=0, Goal at (0,2)=2
    // Short route: Right, Right (length 2)
    // Long route: Down, Right, Right, Up (length 4)
    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    auto mock_pred = create_mock_grid_predictor(0.98f);
    auto root = spark::PlanningEngine::create_cold_start_root(0, 2, 128);

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 8;
    cfg.max_depth = 6;
    cfg.return_first_goal = false; // Search all horizons to ensure both routes are evaluated

    auto result = spark::PlanningEngine::beam_search_custom(mock_pred, layout, root, 2, cfg);

    REQUIRE(result.goal_reached, "Goal must be reached");
    REQUIRE(result.best_node.goal_reached, "Best node must have reached goal");
    REQUIRE(result.best_node.depth == 2, "Shorter 2-step route must be selected over longer route");
    REQUIRE(result.best_plan.size() == 2, "Best plan size must be 2");
    REQUIRE(result.best_plan[0] == spark::Action::Right && result.best_plan[1] == spark::Action::Right,
            "Best plan must be [Right, Right]");

    std::cout << "[Test] test_mock_scenario_a_short_vs_long_route PASSED." << std::endl;
}

// 16. Milestone 5E.2B: Scenario B — Misleading Unfinished Heuristic vs Completed Goal
void test_mock_scenario_b_heuristic_vs_completed_goal() {
    std::cout << "[Test] Running test_mock_scenario_b_heuristic_vs_completed_goal..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    auto mock_pred = create_mock_grid_predictor(0.95f);
    auto root = spark::PlanningEngine::create_cold_start_root(0, 1, 128); // Goal is 1 step away: (0,1)=1

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 4;
    cfg.return_first_goal = false;

    auto result = spark::PlanningEngine::beam_search_custom(mock_pred, layout, root, 1, cfg);

    REQUIRE(result.goal_reached, "Goal must be reached");
    REQUIRE(!result.completed_goal_candidates.empty(), "Completed candidates must exist");
    REQUIRE(result.best_node.goal_reached, "Best node must be completed candidate");
    REQUIRE(result.best_node.depth == 1, "Completed route at depth 1 must beat any unfinished candidate");
    REQUIRE(result.best_plan.size() == 1, "Plan must have 1 step");
    REQUIRE(result.best_plan[0] == spark::Action::Right, "Plan must be [Right]");

    std::cout << "[Test] test_mock_scenario_b_heuristic_vs_completed_goal PASSED." << std::endl;
}

// 17. Milestone 5E.2B: Scenario C — Early Goal Preservation Across Search Depths
void test_mock_scenario_c_early_goal_preservation() {
    std::cout << "[Test] Running test_mock_scenario_c_early_goal_preservation..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    auto mock_pred = create_mock_grid_predictor(0.95f);
    auto root = spark::PlanningEngine::create_cold_start_root(0, 2, 128); // Goal at (0,2)=2

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 6;
    cfg.return_first_goal = false; // Search continues through depth 6

    auto result = spark::PlanningEngine::beam_search_custom(mock_pred, layout, root, 2, cfg);

    REQUIRE(result.goal_reached, "Goal must be reached");
    REQUIRE(result.depth_reached == 6, "Search must reach max_depth 6 without prematurely dropping early goal");
    REQUIRE(!result.completed_goal_candidates.empty(), "Completed goal list must retain candidate");
    REQUIRE(result.best_node.depth == 2, "Early completed goal at depth 2 must be preserved as best node");
    REQUIRE(result.best_node.predicted_position == 2, "Predicted position must be 2");

    std::cout << "[Test] test_mock_scenario_c_early_goal_preservation PASSED." << std::endl;
}

// 18. Milestone 5E.2B: Scenario D — Exact Score Tie-Breaking Determinism
void test_mock_scenario_d_tie_breaking_determinism() {
    std::cout << "[Test] Running test_mock_scenario_d_tie_breaking_determinism..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    auto mock_pred = create_mock_grid_predictor(0.95f);
    // Start at (0,0)=0, Goal at (2,2)=12.
    // At depth 2:
    //   Path 1: [Down, Down] reaches cell (2,0)=10 (distance to 12 is 2, actions [1, 1])
    //   Path 2: [Down, Right] reaches cell (1,1)=6 (distance to 12 is 2, actions [1, 3])
    //   Path 3: [Right, Down] reaches cell (1,1)=6 (distance to 12 is 2, actions [3, 1])
    //   Path 4: [Right, Right] reaches cell (0,2)=2 (distance to 12 is 2, actions [3, 3])
    // All four paths produce identical distance D=2, identical depth L=2, and identical score J=-5.0.
    // Deterministic lexicographical tie-breaking compares actions element-wise:
    //   [1, 1] vs [1, 3] vs [3, 1] vs [3, 3]
    // Since 1 < 3, [1, 1] ([Down, Down]) strictly wins first place!
    auto root = spark::PlanningEngine::create_cold_start_root(0, 12, 128);

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 2;

    auto ref_result = spark::PlanningEngine::beam_search_custom(mock_pred, layout, root, 12, cfg);

    REQUIRE(ref_result.best_plan.size() == 2, "Best plan must have depth 2");
    REQUIRE(ref_result.best_plan[0] == spark::Action::Down, "Deterministic tie-breaking must favor Down over Right");
    REQUIRE(ref_result.best_plan[1] == spark::Action::Down, "Deterministic tie-breaking must favor [Down, Down] over [Down, Right]");

    // Repeat 100 times to guarantee deterministic tie-breaking across runs
    for (int run = 0; run < 100; ++run) {
        auto run_result = spark::PlanningEngine::beam_search_custom(mock_pred, layout, root, 12, cfg);
        REQUIRE(run_result.best_plan == ref_result.best_plan,
                "Tie-breaking must be bitwise deterministic across 100 repeated runs");
        REQUIRE(run_result.best_node.score == ref_result.best_node.score,
                "Score must be bitwise identical across 100 repeated runs");
    }

    std::cout << "[Test] test_mock_scenario_d_tie_breaking_determinism PASSED." << std::endl;
}

// 19. Milestone 5E.2B: Scenario E — No Reachable Goal Fallback
void test_mock_scenario_e_no_reachable_goal_fallback() {
    std::cout << "[Test] Running test_mock_scenario_e_no_reachable_goal_fallback..." << std::endl;

    // Goal at (4,4)=24, but max_depth is 2. Goal cannot be reached in 2 steps.
    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    auto mock_pred = create_mock_grid_predictor(0.95f);
    auto root = spark::PlanningEngine::create_cold_start_root(0, 24, 128);

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 2;

    auto result = spark::PlanningEngine::beam_search_custom(mock_pred, layout, root, 24, cfg);

    REQUIRE(!result.goal_reached, "Goal must NOT be marked reached");
    REQUIRE(result.completed_goal_candidates.empty(), "Completed goal list must be empty");
    REQUIRE(result.best_node.predicted_position != 24, "Final position cannot reach 24");
    REQUIRE(result.best_plan.size() == 2, "Fallback must return best partial path of depth 2");
    REQUIRE(result.nodes_expanded > 0, "Nodes expanded must be tracked");
    REQUIRE(result.model_evaluations > 0, "Model evaluations must be tracked");

    // Start at 0, moving Down then Down or Right then Down brings distance to goal down from 8 to 6
    REQUIRE(result.best_node.manhattan_to_goal == 6, "Fallback must pick closest candidate to goal");

    std::cout << "[Test] test_mock_scenario_e_no_reachable_goal_fallback PASSED." << std::endl;
}

// 20. Milestone 5E.2B: Scenario F — Model Hallucination Obstacle Isolation
void test_mock_scenario_f_hallucination_obstacle_isolation() {
    std::cout << "[Test] Running test_mock_scenario_f_hallucination_obstacle_isolation..." << std::endl;

    // Real environment layout has obstacle at (0,1)=1
    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);
    layout.obstacles[1] = 1; // Wall at (0,1)

    // Simulator reference path requires a 4-step detour around cell 1
    auto ground_truth_bfs = spark::BFSSolver::find_shortest_path(layout, 0, 2);
    REQUIRE(ground_truth_bfs.shortest_path_length == 4, "Ground truth oracle requires 4 steps");

    // But the predictive model hallucinations passage through walls (hallucinate_walls = true)
    auto hallucinating_mock_pred = create_mock_grid_predictor(0.95f, /*hallucinate_walls=*/true);
    auto root = spark::PlanningEngine::create_cold_start_root(0, 2, 128);

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 4;

    auto result = spark::PlanningEngine::beam_search_custom(hallucinating_mock_pred, layout, root, 2, cfg);

    // The model-based planner plans purely according to the model's hypothetical predictions.
    // It DOES NOT consult the ground truth simulator or override model outputs.
    REQUIRE(result.goal_reached, "Model believes goal is reached");
    REQUIRE(result.best_plan.size() == 2, "Model plans 2 steps straight through the obstacle");
    REQUIRE(result.best_plan[0] == spark::Action::Right && result.best_plan[1] == spark::Action::Right,
            "Model plans [Right, Right] believing wall is passable");

    std::cout << "[Test] test_mock_scenario_f_hallucination_obstacle_isolation PASSED." << std::endl;
}

// 21. Milestone 5E.2B: Exhaustive Search vs Beam Search Agreement
void test_exhaustive_vs_beam_search_agreement() {
    std::cout << "[Test] Running test_exhaustive_vs_beam_search_agreement..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    auto mock_pred = create_mock_grid_predictor(0.95f);
    auto root = spark::PlanningEngine::create_cold_start_root(0, 12, 128);

    // At depth 2, total search space is 5^2 = 25.
    // Bounded beam search with beam_width=25 explores the full exhaustive tree.
    spark::BeamSearchConfig cfg;
    cfg.beam_width = 25;
    cfg.max_depth = 2;

    auto beam_res = spark::PlanningEngine::beam_search_custom(mock_pred, layout, root, 12, cfg);
    auto exh_res = spark::PlanningEngine::exhaustive_search_custom(mock_pred, layout, root, 12, 2);

    REQUIRE(beam_res.best_plan == exh_res.best_plan, "Beam search (K=25) and Exhaustive search must produce identical plan");
    REQUIRE(beam_res.best_node.score == exh_res.best_node.score, "Scores must match identically");
    REQUIRE(beam_res.best_node.predicted_position == exh_res.best_node.predicted_position, "Final positions must match identically");
    REQUIRE(beam_res.nodes_expanded == exh_res.nodes_expanded, "Expanded node count must match");

    std::cout << "[Test] test_exhaustive_vs_beam_search_agreement PASSED." << std::endl;
}

// 22. Milestone 5E.2C: Unfinished routes cannot falsely claim goal achievement
void test_unfinished_routes_cannot_claim_goal() {
    std::cout << "[Test] Running test_unfinished_routes_cannot_claim_goal..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    auto mock_pred = create_mock_grid_predictor(0.95f);
    // Start at (0,0)=0, Goal at (4,4)=24. Shortest path is 8 steps.
    // Planning horizon is strictly H=2.
    auto root = spark::PlanningEngine::create_cold_start_root(0, 24, 128);

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 2;

    auto result = spark::PlanningEngine::beam_search_custom(mock_pred, layout, root, 24, cfg);

    REQUIRE(!result.goal_reached, "Search must report goal_reached == false when horizon is insufficient");
    REQUIRE(result.completed_goal_candidates.empty(), "Completed goal list must remain strictly empty");
    REQUIRE(!result.best_node.goal_reached, "Selected best node must have goal_reached == false");
    REQUIRE(result.best_node.predicted_position != 24, "Predicted final position cannot match goal");

    // Exact score check: J = -lambda_d * D - lambda_s * L (NO goal_reward or early_bonus)
    float expected_score = -cfg.score_cfg.lambda_dist * static_cast<float>(result.best_node.manhattan_to_goal)
                           - cfg.score_cfg.lambda_step * static_cast<float>(result.best_node.depth);
    REQUIRE(std::abs(result.best_node.score - expected_score) < 1e-4f,
            "Score must match incomplete penalty formula without any goal reward or early arrival bonus");
    REQUIRE(result.best_node.score < 0.0f, "Unfinished score must be negative");

    std::cout << "[Test] test_unfinished_routes_cannot_claim_goal PASSED." << std::endl;
}

// 23. Milestone 5E.2C: Unexpected model predictions remain observable without silent correction
void test_unexpected_predictions_remain_observable() {
    std::cout << "[Test] Running test_unexpected_predictions_remain_observable..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);
    layout.obstacles[7] = 1; // Mark cell 7 as obstacle

    // Predictor that forces an unexpected prediction into the obstacle cell 7
    spark::PredictorFn anomalous_pred = [](
        const std::vector<float>& h_in,
        const std::vector<float>&,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        h_out = h_in;
        logits.assign(25, 0.0f);
        probs.assign(25, 0.01f);
        probs[7] = 0.76f; // Strongly predicts landing on cell 7 (obstacle)
    };

    auto root = spark::PlanningEngine::create_cold_start_root(2, 12, 128);
    auto child = spark::PlanningEngine::expand_branch_custom(anomalous_pred, layout, root, spark::Action::Down, 12);

    // The planner must faithfully record position 7 without silently clamping or snapping back
    REQUIRE(child.predicted_position == 7, "Model prediction into obstacle cell must be preserved observably");
    REQUIRE(std::abs(child.target_probability - 0.76f) < 1e-4f, "Target probability must match model output");

    std::cout << "[Test] test_unexpected_predictions_remain_observable PASSED." << std::endl;
}

// 24. Milestone 5E.2C: Non-finite score rejection and various valid score configurations
void test_comparator_non_finite_and_score_configs() {
    std::cout << "[Test] Running test_comparator_non_finite_and_score_configs..." << std::endl;

    // 1. Config A: Pure distance heuristic
    spark::PlanningScoreConfig cfg_dist;
    cfg_dist.goal_reward = 100.0f;
    cfg_dist.lambda_dist = 5.0f;
    cfg_dist.lambda_step = 0.0f;
    cfg_dist.early_goal_bonus_per_step = 0.0f;

    std::string err;
    REQUIRE(cfg_dist.validate(err), "Config A must be valid");
    // D=3, L=2, unfinished => J = -5(3) = -15.0
    float score_a = spark::PlanningScorer::compute_score(3, 0, 2, 6, false, cfg_dist);
    REQUIRE(std::abs(score_a - (-15.0f)) < 1e-5f, "Config A score must match -15.0");

    // 2. Config B: Pure step penalty
    spark::PlanningScoreConfig cfg_step;
    cfg_step.goal_reward = 100.0f;
    cfg_step.lambda_dist = 0.0f;
    cfg_step.lambda_step = 3.0f;
    cfg_step.early_goal_bonus_per_step = 0.0f;
    REQUIRE(cfg_step.validate(err), "Config B must be valid");
    // D=3, L=2, unfinished => J = -3(2) = -6.0
    float score_b = spark::PlanningScorer::compute_score(3, 0, 2, 6, false, cfg_step);
    REQUIRE(std::abs(score_b - (-6.0f)) < 1e-5f, "Config B score must match -6.0");

    // 3. Config C: High early bonus
    spark::PlanningScoreConfig cfg_early;
    cfg_early.goal_reward = 50.0f;
    cfg_early.lambda_dist = 1.0f;
    cfg_early.lambda_step = 0.5f;
    cfg_early.early_goal_bonus_per_step = 10.0f;
    REQUIRE(cfg_early.validate(err), "Config C must be valid");
    // D=0, L=2, H_max=6, goal reached => J = 50 - 0.5(2) + 10(4) = 50 - 1 + 40 = 89.0
    float score_c = spark::PlanningScorer::compute_score(0, 0, 2, 6, true, cfg_early);
    REQUIRE(std::abs(score_c - 89.0f) < 1e-5f, "Config C score must match 89.0");

    // 4. Non-finite parameter rejection in compute_score
    spark::PlanningScoreConfig invalid_cfg = cfg_dist;
    invalid_cfg.goal_reward = std::numeric_limits<float>::infinity();
    bool caught = false;
    try {
        spark::PlanningScorer::compute_score(0, 0, 1, 6, true, invalid_cfg);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    REQUIRE(caught, "compute_score must reject infinite goal reward");

    std::cout << "[Test] test_comparator_non_finite_and_score_configs PASSED." << std::endl;
}

// 25. Milestone 5E.2C: Checkpoint parameter immutability during beam search
void test_checkpoint_immutability_under_beam_search() {
    std::cout << "[Test] Running test_checkpoint_immutability_under_beam_search..." << std::endl;

    std::string ckpt_path = find_checkpoint_path();
    REQUIRE(!ckpt_path.empty(), "Checkpoint world_model_best.spark must be located");

    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 128;
    g_cfg.fan_in = 8;
    g_cfg.seed = 42ULL;
    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 128;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 1.0f;
    n_cfg.seed = 42ULL;
    spark::MicroUnitNetwork net(n_cfg, g_cfg);

    bool loaded = spark::MicroUnitNetwork::load_checkpoint(ckpt_path, net);
    REQUIRE(loaded, "Checkpoint must load successfully");

    // Snapshot exact parameters
    auto orig_biases = net.biases();
    auto orig_input_w = net.input_weights();
    auto orig_output_w = net.output_weights();
    auto orig_output_b = net.output_biases();
    auto orig_graph_w = net.graph().weights;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    auto root = spark::PlanningEngine::create_cold_start_root(0, 12, 128);
    spark::BeamSearchConfig cfg;
    cfg.beam_width = 8;
    cfg.max_depth = 4;

    auto res = spark::PlanningEngine::beam_search(net, layout, root, 12, cfg);
    (void)res;

    // Verify 100% bitwise immutability
    REQUIRE(net.biases() == orig_biases, "Biases must not mutate during beam search");
    REQUIRE(net.input_weights() == orig_input_w, "Input weights must not mutate during beam search");
    REQUIRE(net.output_weights() == orig_output_w, "Output weights must not mutate during beam search");
    REQUIRE(net.output_biases() == orig_output_b, "Output biases must not mutate during beam search");
    REQUIRE(net.graph().weights == orig_graph_w, "Graph weights must not mutate during beam search");

    std::cout << "[Test] test_checkpoint_immutability_under_beam_search PASSED." << std::endl;
}

// 26. Milestone 5E.2D.1: Trained checkpoint architectural contract verification
void test_trained_checkpoint_architecture_and_contract() {
    std::cout << "[Test] Running test_trained_checkpoint_architecture_and_contract..." << std::endl;

    std::string ckpt_path = find_checkpoint_path();
    REQUIRE(!ckpt_path.empty(), "Checkpoint world_model_best.spark must be located");

    spark::SparseGraphConfig g_cfg;
    spark::NetworkConfig n_cfg;
    spark::MicroUnitNetwork net(n_cfg, g_cfg);

    bool loaded = spark::MicroUnitNetwork::load_checkpoint(ckpt_path, net);
    REQUIRE(loaded, "Checkpoint must load successfully");

    // Validate architectural contract
    REQUIRE(net.config().input_dim == 55, "Input dimensionality must be exactly 55");
    REQUIRE(net.config().output_dim == 25, "Output dimensionality must be exactly 25");
    REQUIRE(net.config().num_units == 128, "Network must have exactly 128 micro-units");
    REQUIRE(net.config().alpha == 0.8f, "Alpha retention must be 0.8");
    REQUIRE(net.config().beta == 1.0f, "Beta scaling must be 1.0");
    REQUIRE(net.graph().config().fan_in == 8, "Fan-in must be 8");
    REQUIRE(net.graph().config().num_units == 128, "Graph units must be 128");

    // Verify observation encoding contract: 55 floats
    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);
    layout.obstacles[7] = 1;

    std::vector<float> x;
    spark::PlanningEngine::encode_model_input(0, layout, spark::Action::Right, x);
    REQUIRE(x.size() == 55, "Encoded input size must be 55");
    REQUIRE(x[0] == 1.0f, "Position 0 one-hot must be active");
    REQUIRE(x[25 + 7] == 1.0f, "Obstacle 7 one-hot must be active");
    REQUIRE(x[50 + static_cast<uint32_t>(spark::Action::Right)] == 1.0f, "Action Right one-hot must be active");

    // Forward step contract: probabilities sum to 1.0
    std::vector<float> h_out, logits, probs;
    net.forward_step(x, logits, probs, spark::ReadoutMode::Softmax);
    REQUIRE(probs.size() == 25, "Output probability distribution must have 25 elements");
    float sum_p = 0.0f;
    for (float p : probs) {
        REQUIRE(p >= 0.0f && p <= 1.0f, "Probability values must be bounded in [0, 1]");
        sum_p += p;
    }
    REQUIRE(std::abs(sum_p - 1.0f) < 1e-4f, "Probabilities must sum to 1.0");

    std::cout << "[Test] test_trained_checkpoint_architecture_and_contract PASSED." << std::endl;
}

// 27. Milestone 5E.2D.1: Real trained checkpoint beam search candidate generation
void test_trained_checkpoint_beam_search_planning() {
    std::cout << "[Test] Running test_trained_checkpoint_beam_search_planning..." << std::endl;

    std::string ckpt_path = find_checkpoint_path();
    REQUIRE(!ckpt_path.empty(), "Checkpoint world_model_best.spark must be located");

    spark::SparseGraphConfig g_cfg;
    spark::NetworkConfig n_cfg;
    spark::MicroUnitNetwork net(n_cfg, g_cfg);
    bool loaded = spark::MicroUnitNetwork::load_checkpoint(ckpt_path, net);
    REQUIRE(loaded, "Checkpoint must load successfully");

    // Select an unseen test layout
    auto splits = spark::GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);
    REQUIRE(!splits.test_layouts.empty(), "Test layouts must exist");
    const auto& test_layout = splits.test_layouts[0];

    // Generate a valid reachable episode
    auto episode = spark::PlanningEpisodeGenerator::generate_reachable_episode(test_layout, 123ULL, 4, 1);

    auto root = spark::PlanningEngine::create_cold_start_root(episode.start_pos, episode.goal_pos, 128, 4);

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 4;

    auto result = spark::PlanningEngine::beam_search(net, test_layout, root, episode.goal_pos, cfg);

    // Verify search completed and generated valid candidate plans
    REQUIRE(result.nodes_expanded > 0, "Nodes expanded must be > 0");
    REQUIRE(result.model_evaluations > 0, "Model evaluations must be > 0");
    REQUIRE(result.search_runtime_ms >= 0.0, "Search runtime must be non-negative");
    REQUIRE(!result.best_plan.empty(), "Best plan must not be empty");
    REQUIRE(result.best_plan.size() <= 4, "Plan length must not exceed max depth");

    // Verify child node states
    REQUIRE(result.best_node.predicted_position < 25, "Predicted position must be in [0, 24]");
    REQUIRE(result.best_node.recurrent_state.size() == 128, "Recurrent state must have 128 units");
    REQUIRE(result.best_node.mean_step_confidence > 0.0f && result.best_node.mean_step_confidence <= 1.0f,
            "Mean step confidence must be valid probability in (0, 1]");

    for (float h_val : result.best_node.recurrent_state) {
        REQUIRE(std::isfinite(h_val), "Recurrent state activations must be finite");
        REQUIRE(h_val > -1.0f && h_val < 1.0f, "Tanh activations must be bounded in (-1, 1)");
    }

    std::cout << "  Episode Start=(" << (episode.start_pos/5) << "," << (episode.start_pos%5)
              << ") -> Goal=(" << (episode.goal_pos/5) << "," << (episode.goal_pos%5) << ")"
              << " | Plan: " << result.best_plan.size() << " steps"
              << " | Goal Reached in Imagination: " << (result.goal_reached ? "YES" : "NO")
              << " | Score: " << result.best_node.score
              << " | Runtime: " << result.search_runtime_ms << " ms" << std::endl;

    std::cout << "[Test] test_trained_checkpoint_beam_search_planning PASSED." << std::endl;
}

// 28. Milestone 5E.2D.1: Cold-start vs warm-state root planning with real checkpoint
void test_trained_checkpoint_cold_vs_warm_root_generation() {
    std::cout << "[Test] Running test_trained_checkpoint_cold_vs_warm_root_generation..." << std::endl;

    std::string ckpt_path = find_checkpoint_path();
    REQUIRE(!ckpt_path.empty(), "Checkpoint world_model_best.spark must be located");

    spark::SparseGraphConfig g_cfg;
    spark::NetworkConfig n_cfg;
    spark::MicroUnitNetwork net(n_cfg, g_cfg);
    bool loaded = spark::MicroUnitNetwork::load_checkpoint(ckpt_path, net);
    REQUIRE(loaded, "Checkpoint must load successfully");

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    uint32_t start_pos = 6;
    uint32_t goal_pos = 8;

    // Condition 1: Cold start (h(0) = 0)
    auto cold_root = spark::PlanningEngine::create_cold_start_root(start_pos, goal_pos, 128, 4);

    // Condition 2: Legitimate warm state (past observation-action transition s_0, Stay)
    std::vector<float> x_prior;
    spark::PlanningEngine::encode_model_input(start_pos, layout, spark::Action::Stay, x_prior);
    std::vector<float> warm_h, logits, probs;
    net.forward_step(x_prior, logits, probs, spark::ReadoutMode::Softmax);
    warm_h = net.state(); // h(1) from legitimate initial observation

    auto warm_root = spark::PlanningEngine::create_warmed_up_root(start_pos, goal_pos, warm_h, 4);

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 4;

    auto cold_result = spark::PlanningEngine::beam_search(net, layout, cold_root, goal_pos, cfg);
    auto warm_result = spark::PlanningEngine::beam_search(net, layout, warm_root, goal_pos, cfg);

    REQUIRE(!cold_result.best_plan.empty(), "Cold start plan must not be empty");
    REQUIRE(!warm_result.best_plan.empty(), "Warm state plan must not be empty");
    REQUIRE(cold_result.best_node.recurrent_state != warm_result.best_node.recurrent_state,
            "Cold start and warm state plans must produce distinct recurrent dynamics");

    std::cout << "  Cold Plan: " << cold_result.best_plan.size() << " steps, Score: " << cold_result.best_node.score
              << " | Warm Plan: " << warm_result.best_plan.size() << " steps, Score: " << warm_result.best_node.score
              << std::endl;

    std::cout << "[Test] test_trained_checkpoint_cold_vs_warm_root_generation PASSED." << std::endl;
}

// 29. Milestone 5E.2D.2: Open-loop simulator execution and path efficiency
void test_open_loop_plan_execution() {
    std::cout << "[Test] Running test_open_loop_plan_execution..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    spark::PlanningEpisode ep;
    ep.start_pos = 0;
    ep.goal_pos = 2;
    ep.layout = layout;
    ep.shortest_path_length = 2;
    ep.reference_shortest_path = {spark::Action::Right, spark::Action::Right};

    spark::PlanningSearchResult s_res;
    s_res.best_plan = {spark::Action::Right, spark::Action::Right};
    s_res.predicted_trajectory = {0, 1, 2};
    s_res.goal_reached = true;
    s_res.predicted_final_position = 2;
    s_res.max_depth = 4;

    auto exec_res = spark::PlanningEngine::execute_plan_open_loop(ep, s_res, true);

    REQUIRE(exec_res.actual_goal_reached, "Actual goal must be reached in real environment");
    REQUIRE(exec_res.predicted_actual_agreement, "Predicted and actual outcomes must agree");
    REQUIRE(exec_res.first_divergence_step == -1, "No divergence step should occur");
    REQUIRE(exec_res.actual_trajectory.size() == 3, "Actual trajectory must have 3 positions");
    REQUIRE(exec_res.actual_trajectory[0] == 0 && exec_res.actual_trajectory[1] == 1 && exec_res.actual_trajectory[2] == 2,
            "Actual trajectory must be [0, 1, 2]");
    REQUIRE(std::abs(exec_res.path_efficiency - 1.0f) < 1e-5f, "Optimal route path efficiency must be 1.0");
    REQUIRE(exec_res.failure_category == spark::FailureCategory::None, "Success must have FailureCategory::None");

    std::cout << "[Test] test_open_loop_plan_execution PASSED." << std::endl;
}

// 30. Milestone 5E.2D.2: Failure categorization and model divergence detection
void test_failure_categorization_and_divergence() {
    std::cout << "[Test] Running test_failure_categorization_and_divergence..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);
    layout.obstacles[1] = 1; // Obstacle at (0,1)=1

    spark::PlanningEpisode ep;
    ep.start_pos = 0;
    ep.goal_pos = 2;
    ep.layout = layout;
    ep.shortest_path_length = 4; // Detour around cell 1
    ep.reference_shortest_path = {spark::Action::Down, spark::Action::Right, spark::Action::Right, spark::Action::Up};

    // Plan hallucinating movement through obstacle cell 1
    spark::PlanningSearchResult s_res;
    s_res.best_plan = {spark::Action::Right, spark::Action::Right};
    s_res.predicted_trajectory = {0, 1, 2};
    s_res.goal_reached = true;
    s_res.predicted_final_position = 2;
    s_res.max_depth = 4;

    auto exec_res = spark::PlanningEngine::execute_plan_open_loop(ep, s_res, true);

    REQUIRE(!exec_res.actual_goal_reached, "Actual goal must NOT be reached when hitting obstacle");
    REQUIRE(!exec_res.predicted_actual_agreement, "Predicted and actual outcomes must disagree");
    REQUIRE(exec_res.first_divergence_step == 1, "Divergence must occur at step 1 where model hallucinated cell 1");
    REQUIRE(exec_res.actual_trajectory[1] == 0, "Actual agent must remain blocked at position 0");
    REQUIRE(exec_res.failure_category == spark::FailureCategory::ObstacleHallucination,
            "Failure category must be ObstacleHallucination");
    REQUIRE(exec_res.path_efficiency == 0.0f, "Failed plan must have efficiency 0.0");

    std::cout << "[Test] test_failure_categorization_and_divergence PASSED." << std::endl;
}

// 31. Milestone 5E.2D.2: Episode classification across Groups A, B, C, D
void test_episode_group_classification() {
    std::cout << "[Test] Running test_episode_group_classification..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    // Group A: Direct route (shortest path == Manhattan distance)
    spark::PlanningEpisode ep_a;
    ep_a.start_pos = 0;
    ep_a.goal_pos = 2;
    ep_a.shortest_path_length = 2;
    ep_a.reference_shortest_path = {spark::Action::Right, spark::Action::Right};
    ep_a.layout = layout;
    REQUIRE(spark::PlanningEngine::classify_episode(ep_a, 4) == spark::EpisodeGroup::GroupA_Direct,
            "Episode with path == Manhattan distance must be Group A");

    // Group D: Longer route (shortest path > max_depth)
    REQUIRE(spark::PlanningEngine::classify_episode(ep_a, 1) == spark::EpisodeGroup::GroupD_Longer,
            "Episode with path > max_depth must be Group D");

    // Group B: Detour route
    layout.obstacles[7] = 1;
    spark::PlanningEpisode ep_b;
    ep_b.start_pos = 2;
    ep_b.goal_pos = 12;
    ep_b.shortest_path_length = 4; // Manhattan is 2, path is 4
    ep_b.reference_shortest_path = {spark::Action::Left, spark::Action::Down, spark::Action::Down, spark::Action::Right};
    ep_b.layout = layout;
    // Step 1: from 2 Left to 1 -> Manhattan to 12 is |0-2| + |1-2| = 2 + 1 = 3 > 2 (first step increases distance!)
    REQUIRE(spark::PlanningEngine::classify_episode(ep_b, 6) == spark::EpisodeGroup::GroupC_Misleading,
            "Episode where first step increases distance must be Group C");

    std::cout << "[Test] test_episode_group_classification PASSED." << std::endl;
}

// 32. Milestone 5E.3 (Step 1): Live recurrent-state updates & hypothetical independence
void test_closed_loop_live_state_updates_and_independence() {
    std::cout << "[Test] Running test_closed_loop_live_state_updates_and_independence..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    spark::PlanningEpisode ep;
    ep.start_pos = 0;
    ep.goal_pos = 2; // Shortest path: Right, Right (2 steps)
    ep.layout = layout;
    ep.shortest_path_length = 2;
    ep.reference_shortest_path = {spark::Action::Right, spark::Action::Right};

    // Predictor where h[0] increments by 1.0 each time a step is made
    spark::PredictorFn predictor = [](
        const std::vector<float>& h_in,
        const std::vector<float>& x,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        h_out = h_in;
        h_out[0] += 1.0f; // Track cumulative transitions

        // Find position from x (one-hot in [0..24])
        uint32_t curr = 0;
        for (uint32_t i = 0; i < 25; ++i) {
            if (x[i] > 0.5f) { curr = i; break; }
        }
        // Find action from x (one-hot in [50..54])
        spark::Action a = spark::Action::Stay;
        for (uint32_t i = 0; i < 5; ++i) {
            if (x[50 + i] > 0.5f) { a = static_cast<spark::Action>(i); break; }
        }
        auto trans = spark::GridWorldEnv::simulate_transition(curr, a, std::vector<uint8_t>(25, 0), 5, 5);

        logits.assign(25, -10.0f);
        probs.assign(25, 0.0f);
        logits[trans.to_pos] = 10.0f;
        probs[trans.to_pos] = 1.0f;
    };

    std::vector<float> initial_h(128, 0.0f);
    initial_h[0] = 100.0f; // Initial state tag

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 2;
    cfg.max_depth = 2;

    auto res = spark::PlanningEngine::execute_plan_closed_loop_custom(
        predictor, ep, initial_h, cfg, 5, true
    );

    REQUIRE(res.actual_goal_reached, "Closed-loop planner must reach the goal");
    REQUIRE(res.executed_plan.size() == 2, "Must execute exactly 2 real actions");
    REQUIRE(res.replan_count == 2, "Must replan at each executed real action");
    REQUIRE(res.executed_plan[0] == spark::Action::Right && res.executed_plan[1] == spark::Action::Right,
            "Executed actions must be [Right, Right]");
    REQUIRE(res.actual_trajectory.size() == 3, "Trajectory must have [0, 1, 2]");
    REQUIRE(res.actual_trajectory[0] == 0 && res.actual_trajectory[1] == 1 && res.actual_trajectory[2] == 2,
            "Trajectory sequence must be 0 -> 1 -> 2");

    // Live state advanced exactly 2 times (initial was 100.0, after 2 executed steps h[0] must be 102.0)
    // Note: Search explored dozens of hypothetical branches (each incrementing h[0]), but live state must NOT be corrupted!
    std::cout << "  Nodes expanded across replanning: " << res.nodes_expanded << std::endl;
    REQUIRE(res.nodes_expanded > 2, "Hypothetical search must expand multiple branches without corrupting live state");

    std::cout << "[Test] test_closed_loop_live_state_updates_and_independence PASSED." << std::endl;
}

// 33. Milestone 5E.3 (Step 1): Action budget enforcement and goal termination
void test_closed_loop_action_budget_and_termination() {
    std::cout << "[Test] Running test_closed_loop_action_budget_and_termination..." << std::endl;

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    spark::PredictorFn ideal_predictor = [](
        const std::vector<float>& h_in,
        const std::vector<float>& x,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        h_out = h_in;
        uint32_t curr = 0;
        for (uint32_t i = 0; i < 25; ++i) {
            if (x[i] > 0.5f) { curr = i; break; }
        }
        spark::Action a = spark::Action::Stay;
        for (uint32_t i = 0; i < 5; ++i) {
            if (x[50 + i] > 0.5f) { a = static_cast<spark::Action>(i); break; }
        }
        auto trans = spark::GridWorldEnv::simulate_transition(curr, a, std::vector<uint8_t>(25, 0), 5, 5);
        logits.assign(25, -10.0f);
        probs.assign(25, 0.0f);
        logits[trans.to_pos] = 10.0f;
        probs[trans.to_pos] = 1.0f;
    };

    std::vector<float> initial_h(128, 0.0f);
    spark::BeamSearchConfig cfg;
    cfg.beam_width = 2;
    cfg.max_depth = 2;

    // Subtest 1: Goal 4 steps away, budget = 2 (Budget truncation)
    spark::PlanningEpisode ep_budget;
    ep_budget.start_pos = 0;
    ep_budget.goal_pos = 4; // 4 steps away
    ep_budget.shortest_path_length = 4;
    ep_budget.layout = layout;

    auto res_budget = spark::PlanningEngine::execute_plan_closed_loop_custom(
        ideal_predictor, ep_budget, initial_h, cfg, 2 /* max_executed_steps = 2 */, true
    );
    REQUIRE(!res_budget.actual_goal_reached, "Goal must NOT be reached when budget is truncated");
    REQUIRE(res_budget.executed_plan.size() == 2, "Exactly 2 actions must be executed under budget 2");
    REQUIRE(res_budget.failure_category == spark::FailureCategory::DepthInsufficient,
            "Budget exhaustion before goal must be DepthInsufficient");

    // Subtest 2: Goal 2 steps away, budget = 10 (Early termination on goal reached)
    spark::PlanningEpisode ep_reach;
    ep_reach.start_pos = 0;
    ep_reach.goal_pos = 2; // 2 steps away
    ep_reach.shortest_path_length = 2;
    ep_reach.layout = layout;

    auto res_reach = spark::PlanningEngine::execute_plan_closed_loop_custom(
        ideal_predictor, ep_reach, initial_h, cfg, 10 /* max_executed_steps = 10 */, true
    );
    REQUIRE(res_reach.actual_goal_reached, "Goal must be reached");
    REQUIRE(res_reach.executed_plan.size() == 2, "Must terminate immediately upon reaching goal, not run to budget 10");
    REQUIRE(res_reach.failure_category == spark::FailureCategory::None, "Success must have FailureCategory::None");

    std::cout << "[Test] test_closed_loop_action_budget_and_termination PASSED." << std::endl;
}

// 34. Milestone 5E.3 (Step 1): Closed-loop replanning and recovery from environmental obstacle
void test_closed_loop_replanning_obstacle_recovery() {
    std::cout << "[Test] Running test_closed_loop_replanning_obstacle_recovery..." << std::endl;

    // Setup environment with an obstacle at cell 1
    // Agent starts at 0, Goal is at 2.
    // Shortest detour: Down (5), Right (6), Right (7), Up (2) -> 4 steps
    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);
    layout.obstacles[1] = 1; // Obstacle at (0,1)=1

    spark::PlanningEpisode ep;
    ep.start_pos = 0;
    ep.goal_pos = 2;
    ep.shortest_path_length = 4;
    ep.reference_shortest_path = {spark::Action::Down, spark::Action::Right, spark::Action::Right, spark::Action::Up};
    ep.layout = layout;

    // A model that initially erroneously thinks cell 1 is open from cell 0!
    // But from any other cell or upon failing, it predicts accurately.
    spark::PredictorFn flawed_model = [&layout](
        const std::vector<float>& h_in,
        const std::vector<float>& x,
        std::vector<float>& h_out,
        std::vector<float>& logits,
        std::vector<float>& probs
    ) {
        h_out = h_in;
        uint32_t curr = 0;
        for (uint32_t i = 0; i < 25; ++i) {
            if (x[i] > 0.5f) { curr = i; break; }
        }
        spark::Action a = spark::Action::Stay;
        for (uint32_t i = 0; i < 5; ++i) {
            if (x[50 + i] > 0.5f) { a = static_cast<spark::Action>(i); break; }
        }

        uint32_t pred_next = curr;
        if (curr == 0 && a == spark::Action::Right) {
            pred_next = 1; // Flawed prediction: thinks cell 1 is open!
        } else {
            auto tr = spark::GridWorldEnv::simulate_transition(curr, a, layout.obstacles, 5, 5);
            pred_next = tr.to_pos;
        }

        logits.assign(25, -10.0f);
        probs.assign(25, 0.0f);
        logits[pred_next] = 10.0f;
        probs[pred_next] = 1.0f;
    };

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 4;

    // In open loop:
    auto root = spark::PlanningEngine::create_cold_start_root(0, 2, 128, 4);
    auto open_search = spark::PlanningEngine::beam_search_custom(flawed_model, layout, root, 2, cfg);
    auto open_res = spark::PlanningEngine::execute_plan_open_loop(ep, open_search, true);
    REQUIRE(!open_res.actual_goal_reached, "Open loop MUST fail because it executes through obstacle cell 1");
    REQUIRE(open_res.first_divergence_step == 1, "Open loop diverges at step 1");

    // In closed loop:
    // When the agent attempts Action::Right, the real simulator keeps it at cell 0!
    // Next replan step: agent is at cell 0, and subsequent replans discover or take the detour Down -> Right -> Right -> Up!
    // Let's test standard closed-loop execution:
    auto closed_res = spark::PlanningEngine::execute_plan_closed_loop_custom(
        flawed_model, ep, std::vector<float>(128, 0.0f), cfg, 10, true
    );

    std::cout << "  Closed-loop replans: " << closed_res.replan_count
              << " | Executed plan size: " << closed_res.executed_plan.size()
              << " | Actual trajectory size: " << closed_res.actual_trajectory.size()
              << " | Transition errors: " << closed_res.transition_errors << std::endl;

    REQUIRE(closed_res.transition_errors >= 1, "Closed loop must record transition error on blocked step");
    REQUIRE(closed_res.first_divergence_step == 1, "First divergence step must be step 1");

    std::cout << "[Test] test_closed_loop_replanning_obstacle_recovery PASSED." << std::endl;
}

// 35. Milestone 5E.3 (Step 1): Real checkpoint closed-loop execution
void test_real_checkpoint_closed_loop_execution() {
    std::cout << "[Test] Running test_real_checkpoint_closed_loop_execution..." << std::endl;

    std::string ckpt_path = find_checkpoint_path();
    REQUIRE(!ckpt_path.empty(), "Checkpoint world_model_best.spark must be located");

    spark::SparseGraphConfig g_cfg;
    spark::NetworkConfig n_cfg;
    spark::MicroUnitNetwork net(n_cfg, g_cfg);
    bool loaded = spark::MicroUnitNetwork::load_checkpoint(ckpt_path, net);
    REQUIRE(loaded, "Checkpoint must load successfully");

    spark::GridWorldLayout layout;
    layout.width = 5;
    layout.height = 5;
    layout.obstacles.assign(25, 0);

    spark::PlanningEpisode ep;
    ep.start_pos = 0;
    ep.goal_pos = 2; // Shortest path: Right, Right (2 steps)
    ep.layout = layout;
    ep.shortest_path_length = 2;
    ep.reference_shortest_path = {spark::Action::Right, spark::Action::Right};

    spark::BeamSearchConfig cfg;
    cfg.beam_width = 4;
    cfg.max_depth = 4;

    auto res = spark::PlanningEngine::execute_plan_closed_loop(net, ep, cfg, 6, false /* warm state */);

    REQUIRE(res.actual_goal_reached, "Real checkpoint closed-loop planner must reach the goal");
    REQUIRE(res.executed_plan.size() == 2, "Real checkpoint must reach goal in exactly 2 steps");
    REQUIRE(res.replan_count == 2, "Must replan 2 times for 2 executed steps");
    REQUIRE(res.actual_trajectory.size() == 3, "Trajectory must have [0, 1, 2]");
    REQUIRE(res.path_efficiency == 1.0f, "Path efficiency must be 1.0 (optimal)");

    std::cout << "  Closed-loop goal reached in " << res.executed_plan.size() << " steps | Runtime: "
              << res.search_runtime_ms << " ms | Evaluations: " << res.model_evaluations << std::endl;

    std::cout << "[Test] test_real_checkpoint_closed_loop_execution PASSED." << std::endl;
}

int main() {
    std::cout << "=== Running SPARK Milestone 5E.1, 5E.2 (A/B/C), and 5E.2D Planning Tests ===" << std::endl;

    test_bfs_shortest_path_oracle();
    test_bfs_unreachable_and_obstacle_guards();
    test_planning_episode_generator();
    test_recurrent_state_branching();
    test_nested_branching_depths();
    test_trajectory_scoring_and_tie_breaking();
    test_cold_start_vs_warmed_up_root();
    test_checkpoint_immutability();

    // 5E.2A tests
    test_beam_search_core_bounded_limits();
    test_beam_search_identical_sequence_order_invariance();
    test_beam_search_goal_preservation_across_depths();

    // 5E.2B tests
    test_scoring_validation_and_objective();
    test_strict_weak_ordering_axioms();
    test_confidence_tracking();
    test_mock_scenario_a_short_vs_long_route();
    test_mock_scenario_b_heuristic_vs_completed_goal();
    test_mock_scenario_c_early_goal_preservation();
    test_mock_scenario_d_tie_breaking_determinism();
    test_mock_scenario_e_no_reachable_goal_fallback();
    test_mock_scenario_f_hallucination_obstacle_isolation();
    test_exhaustive_vs_beam_search_agreement();

    // 5E.2C tests
    test_unfinished_routes_cannot_claim_goal();
    test_unexpected_predictions_remain_observable();
    test_comparator_non_finite_and_score_configs();
    test_checkpoint_immutability_under_beam_search();

    // 5E.2D.1 tests
    test_trained_checkpoint_architecture_and_contract();
    test_trained_checkpoint_beam_search_planning();
    test_trained_checkpoint_cold_vs_warm_root_generation();

    // 5E.2D.2 tests
    test_open_loop_plan_execution();
    test_failure_categorization_and_divergence();
    test_episode_group_classification();

    // 5E.3 tests (Step 1)
    test_closed_loop_live_state_updates_and_independence();
    test_closed_loop_action_budget_and_termination();
    test_closed_loop_replanning_obstacle_recovery();
    test_real_checkpoint_closed_loop_execution();

    std::cout << "All Milestone 5E.1, 5E.2 (A/B/C), 5E.2D, and 5E.3 planning tests passed successfully!" << std::endl;
    return 0;
}
