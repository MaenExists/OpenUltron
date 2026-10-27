#include "spark/planning.hpp"
#include "spark/gridworld.hpp"
#include "spark/network.hpp"
#include <iostream>
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

int main() {
    std::cout << "=== Running SPARK Milestone 5E.1 & 5E.2 (A/B) Planning Tests ===" << std::endl;

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

    std::cout << "All Milestone 5E.1, 5E.2A, and 5E.2B planning tests passed successfully!" << std::endl;
    return 0;
}
