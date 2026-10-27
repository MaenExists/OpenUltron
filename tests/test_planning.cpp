#include "spark/planning.hpp"
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

int main() {
    std::cout << "=== Running SPARK Milestone 5E.1 & 5E.2A Planning Tests ===" << std::endl;

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

    std::cout << "All Milestone 5E.1 & 5E.2A planning tests passed successfully!" << std::endl;
    return 0;
}
