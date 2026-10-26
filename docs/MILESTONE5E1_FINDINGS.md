# SPARK Phase 5 — Milestone 5E.1 Findings Report
## Goal Planning Foundation, Independent Recurrent-State Branching & Trajectory Scoring

**Date:** October 26, 2026  
**Status:** Milestone 5E.1 Complete  
**Repository:** `MaenExists/OpenUltron`  
**Language & Build:** Modern C++17, Standard Library Only, CMake, Release mode  
**Authors:** Principal AI Research Engineer & C++ Systems Engineer (Google Antigravity / Gemini Flash)

---

## 1. Executive Summary

Milestone 5E.1 builds the formal algorithmic and engineering foundation for goal-directed planning using SPARK's learned world model. In earlier Phase 5 milestones (5B, 5C, 5D), SPARK was established as a one-step transition predictor, a partially observable state tracker, and an autoregressive multi-step generative model.

Milestone 5E.1 establishes the search tree infrastructure required to evaluate hypothetical alternative futures without modifying the underlying model weights, leaking ground-truth simulator transitions, or cross-contaminating recurrent states across search branches.

### Core Deliverables Built & Verified:
1. **Planning Episode Representation & Generator (`PlanningEpisodeGenerator`)**:
   - Compact structure specifying start cell, goal cell, static obstacle layout, maximum horizon, and episode seed.
   - Guaranteed reachability and reference shortest path lengths calculated via a reference Breadth-First Search (BFS) solver.
2. **Ground-Truth BFS Reference Oracle (`BFSSolver`)**:
   - Reference shortest-path solver on the discrete grid graph.
   - Strictly encapsulated: used solely for reachability guarantees during dataset creation and oracle benchmark baselines. **The learned model planner never invokes BFS during candidate search.**
3. **Lightweight Planning Node (`PlanningNode`)**:
   - Contains predicted agent position, recurrent hidden state snapshot ($N=128$ floats), action history, parent pointer, depth, confidence, and objective score.
   - Model weights remain shared and strictly `const` (read-only).
4. **Independent Recurrent-State Branching (`forward_step_branching`)**:
   - Thread-safe, reentrant branching mechanism: child states are computed from parent state snapshots without mutating the network instance.
   - Verified that evaluating candidate branches in different permutations produces bitwise identical states and predictions.
5. **Deterministic Trajectory Scoring (`PlanningScorer`)**:
   - Objective: $J = R_{\text{goal}} - \lambda_d D - \lambda_s L + \text{early\_bonus}$.
   - Strict 6-tier deterministic tie-breaking ensuring 100% reproducible candidate rankings.
6. **Preservation of Cold-Start vs Warmed-Up Roots**:
   - Formal initialization policies distinguishing planning from $h(0) = \mathbf{0}$ vs planning conditioned on an existing observation trajectory.
7. **Comprehensive Unit Test Suite (`test_planning`)**:
   - 8 automated test suites passing 100% (all 10 CTest suites passing in 0.16s).

---

## 2. Architecture & Design Details

### 2.1 BFS Shortest Path Reference Oracle
The BFS reference solver finds the exact ground-truth shortest path on the $5 \times 5$ grid graph while accounting for boundary walls and static obstacles:
- **Reachability Verification**: Guarantees that generated planning episodes have valid non-obstacle start and goal coordinates and at least one navigable path connecting them.
- **Strict Isolation**: `BFSSolver` is used only by `PlanningEpisodeGenerator` and offline oracle evaluations. No component of the learned model search engine queries BFS during candidate evaluation.

### 2.2 Planning Episode Representation
```cpp
struct PlanningEpisode {
    uint32_t start_pos = 0;                  // Initial agent position (0..24)
    uint32_t goal_pos = 0;                   // Goal position (0..24)
    GridWorldLayout layout;                  // 5x5 layout with static obstacles
    uint32_t max_steps = 16;                 // Maximum planning horizon
    uint64_t layout_id = 0;                  // Layout identifier
    uint64_t seed = 42ULL;                   // Episode seed
    uint32_t shortest_path_length = 0;       // Reference BFS distance
    std::vector<Action> reference_shortest_path; // Reference path for validation only
};
```

### 2.3 Lightweight Planning Node Representation
To prevent large heap allocations during tree exploration, `PlanningNode` captures only the necessary state for a hypothetical future:
```cpp
struct PlanningNode {
    uint32_t predicted_position = 0;          // Model predicted agent position
    std::vector<float> recurrent_state;       // Snapshot of recurrent state h(t) (128 floats = 512 bytes)
    std::vector<Action> actions_from_root;    // Full action trajectory from root
    int32_t parent_idx = -1;                  // Parent node index in search tree
    Action action_from_parent = Action::Stay; // Action leading to this node
    float score = 0.0f;                       // Trajectory score J
    uint32_t depth = 0;                       // Candidate path length L
    bool goal_reached = false;                // Goal achievement flag
    uint32_t manhattan_to_goal = 0;           // Manhattan distance D
    float target_probability = 1.0f;          // Model argmax confidence

    bool is_better_than(const PlanningNode& other) const;
};
```

### 2.4 Recurrent-State Branching & Stateless Forward Step
A recurrent neural network naturally maintains temporal state in internal memory buffers. In search algorithms (e.g. Beam Search, MCTS), multiple hypothetical futures must branch from the same parent state:
- **Branch A**: Right $\to$ Up
- **Branch B**: Left $\to$ Down

If `net.forward_step()` were called in-place, evaluating Branch A would leave the network's internal buffers modified, contaminating Branch B.

To guarantee complete independence, `MicroUnitNetwork` provides a stateless forward pass:
```cpp
void forward_step_branching(
    const std::vector<float>& h_in,
    const std::vector<float>& input,
    std::vector<float>& h_out,
    std::vector<float>& out_logits,
    std::vector<float>& out_probs,
    ReadoutMode mode = ReadoutMode::Softmax
) const;
```
- The network object is marked `const` (read-only).
- `h_in` is read as input; `h_out` is produced as an independent vector.
- No internal network buffers are modified.
- Network parameters ($U, V, W, b, c$) remain completely immutable throughout search.

### 2.5 Trajectory Scoring & Deterministic Tie-Breaking
The planning objective function evaluates candidate nodes using only model-predicted state and the episode goal:
$$J = R_{\text{goal}} - \lambda_d D - \lambda_s L + \text{early\_bonus}$$
Where:
- $R_{\text{goal}} = 100.0$: Reward granted if the predicted position matches `goal_pos`.
- $D = \text{Manhattan}(\text{pred\_pos}, \text{goal\_pos})$: Distance penalty ($\lambda_d = 2.0$).
- $L = \text{depth}$: Path length / action cost ($\lambda_s = 0.5$).
- $\text{early\_bonus} = \lambda_{\text{early}} \cdot (L_{\text{max}} - L)$: Bonus for reaching the goal in fewer steps ($\lambda_{\text{early}} = 1.0$).

#### Deterministic Tie-Breaking Comparator (`is_better_than`):
1. **Score $J$**: Difference $> 10^{-4}$ determines superiority.
2. **Goal Reached**: Nodes that reached the goal win over nodes that did not.
3. **Manhattan Distance $D$**: Lower distance to goal wins.
4. **Depth $L$**: Shorter candidate trajectories win.
5. **Confidence $P(\text{pred})$**: Higher argmax softmax probability wins.
6. **Lexicographical Action Order**: Earliest divergence in action sequence (Up < Down < Left < Right < Stay) breaks ties deterministically.

### 2.6 Cold-Start vs Warmed-Up Root Initialization
Every planning episode must explicitly define its root context:
1. **Cold-Start Root** (`create_cold_start_root`):
   - Used when the planner plans from the very beginning of an episode before any prior observations have been made.
   - Initialized with $h(0) = \mathbf{0}$.
2. **Warmed-Up Root** (`create_warmed_up_root`):
   - Used when the agent has already executed transitions in the environment and accumulated internal state $h(t)$.
   - Initialized with the live recurrent hidden state $h(t)$.
   - Prevents artificial information leakage while preserving valid historical context.

---

## 3. Automated Test Verification (`test_planning.cpp`)

The test harness in `tests/test_planning.cpp` executes 8 comprehensive verification suites:

1. **`test_bfs_shortest_path_oracle`**:
   - Empty grid: verified shortest path from $(0,0)$ to $(2,3)$ has length 5.
   - Detour layout: verified path around a blocked cell $(1,2)$ has length 4; path around a 3-cell wall has length 6.
2. **`test_bfs_unreachable_and_obstacle_guards`**:
   - Verified that a walled corner is recognized as `reachable = false`.
   - Verified that placing start or goal inside an obstacle is safely rejected.
   - Verified identical start and goal returns `length = 0, reachable = true`.
3. **`test_planning_episode_generator`**:
   - Verified generation across unseen test layouts produces valid, distinct, non-obstacle start and goal coordinates.
   - Verified reference BFS shortest path matches the generated start/goal pair.
4. **`test_recurrent_state_branching`**:
   - Order Invariance Test: evaluated Branch A (Right), B (Left), C (Up) in Order 1 (Right $\to$ Left $\to$ Up) vs Order 2 (Up $\to$ Right $\to$ Left).
   - **Result**: Bitwise identical predicted positions, scores, and recurrent state vectors ($0.0$ difference).
   - Verified child states differ from root and from sibling branches.
5. **`test_nested_branching_depths`**:
   - Expanded 2 levels of nested branching (Depth 1 Right $\to$ Depth 2 Up vs Right).
   - Verified correct depth progression, action history concatenation, and state divergence.
6. **`test_trajectory_scoring_and_tie_breaking`**:
   - Verified exact score calculations for intermediate steps and goal-reaching steps.
   - Verified early goal bonus awards higher scores to shorter paths reaching the goal.
   - Verified all 6 tie-breaking tiers.
7. **`test_cold_start_vs_warmed_up_root`**:
   - Verified cold start roots have strictly zeroed recurrent state vectors.
   - Verified warmed-up roots preserve non-zero contextual vectors.
   - Verified branching from cold start vs warmed-up roots produces distinct, appropriate trajectories.
8. **`test_checkpoint_immutability`**:
   - Saved snapshots of network biases, input weights, output weights, and graph weights.
   - Executed deep multi-branch expansions.
   - **Result**: Network parameters remain 100% bitwise identical (zero mutation).

---

## 4. Test Suite Status

```
Test project /home/maen/Builds/OpenUltron/build
      Start  1: test_graph
 1/10 Test  #1: test_graph .......................   Passed    0.01 sec
      Start  2: test_forward
 2/10 Test  #2: test_forward .....................   Passed    0.00 sec
      Start  3: test_gradients
 3/10 Test  #3: test_gradients ...................   Passed    0.00 sec
      Start  4: test_learning
 4/10 Test  #4: test_learning ....................   Passed    0.03 sec
      Start  5: test_baselines
 5/10 Test  #5: test_baselines ...................   Passed    0.01 sec
      Start  6: test_regression
 6/10 Test  #6: test_regression ..................   Passed    0.00 sec
      Start  7: test_local_learning
 7/10 Test  #7: test_local_learning ..............   Passed    0.03 sec
      Start  8: test_gridworld
 8/10 Test  #8: test_gridworld ...................   Passed    0.05 sec
      Start  9: test_rollout
 9/10 Test  #9: test_rollout .....................   Passed    0.00 sec
      Start 10: test_planning
10/10 Test #10: test_planning ....................   Passed    0.01 sec

100% tests passed out of 10
Total Test time (real) = 0.16 sec
```

---

## 5. Scope for Milestone 5E.2 (Next Steps)

With the planning foundation, independent branching engine, and episode generator fully verified, Milestone 5E.2 will implement and evaluate complete learned-model search algorithms:

1. **Search Algorithm Implementations**:
   - **Greedy Best-First Rollout**: Depth-limited greedy action selection.
   - **Beam Search Planner**: Maintaining top-$K$ candidate branches per depth level.
   - **Monte Carlo Rollout Search**: Stochastic candidate rollout sampling.
2. **Closed-Loop Execution & Evaluation**:
   - Execute planned action sequences in the true environment simulator.
   - Measure **Plan Success Rate** (fraction of episodes where the agent reaches the goal).
   - Measure **Plan Optimality Ratio** ($\text{Length}_{\text{planned}} / \text{Length}_{\text{BFS}}$).
   - Quantify model mismatch: cases where the learned model predicted goal arrival but the real environment collided with an obstacle.
3. **Comparison Against Baselines**:
   - Compare SPARK learned planning against Random Search, Naive Straight-Line Heuristic, and Ground-Truth BFS Oracle.
