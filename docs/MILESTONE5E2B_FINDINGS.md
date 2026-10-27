# SPARK Milestone 5E.2B — Planning Scores, Goal Preservation & Deterministic Decision Ranking
**OpenUltron / SPARK: Sparse Predictive Adaptive Recurrent Kernel**
**Author:** Principal AI Research Engineer & Systems Developer
**Date:** October 27, 2026
**Status:** Completed & Empirically Verified

---

## 1. Executive Summary & Objective

In **Milestone 5E.2B**, we mathematically formalize, defensively harden, and systematically validate the decision-making core of SPARK's model-based planning engine. 

Milestone 5E.2A implemented the bounded beam-search harness; Milestone 5E.2B establishes:
1. **Exact Mathematical Planning Objective & Defensive Validation**: Explicit implementation of the trajectory value function $J(n)$ with strict runtime checks preventing non-finite or negative parameters.
2. **Terminal Goal Preservation & Separation**: Uncompromising architectural separation between completed goal trajectories and in-flight search candidates, preventing search truncation from degrading early goal solutions.
3. **Formal Strict Weak Ordering Compliance**: Elimination of floating-point epsilon comparison anomalies in `PlanningNode::is_better_than`, ensuring strict compliance with C++ `std::sort` requirements (irreflexivity, asymmetry, transitivity, and transitivity of incomparability).
4. **Three-Tier Confidence Tracking**: Unambiguous tracking of step transition probability ($p_L$), cumulative log confidence ($\sum \log p_t$), and geometric mean step confidence ($\exp(\frac{1}{L}\sum \log p_t)$).
5. **Deterministic Validation Across Scenarios A through G**: Complete verification using deterministic mock transition models and real neural checkpoints across 7 benchmark test scenarios.

---

## 2. Mathematical Planning Objective & Validation

### 2.1 The Value Function Formula
The objective function evaluates any hypothetical node $n$ reached via action sequence $a_{1:L}$:

$$J(n) = R_g \cdot \mathbf{1}_{\text{goal}}(n) - \lambda_d \cdot D(n, g) - \lambda_s \cdot L(n) + \lambda_e \cdot \mathbf{1}_{\text{goal}}(n) \cdot (H_{\max} - L(n))$$

Where:
- $R_g \ge 0$: Large terminal reward for reaching the target goal state ($R_g = 100.0$).
- $\mathbf{1}_{\text{goal}}(n) \in \{0, 1\}$: Indicator function, evaluating to $1$ if node $n$'s predicted position matches the goal cell, and $0$ otherwise.
- $D(n, g) = |r_n - r_g| + |c_n - c_g|$: Manhattan distance on the $5 \times 5$ grid from predicted position $n$ to goal position $g$.
- $L(n) = \text{depth}(n)$: Candidate sequence length (number of planning steps executed from root).
- $\lambda_d \ge 0$: Distance penalty coefficient ($\lambda_d = 2.0$), guiding search heuristics toward the goal.
- $\lambda_s \ge 0$: Step penalty coefficient ($\lambda_s = 0.5$), penalizing unnecessary transit time.
- $\lambda_e \ge 0$: Early arrival bonus coefficient ($\lambda_e = 1.0$), explicitly rewarding shorter plans that achieve the goal ahead of the maximum planning horizon $H_{\max}$.

### 2.2 Defensive Invariant Checks
All configuration parameters are validated via `PlanningScoreConfig::validate(std::string& err)`:
- $R_g, \lambda_d, \lambda_s, \lambda_e$ must be finite (`std::isfinite`).
- All coefficients must be non-negative ($\ge 0.0f$).
- If any parameter is invalid, `PlanningScorer::compute_score` rejects execution immediately by throwing `std::invalid_argument`.
- Computed scores are defensively asserted with `std::isfinite(score)`, throwing `std::runtime_error` if NaN or Inf is ever detected.

---

## 3. Strict Weak Ordering & Deterministic Tie-Breaking

### 3.1 Resolving the Floating-Point Epsilon Failure
In Milestone 5E.2A, `PlanningNode::is_better_than` utilized an epsilon tolerance threshold:
```cpp
// FLAWED: Violates transitivity of incomparability
if (std::abs(score - other.score) > 1e-4f) {
    return (score - other.score) > 0.0f;
}
```
**Why this breaks `std::sort`**:
Consider three candidate nodes with scores $a = 0.00000$, $b = 0.00008$, $c = 0.00015$:
- $|a - b| = 0.00008 < 10^{-4} \implies a \sim b$ (incomparable under epsilon).
- $|b - c| = 0.00007 < 10^{-4} \implies b \sim c$ (incomparable under epsilon).
- But $|a - c| = 0.00015 > 10^{-4} \implies c > a$ (comparable!).
This violates the fundamental axiom of **transitivity of incomparability** required by C++ Strict Weak Ordering, leading to undefined behavior or segmentation faults in quicksort implementations.

### 3.2 The Strict Scalar Hierarchy
Milestone 5E.2B replaces epsilon band comparisons with direct scalar comparisons:
```cpp
bool PlanningNode::is_better_than(const PlanningNode& other) const {
    // 1. Primary: Trajectory Score
    if (score != other.score) return score > other.score;

    // 2. Goal reached indicator
    if (goal_reached != other.goal_reached) return goal_reached;

    // 3. Manhattan distance to goal (smaller is better)
    if (manhattan_to_goal != other.manhattan_to_goal) return manhattan_to_goal < other.manhattan_to_goal;

    // 4. Candidate path length (shorter depth is better)
    if (depth != other.depth) return depth < other.depth;

    // 5. Cumulative log confidence (higher is better)
    if (cum_log_confidence != other.cum_log_confidence) return cum_log_confidence > other.cum_log_confidence;

    // 6. Target step probability (higher is better)
    if (target_probability != other.target_probability) return target_probability > other.target_probability;

    // 7. Strict deterministic tie-break: Lexicographical action sequence
    const size_t min_len = std::min(actions_from_root.size(), other.actions_from_root.size());
    for (size_t i = 0; i < min_len; ++i) {
        auto a1 = static_cast<uint8_t>(actions_from_root[i]);
        auto a2 = static_cast<uint8_t>(other.actions_from_root[i]);
        if (a1 != a2) return a1 < a2;
    }

    if (actions_from_root.size() != other.actions_from_root.size()) {
        return actions_from_root.size() < other.actions_from_root.size();
    }

    return false; // Identical nodes
}
```

### 3.3 Formal Verification of Axioms
The mathematical properties of `is_better_than` were empirically verified in `test_strict_weak_ordering_axioms` on 1,600 node comparison pairs and 64,000 triplet permutations:
1. **Irreflexivity**: $\forall a, \neg(a \prec a)$ — **Verified (0 violations)**.
2. **Asymmetry**: $\forall a, b, a \prec b \implies \neg(b \prec a)$ — **Verified (0 violations)**.
3. **Transitivity**: $\forall a, b, c, (a \prec b \land b \prec c) \implies a \prec c$ — **Verified (0 violations)**.
4. **Transitivity of Incomparability**: $\forall a, b, c, (a \sim b \land b \sim c) \implies a \sim c$ — **Verified (0 violations)**.

---

## 4. Multi-Tier Confidence Metrics

A common error in sequence search is confusing the instantaneous transition probability at the horizon leaf ($p_L$) with the joint probability of the entire rollout path ($\prod_{t=1}^L p_t$). `PlanningNode` now maintains three separate metrics:

1. **Step Transition Probability** ($p_L$):
   $$\text{target\_probability} = P(s_L \mid s_{L-1}, a_L)$$
2. **Cumulative Log Confidence** ($\log C$):
   $$\text{cum\_log\_confidence} = \sum_{t=1}^L \log(\max(p_t, \epsilon)), \quad \epsilon = 10^{-7}$$
   Numerically clamped to avoid $-\infty$. Root initialized at $0.0$.
3. **Geometric Mean Step Confidence** ($\bar{C}$):
   $$\text{mean\_step\_confidence} = \exp\left(\frac{1}{L} \sum_{t=1}^L \log(\max(p_t, \epsilon))\right)$$
   Normalized measure representing the effective per-step reliability along the hypothetical trajectory.

---

## 5. Empirical Validation Across Mock Scenarios

To verify planning logic independently of neural weight training dynamics, we designed a suite of deterministic mock transition predictors and tested 7 distinct scenarios.

| Scenario | Objective | Ground Truth / Model Setup | Observed Outcome | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Scenario A** | Short vs Long Route | Goal at cell 2. Route 1: 2 steps. Route 2: 4 steps. | Route 1 ($J=103.0$) chosen over Route 2 ($J=100.0$). | **PASSED** |
| **Scenario B** | Misleading Heuristic vs Goal | Completed route at depth 1 vs $D=0$ unfinished node. | Completed candidate preserved & selected ($J=104.5$). | **PASSED** |
| **Scenario C** | Early Goal Preservation | Goal reached at depth 2; search continues to depth 6. | Depth 2 node preserved across horizons 3, 4, 5, 6. | **PASSED** |
| **Scenario D** | Exact Tie-Breaking Determinism | 4 candidate paths with identical $D=2, L=2, J=-5.0$. | Lexicographical order selects $[1, 1]$ across 100/100 runs. | **PASSED** |
| **Scenario E** | Unreachable Goal Fallback | Goal at 24 unreachable within depth 2 horizon. | Clean fallback to best partial candidate; `goal_reached=false`. | **PASSED** |
| **Scenario F** | Model Hallucination Isolation | Wall at cell 1 (BFS=4). Mock hallucinates passage. | Search plans $[R, R]$ without simulator leakage. | **PASSED** |
| **Scenario G** | Search Consistency | Exhaustive search ($5^2=25$) vs Beam Search ($K=25$). | Bitwise identical best plan, score, and nodes expanded. | **PASSED** |

### Key Scientific Insight on Scenario F (Model Autonomy):
In Scenario F, the ground-truth environment placed an obstacle at cell $(0, 1)$, requiring a 4-step detour in the real grid. However, the mock model hallucinated that cell $(0, 1)$ was free. The planning engine produced the 2-step plan `[Right, Right]`. 
This proves that **the planning search operates strictly in the model's subjective hypothesis space and never leaks or queries ground-truth simulator checks during planning**.

---

## 6. Integration and CLI Demonstration

We introduced the CLI command `spark plan-beam` in `src/main.cpp`. Testing against the trained checkpoint `checkpoints/world_model_best.spark`:

```bash
$ ./build/spark plan-beam --width 4 --depth 4 --episodes 3
Configuration: Beam Width K=4, Max Depth H=4, Episodes=3
Loaded world model from: checkpoints/world_model_best.spark

Evaluating Bounded Beam Search Across Episodes:
  Episode 1: Start=(1,1) -> Goal=(1,0) | Ref BFS: 1 steps
    Plan Found: 1 steps (Left ) | Goal Reached: YES | Final Pos: (1,0) | Score: 102.50 | Mean Conf: 99.9% | Expanded: 13 | Time: 12.48 ms
  Episode 2: Start=(3,4) -> Goal=(4,1) | Ref BFS: 4 steps
    Plan Found: 2 steps (Left Down ) | Goal Reached: NO | Final Pos: (4,3) | Score: -5.00 | Mean Conf: 99.3% | Expanded: 13 | Time: 6.35 ms
  Episode 3: Start=(1,2) -> Goal=(0,4) | Ref BFS: 3 steps
    Plan Found: 3 steps (Right Right Up ) | Goal Reached: YES | Final Pos: (0,4) | Score: 99.50 | Mean Conf: 93.9% | Expanded: 13 | Time: 4.27 ms

Evaluation Summary: 2/3 (66.67%) goals reached in model imagination.
```

---

## 7. Test Suite Summary

All 10 test suites pass with 100% success rate:
```
Test project /home/maen/Builds/OpenUltron/build
 1/10 Test  #1: test_graph .......................   Passed    0.01 sec
 2/10 Test  #2: test_forward .....................   Passed    0.01 sec
 3/10 Test  #3: test_gradients ...................   Passed    0.02 sec
 4/10 Test  #4: test_learning ....................   Passed    0.04 sec
 5/10 Test  #5: test_baselines ...................   Passed    0.02 sec
 6/10 Test  #6: test_regression ..................   Passed    0.01 sec
 7/10 Test  #7: test_local_learning ..............   Passed    0.08 sec
 8/10 Test  #8: test_gridworld ...................   Passed    0.20 sec
 9/10 Test  #9: test_rollout .....................   Passed    0.04 sec
10/10 Test #10: test_planning ....................   Passed    0.08 sec

100% tests passed out of 10
Total Test time (real) = 0.54 sec
```
