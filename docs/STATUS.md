# Project Status Tracking

## Completed Phases
- **Phase 0: Research & Repository Foundation** (Completed: 2026-10-25)
  - CMake C++17 build harness with strict warnings and native optimizations.
  - Reproducible PRNG engine (`spark::Random`).
  - Configuration structs and range validation (`spark::Config`).
  - Core documentation (`AGENTS.md`, `README.md`, `ARCHITECTURE.md`, `SCIENTIFIC_MODEL.md`, `RESEARCH_PLAN.md`, `DECISIONS.md`).
- **Phase 1: Sparse Recurrent Computation** (Completed: 2026-10-25)
  - Compressed Sparse Row graph representation (`spark::SparseGraph`) with Random and Clustered topologies.
  - Homogeneous MicroUnit state arrays with contiguous FP32 vectors and synchronous double buffering (`spark::MicroUnitNetwork`).
  - Mathematical recurrence equations:
    $a_i(t) = \alpha h_i(t-1) + \frac{\beta}{\sqrt{K_i}} \sum_{j \in \mathcal{N}(i)} w_{ij} h_j(t-1) + \sum_d U_{id} x_d(t) + b_i$, $h_i(t) = \tanh(a_i(t))$.
  - Numerical safeguards and stable activations (sigmoid, softmax).
  - Profiling metrics (`spark::Metrics`, high-res timer, memory profiler).
  - CLI binary `spark` / `openultron` (`info`, `demo`, `benchmark`).
  - 100% CTest pass rate (`test_graph`, `test_forward`).
  - Verified 10k-unit CPU inference benchmark (~194.59 M-edges/s, 0.82 ms/tick).

- **Phase 2: Learning with Exact Gradients (TBPTT)** (Completed: 2026-10-25)
  - Abstract `Environment` interface and `TemporalMemoryEnv` (delay-copy task with independent seeds).
  - Exact Truncated Backpropagation Through Time (`spark::TBPTTTrainer`, activation records, backward pass through recurrent graph, projections, unit biases, and self-retention).
  - Stable binary cross-entropy loss with logits and exact gradient accumulation.
  - Global gradient norm computation and clipping (`Gradients::clip_by_global_norm`).
  - `SGDOptimizer` with momentum and weight decay.
  - Rigorous finite-difference gradient checker (`verify_gradients_finite_difference`) with relative error $< 2.4 \times 10^{-6}$.
  - Automated unit tests (`test_gradients`, `test_learning`) with 100% pass rate.
  - Verified learning on delay-copy task: 100% test accuracy on delay-1, 72.59% on delay-3.
  - Command-line training interface `spark train`.

- **Phase 3: Scientific Validation & Memory Investigation** (Completed: 2026-10-25)
  - Code audit confirming target alignment, warmup exclusion, and generator isolation.
  - Reproduced Phase 2 baseline results (100% on Delay-1, 72.59% on Delay-3).
  - Implemented 5 control baselines: Shift-Register Oracle (100%), Constant Predictor (49.25%), Memoryless Predictor (54.77%), Untrained SPARK (46.73%), Trained SPARK (100.00%).
  - Evaluated memory capacity across Delays 1, 2, 3, 5 with 64 and 128 units over 80 seeded runs (Delay-1: 100%, Delay-3: 82.28%, Delay-5: 71.90%).
  - Diagnosed Delay-3 degradation: identified recurrent signal attenuation $\beta^d$; confirmed scaling $\beta=1.5$ reaches **100.00% accuracy** on Delay-3.
  - Inference-time ablations verified memory depends on learned recurrent weights and dynamics (state reset drops accuracy to 44.72%, $W=0$ drops to 51.76%).
  - Automated regression suite (`test_baselines`, `test_regression`) passing 100%.
  - Produced `docs/PHASE3_FINDINGS.md` and CSV export registries.

- **Phase 4: Local Learning & Multi-Mode Comparison** (Completed: 2026-10-25)
  - Resolved Step 0 audit discrepancies (143/197 reporting breakdown, Jacobian singular value analysis, 10,000-bit test sequences).
  - Implemented CSR-aligned local `EligibilityTraces` ($E_{ij}, E^U_{id}, E^b_i$) with decay $\lambda$.
  - Implemented `FeedbackProjection` supporting Mode B (Random Feedback matrix $B$) and Mode C (Readout Transpose $V^T$).
  - Implemented `LocalLearner` with exact direct readout gradient and local recurrent/input/bias updates.
  - Benchmarked Modes A, B, C, D across Delays 1, 3, 5 for $N \in \{64, 128\}$ across 240 runs with 10 seeds each.
  - Performed representation freezing ablations confirming active recurrent adaptation ($+9.23\%$) and trace criticality ($-17.40\%$ without traces).
  - Verified ~48% training memory reduction of local learning vs TBPTT (9.5 KB vs 18.4 KB).
  - Automated tests passing 100% (`test_local_learning`, all 7 CTest suites passing).
  - Produced comprehensive scientific report `docs/PHASE4_FINDINGS.md` and CSV registries.

## In Progress
- **Phase 5: World Models, Causal Learning & Predictive Generation**
  - **Milestone 5A — Grid-world foundation** (Completed: 2026-10-26)
    - Implemented 5x5 `GridWorldEnv` with deterministic movement rules, boundary collisions, obstacle collisions, and agent placement guards (`include/spark/gridworld.hpp`, `src/gridworld.cpp`).
    - Implemented fully observable (55 dims) and partially observable (56 dims with mask) observation encodings.
    - Implemented reproducible layout generator and dataset split generator with 0-overlap uniqueness guarantees across train/val/test splits.
    - Extended `TBPTTTrainer` and `Trainer` with `LossType::CategoricalCrossEntropy` using stable log-sum-exp and softmax readout.
    - Implemented `verify_categorical_gradients_finite_difference` with verified relative error $< 4.3 \times 10^{-5}$ across all 5 parameter groups.
    - 10-point test suite `test_gridworld` passing 100%; all 8/8 CTests passing in 0.16s.
  - **Milestone 5B — One-step world-model learning** (Completed: 2026-10-26)
    - Implemented multi-layout cycling in `GridWorldEnv` (100 train, 20 val, 20 unseen test layouts).
    - Implemented parameter training toggle controls in `SGDOptimizer` and `Trainer` to support Readout-Only ablation.
    - Implemented `WorldModelMetrics` with granular breakdown (successful moves, blocked moves, boundary collisions, obstacle collisions).
    - Evaluated 4 non-learning baselines on 10,000 transitions on unseen test layouts: Random (4.28%), Copy Current (49.66%), Naive Move (86.05%), Ground-Truth Oracle (100.00%).
    - Evaluated Readout-Only SPARK reaching 41.35% accuracy.
    - Evaluated SPARK (TBPTT, 128 units, fan-in 8) across 5 independent seeds on 10,000 held-out transitions per seed: **87.46% ± 1.55%** overall accuracy, **98.26%** boundary collision accuracy, **89.58%** successful move accuracy, outperforming the Naive Movement baseline.
    - Exported machine-readable results to `experiments/results/phase5/milestone5b_transitions.csv`.
    - Unit test suite extended to 11 tests passing 100%.
  - **Milestone 5C — Partial observability and memory** (Completed: 2026-10-26)
    - Implemented 56-dimensional partially observable environment protocol (`pos_mask = 1.0` at $t=0$, `pos_mask = 0.0` with strictly zeroed position vector for $t \ge 1$).
    - Implemented `PartialObsMetrics`, step-level diagnostics, and automated evaluators for intact models, ground-truth oracle, and random baselines.
    - Implemented inference ablations: `ResetStateEachStep` (severing recurrent belief state) and `ZeroRecurrentWeights` ($W=0$).
    - Implemented training ablations: Readout-Only and Frozen Recurrent ($W$ fixed at random initialization).
    - Benchmarked multi-step tracking over horizons $H \in \{2, 4, 8, 16\}$ on 20 unseen test layouts across 5 seeds:
      - Horizon 2: Intact SPARK **40.50% ± 3.34%** vs Reset State **28.40%** vs Random **4.50%**.
      - Horizon 4: Intact SPARK **31.60% ± 1.47%** vs Reset State **17.95%** vs Random **3.75%**.
      - Horizon 8: Intact SPARK **25.95% ± 0.99%** vs Reset State **13.21%** vs Random **3.94%**.
      - Horizon 16: Intact SPARK **22.41% ± 0.72%** vs Reset State **10.59% ± 1.18%** vs Zero Weights **14.32% ± 1.75%** vs Readout-Only **12.81%** vs Random **3.78%**.
      - Step-level memory: at step 2 (1 step after observation hidden), intact SPARK scores **31.4%** vs reset state **9.3%** (+22.1% advantage); at step 4, **25.5%** vs **7.5%** (+18.0% advantage); at step 8, **18.3%** vs **7.8%** (+10.5% advantage); at step 16, **17.6%** vs **10.0%** (+7.6% advantage).
      - Ground-truth oracle achieved **100.00%** across all horizons.
    - Exported 76 experimental runs to `experiments/results/phase5/milestone5c_partial_obs.csv`.
    - Unit test suite extended to 13 tests passing 100%; all 8 CTest suites passing in 0.15s.
  - **Milestone 5D — Predictive generation, open-loop world modeling, and error propagation** (Completed: 2026-10-26)
    - Sub-milestone 5D.0 (Markovian Diagnosis): Investigated 1-step transition error modes; extended training to 20k steps yielding **96.82%** accuracy on unseen test layouts (100% boundary, 98.11% free moves); established feedforward MLP benchmark (99.90%).
    - Sub-milestone 5D.1 (Rollout Engine Infrastructure): Implemented binary checkpoint serialization/deserialization (`save_checkpoint`, `load_checkpoint`); built `RolloutEngine` supporting Teacher-Forced (Mode A) and Autoregressive Open-Loop (Mode B) with deterministic oracle, zero ground-truth leakage, and CLI `spark rollout`. Added `test_rollout` unit tests (9/9 CTests passing in 0.15s).
    - Sub-milestone 5D.2 (Multi-Horizon Statistical Experiments): Evaluated $H \in \{1, 2, 4, 8, 16\}$ across 5 seeds on 20 unseen test layouts (500 episodes / horizon). Quantified open-loop compounding error divergence growing from **0.00%** ($H=1$) to **12.99%** ($H=16$), while maintaining **66.4%** full trajectory accuracy and 0.370 Manhattan error at $H=16$.
    - Sub-milestone 5D.3 (Recurrence Memory Ablations): Demonstrated that recurrence stabilizes open-loop generation. At $H=16$, Intact SPARK (**66.4%** traj acc) strongly outperforms State Reset (**28.0%**), Zero Weights (**22.4%**), and Scrambled Weights (**14.6%**).
    - Sub-milestone 5D.4 (Scientific Baselines & Controls): Compared against Oracle (100%), Feedforward MLP (100% in fully observable Markov setting), Naive-Move (51.8%), Copy-Current (23.4%), and Random (4.0%).
    - Exported empirical CSV datasets to `experiments/results/phase5/milestone5d2_multi_horizon.csv`, `milestone5d3_ablations.csv`, and `milestone5d4_controls.csv`.
    - Documented comprehensive research findings in `docs/MILESTONE5D_FINDINGS.md`.
  - **Milestone 5E — Goal-directed planning & active world models** (In Progress)
    - Sub-milestone 5E.0 (Audit & Diagnostics): Reconciled cold-start vs warmed-up recurrent dynamics ($89.6\%$ cold start vs $97.4\%$ warmed-up), explained evaluation weighting across horizons, and verified exact step sample counts.
    - Sub-milestone 5E.1 (Goal Planning Foundation & Recurrent Branching): Built reachable planning episode generator (`PlanningEpisodeGenerator`); ground-truth BFS oracle (`BFSSolver`) strictly encapsulated for reachability/metrics; lightweight `PlanningNode`; independent recurrent-state branching (`forward_step_branching`) with proven order invariance; configurable trajectory scoring ($J = R_{\text{goal}} - \lambda_d D - \lambda_s L + \text{early\_bonus}$) with 6-tier deterministic tie-breaking; cold-start vs warmed-up root policies; automated test suite `test_planning` (10/10 CTests passing in 0.16s); CLI command `spark plan-foundation`; findings documented in `docs/MILESTONE5E1_FINDINGS.md`.
    - Sub-milestone 5E.2A (Search Algorithms & Bounded Beam Search Core): Implemented `BeamSearchConfig`, `PlanningSearchResult`, `beam_search`, `greedy_search`, and `exhaustive_search`; unpruned depth exploration and model evaluation tracking; automated unit tests for beam bounds, order invariance, and goal preservation.
    - Sub-milestone 5E.2B (Planning Scores, Goal Preservation & Deterministic Decision Ranking): Formalized trajectory value function $J(n) = R_g \mathbf{1}_{goal} - \lambda_d D - \lambda_s L + \lambda_e \mathbf{1}_{goal} (H_{max} - L)$; added defensive parameter validation (`validate`) rejecting non-finite/negative coefficients; fixed floating-point comparison flaw in `PlanningNode::is_better_than` ensuring strict weak ordering compliance; added 3-tier confidence tracking (step prob, cumulative log confidence, mean step confidence); implemented custom predictor harness (`expand_branch_custom`, `beam_search_custom`); verified 7 deterministic mock scenarios (A through G) covering short vs long paths, misleading heuristics, early preservation, lexicographical tie-breaking, fallback on unreachable goals, and model hallucination isolation; added CLI `spark plan-beam`; documented findings in `docs/MILESTONE5E2B_FINDINGS.md`.
    - Sub-milestone 5E.2C (Evaluation Protocol & Baselines): Next.

## Pending Phases
- Phase 6: Iterative Internal Computation
- Phase 7: Adaptive Topology
- Phase 8: INT8 Quantization & Memory Optimization
- Phase 9: Compositional Generalization & Planning
- Phase 10: Final Scientific Research Report
