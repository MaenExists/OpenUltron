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
  - **Milestone 5C — Partial observability and memory** (Next)

## Pending Phases
- Phase 6: Iterative Internal Computation
- Phase 7: Adaptive Topology
- Phase 8: INT8 Quantization & Memory Optimization
- Phase 9: Compositional Generalization & Planning
- Phase 10: Final Scientific Research Report
