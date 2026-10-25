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

## Pending Phases
- Phase 5: World-Model Experiments
- Phase 6: Iterative Internal Computation
- Phase 7: Adaptive Topology
- Phase 8: INT8 Quantization & Memory Optimization
- Phase 9: Compositional Generalization & Planning
- Phase 10: Final Scientific Research Report
