# Research Plan & Empirical Roadmap

## Central Scientific Hypothesis
> A sufficiently organized network of small computational units may acquire useful, transferable cognitive capabilities through distributed learning, recurrent computation, and adaptive connectivity, without requiring explicit programming of individual concepts or problem-solving rules.

## Roadmap & Milestones
- **Phase 0: Research & Repository Foundation**
  - CMake build harness, C++17 configuration, reproducible PRNG, strict compiler warnings.
- **Phase 1: Sparse Recurrent Computation**
  - Contiguous MicroUnit arrays, CSR graph, synchronous double-buffering, forward dynamics, projections, baseline CLI benchmark.
- **Phase 2: Learning with Exact Gradients**
  - TBPTT, numerical finite difference gradient verification, SGD optimizer, gradient clipping, delay-copy task.
- **Phase 3: Scientific Comparison**
  - Baselines (random, majority/Markov), multi-seed evaluation, CSV metric tracking.
- **Phase 4: Local Learning**
  - Eligibility traces, random feedback alignment ($B$), comparative study vs TBPTT.
- **Phase 5: World-Model Experiments**
  - Action-conditioned grid world, held-out layout generalization.
- **Phase 6: Iterative Internal Computation**
  - Microstep recurrence before emission; compute vs accuracy trade-off.
- **Phase 7: Adaptive Topology**
  - Synapse pruning and regrowth under fixed edge budgets.
- **Phase 8: Memory Optimization**
  - INT8 quantization and memory profiling up to 100k units.
- **Phase 9: Compositional Generalization & Planning**
- **Phase 10: Final Scientific Research Report**
