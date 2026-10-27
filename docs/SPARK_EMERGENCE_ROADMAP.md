# SPARK Emergence Research Roadmap

## 1. Long-Term Research Vision
OpenUltron investigates whether increasingly sophisticated, intelligence-related capabilities can emerge from large populations of tiny, structurally identical, sparsely interconnected computational units (**MicroUnits**) operating on continuous environmental interaction.

Unlike conventional deep learning approaches reliant on dense transformer blocks, billions of static floating-point parameters, and non-local backpropagation through time across large batches, SPARK seeks an alternative computational substrate characterized by:
1. **Ultra-Compact Units**: Progressing from initial FP32 prototyping toward 4-bit discrete internal states and low-precision synaptic strengths.
2. **Homogeneous Computational Primitives**: Strictly identical mathematical update equations across all units. No semantic assignment or architectural asymmetry (no dedicated "concept neurons" or "rule units").
3. **Sparse & Contiguous Topologies**: Fixed local fan-in ($K \ll N$) embedded in flat, cache-efficient CSR memory structures.
4. **Locally Adaptive Online Learning**: Synaptic adaptation driven by local activity eligibility traces and approximate feedback signals, learning continuously from online environmental interaction without global backward passes.
5. **Emergence as an Empirical Hypothesis**: Capabilities (such as long-delay retention, compositional generalization, adaptive recovery, and goal-directed action) must be demonstrated through rigorous, controlled behavioral tests, not assumed simply by increasing parameter count.

---

## 2. Emergence Criteria & Behavioral Evaluation
Scaling network size alone is not proof of intelligence. A capability is considered candidate evidence for emergence only if it satisfies predefined behavioral criteria that smaller or ablated networks fail to exhibit:

| Behavioral Dimension | Evaluation Protocol | Failure Control / Baseline | Emergence Criterion |
|---|---|---|---|
| **A. Extended Temporal Memory** | Delay-$D$ recall where input history must be preserved across non-linear distracting intervals | Memoryless Predictor, Reset State, Frozen Graph | Retention accuracy $>80\%$ at delays where small networks collapse to chance ($1/N$). |
| **B. Multi-Rule Composition** | Learning separate environmental navigation invariants independently, then testing simultaneous application in unseen layouts | Single-rule memorization baselines | Correct combined-rule navigation on composite layouts with zero cross-rule training. |
| **C. Adaptive Online Recovery** | Injecting unexpected environmental transitions (e.g., dynamically blocked pathways) during active rollouts | Static policy rollouts, Open-loop execution | Real-time adjustment of predictive belief state and avoidance of secondary traps without full network reset. |
| **D. History-Dependent Disambiguation** | Partially observable environments where identical current observations require different predictions based on past trajectory | Feedforward Markov baselines, State-Reset models | High predictive accuracy under aliased states solely supported by recurrent state trajectory. |
| **E. Autonomous Goal-Conditioned Dynamics** | Removing external heuristic beam search; evaluating whether recurrent dynamics can directly select optimal actions given a goal vector | Pure random exploration, Greedy Manhattan heuristic | Direct goal arrival rate exceeding external heuristic search without simulator queries during deliberation. |

---

## 3. Phased Research Trajectory

### Milestone 6A: Network Scaling & Local Online Learning Feasibility (Current)
- Establish configurable network sizing ($N \in \{128, 256, 512, 1024, 2048\}$) with fixed fan-in $K=8$.
- Verify contiguous memory scaling, parameter counting, numerical stability, and checkpoint serialization.
- Audit Phase 4 local eligibility-trace learning mechanism against 25-class categorical world-model prediction.
- Implement minimal categorical local-learning extension (Softmax + CCE feedback).
- Measure empirical scaling behavior on equal environmental experience across multiple seeds.

### Milestone 6B: Native 4-bit MicroUnit Architecture
- Author native 4-bit architectural specification (`docs/SPARK_4BIT_MICROUNIT_DESIGN.md`).
- Implement true discrete 4-bit states (16 distinct states per unit) and quantized 4-bit synaptic connection strengths.
- Integer accumulation and fixed-point arithmetic eliminating persistent FP32 shadow buffers.
- Stochastic rounding and integer eligibility accumulation mechanisms.

### Milestone 6C: Continuous Online Interaction & Self-Supervised World Modeling
- Continuous environmental interaction without sequence-boundary network wipes.
- Real-time prediction error driven synaptic adaptation.
- Paired comparison: Local Online Adaptation vs TBPTT control vs Static Representations.

### Phase 7: Dynamic Topology & Structural Plasticity
- Activity-dependent synaptic pruning and rewiring.
- Emergent clustering and modular sparse routing within homogeneous units.
- Homeostatic regulation to prevent network silence or saturation.

---

## 4. Architectural Invariants
All research code developed under the SPARK Emergence Track must strictly adhere to the project invariants defined in `AGENTS.md`:
1. **C++17 Standard Library Only**: No external ML frameworks (PyTorch, TensorFlow, JAX, Python backends).
2. **Strict Homogeneity**: Every MicroUnit executes the exact same mathematical update equations.
3. **Contiguous Memory**: State vectors, adjacency lists, and weights reside in flat contiguous arrays.
4. **Synchronous Double-Buffering**: Recurrent states update from $h(t-1)$ to $h(t)$ synchronously without in-place state corruption.
5. **Scientific Integrity**: Separate exploratory validation from held-out confirmatory evaluation; never invent benchmarks; record all negative results transparently.
