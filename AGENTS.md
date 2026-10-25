# AGENTS.md — Permanent Guidelines & Architectural Invariants

## 1. Project Mission & Identity
OpenUltron's core computational engine is **SPARK (Sparse Predictive Adaptive Recurrent Kernel)**: an experimental, CPU-first, distributed recurrent intelligence engine implemented in C++17.
The core hypothesis is that useful cognitive and predictive capabilities can emerge from large networks of simple, sparsely connected computational units (MicroUnits) via recurrent interaction, without relying on dense layered transformers or predefined semantic labels.

## 2. Invariants & Non-Negotiable Rules
1. **Language & Build**: C++17, standard library only, CMake build system. No Python, PyTorch, TensorFlow, JAX, or LLM backends for core computation.
2. **Homogeneous MicroUnits**: No neuron or unit is assigned a semantic role (no "math neuron", "language neuron", etc.). All internal units are structurally identical.
3. **Contiguous Memory**: Avoid per-unit or per-connection heap allocations. Use flat contiguous `std::vector<float>` and CSR index structures.
4. **Synchronous Double-Buffering**: Recurrent state updates read from $h(t-1)$ and write to $h(t)$. Buffer swap occurs after all units are calculated. Never update recurrent states in-place.
5. **Exact Mathematical Equations**:
   $$a_i(t) = \alpha h_i(t-1) + \frac{\beta}{\sqrt{K_i}} \sum_{j \in \mathcal{N}(i)} w_{ij} h_j(t-1) + \sum_d U_{id} x_d(t) + b_i$$
   $$h_i(t) = \tanh(a_i(t))$$
6. **Defensive Checks**: Numerical safety checks for NaN, Inf, and array bounds must exist on all critical paths.
7. **Empirical Scientific Integrity**:
   - Never invent benchmark results.
   - Always run the tests and record actual terminal outputs.
   - Separate training metrics from held-out evaluation metrics.
   - Document negative results honestly.

## 3. Incremental Development Protocol
For each milestone:
1. Inspect `AGENTS.md` and `docs/STATUS.md`.
2. State intended files and testing criteria.
3. Implement minimal, complete, compilable components.
4. Compile with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2`.
5. Execute CTest: `ctest --test-dir build --output-on-failure`.
6. Fix failures immediately before proceeding.
7. Update `docs/STATUS.md` and `docs/EXPERIMENTS.md` with verified metrics.
8. Commit working milestones to git and push to GitHub.
