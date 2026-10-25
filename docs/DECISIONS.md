# Architectural Decision Records (ADR)

## ADR 001: Contiguous Flat Arrays Over Pointer-Graph Nodes
- **Context**: Object-oriented neural networks often allocate a `Neuron` object per unit and pointer-based `Synapse` objects.
- **Decision**: Store all unit activations and biases in flat `std::vector<float>`, and graph topology in Compressed Sparse Row (CSR) vectors (`offsets`, `sources`, `weights`).
- **Rationale**: Eliminates heap fragmentation, pointer chasing, and cache misses. Critical for scaling to 100k+ units on consumer CPUs.

## ADR 002: Synchronous Double Buffering
- **Context**: Asynchronous or in-place recurrent state updates introduce order dependence and race conditions.
- **Decision**: Read all state values from $h(t-1)$, write updates into $h(t)$, and swap buffer pointers/indices.
- **Rationale**: Guarantees deterministic, mathematically rigorous recurrence invariant to unit evaluation order.

## ADR 003: Deterministic Pseudo-Random Generation
- **Context**: Scientific reproducibility requires identical runs across platforms and runs.
- **Decision**: Wrap standard `std::mt19937_64` inside `spark::Random` with explicit seed control. Never use unseeded global RNG.
- **Rationale**: Guarantees reproducibility across all experiments.
