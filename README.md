# OpenUltron / SPARK

**Sparse Predictive Adaptive Recurrent Kernel (SPARK)** is an experimental, CPU-first distributed recurrent intelligence architecture implemented in modern C++17.

## Key Features
- **Homogeneous MicroUnits**: Uniform computational elements operating without hand-crafted semantic specialization.
- **Sparse Recurrent Graphs**: High-performance CSR (Compressed Sparse Row) representation supporting random and clustered topologies.
- **Cache-Optimized Memory Layout**: Contiguous flat vectors for pre-activations, activations, weights, and double-buffered states.
- **Strict Determinism**: Seeded reproducible PRNG engine and automated verification suites.
- **Empirical Scientific Workflow**: Grounded on baseline comparisons, ablation tests, and resource profiling.

## Building and Testing

### Prerequisites
- CMake >= 3.16
- C++17 compliant compiler (GCC 9+, Clang 10+)
- Linux recommended

### Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

### Run Tests
```bash
ctest --test-dir build --output-on-failure
```

### CLI Commands
```bash
# Print system and build information
./build/spark info

# Run forward-pass demonstration and perturbation analysis
./build/spark demo

# Benchmark forward inference throughput
./build/spark benchmark --units 10000 --fan-in 16 --ticks 100 --seed 42
```
