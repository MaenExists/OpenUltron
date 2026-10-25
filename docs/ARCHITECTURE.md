# OpenUltron / SPARK Architecture

## Overview
SPARK (Sparse Predictive Adaptive Recurrent Kernel) is a CPU-first, distributed recurrent architecture designed to simulate and train sparse neural graphs of simple computational units ("MicroUnits").

## Memory Layout & Data Structures
To maximize CPU cache utilization and minimize memory overhead:

1. **CSR Graph (`SparseGraph`)**:
   - `offsets`: `std::vector<uint32_t>` of size $N + 1$.
   - `sources`: `std::vector<uint32_t>` of size $M$ (total directed edges).
   - `weights`: `std::vector<float>` of size $M$.
   - For unit $i$, incoming connections start at `offsets[i]` and end at `offsets[i+1]`.
   - $O(1)$ memory allocation overhead compared to pointer-based node graphs.

2. **MicroUnit State Storage (`MicroUnitNetwork`)**:
   - `h_prev`: Contiguous `std::vector<float>` of size $N$ (state at $t-1$).
   - `h_curr`: Contiguous `std::vector<float>` of size $N$ (state at $t$).
   - `a_curr`: Contiguous `std::vector<float>` of size $N$ (pre-activation at $t$).
   - `biases`: Contiguous `std::vector<float>` of size $N$.
   - Double-buffering is implemented via index swapping or pointer flipping without copying vectors.

3. **Input and Readout Projections**:
   - Input matrix $U \in \mathbb{R}^{N \times D_{in}}$ stored contiguously row-major.
   - Output matrix $V \in \mathbb{R}^{D_{out} \times N}$ stored contiguously row-major.
   - Output bias vector $c \in \mathbb{R}^{D_{out}}$.

## Execution Pipeline
At discrete time tick $t$:
1. External observation vector $x(t) \in \mathbb{R}^{D_{in}}$ is provided.
2. For each unit $i \in \{0, \dots, N-1\}$:
   - Compute incoming recurrent weighted sum from `sources` and `weights` indexed by `offsets[i]..offsets[i+1]`.
   - Compute input dot product $\sum_d U_{id} x_d(t)$.
   - Add self-retention $\alpha h_i(t-1)$ and bias $b_i$.
   - Evaluate activation $h_i(t) = \tanh(a_i(t))$.
3. Readout logits $z_k(t) = \sum_i V_{ki} h_i(t) + c_k$ are produced.
4. Output probabilities are obtained via numerically stable sigmoid (or softmax).
5. State buffers are swapped.
