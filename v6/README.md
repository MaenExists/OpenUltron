# OpenUltron v6 (NanoUltron)
## Pure Neural Transformer Language Model Built From First Mathematical Principles

**NanoUltron v6** is an authentic, un-faked Neural Transformer Language Model implemented completely in native Rust without PyTorch, LibTorch, or any external machine learning frameworks.

---

### 1. Mathematical Architecture
- **Causal Multi-Head Self-Attention (MHA)**:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$
- **Pre-Layer Normalization (RMSNorm)**:
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
- **Feed-Forward MLP with Gaussian Error Linear Units (GELU)**:
  $$\text{GELU}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$$
- **Analytical Gradient Backpropagation Engine**:
  - Full reverse-mode automatic differentiation through attention heads, softmax Jacobians, GELU derivatives, and RMSNorm scale parameters.
- **AdamW Optimizer**:
  - Decoupled $L_2$ weight decay with first-moment ($m_t$) and second-moment ($v_t$) running averages.

---

### 2. Verified Hardware Specifications Audit

Audited directly on the host system:
```
========================================================================
|                      NANOSPEC HARDWARE AUDIT                         |
========================================================================
[HOST HARDWARE PROFILE]
• Processor           : Intel(R) Core(TM) i3-5005U CPU @ 2.00GHz
• Physical Cores      : 2 (Threads: 4)
• CPU L3 Cache        : 3072 KB (Fits model in cache: YES (Zero L3 miss penalty))
• Total Host RAM      : 3821.1 MB
• Available Host RAM  : 749.3 MB
------------------------------------------------------------------------
[NEURAL ARCHITECTURE SPECS]
• Model Architecture  : NanoUltron Autoregressive Transformer
• Layers (Depth)      : 3
• Hidden Dimension    : 64 (Attention Heads: 4)
• Feed-Forward (MLP)  : 256
• Context Window      : 64 tokens
• Vocabulary Size     : 376 tokens
• Total Parameters    : 200,128 weights
• Model Weights RAM   : 0.76 MB (Float32 precision)
• Theoretical FLOPs   : 400,256 FLOPs / token
------------------------------------------------------------------------
[RUNTIME PERFORMANCE BENCHMARK]
• Process Resident RSS: 4.47 MB
• Single-Token Latency: 1.64 ms / token
• Inference Throughput: 610.7 tokens / second
• Compute Throughput  : 0.24 GFLOPs (Parallel Multi-Threaded Rayon)
========================================================================
```

---

### 3. Quickstart & Usage

#### Run the Interactive Shell
```bash
cd v6
cargo run --release
```

#### In-App Commands
- `/specs`: Displays a live hardware and memory audit from `/proc/self/status` and `/proc/cpuinfo`.
- `/bench`: Runs a 100-step token generation benchmark measuring latency and throughput.
- `exit`: Terminates the session.

#### Run Unit Tests
```bash
cargo test
```
All unit tests verify tensor arithmetic, matrix multiplications, softmax normalization, forward pass validity, and loss reduction under gradient backpropagation.
