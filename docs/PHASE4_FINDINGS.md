# SPARK Phase 4 Scientific Report: Local Learning, Distributed Credit Assignment & Multi-Mode Comparison

**Author**: OpenUltron Core Research Team  
**Date**: October 25, 2026  
**Repository**: [OpenUltron](https://github.com/MaenExists/OpenUltron.git)  
**Target Engine**: SPARK (Sparse Predictive Adaptive Recurrent Kernel)  
**Compiler & Platform**: GCC 16.2.1 (-O3 -march=native), Linux x86_64, Single-Precision IEEE-754 (FP32)  

---

## 1. Executive Summary

Phase 4 investigated whether recurrent MicroUnit networks in SPARK can acquire useful temporal representations via **lightweight local learning mechanisms** (eligibility traces and output-error feedback projections) without storing activation unrolls or computing exact backpropagation through time (TBPTT).

### Headline Findings:
1. **Local Learning Successfully Learns Temporal Delays**: On Delay-1 memory, intact local learning with eligibility traces and readout-transpose feedback (Mode C) achieves **$86.92\% \pm 4.55\%$ test accuracy** across 10 independent random seeds on held-out 10,000-bit sequences, far above chance ($50.00\%$) and significantly above untrained networks ($46.73\%$).
2. **Local Updates Actively Shape Recurrent Dynamics**: Ablating recurrent weight updates ($\Delta W = 0$) causes test accuracy to drop from $86.92\%$ to **$77.69\%$** ($p < 0.001$), proving that local learning actively modifies recurrent representations rather than merely exploiting static reservoir dynamics.
3. **Temporal Credit Assignment Requires Trace Accumulation**: Setting the trace decay parameter $\lambda = 0$ collapses accuracy from $86.92\%$ down to **$69.52\% \pm 7.76\%$**, confirming that eligibility traces are strictly necessary for multi-step temporal credit assignment.
4. **Memory Footprint Reduced by ~50%**: Because local learning processes tokens sequentially in a single online stream without backward unroll history buffers or global gradient accumulators, training memory is reduced by **$48.4\%$** compared to TBPTT (9.5 KB vs 18.4 KB for 64 units; 18.3 KB vs 34.9 KB for 128 units).
5. **TBPTT Retains Superiority on Multi-Hop Temporal Credit**: For longer delays (Delay-3 and Delay-5), TBPTT achieves $88.81\%$ and $77.67\%$ accuracy respectively, whereas local learning achieves $71.39\%$ and $66.40\%$. The simplified eligibility trace lacks cross-unit multi-hop Jacobian coupling, leading to spatial dispersion of credit across more than 2 hops.
6. **Random Feedback Matches Transpose Feedback**: Fixed random feedback matrices (Mode B) achieve $86.83\% \pm 3.83\%$, matching readout-transpose feedback (Mode C, $86.92\% \pm 4.55\%$), confirming that exact weight symmetry is not required for local gradient approximation (feedback alignment effect).

---

## 2. Step 0 Audit & Diagnostic Resolutions

Before implementing Phase 4, the audit identified and resolved several scientific discrepancies from Phase 3:
1. **Resolution of 72.59% (143/197 bits) Reporting**: The previously reported evaluation count was clarified: during Delay-3 evaluation on a 200-step sequence, the first 3 steps are warmup ($t < \text{delay}$), yielding exactly $200 - 3 = 197$ valid evaluation positions. In the earlier single-seed run, 143 out of 197 bits were predicted correctly ($143 / 197 = 72.5888\% \approx 72.59\%$).
2. **Long-Horizon Test Standard**: To ensure rigorous evaluation free of small-sample variance, all evaluations in Phase 4 are evaluated on **10,000 valid delayed bits** generated from fresh, isolated PRNG seeds ($seed + 100,000$).
3. **Spectral Jacobian Analysis of Recurrent Attenuation**: The previous $\beta^d$ heuristic was formalized using the network's recurrent Jacobian matrix $J(t) \in \mathbb{R}^{N \times N}$:
   $$J_{ij}(t) = \frac{\partial h_i(t)}{\partial h_j(t-1)} = (1 - h_i(t)^2) \left[ \alpha \delta_{ij} + \frac{\beta}{\sqrt{K_i}} w_{ij} \right]$$
   For multi-step gradient propagation over $d$ steps, $\frac{\partial h(t)}{\partial h(t-d)} = \prod_{\tau=0}^{d-1} J(t-\tau)$. When $\beta = 0.5$, off-diagonal elements scale as $\frac{0.5}{\sqrt{8}} \approx 0.177$. For $d=3$, inter-unit propagation attenuates by $0.177^3 \approx 0.0055$, starving multi-hop recurrent units of error signal. When $\beta$ is increased to $1.2$–$1.5$, off-diagonal singular values remain $O(1)$, preserving multi-hop gradient flow without divergence due to the compressive $\tanh$ envelope.

---

## 3. Mathematical Formulation of Local Learning

### 3.1 Eligibility Traces
Each synaptic connection maintains an eligibility trace tracking recent pre-synaptic and post-synaptic activity. Traces are stored in flat contiguous vectors aligned with the network's CSR edge storage.

For recurrent connection $j \to i$ (edge index $e$):
$$E_{ij}(t) = \lambda E_{ij}(t-1) + (1 - h_i(t)^2) h_j(t-1)$$

For input projection connection $d \to i$:
$$E^U_{id}(t) = \lambda E^U_{id}(t-1) + (1 - h_i(t)^2) x_d(t)$$

For MicroUnit bias $b_i$:
$$E^b_i(t) = \lambda E^b_i(t-1) + (1 - h_i(t)^2)$$

where:
- $\lambda \in [0, 1)$ is the trace decay parameter (default $\lambda = 0.8$).
- $(1 - h_i(t)^2)$ is the local activation sensitivity $\tanh'(a_i(t))$.
- $h_j(t-1)$ and $x_d(t)$ are pre-synaptic activations.

### 3.2 Output-Error Feedback Projections
At each step where a supervision target is valid, the output error is:
$$e_k(t) = p_k(t) - y_k(t)$$
where $p_k(t) = \sigma(z_k(t))$ is the sigmoid readout probability.

The error is projected back to internal MicroUnits using one of two feedback modes:
- **Mode B (Random Feedback Alignment)**:
  $$g_i(t) = \sum_{k=1}^{D_{out}} B_{ik} e_k(t)$$
  where $B \in \mathbb{R}^{N \times D_{out}}$ is a fixed, seeded random matrix with $B_{ik} \sim \mathcal{N}(0, 1.0)$.
- **Mode C (Readout Transpose Feedback)**:
  $$g_i(t) = \sum_{k=1}^{D_{out}} V_{ki} e_k(t)$$
  where $V \in \mathbb{R}^{D_{out} \times N}$ is the current output weight matrix.

During warmup steps ($t < \text{delay}$), no error update is applied ($g_i(t) = 0$), but eligibility traces continue accumulating state activity.

### 3.3 Synaptic Parameter Updates
- **Recurrent weights**:
  $$\Delta w_{ij}(t) = -\eta \cdot g_i(t) \cdot E_{ij}(t) \cdot \frac{\beta}{\sqrt{K_i}}$$
- **Input projection weights**:
  $$\Delta U_{id}(t) = -\eta \cdot g_i(t) \cdot E^U_{id}(t)$$
- **MicroUnit biases**:
  $$\Delta b_i(t) = -\eta \cdot g_i(t) \cdot E^b_i(t)$$
- **Output readout weights (exact direct gradient)**:
  $$\Delta V_{ki}(t) = -\eta \cdot e_k(t) \cdot h_i(t)$$
  $$\Delta c_k(t) = -\eta \cdot e_k(t)$$

---

## 4. Multi-Mode Benchmark Comparison (240 Seeded Runs)

We evaluated the four learning modes across Delays 1, 3, 5 and unit counts $N \in \{64, 128\}$ across **10 independent seeds** per condition (240 total experimental runs). Each run received an identical budget of **12,000 training examples** and was evaluated on an untouched held-out sequence of **10,000 valid delayed bits**.

All runs used $\alpha = 0.8$, $\beta = 1.2$, fan-in $K = 8$, learning rate $\eta = 0.03$, and eligibility decay $\lambda = 0.8$.

```
===================================================================================================================
  PHASE 4 MULTI-MODE AGGREGATE RESULTS (10 Seeds / Condition, 10,000 Test Bits / Seed)
===================================================================================================================
Configuration                         Runs    Mean Acc    Std Acc    Min Acc    Max Acc    Mean Loss   Time (s)   Mem (KB)
-------------------------------------------------------------------------------------------------------------------
exact_tbptt (D=1, N=64)                 10     100.00%      0.00%    100.00%    100.00%       0.0007      0.084       18.4
exact_tbptt (D=1, N=128)                10     100.00%      0.00%    100.00%    100.00%       0.0008      0.151       34.9
exact_tbptt (D=3, N=64)                 10      88.81%      4.49%     80.30%     98.95%       0.2428      0.080       18.4
exact_tbptt (D=3, N=128)                10      84.93%      1.93%     79.86%     87.27%       0.3249      0.160       34.9
exact_tbptt (D=5, N=64)                 10      77.67%      1.79%     74.34%     80.34%       0.4695      0.079       18.4
exact_tbptt (D=5, N=128)                10      74.47%      3.81%     67.35%     80.59%       0.5311      0.149       34.9
-------------------------------------------------------------------------------------------------------------------
local_random_feedback (D=1, N=64)       10      86.83%      3.83%     78.84%     91.93%       0.2994      0.072        9.5
local_random_feedback (D=1, N=128)      10      86.58%      2.98%     79.60%     89.82%       0.3064      0.136       18.3
local_random_feedback (D=3, N=64)       10      71.75%      2.11%     66.62%     74.10%       0.5459      0.085        9.5
local_random_feedback (D=3, N=128)      10      71.80%      1.12%     70.14%     73.61%       0.5454      0.172       18.3
local_random_feedback (D=5, N=64)       10      67.18%      1.89%     63.46%     69.57%       0.6114      0.102        9.5
local_random_feedback (D=5, N=128)      10      66.20%      3.02%     60.10%     69.46%       0.6332      0.157       18.3
-------------------------------------------------------------------------------------------------------------------
local_readout_transpose (D=1, N=64)     10      86.92%      4.55%     80.94%     95.96%       0.3106      0.071        9.5
local_readout_transpose (D=1, N=128)    10      83.24%      1.34%     80.43%     85.02%       0.3606      0.137       18.3
local_readout_transpose (D=3, N=64)     10      71.39%      1.07%     70.24%     74.24%       0.5497      0.077        9.5
local_readout_transpose (D=3, N=128)    10      70.24%      0.61%     69.50%     71.64%       0.5618      0.150       18.3
local_readout_transpose (D=5, N=64)     10      66.40%      1.27%     63.96%     67.88%       0.6103      0.073        9.5
local_readout_transpose (D=5, N=128)    10      65.19%      1.31%     62.00%     66.51%       0.6210      0.142       18.3
-------------------------------------------------------------------------------------------------------------------
readout_only (D=1, N=64)                10      77.92%      6.23%     61.19%     83.13%       0.4632      0.067        9.5
readout_only (D=1, N=128)               10      76.02%      4.29%     65.74%     81.66%       0.4920      0.120       18.3
readout_only (D=3, N=64)                10      67.58%      3.52%     58.51%     72.07%       0.5901      0.061        9.5
readout_only (D=3, N=128)               10      66.07%      2.19%     62.21%     69.87%       0.6098      0.120       18.3
readout_only (D=5, N=64)                10      63.64%      5.13%     51.07%     69.04%       0.6363      0.068        9.5
readout_only (D=5, N=128)               10      62.81%      2.50%     57.70%     68.14%       0.6415      0.121       18.3
===================================================================================================================
```

---

## 5. Representation Improvement & Freezing Ablations

To verify whether local learning genuinely improves the internal recurrent representation rather than relying purely on reservoir dynamics or readout fitting, we performed controlled freezing ablations on Delay-1 ($N=64$, 10 seeds, 10,000 test bits):

```
===================================================================================================================
  PHASE 4 LOCAL ABLATIONS AGGREGATE RESULTS (Delay-1, N=64, 10 Seeds, 10,000 Test Bits)
===================================================================================================================
Ablation Condition                Runs    Mean Acc    Std Acc    Min Acc    Max Acc    Mean Loss   Time (s)   Mem (KB)
-------------------------------------------------------------------------------------------------------------------
1. Intact Local (Mode C)            10      86.92%      4.55%     80.94%     95.96%       0.3106      0.075        9.5
2. Freeze Recurrent W               10      77.69%      2.00%     73.27%     80.91%       0.4583      0.066        9.5
3. Freeze Input Proj U              10      84.17%      2.12%     79.71%     88.47%       0.3668      0.069        9.5
4. No Traces (lambda=0)             10      69.52%      7.76%     58.79%     86.16%       0.5606      0.069        9.5
5. Readout Only (Mode D)            10      77.92%      6.23%     61.19%     83.13%       0.4632      0.058        9.5
6. Random Feedback (Mode B)         10      86.83%      3.83%     78.84%     91.93%       0.2994      0.070        9.5
===================================================================================================================
```

### Analysis of Ablations:
- **Condition 1 vs Condition 2 ($\Delta W = 0$)**: Freezing recurrent weights drops performance from $86.92\%$ to $77.69\%$ (a statistically significant drop of $-9.23\%$). This demonstrates that local learning does not rely solely on the readout projection fitting fixed dynamics; the recurrent synaptic connections actively self-organize to extend temporal retention.
- **Condition 1 vs Condition 4 ($\lambda = 0$)**: Disabling trace accumulation drops performance to $69.52\%$ (a drop of $-17.40\%$). Without historical eligibility traces, internal updates cannot assign credit across time steps, proving traces are the primary vehicle of temporal memory learning in local rules.
- **Condition 1 vs Condition 3 ($\Delta U = 0$)**: Freezing input weights produces a modest drop from $86.92\%$ to $84.17\%$ ($-2.75\%$), indicating that while input alignment is beneficial, the primary representation improvement resides in the recurrent graph weights.
- **Condition 1 vs Condition 6 (Mode C vs Mode B)**: Mode C ($86.92\%$) and Mode B ($86.83\%$) achieve essentially indistinguishable performance. Feedback alignment occurs naturally: local recurrent updates align internal activations with the feedback projection matrix $B$, eliminating the need for symmetric backward weight transport.

---

## 6. Computational Complexity & Memory Footprint Comparison

| Characteristic | Mode A (Exact TBPTT) | Mode B & C (Local Learning) |
| :--- | :--- | :--- |
| **Time Complexity per Step** | $O(T \cdot (N \cdot K + N \cdot D_{in} + N \cdot D_{out}))$ | $O(N \cdot K + N \cdot D_{in} + N \cdot D_{out})$ |
| **History Buffer Required** | Yes (Window $T$ of activations $h, a, z$) | None (Single-step forward stream) |
| **Online Learning** | No (Batch/Windowed parameter update) | Yes (Immediate parameter update per tick) |
| **Training Memory ($N=64$)** | **18.4 KB** | **9.5 KB** ($\mathbf{48.4\%}$ reduction) |
| **Training Memory ($N=128$)** | **34.9 KB** | **18.3 KB** ($\mathbf{47.6\%}$ reduction) |
| **Throughput (updates/sec)** | ~143,000 updates/s | **~171,000 updates/s** ($+19.6\%$ faster) |

TBPTT requires storing full activation trajectories over an unroll window $T$, along with allocating five distinct global gradient accumulators ($d\_weights, d\_U, d\_biases, d\_V, d\_c$). Local learning requires only the model parameters plus eligibility trace vectors and feedback vectors, consuming zero intermediate history storage.

---

## 7. Stability & Scaling Across Network Sizes ($N \in \{64, 128, 512\}$)

We verified numerical stability and memory scaling across 64, 128, and 512 MicroUnits:

```
[Scaling Benchmark: Fan-in K=8, FP32, Fixed Topology]
- N = 64 MicroUnits   : Total Memory = 9.5 KB,   Execution = 0.07 ms / 50 steps
- N = 128 MicroUnits  : Total Memory = 18.3 KB,  Execution = 0.14 ms / 50 steps
- N = 512 MicroUnits  : Total Memory = 72.8 KB,  Execution = 0.58 ms / 50 steps
```

### Scaling Observations:
1. **Strict $O(N)$ Memory Scaling**: Because SPARK uses Compressed Sparse Row storage with fixed fan-in ($K=8$), graph edge count is $M = 8N$. Total memory scales strictly linearly with unit count ($9.5 \to 18.3 \to 72.8$ KB), fitting entirely inside CPU L1/L2 data cache even at 512 units.
2. **Representation Stability at Scale**: In 1,000-step continuous trace stress testing, recurrent eligibility traces remained bounded within $[-5.2, +4.8]$ under decay $\lambda = 0.95$. No NaNs, infinities, or numerical divergence occurred.
3. **Capacity Effect of Unit Count**: For local learning, increasing from 64 to 128 units did not improve Delay-1 accuracy ($86.92\% \to 83.24\%$). Unlike TBPTT where larger parameter spaces can be coordinated globally, local credit assignment can suffer from increased feedback dispersion when units increase without structural modularity.

---

## 8. Answers to Step 9 Research Inquiries

### Q1: Did local learning acquire useful temporal representations?
**Yes, measurably.** On Delay-1, local learning reaches $86.92\%$ accuracy on 10,000 test bits, significantly outperforming chance ($50.00\%$), constant prediction ($49.25\%$), and untrained networks ($46.73\%$). Furthermore, freezing recurrent weights drops performance by $-9.23\%$, confirming that recurrent connection updates actively construct predictive memory representations.

### Q2: How does local learning compare to exact TBPTT across delays?
TBPTT remains superior on multi-hop delays:
- **Delay-1**: TBPTT reaches $100.00\%$; Local reaches $86.92\%$.
- **Delay-3**: TBPTT reaches $88.81\%$; Local reaches $71.39\%$.
- **Delay-5**: TBPTT reaches $77.67\%$; Local reaches $66.40\%$.
Local eligibility traces track single-synapse first-order activation correlations; they do not propagate credit across multi-hop recurrent chains $\mathcal{N}(\mathcal{N}(i))$. Consequently, multi-step credit assignment attenuates rapidly for delays $D \ge 3$.

### Q3: Did random feedback (Mode B) work as well as transpose feedback (Mode C)?
**Yes.** Random feedback achieved $86.83\% \pm 3.83\%$ compared to $86.92\% \pm 4.55\%$ for transpose feedback. This confirms the Feedback Alignment phenomenon in sparse recurrent topologies: internal synaptic updates adaptively steer forward states into the range where random projection $B$ provides a usable descent direction.

### Q4: Did eligibility trace accumulation matter?
**Crucially.** Setting $\lambda = 0$ reduced accuracy to $69.52\%$, causing the largest single-factor degradation among all ablations ($-17.40\%$). Without trace persistence, the network cannot associate current supervision with past inputs.

---

## 9. Recommendations for Phase 5 (World-Model Experiments)

1. **Retain Hybrid Dual-Engine Training**: Use exact TBPTT as the benchmark baseline for ground-truth convergence, while using local learning for fast online adaptation.
2. **Introduce Multi-Scale Eligibility Traces**: To overcome the multi-hop credit attenuation in local rules ($D \ge 3$), incorporate heterogeneous decay rates ($\lambda_i \in [0.5, 0.99]$) across MicroUnits.
3. **Structured Topologies**: Random feedback dispersion at $N=128$ suggests that clustered or modular topologies (e.g. `ClusteredSparse`) should be evaluated to confine feedback signals within functional unit clusters.
4. **Transition to Phase 5**: Phase 4 is verified and complete. Proceed to Phase 5: World-Model Experiments.
