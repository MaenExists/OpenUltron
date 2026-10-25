# Experiments & Benchmark Registry

## Logging Format
Columns tracked for all experiments:
`experiment_id,seed,task,network_units,fan_in,topology,trainer,learning_rate,training_steps,train_loss,validation_loss,test_accuracy,peak_memory_bytes,training_seconds,inference_seconds`

## Phase 1 Benchmarks

```csv
experiment_id,seed,task,network_units,fan_in,topology,trainer,learning_rate,training_steps,train_loss,validation_loss,test_accuracy,peak_memory_bytes,training_seconds,inference_seconds
bench_p1_10k_rnd,42,forward_inference,10000,16,RandomSparse,none,0.0,0,N/A,N/A,N/A,333578240,0.0,0.0822
```

### Hardware & Environment
- **Platform**: Linux x86_64
- **Compiler**: GCC 16.2.1 (-O3 -march=native)
- **Precision**: FP32
- **Network Parameters**: 10,000 units, 16 fan-in (160,000 directed edges, 290,004 total weights)
- **Model Allocation**: 1.87 MB
- **Throughput**: 1.22e+07 units/second (194.59 Million synaptic updates/second)
- **Latency**: 0.8223 ms per tick (100 forward ticks)
- **Peak Resident Memory**: 318.13 MB

## Phase 2 Experiments (TBPTT & Temporal Memory)

```csv
experiment_id,seed,task,network_units,fan_in,topology,trainer,learning_rate,training_steps,train_loss,validation_loss,test_accuracy,peak_memory_bytes,training_seconds,inference_seconds
exp_p2_delay1_64u,42,delay-copy-1,64,8,RandomSparse,tbptt,0.03,1200,0.00882,0.00680,1.0000,12582912,0.07,N/A
exp_p2_delay3_128u,42,delay-copy-3,128,8,RandomSparse,tbptt,0.03,2000,0.51601,0.50241,0.7259,16777216,0.31,N/A
```

### Numerical Gradient Verification (Finite Difference)
- **Architecture**: 8 MicroUnits, fan-in 3, input dim 2, output dim 1, unroll window $T=4$.
- **Analytical vs Numerical Tolerance**: $\epsilon = 10^{-6}$, max tolerance $10^{-3}$.
- **Verified Max Relative Error**: $2.336 \times 10^{-6}$ (Worst parameter: `recurrent_weights[9]`).
- **Conclusion**: Analytical gradient equations for recurrent weights, unit biases, input projection, and output readout are verified down to micro-tolerances.

### Temporal Memory Findings
- **Delay-1 Task (64 units)**: Rapidly achieved **100.00% evaluation accuracy** on held-out test seed within 750 window steps (loss reduced from $0.66014$ to $0.00680$).
- **Delay-3 Task (128 units)**: Reached **72.59% evaluation accuracy** (baseline random guess is 50.00%), verifying multi-step temporal information retention across recurrent steps.

## Phase 3 Experiments (Baselines, Capacity & Delay-3 Diagnosis)

### Baseline Comparisons (Delay-1 & Delay-3, Seed 42)
- **Shift-Register Oracle**: $100.00\%$ accuracy (Loss: $0.00000$)
- **Constant Predictor (Majority)**: $49.25\%$ accuracy (Loss: $5.84256$)
- **Memoryless Predictor (Logistic Regression)**: $54.77\%$ accuracy (Loss: $0.69112$)
- **Untrained SPARK**: $46.73\%$ accuracy (Loss: $0.77433$)
- **Trained SPARK (Delay-1)**: $100.00\%$ accuracy (Loss: $0.00698$)
- **Trained SPARK (Delay-3)**: $76.65\%$ accuracy (Loss: $0.51180$)

### Memory Capacity Scaling (80 Seeded Runs across Delays 1, 2, 3, 5)
Data exported to `experiments/results/capacity_results.csv`:
- **Delay 1 (N=64)**: $100.00\% \pm 0.00\%$ accuracy
- **Delay 1 (N=128)**: $100.00\% \pm 0.00\%$ accuracy
- **Delay 2 (N=64)**: $76.87\% \pm 5.02\%$ accuracy
- **Delay 2 (N=128)**: $76.41\% \pm 4.97\%$ accuracy
- **Delay 3 (N=64)**: $82.28\% \pm 3.21\%$ accuracy
- **Delay 3 (N=128)**: $80.91\% \pm 3.35\%$ accuracy
- **Delay 5 (N=64)**: $71.90\% \pm 2.49\%$ accuracy
- **Delay 5 (N=128)**: $70.31\% \pm 3.33\%$ accuracy

### Delay-3 Diagnostic Factorial Diagnosis
Data exported to `experiments/results/delay3_diagnostic.csv`:
- Attenuation identified as multi-hop propagation scaling with $\beta^d$.
- Default $\beta=0.5$: accuracy $82.23\%$
- Scaling $\beta=1.5$: accuracy **$100.00\%$** (Loss: $0.00949$)

### Inference-Time Recurrence Ablations
- State Reset Every Step: Drops accuracy to $44.72\%$ ($D=1$) and $48.73\%$ ($D=3$).
- Zero Recurrent Weights ($W=0$): Drops accuracy to $51.76\%$ ($D=1$) and $52.79\%$ ($D=3$).
- Confirmed: Temporal retention is stored strictly within learned recurrent dynamics.

## Phase 4 Experiments (Local Learning & Multi-Mode Comparison)

### Multi-Mode Comparison (240 Seeded Runs across Delays 1, 3, 5; N in {64, 128})
Data exported to `experiments/results/phase4_modes_comparison.csv` (10 seeds/config, 10,000 valid bits/seed):

| Mode | Delay 1 (N=64) | Delay 1 (N=128) | Delay 3 (N=64) | Delay 3 (N=128) | Delay 5 (N=64) | Delay 5 (N=128) | Memory (64u / 128u) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode A (Exact TBPTT)** | **100.00%** | **100.00%** | **88.81%** | **84.93%** | **77.67%** | **74.47%** | 18.4 KB / 34.9 KB |
| **Mode B (Random Feedback)** | 86.83% | 86.58% | 71.75% | 71.80% | 67.18% | 66.20% | **9.5 KB / 18.3 KB** |
| **Mode C (Readout Transpose)**| 86.92% | 83.24% | 71.39% | 70.24% | 66.40% | 65.19% | **9.5 KB / 18.3 KB** |
| **Mode D (Readout Only)** | 77.92% | 76.02% | 67.58% | 66.07% | 63.64% | 62.81% | **9.5 KB / 18.3 KB** |

### Local Representation & Freezing Ablations (Delay 1, N=64, 10 Seeds)
Data exported to `experiments/results/phase4_ablations.csv` (10,000 test bits/seed):
- **1. Intact Local (Mode C)**: $86.92\% \pm 4.55\%$ (Loss: $0.3106$)
- **2. Freeze Recurrent W**: $77.69\% \pm 2.00\%$ (Loss: $0.4583$) — **$\Delta = -9.23\%$**
- **3. Freeze Input Proj U**: $84.17\% \pm 2.12\%$ (Loss: $0.3668$) — **$\Delta = -2.75\%$**
- **4. No Traces ($\lambda = 0$)**: $69.52\% \pm 7.76\%$ (Loss: $0.5606$) — **$\Delta = -17.40\%$**
- **5. Readout Only (Mode D)**: $77.92\% \pm 6.23\%$ (Loss: $0.4632$) — **$\Delta = -9.00\%$**
- **6. Random Feedback (Mode B)**: $86.83\% \pm 3.83\%$ (Loss: $0.2994$) — **$\Delta = -0.09\%$**

### Key Takeaways
1. **Recurrent Representation Adaptation**: Recurrent updates in local learning provide a statistically significant $+9.23\%$ improvement over freezing the recurrent reservoir.
2. **Trace Accumulation**: Eligibility traces are strictly required ($\lambda=0$ collapses performance by $-17.40\%$).
3. **Memory Savings**: Local learning reduces training memory overhead by $\sim 48\%$ while delivering online learning without unroll window buffers.




