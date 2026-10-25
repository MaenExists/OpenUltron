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


