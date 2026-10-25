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

