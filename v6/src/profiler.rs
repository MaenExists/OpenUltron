use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use crate::model::NanoUltron;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemSpecs {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub l3_cache_kb: usize,
    pub total_ram_mb: f64,
    pub available_ram_mb: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelSpecs {
    pub num_layers: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub total_parameters: usize,
    pub weights_size_mb: f64,
    pub flops_per_token: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RuntimeBenchmark {
    pub process_rss_mb: f64,
    pub avg_latency_ms_per_token: f64,
    pub tokens_per_second: f64,
    pub gflops_achieved: f64,
}

pub struct HardwareProfiler;

impl HardwareProfiler {
    pub fn get_system_specs() -> SystemSpecs {
        let mut cpu_model = "Unknown CPU".to_string();
        let mut cpu_cores = 1;
        let cpu_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let mut l3_cache_kb = 0;
        let mut total_ram_mb = 0.0;
        let mut available_ram_mb = 0.0;

        // Read /proc/cpuinfo
        if let Ok(file) = File::open("/proc/cpuinfo") {
            let reader = BufReader::new(file);
            let mut physical_ids = std::collections::HashSet::new();
            let mut core_ids = std::collections::HashSet::new();

            for line in reader.lines().flatten() {
                if line.starts_with("model name") {
                    if let Some(pos) = line.find(':') {
                        cpu_model = line[pos + 1..].trim().to_string();
                    }
                } else if line.starts_with("cache size") {
                    if let Some(pos) = line.find(':') {
                        let val_str = line[pos + 1..].trim();
                        if let Some(kb) = val_str.strip_suffix("KB") {
                            if let Ok(parsed) = kb.trim().parse::<usize>() {
                                l3_cache_kb = parsed;
                            }
                        }
                    }
                } else if line.starts_with("physical id") {
                    if let Some(pos) = line.find(':') {
                        physical_ids.insert(line[pos + 1..].trim().to_string());
                    }
                } else if line.starts_with("core id") {
                    if let Some(pos) = line.find(':') {
                        core_ids.insert(line[pos + 1..].trim().to_string());
                    }
                }
            }
            if !core_ids.is_empty() {
                cpu_cores = core_ids.len() * physical_ids.len().max(1);
            }
        }

        // Read /proc/meminfo
        if let Ok(file) = File::open("/proc/meminfo") {
            let reader = BufReader::new(file);
            for line in reader.lines().flatten() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(val) = kb.parse::<f64>() {
                            total_ram_mb = val / 1024.0;
                        }
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(val) = kb.parse::<f64>() {
                            available_ram_mb = val / 1024.0;
                        }
                    }
                }
            }
        }

        SystemSpecs {
            cpu_model,
            cpu_cores,
            cpu_threads,
            l3_cache_kb,
            total_ram_mb,
            available_ram_mb,
        }
    }

    pub fn get_process_rss_mb() -> f64 {
        if let Ok(file) = File::open("/proc/self/status") {
            let reader = BufReader::new(file);
            for line in reader.lines().flatten() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(val) = kb.parse::<f64>() {
                            return val / 1024.0;
                        }
                    }
                }
            }
        }
        0.0
    }

    pub fn get_model_specs(model: &NanoUltron) -> ModelSpecs {
        let total_params = model.param_count();
        let weights_size_mb = (total_params * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0);

        // Theoretical FLOPs per token:
        // ~2 FLOPs per param for forward pass (matmul has 1 multiply + 1 accumulate = 2 FLOPs per weight)
        let flops_per_token = (2 * total_params) as u64;

        ModelSpecs {
            num_layers: model.config.n_layers,
            d_model: model.config.d_model,
            num_heads: model.config.n_heads,
            d_ff: model.config.d_ff,
            max_seq_len: model.config.max_seq_len,
            vocab_size: model.config.vocab_size,
            total_parameters: total_params,
            weights_size_mb,
            flops_per_token,
        }
    }

    pub fn benchmark_inference(model: &NanoUltron, warmup_iters: usize, bench_iters: usize) -> RuntimeBenchmark {
        let dummy_tokens = vec![1, 2, 3, 4, 5];
        
        // Warmup
        for _ in 0..warmup_iters {
            let _ = model.forward(&dummy_tokens);
        }

        let start = Instant::now();
        for _ in 0..bench_iters {
            let _ = model.forward(&dummy_tokens);
        }
        let elapsed = start.elapsed();
        let total_tokens_gen = bench_iters; // 1 token per forward step
        let total_seconds = elapsed.as_secs_f64();
        
        let tokens_per_second = total_tokens_gen as f64 / total_seconds;
        let avg_latency_ms = (total_seconds / total_tokens_gen as f64) * 1000.0;
        
        let flops_per_tok = (2 * model.param_count()) as f64;
        let gflops_achieved = (flops_per_tok * tokens_per_second) / 1e9;
        let process_rss_mb = Self::get_process_rss_mb();

        RuntimeBenchmark {
            process_rss_mb,
            avg_latency_ms_per_token: avg_latency_ms,
            tokens_per_second,
            gflops_achieved,
        }
    }

    pub fn format_full_report(sys: &SystemSpecs, model: &ModelSpecs, bench: &RuntimeBenchmark) -> String {
        format!(
            "========================================================================\n\
             |                      NANOSPEC HARDWARE AUDIT                         |\n\
             ========================================================================\n\
             [HOST HARDWARE PROFILE]\n\
             • Processor           : {}\n\
             • Physical Cores      : {} (Threads: {})\n\
             • CPU L3 Cache        : {} KB (Fits model in cache: {})\n\
             • Total Host RAM      : {:.1} MB\n\
             • Available Host RAM  : {:.1} MB\n\
             ------------------------------------------------------------------------\n\
             [NEURAL ARCHITECTURE SPECS]\n\
             • Model Architecture  : NanoUltron Autoregressive Transformer\n\
             • Layers (Depth)      : {}\n\
             • Hidden Dimension    : {} (Attention Heads: {})\n\
             • Feed-Forward (MLP)  : {}\n\
             • Context Window      : {} tokens\n\
             • Vocabulary Size     : {} tokens\n\
             • Total Parameters    : {} weights\n\
             • Model Weights RAM   : {:.2} MB (Float32 precision)\n\
             • Theoretical FLOPs   : {} FLOPs / token\n\
             ------------------------------------------------------------------------\n\
             [RUNTIME PERFORMANCE BENCHMARK]\n\
             • Process Resident RSS: {:.2} MB\n\
             • Single-Token Latency: {:.2} ms / token\n\
             • Inference Throughput: {:.1} tokens / second\n\
             • Compute Throughput  : {:.2} GFLOPs (Parallel Multi-Threaded Rayon)\n\
             ========================================================================",
            sys.cpu_model,
            sys.cpu_cores,
            sys.cpu_threads,
            sys.l3_cache_kb,
            if (model.weights_size_mb * 1024.0) < sys.l3_cache_kb as f64 { "YES (Zero L3 miss penalty)" } else { "NO (Spills to L3/DRAM)" },
            sys.total_ram_mb,
            sys.available_ram_mb,
            model.num_layers,
            model.d_model,
            model.num_heads,
            model.d_ff,
            model.max_seq_len,
            model.vocab_size,
            model.total_parameters,
            model.weights_size_mb,
            model.flops_per_token,
            bench.process_rss_mb,
            bench.avg_latency_ms_per_token,
            bench.tokens_per_second,
            bench.gflops_achieved,
        )
    }
}
