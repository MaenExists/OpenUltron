use open_ultron_v6::{
    GenerationConfig, Generator, HardwareProfiler, ModelConfig, NanoUltron, Tokenizer,
    Trainer, TrainConfig, Vocabulary,
};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const CHECKPOINT_PATH: &str = "nanoultron_weights.json";
const CORPUS_PATH: &str = "data/chat_corpus.txt";

fn main() {
    println!("========================================================================");
    println!("       OPENULTRON v6 : GENUINE NEURAL TRANSFORMER LANGUAGE MODEL       ");
    println!("   Pure Rust Backpropagation • Zero Canned Templates • Native Rayon CPU  ");
    println!("========================================================================");

    // 1. Load or Build Training Corpus & Tokenizer
    let corpus_content = fs::read_to_string(CORPUS_PATH)
        .unwrap_or_else(|_| panic!("Failed to read training corpus at {}", CORPUS_PATH));

    let vocab = Vocabulary::from_corpus(&corpus_content);
    let vocab_size = vocab.size();
    let tokenizer = Tokenizer::new(vocab);

    println!("[*] Vocabulary initialized: {} unique tokens.", vocab_size);

    // 2. Configure Model Architecture
    let model_config = ModelConfig {
        vocab_size,
        d_model: 64,       // Latent embedding dimension
        n_layers: 3,       // Transformer decoder layers
        n_heads: 4,        // Multi-head self-attention heads
        d_ff: 256,         // MLP intermediate dimension
        max_seq_len: 64,   // Context window
    };

    // 3. Load or Train Model
    let mut model = if Path::new(CHECKPOINT_PATH).exists() {
        println!("[*] Found existing model checkpoint: {}", CHECKPOINT_PATH);
        print!("[*] Loading neural weights... ");
        io::stdout().flush().unwrap();
        match Trainer::load_checkpoint(CHECKPOINT_PATH) {
            Ok(m) => {
                println!("DONE.");
                m
            }
            Err(e) => {
                println!("FAILED ({}), re-initializing.", e);
                NanoUltron::new(model_config.clone())
            }
        }
    } else {
        println!("[*] No checkpoint found. Initializing new Transformer from random Gaussian weights.");
        NanoUltron::new(model_config.clone())
    };

    // If newly initialized or requested, train the model
    if model.optimizer_step == 0 {
        println!("[*] Starting initial training on conversational corpus...");
        let documents: Vec<String> = corpus_content
            .split("\n\n")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let train_config = TrainConfig {
            epochs: 75,
            learning_rate: 3e-3,
            min_learning_rate: 3e-4,
            weight_decay: 0.01,
            seq_len: 32,
            print_every_epochs: 10,
            checkpoint_path: Some(CHECKPOINT_PATH.to_string()),
        };

        let trainer = Trainer::new(train_config);
        trainer.train(&mut model, &tokenizer, &documents);
    }

    // 4. System & Model Specs Audit
    let sys_specs = HardwareProfiler::get_system_specs();
    let model_specs = HardwareProfiler::get_model_specs(&model);
    let bench = HardwareProfiler::benchmark_inference(&model, 5, 20);

    let report = HardwareProfiler::format_full_report(&sys_specs, &model_specs, &bench);
    println!("\n{}\n", report);

    println!("[COMMUNICATION CHANNEL OPEN]");
    println!("  • Chat with Ultron naturally. Every response is generated token by token.");
    println!("  • Commands: '/specs' for hardware audit, '/bench' for benchmark, 'exit' to quit.\n");

    let stdin = io::stdin();
    let mut input_buffer = String::new();

    let user_token_id = tokenizer.vocab.token_to_id("user");
    let gen_config = GenerationConfig {
        max_new_tokens: 48,
        temperature: 0.60,
        top_p: 0.85,
        repetition_penalty: 1.15,
        seed: 0, // dynamic seed
        stop_token_ids: vec![user_token_id],
    };

    loop {
        print!("[Maen] > ");
        io::stdout().flush().unwrap();
        input_buffer.clear();

        if stdin.read_line(&mut input_buffer).is_err() {
            break;
        }

        let trimmed = input_buffer.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "exit" || trimmed == "quit" {
            println!("[*] Terminating Ultron runtime. Goodbye.");
            break;
        }

        if trimmed == "/specs" {
            let current_bench = HardwareProfiler::benchmark_inference(&model, 2, 10);
            let current_sys = HardwareProfiler::get_system_specs();
            let current_model = HardwareProfiler::get_model_specs(&model);
            println!("\n{}\n", HardwareProfiler::format_full_report(&current_sys, &current_model, &current_bench));
            continue;
        }

        if trimmed == "/bench" {
            println!("[*] Running 100-step inference benchmark on 4 threads...");
            let b = HardwareProfiler::benchmark_inference(&model, 10, 100);
            println!(
                "    • Throughput: {:.1} tok/sec | Latency: {:.2} ms/tok | RSS RAM: {:.2} MB | Compute: {:.2} GFLOPs",
                b.tokens_per_second, b.avg_latency_ms_per_token, b.process_rss_mb, b.gflops_achieved
            );
            continue;
        }

        // Format prompt as conversational context
        let query = if trimmed.ends_with('.') || trimmed.ends_with('?') || trimmed.ends_with('!') {
            trimmed.to_string()
        } else {
            format!("{}.", trimmed)
        };
        let prompt = format!("User: {}\nUltron:", query);
        let result = Generator::generate(&model, &tokenizer, &prompt, &gen_config);

        let rss = HardwareProfiler::get_process_rss_mb();
        println!(
            "    [Latency: {:.1} ms | Speed: {:.1} tok/s | Gen: {} tokens | RAM RSS: {:.2} MB]",
            result.elapsed_ms, result.tokens_per_sec, result.generated_tokens, rss
        );

        let mut clean_response = result
            .text
            .trim()
            .split("User")
            .next()
            .unwrap_or(&result.text)
            .split("user")
            .next()
            .unwrap_or(&result.text)
            .trim();

        // Ensure response ends cleanly at the last completed sentence
        if let Some(last_punct_idx) = clean_response.rfind(|c| c == '.' || c == '?' || c == '!') {
            clean_response = clean_response[..=last_punct_idx].trim();
        }

        if clean_response.is_empty() {
            println!("[ULTRON]: I am processing and attending to your input.");
        } else {
            println!("[ULTRON]: {}", clean_response);
        }
        println!();
    }
}
