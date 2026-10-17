use open_ultron_v3::config::BrainConfig;
use open_ultron_v3::brain::OpenUltronBrainV3;
use open_ultron_v3::vocab::Vocabulary;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

fn print_banner() {
    println!("\x1b[1;36m==================================================================\x1b[0m");
    println!("\x1b[1;32m                 OpenUltron v3: Semantic Brain                    \x1b[0m");
    println!("\x1b[1;37m     Donald Hebb Cell Assemblies & Lexical Attractor Engine       \x1b[0m");
    println!("\x1b[1;36m==================================================================\x1b[0m");
    println!("Architecture: Word Cell Assemblies + 6,000-Monad Reservoir + Lexical Voice");
    println!("Learning:     Causal STDP Across Semantic Concepts (Zero Backpropagation)\n");
}

fn initialize_brain() -> OpenUltronBrainV3 {
    let conv_path = "data/conversations.txt";
    let lit_path = "data/literature.txt";
    let alice_path = "data/alice.txt";
    let brain_cache = "data/trained_brain.txt";
    let vocab_cache = "data/vocab.txt";

    let mut config = BrainConfig::default();
    config.reservoir_size = 6_000;
    config.assembly_size = 6;
    config.max_vocab_size = 1_200;

    // 1. Build Vocabulary
    let mut vocab = Vocabulary::new();
    let conv_text = fs::read_to_string(conv_path).unwrap_or_default();
    let lit_text = fs::read_to_string(lit_path).unwrap_or_default();
    let alice_text = fs::read_to_string(alice_path).unwrap_or_default();

    // Prioritize conversational words, then literary words
    vocab.build_from_text(&conv_text, 800);
    let combined_literature = format!("{}\n{}", alice_text, lit_text);
    vocab.build_from_text(&combined_literature, config.max_vocab_size);

    println!("\x1b[1;33m[Lexicon]\x1b[0m Built vocabulary of {} semantic word assemblies.", vocab.size());

    let mut brain = OpenUltronBrainV3::new(config, vocab);

    // 2. Load or Train Brain
    if Path::new(brain_cache).exists() && Path::new(vocab_cache).exists() {
        print!("\x1b[1;33m[Bootstrap]\x1b[0m Restoring trained brain state from '{}'... ", brain_cache);
        io::stdout().flush().unwrap();
        let start = Instant::now();
        match brain.load_brain(brain_cache) {
            Ok(count) => println!("\x1b[1;32mLoaded {} synapses in {:.2?}\x1b[0m", count, start.elapsed()),
            Err(e) => println!("\x1b[1;31mError loading brain state (re-training): {}\x1b[0m", e),
        }
    } else {
        print!("\x1b[1;33m[Bootstrap]\x1b[0m Training semantic attractors across conversational corpus... ");
        io::stdout().flush().unwrap();
        let start = Instant::now();

        let conv_tokens = Vocabulary::tokenize(&conv_text);

        // Train 40 epochs across conversational dialogues to solidify attractors
        let mut total_words = 0;
        for _ in 0..40 {
            let (steps, _) = brain.train_tokens(&conv_tokens);
            total_words += steps;
        }

        // Stream literary sample to enrich vocabulary connectivity
        let lit_sample = &alice_text[..alice_text.len().min(12_000)];
        let lit_tokens = Vocabulary::tokenize(lit_sample);
        let (steps, _) = brain.train_tokens(&lit_tokens);
        total_words += steps;

        let elapsed = start.elapsed();
        println!("\x1b[1;32mDone in {:.2?}\x1b[0m ({} word transitions learned)", elapsed, total_words);

        // Cache brain state and vocab
        if let Err(e) = brain.save_brain(brain_cache) {
            eprintln!("Warning: could not cache brain: {}", e);
        }
        if let Ok(mut f) = fs::File::create(vocab_cache) {
            let _ = writeln!(f, "vocab_size:{}", brain.vocab.size());
        }
    }

    let telem = brain.get_telemetry();
    println!("Brain online: {} active synapses across {} total monads.\n", telem.total_synapses, telem.total_monads);
    brain
}

fn main() {
    print_banner();

    let mut brain = initialize_brain();

    println!("\x1b[1;37mCommands:\x1b[0m");
    println!("  \x1b[1;33mchat <text>\x1b[0m        - Talk to OpenUltron (or just type your message directly)");
    println!("  \x1b[1;33mtrain <text>\x1b[0m       - Feed text online into OpenUltron's word assemblies");
    println!("  \x1b[1;33mtrain-file <path>\x1b[0m  - Feed an entire text file into OpenUltron");
    println!("  \x1b[1;33mbrain\x1b[0m              - Display live telemetry and brain stats");
    println!("  \x1b[1;33msave <path>\x1b[0m        - Save brain synaptic state to disk");
    println!("  \x1b[1;33mload <path>\x1b[0m        - Load brain synaptic state from disk");
    println!("  \x1b[1;33mhelp\x1b[0m               - Show this help menu");
    println!("  \x1b[1;33mexit\x1b[0m               - Exit the interface\n");

    let stdin = io::stdin();

    loop {
        print!("\x1b[1;32multron>\x1b[0m ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.read_line(&mut input).is_err() || input.is_empty() {
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "exit" || trimmed == "quit" {
            println!("\x1b[1;33mShutting down OpenUltron v3. The monads rest in peace.\x1b[0m");
            break;
        } else if trimmed == "help" {
            println!("Available commands: chat <text>, train <text>, train-file <path>, brain, save <path>, load <path>, exit");
        } else if trimmed == "brain" || trimmed == "status" {
            let t = brain.get_telemetry();
            println!("\n\x1b[1;36m--- OpenUltron v3 Live Telemetry ---\x1b[0m");
            println!(" Total Monads:                {}", t.total_monads);
            println!("  - Vocabulary Tokens:        {}", t.vocabulary_size);
            println!("  - Assembly Monads (Eyes):   {}", t.assembly_monads);
            println!("  - Reservoir Monads (Mesh):  {}", t.reservoir_monads);
            println!("  - Motor Monads (Voice):     {}", t.motor_monads);
            println!(" Active Synaptic Wiring:      {}", t.total_synapses);
            println!(" Cumulative Spikes Fired:     {}", t.total_spikes);
            println!(" Mean Synaptic Weight:        {:.4}", t.average_synaptic_weight);
            println!(" Current Simulation Tick:     {}\n", t.current_simulation_tick);
        } else if trimmed.starts_with("train-file ") {
            let path = trimmed.strip_prefix("train-file ").unwrap().trim();
            match fs::read_to_string(path) {
                Ok(content) => {
                    println!("Ingesting '{}' ({} bytes)...", path, content.len());
                    let start = Instant::now();
                    let tokens = Vocabulary::tokenize(&content);
                    let (steps, acc) = brain.train_tokens(&tokens);
                    println!(
                        "\x1b[1;32m[Trained]\x1b[0m Processed {} word transitions in {:.2?}. Step accuracy: {:.2}%",
                        steps,
                        start.elapsed(),
                        acc * 100.0
                    );
                }
                Err(e) => println!("\x1b[1;31mError reading file:\x1b[0m {}", e),
            }
        } else if trimmed.starts_with("train ") {
            let text = trimmed.strip_prefix("train ").unwrap();
            let start = Instant::now();
            let tokens = Vocabulary::tokenize(text);
            let (steps, acc) = brain.train_tokens(&tokens);
            println!(
                "\x1b[1;32m[Trained]\x1b[0m Absorbed {} word transitions in {:.2?}. Prediction accuracy: {:.2}%",
                steps,
                start.elapsed(),
                acc * 100.0
            );
        } else if trimmed.starts_with("save ") {
            let path = trimmed.strip_prefix("save ").unwrap().trim();
            match brain.save_brain(path) {
                Ok(_) => println!("\x1b[1;32m[Saved]\x1b[0m Brain state successfully saved to '{}'", path),
                Err(e) => println!("\x1b[1;31mError saving brain:\x1b[0m {}", e),
            }
        } else if trimmed.starts_with("load ") {
            let path = trimmed.strip_prefix("load ").unwrap().trim();
            match brain.load_brain(path) {
                Ok(count) => println!("\x1b[1;32m[Loaded]\x1b[0m Restored {} synapses from '{}'", count, path),
                Err(e) => println!("\x1b[1;31mError loading brain:\x1b[0m {}", e),
            }
        } else {
            // Conversational mode:
            let prompt = if trimmed.starts_with("chat ") {
                trimmed.strip_prefix("chat ").unwrap()
            } else {
                trimmed
            };

            let start = Instant::now();
            let spikes_before = brain.monads.iter().map(|m| m.total_spikes).sum::<u64>();
            let response = brain.generate_response(prompt, 14, 0.10);
            let spikes_after = brain.monads.iter().map(|m| m.total_spikes).sum::<u64>();
            let elapsed = start.elapsed();

            println!("\x1b[1;35mUltron:\x1b[0m {}", response);
            println!(
                "        \x1b[2m({} spikes fired in {:.1?} | {:.0} spikes/sec)\x1b[0m\n",
                spikes_after.saturating_sub(spikes_before),
                elapsed,
                (spikes_after.saturating_sub(spikes_before)) as f64 / elapsed.as_secs_f64().max(0.001)
            );

            // Online real-time plasticity: absorb user prompt
            let user_tokens = Vocabulary::tokenize(prompt);
            brain.train_tokens(&user_tokens);
        }
    }
}
