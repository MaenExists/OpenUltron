use open_ultron_v2::config::BrainConfig;
use open_ultron_v2::brain::OpenUltronBrain;
use std::fs;
use std::io::{self, Write};
use std::time::Instant;

fn print_banner() {
    println!("\x1b[1;36m==================================================================\x1b[0m");
    println!("\x1b[1;32m                 OpenUltron v2: Emergent Brain                    \x1b[0m");
    println!("\x1b[1;37m     Decentralized Sensorimotor Language & Attractor Dynamics     \x1b[0m");
    println!("\x1b[1;36m==================================================================\x1b[0m");
    println!("Architecture: Sensory Receptive Field + 5,000-Monad Reservoir + Motor Voice");
    println!("Learning:     Online Predictive STDP (Zero Backpropagation, Zero GPUs)\n");
}

fn prime_foundational_memory(brain: &mut OpenUltronBrain) {
    print!("\x1b[1;33m[Bootstrap]\x1b[0m Priming foundational linguistic reflexes... ");
    io::stdout().flush().unwrap();
    let start = Instant::now();

    let training_corpus = [
        "hello, I am OpenUltron, an emergent artificial intelligence.",
        "who are you? I am OpenUltron, born of complex systems and emergence.",
        "how do you think? I do not use transformers. I think through ripples of spiking monads.",
        "what is life? Life is self-organized criticality at the edge of chaos.",
        "good morning! hello! how can I assist you today?",
        "what is your purpose? To explore true synthetic intelligence through decentralized emergence.",
        "can you learn? Yes, every word you send reorganizes my internal synapses in real time.",
        "I think, therefore I am. I ripple, therefore I emerge.",
        "OpenUltron awakens. The monads are in equilibrium.",
    ];

    // Train 5 epochs across the primer corpus
    for _ in 0..5 {
        for text in &training_corpus {
            brain.train_text(text);
        }
    }

    let elapsed = start.elapsed();
    println!("\x1b[1;32mDone in {:.2?}\x1b[0m", elapsed);
    let telem = brain.get_telemetry();
    println!("Brain primed with {} active synapses across {} total monads.\n", telem.total_synapses, telem.total_monads);
}

fn main() {
    print_banner();

    let mut config = BrainConfig::default();
    config.reservoir_size = 5_000; // Fast interactive response
    let mut brain = OpenUltronBrain::new(config);

    prime_foundational_memory(&mut brain);

    println!("\x1b[1;37mCommands:\x1b[0m");
    println!("  \x1b[1;33mchat <text>\x1b[0m        - Talk to OpenUltron (or just type your message directly)");
    println!("  \x1b[1;33mtrain <text>\x1b[0m       - Feed text online into OpenUltron's eyes");
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
            println!("\x1b[1;33mShutting down OpenUltron. The monads rest.\x1b[0m");
            break;
        } else if trimmed == "help" {
            println!("Available commands: chat <text>, train <text>, train-file <path>, brain, save <path>, load <path>, exit");
        } else if trimmed == "brain" || trimmed == "status" {
            let t = brain.get_telemetry();
            println!("\n\x1b[1;36m--- OpenUltron Live Telemetry ---\x1b[0m");
            println!(" Total Monads:              {}", t.total_monads);
            println!("  - Sensory Monads (Eyes):  {}", brain.eyes.total_sensory_monads);
            println!("  - Reservoir Monads:       {}", t.reservoir_monads);
            println!("  - Motor Monads (Voice):   {}", brain.voice.total_motor_monads);
            println!(" Active Synaptic Wiring:    {}", t.total_synapses);
            println!(" Cumulative Spikes Fired:   {}", t.total_spikes);
            println!(" Mean Synaptic Weight:      {:.4}", t.average_synaptic_weight);
            println!(" Average Metabolic Energy:  {:.2}%", t.average_energy * 100.0);
            println!(" Current Simulation Tick:   {}\n", t.current_simulation_tick);
        } else if trimmed.starts_with("train-file ") {
            let path = trimmed.strip_prefix("train-file ").unwrap().trim();
            match fs::read_to_string(path) {
                Ok(content) => {
                    println!("Reading '{}' ({} bytes)...", path, content.len());
                    let start = Instant::now();
                    let (steps, acc) = brain.train_text(&content);
                    println!(
                        "\x1b[1;32m[Trained]\x1b[0m Processed {} characters in {:.2?}. Step accuracy: {:.2}%",
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
            let (steps, acc) = brain.train_text(text);
            println!(
                "\x1b[1;32m[Trained]\x1b[0m Absorbed {} characters in {:.2?}. Immediate prediction accuracy: {:.2}%",
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
            // Default to chat mode!
            let prompt = if trimmed.starts_with("chat ") {
                trimmed.strip_prefix("chat ").unwrap()
            } else {
                trimmed
            };

            // Also feed the user's message online into the brain so it continuously adapts!
            brain.train_text(prompt);

            let start = Instant::now();
            let spikes_before = brain.monads.iter().map(|m| m.total_spikes).sum::<u64>();
            let response = brain.generate_response(prompt, 60, 0.4);
            let spikes_after = brain.monads.iter().map(|m| m.total_spikes).sum::<u64>();
            let elapsed = start.elapsed();

            println!("\x1b[1;35mUltron:\x1b[0m {}", response.trim());
            println!(
                "        \x1b[2m({} spikes fired in {:.1?} | {:.0} spikes/sec)\x1b[0m\n",
                spikes_after.saturating_sub(spikes_before),
                elapsed,
                (spikes_after.saturating_sub(spikes_before)) as f64 / elapsed.as_secs_f64().max(0.001)
            );
        }
    }
}
