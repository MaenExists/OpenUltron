use open_ultron_v2::config::BrainConfig;
use open_ultron_v2::brain::OpenUltronBrain;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

fn print_banner() {
    println!("\x1b[1;36m==================================================================\x1b[0m");
    println!("\x1b[1;32m                 OpenUltron v2: Emergent Brain                    \x1b[0m");
    println!("\x1b[1;37m     Decentralized Sensorimotor Language & Attractor Dynamics     \x1b[0m");
    println!("\x1b[1;36m==================================================================\x1b[0m");
    println!("Architecture: Sensory Receptive Field + 5,000-Monad Reservoir + Motor Voice");
    println!("Learning:     Online Predictive STDP (Zero Backpropagation, Zero GPUs)\n");
}

fn initialize_brain() -> OpenUltronBrain {
    let mut config = BrainConfig::default();
    config.reservoir_size = 5_000;
    let mut brain = OpenUltronBrain::new(config);

    let saved_state_path = "data/trained_brain.txt";
    let conversations_path = "data/conversations.txt";

    if Path::new(saved_state_path).exists() {
        print!("\x1b[1;33m[Bootstrap]\x1b[0m Restoring trained brain state from '{}'... ", saved_state_path);
        io::stdout().flush().unwrap();
        let start = Instant::now();
        match brain.load_brain(saved_state_path) {
            Ok(count) => println!("\x1b[1;32mLoaded {} synapses in {:.2?}\x1b[0m", count, start.elapsed()),
            Err(e) => println!("\x1b[1;31mLoad error (rebuilding fresh): {}\x1b[0m", e),
        }
    } else {
        print!("\x1b[1;33m[Bootstrap]\x1b[0m Training conversational language attractors... ");
        io::stdout().flush().unwrap();
        let start = Instant::now();

        let corpus = if Path::new(conversations_path).exists() {
            fs::read_to_string(conversations_path).unwrap_or_default()
        } else {
            String::from("User: hello\nUltron: hello, I am OpenUltron, an emergent artificial intelligence.\n")
        };

        // Train 8 epochs across the dialogue pairs
        let mut total_chars = 0;
        for _ in 0..8 {
            let (steps, _) = brain.train_text(&corpus);
            total_chars += steps;
        }

        let elapsed = start.elapsed();
        println!("\x1b[1;32mDone in {:.2?}\x1b[0m ({} characters processed)", elapsed, total_chars);

        // Persist trained state for instant startup
        if let Err(e) = brain.save_brain(saved_state_path) {
            eprintln!("Warning: could not cache brain state: {}", e);
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
            // Chat mode:
            let prompt = if trimmed.starts_with("chat ") {
                trimmed.strip_prefix("chat ").unwrap()
            } else {
                trimmed
            };

            let start = Instant::now();
            let spikes_before = brain.monads.iter().map(|m| m.total_spikes).sum::<u64>();
            let response = brain.generate_response(prompt, 60, 0.05);
            let spikes_after = brain.monads.iter().map(|m| m.total_spikes).sum::<u64>();
            let elapsed = start.elapsed();

            println!("\x1b[1;35mUltron:\x1b[0m {}", response.trim());
            println!(
                "        \x1b[2m({} spikes fired in {:.1?} | {:.0} spikes/sec)\x1b[0m\n",
                spikes_after.saturating_sub(spikes_before),
                elapsed,
                (spikes_after.saturating_sub(spikes_before)) as f64 / elapsed.as_secs_f64().max(0.001)
            );

            // Online real-time plasticity: user input leaves an echo
            brain.train_text(prompt);
        }
    }
}
