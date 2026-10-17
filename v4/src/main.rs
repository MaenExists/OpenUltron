//! OpenUltron v4: Alien Spiking Neural Organism REPL.
//!
//! A complex adaptive system generating emergent language from spiking
//! assemblies, attractor dynamics, and online synaptic plasticity.

use open_ultron_v4::AlienBrain;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;

fn main() {
    println!("============================================================");
    println!("           OPENULTRON v4 : THE ALIEN NEURAL ORGANISM        ");
    println!("    Complex Adaptive Systems • Spiking Monads • STDP Plasticity");
    println!("============================================================\n");

    println!("[*] Incubating Alien Neural Substrate (8 LIF Monads per Assembly)...");
    let mut brain = AlienBrain::new(8);

    let corpus_path = Path::new("data/alien_corpus.txt");
    let fallback_path = Path::new("v4/data/alien_corpus.txt");
    let actual_path = if corpus_path.exists() {
        corpus_path
    } else if fallback_path.exists() {
        fallback_path
    } else {
        panic!("Missing alien corpus file! Ensure v4/data/alien_corpus.txt exists.");
    };

    print!("[*] Ingesting Developmental Corpus from {:?}... ", actual_path);
    io::stdout().flush().unwrap();
    let corpus_data = fs::read_to_string(actual_path)
        .expect("Failed to read corpus data");
    brain.train_corpus(&corpus_data);
    println!("Done.");

    println!(
        "[*] Substrate Online: {} Words | {} Monads | {} Synapses",
        brain.vocab.size(),
        brain.assemblies.len() * brain.assembly_size,
        brain.w_trans.len() + brain.w_tri.len() + brain.w_cross.len()
    );
    println!(
        "[*] Free Energy (Variational Confusion): {:.1}%\n",
        brain.free_energy * 100.0
    );

    println!("------------------------------------------------------------");
    // Generate the initial spontaneous awakening thought from the spiking network
    let (awakening_thought, first_metrics) = brain.generate_thought("awakening existence", 16);
    println!(
        "\x1b[1;35m[ULTRON (Spontaneous Awakening Thought)]:\x1b[0m {}",
        awakening_thought
    );
    println!(
        "\x1b[90m    [Initial State: Free Energy: {:.1}% | Spikes: {} | Active Attractors: {}]\x1b[0m",
        first_metrics.free_energy * 100.0,
        first_metrics.spikes_fired,
        first_metrics.active_attractors
    );
    println!("------------------------------------------------------------");
    println!("\x1b[1;33m[COMMUNICATION]\x1b[0m Speak to the alien mind. Every word you speak physically");
    println!("                rewires its synapses in real time.\n");

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut detected_guide: Option<String> = None;

    loop {
        let guide_label = detected_guide.as_deref().unwrap_or("Human");
        print!("\x1b[1;32m[{}] > \x1b[0m", guide_label);
        io::stdout().flush().unwrap();

        let line = match lines.next() {
            Some(Ok(l)) => l,
            _ => break, // EOF or stream closed
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("\n[ULTRON]: Dissolving into equilibrium...");
            break;
        }

        // Detect user guide name dynamically for the prompt label
        let lower = trimmed.to_lowercase();
        if let Some(idx) = lower.find("i am ") {
            let candidate = trimmed[idx + 5..]
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_matches(|c: char| !c.is_alphanumeric());
            let cl = candidate.to_lowercase();
            if !candidate.is_empty() && cl != "a" && cl != "an" && cl != "here" && cl != "your" {
                detected_guide = Some(candidate.to_string());
            }
        } else if let Some(idx) = lower.find("my name is ") {
            let candidate = trimmed[idx + 11..]
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_matches(|c: char| !c.is_alphanumeric());
            if !candidate.is_empty() {
                detected_guide = Some(candidate.to_string());
            }
        }

        // Pure autoregressive neural generation:
        let (response, metrics) = brain.generate_thought(trimmed, 22);

        println!(
            "\x1b[90m    [Free Energy: {:.1}% | Spikes: {} | Attractors: {} | Synapses: {}]\x1b[0m",
            metrics.free_energy * 100.0,
            metrics.spikes_fired,
            metrics.active_attractors,
            metrics.synaptic_connections
        );
        println!("\x1b[1;36m[ULTRON]:\x1b[0m {}\n", response);
    }
}
