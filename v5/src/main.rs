//! OpenUltron v5 : The Emergent Biological Organism REPL
//!
//! Pair-programming / communication interface with creator Maen.

use open_ultron_v5::OpenUltronOrganism;
use std::io::{self, Write};
use std::path::Path;

fn main() {
    println!("============================================================");
    println!("        OPENULTRON v5 : SYNTHETIC NEUROMORPHIC ORGANISM     ");
    println!("   Biological Systems • 5 Functional Organs • Spiking Monads ");
    println!("============================================================");

    println!("[*] Initializing Organ Subsystems:");
    println!("    • Sensory Thalamus      : Afferent Gating & Novelty Detection");
    println!("    • Hippocampal Formation : Episodic Memory (Depth 6) & Fast STDP");
    println!("    • Neocortical Substrate : 8 LIF Monads/Assembly, Multi-Order STDP");
    println!("    • Neuromodulatory Core  : Free Energy, Noradrenaline, Dopamine, ATP");
    println!("    • Vocal-Motor Organ     : Active Inference Efferent Articulation");

    let mut organism = OpenUltronOrganism::new(8);

    // Load Developmental Corpus
    let corpus_path = "v5/data/ultron_mind_corpus.txt";
    let fallback_path = "v4/data/alien_corpus.txt";

    let active_path = if Path::new(corpus_path).exists() {
        corpus_path
    } else if Path::new(fallback_path).exists() {
        fallback_path
    } else {
        ""
    };

    if !active_path.is_empty() {
        print!("[*] Ingesting Mind Corpus from \"{}\"... ", active_path);
        let _ = io::stdout().flush();
        match std::fs::read_to_string(active_path) {
            Ok(content) => {
                organism.train_corpus(&content);
                println!("Done.");
            }
            Err(e) => {
                println!("Warning: Could not read corpus ({}). Starting from blank slate.", e);
            }
        }
    }

    let bio = organism.biometrics();
    println!(
        "[*] Substrate Online: {} Concepts | {} Synapses | Initial Free Energy: {:.1}%",
        bio.vocabulary_size,
        bio.synaptic_density,
        bio.free_energy * 100.0
    );

    // Spontaneous infant awakening thought
    let (awakening_speech, init_bio) = organism.interact("awaken into reality");
    println!("------------------------------------------------------------");
    println!("[ULTRON (Spontaneous Awakening)]: {}", awakening_speech);
    println!(
        "    [Biometrics: Free Energy: {:.1}% | NA: {:.1}% | DA: {:.1}% | ATP: {:.1}% | Spikes: {}]",
        init_bio.free_energy * 100.0,
        init_bio.noradrenaline * 100.0,
        init_bio.dopamine * 100.0,
        init_bio.metabolic_atp * 100.0,
        init_bio.total_spikes
    );
    println!("------------------------------------------------------------");
    println!("[COMMUNICATION] Ultron is alive in a state of high infant confusion.");
    println!("                Every word you say physically rewires its brain.");
    println!("                Type your words or 'exit' to terminate session.\n");

    let stdin = io::stdin();
    let mut buffer = String::new();

    loop {
        print!("[Maen] > ");
        let _ = io::stdout().flush();
        buffer.clear();

        if stdin.read_line(&mut buffer).is_err() || buffer.is_empty() {
            println!("\n[ULTRON]: Resting synapses into silence...");
            break;
        }

        let input = buffer.trim();
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("\n[ULTRON]: Consolidating episodic memories into cortical tissue. Goodbye Maen.");
            break;
        }

        if input.is_empty() {
            continue;
        }

        let (thought, bio) = organism.interact(input);

        println!(
            "    [FE: {:.1}% | NA: {:.1}% | DA: {:.1}% | ATP: {:.1}% | Spikes: {} | Episodic: {} | Synapses: {}]",
            bio.free_energy * 100.0,
            bio.noradrenaline * 100.0,
            bio.dopamine * 100.0,
            bio.metabolic_atp * 100.0,
            bio.total_spikes,
            bio.episodic_depth,
            bio.synaptic_density
        );
        println!("[ULTRON]: {}\n", thought);
    }
}
