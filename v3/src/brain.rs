//! OpenUltron v3 Semantic Assembly Brain.
//!
//! Grounded in Donald Hebb's Cell Assembly Theory and Attractor Dynamics:
//! - Words and concepts are embodied as synchronized cell assemblies.
//! - Thoughts propagate as phase sequences across a recurrent liquid mesh.
//! - Motor articulators emit real, coherent words.

use crate::assembly::AssemblyManager;
use crate::config::BrainConfig;
use crate::monad::{Monad, MonadId, Synapse};
use crate::vocab::Vocabulary;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

#[derive(Debug, Clone)]
struct InFlightPulse {
    arrival_time: u64,
    target_id: MonadId,
    weight: f32,
}

pub struct OpenUltronBrainV3 {
    pub config: BrainConfig,
    pub vocab: Vocabulary,
    pub assemblies: AssemblyManager,
    pub monads: Vec<Monad>,
    pub current_time: u64,
    pending_pulses: VecDeque<InFlightPulse>,

    // Segment boundaries
    pub assembly_start: usize,
    pub reservoir_start: usize,
    pub motor_start: usize,
    pub total_monads: usize,
}

impl OpenUltronBrainV3 {
    pub fn new(config: BrainConfig, vocab: Vocabulary) -> Self {
        let vocab_size = vocab.size();
        let assemblies = AssemblyManager::new(vocab_size, config.assembly_size, 0);

        let assembly_start = 0;
        let reservoir_start = assemblies.total_assembly_monads();
        let motor_start = reservoir_start + config.reservoir_size;
        let total_monads = motor_start + vocab_size;

        let mut monads = Vec::with_capacity(total_monads);
        for id in 0..total_monads {
            monads.push(Monad::new(id, &config));
        }

        let mut brain = Self {
            config,
            vocab,
            assemblies,
            monads,
            current_time: 0,
            pending_pulses: VecDeque::new(),
            assembly_start,
            reservoir_start,
            motor_start,
            total_monads,
        };

        brain.wire_architecture();
        brain
    }

    /// Wire the Cell Assembly & Reservoir Network
    fn wire_architecture(&mut self) {
        let mut seed: u64 = 77777777;
        let mut rand_u32 = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };

        let res_len = self.config.reservoir_size;

        // 1. Intra-Assembly Mutual Resonance (Hebb's Rule: Cells that fire together wire together)
        for assembly in &self.assemblies.assemblies {
            for &src in &assembly.monad_ids {
                for &dst in &assembly.monad_ids {
                    if src != dst {
                        self.monads[src].synapses.push(Synapse::new(dst, 0.75, 1));
                    }
                }
            }
        }

        // 2. Assembly -> Reservoir Projections (Reliable Ignition into Cognitive Mesh)
        for (w_idx, assembly) in self.assemblies.assemblies.iter().enumerate() {
            let cluster_start = (w_idx * 17) % res_len;
            for &src in &assembly.monad_ids {
                for offset in 0..6 {
                    let target = self.reservoir_start + ((cluster_start + offset) % res_len);
                    let weight = 0.95 + 0.20 * ((rand_u32() % 1000) as f32 / 1000.0);
                    self.monads[src].synapses.push(Synapse::new(target, weight, 1));
                }
            }
        }

        // 3. Reservoir Recurrent Small-World Mesh (Liquid Core)
        let k = self.config.connections_per_monad;
        let rewire_prob = self.config.small_world_rewire_prob;

        for r_offset in 0..res_len {
            let r_id = self.reservoir_start + r_offset;
            for offset in 1..=(k / 2) {
                let mut target_offset = (r_offset + offset) % res_len;
                let rand_val = (rand_u32() % 10000) as f32 / 10000.0;
                if rand_val < rewire_prob {
                    let rand_target = rand_u32() as usize % res_len;
                    if rand_target != r_offset {
                        target_offset = rand_target;
                    }
                }

                let target_id = self.reservoir_start + target_offset;
                let is_inhibitory = (rand_u32() % 100) < 20; // 80/20 Dale balance
                let w_mag = 0.4 + 0.6 * ((rand_u32() % 1000) as f32 / 1000.0);
                let weight = if is_inhibitory { -w_mag * 0.3 } else { w_mag * 0.65 };
                let delay = 1 + (rand_u32() % 2) as u8;

                self.monads[r_id].synapses.push(Synapse::new(target_id, weight, delay));
            }

            // 4. Reservoir -> Motor Projections (Initial plastic pathways)
            for _ in 0..3 {
                let motor_offset = rand_u32() as usize % self.vocab.size();
                let target_id = self.motor_start + motor_offset;
                let weight = 0.02 * ((rand_u32() % 1000) as f32 / 1000.0);
                self.monads[r_id].synapses.push(Synapse::new(target_id, weight, 1));
            }
        }
    }

    /// Ignite the Cell Assembly for a word
    pub fn perceive_word(&mut self, word: &str) {
        let word_id = self.vocab.token_to_id(word);
        if let Some(assembly) = self.assemblies.get_assembly(word_id) {
            for &id in &assembly.monad_ids {
                if let Some(m) = self.monads.get_mut(id) {
                    m.receive(2.5); // Overwhelming driving current to detonate assembly
                }
            }
        }
    }

    /// Advance simulation by 1 micro-tick
    pub fn tick(&mut self) -> Vec<MonadId> {
        self.current_time += 1;

        // 1. Deliver pending pulses
        let mut remaining = VecDeque::new();
        while let Some(pulse) = self.pending_pulses.pop_front() {
            if pulse.arrival_time <= self.current_time {
                if let Some(target) = self.monads.get_mut(pulse.target_id) {
                    target.receive(pulse.weight);
                }
            } else {
                remaining.push_back(pulse);
            }
        }
        self.pending_pulses = remaining;

        // 2. Step integration
        let mut fired = Vec::new();
        for id in 0..self.total_monads {
            if self.monads[id].step_integration(self.current_time, &self.config) {
                fired.push(id);
            }
        }

        // 3. Step homeostasis
        for id in 0..self.total_monads {
            let f = self.monads[id].last_spike_time == Some(self.current_time);
            self.monads[id].step_homeostasis(f, &self.config);
        }

        // 4. Schedule outgoing pulses
        for &src_id in &fired {
            for syn in &self.monads[src_id].synapses {
                self.pending_pulses.push_back(InFlightPulse {
                    arrival_time: self.current_time + syn.delay as u64,
                    target_id: syn.target_id,
                    weight: syn.weight,
                });
            }
        }

        fired
    }

    /// Train online on a stream of text tokens using associative STDP and contextual phase binding
    pub fn train_tokens(&mut self, tokens: &[String]) -> (usize, f32) {
        if tokens.len() < 2 {
            return (0, 0.0);
        }

        let mut correct = 0;
        let mut total = 0;

        for i in 0..(tokens.len() - 1) {
            let current_word = &tokens[i];
            let target_word = &tokens[i + 1];

            // 1. Ignite current word assembly strongly
            self.perceive_word(current_word);
            let current_id = self.vocab.token_to_id(current_word);
            let mut active_cells = self.assemblies.get_assembly(current_id)
                .map(|a| a.monad_ids.clone())
                .unwrap_or_default();

            // Inject context from previous word if available
            if i > 0 {
                let prev_id = self.vocab.token_to_id(&tokens[i - 1]);
                if let Some(assembly) = self.assemblies.get_assembly(prev_id) {
                    for &id in &assembly.monad_ids {
                        if let Some(m) = self.monads.get_mut(id) {
                            m.receive(0.9);
                        }
                    }
                    active_cells.extend(assembly.monad_ids.iter().cloned());
                }
            }

            // 2. Ripple for 2 micro-ticks
            for _ in 0..2 {
                let fired = self.tick();
                for &id in &fired {
                    if id >= self.reservoir_start && id < self.motor_start {
                        active_cells.push(id);
                    }
                }
            }

            // 3. Predict next word
            let target_id = self.vocab.token_to_id(target_word);
            let target_motor_id = self.motor_start + target_id;

            let mut best_id = 3;
            let mut best_pot = f32::NEG_INFINITY;
            for w_idx in 3..self.vocab.size() {
                let pot = self.monads[self.motor_start + w_idx].potential;
                if pot > best_pot {
                    best_pot = pot;
                    best_id = w_idx;
                }
            }

            if best_id == target_id && best_pot > 0.01 {
                correct += 1;
            }
            total += 1;

            // 4. Associative Plasticity
            for &src in &active_cells {
                let mut found = false;
                for syn in &mut self.monads[src].synapses {
                    if syn.target_id == target_motor_id {
                        syn.weight = (syn.weight + self.config.lr_ltp).min(1.0);
                        found = true;
                        break;
                    }
                }
                if !found && self.monads[src].synapses.len() < 40 {
                    self.monads[src].synapses.push(Synapse::new(target_motor_id, self.config.lr_ltp, 1));
                }
            }

            // Depress incorrect competitor
            if best_id != target_id && best_pot > 0.01 {
                let wrong_motor_id = self.motor_start + best_id;
                for &src in &active_cells {
                    for syn in &mut self.monads[src].synapses {
                        if syn.target_id == wrong_motor_id {
                            syn.weight = (syn.weight - self.config.lr_ltd).max(-0.5);
                        }
                    }
                }
            }

            // Reset motor potentials after step
            for m_id in self.motor_start..self.total_monads {
                self.monads[m_id].potential = 0.0;
            }
        }

        let acc = if total > 0 { correct as f32 / total as f32 } else { 0.0 };
        (total, acc)
    }

    /// Conversational generation using Word Assemblies and 2-step context
    pub fn generate_response(&mut self, prompt: &str, max_words: usize, repetition_penalty: f32) -> String {
        let full_prompt = if prompt.starts_with("User") || prompt.contains("Ultron") {
            prompt.to_string()
        } else {
            format!("User : {} Ultron :", prompt)
        };
        let tokens = Vocabulary::tokenize(&full_prompt);

        // 1. Prime the network with the full prompt sequence
        for window in tokens.windows(2) {
            self.perceive_word(&window[0]);
            self.perceive_word(&window[1]);
            self.tick();
            self.tick();
        }

        let mut generated_words: Vec<String> = Vec::new();
        let mut prev_word = tokens.last().cloned().unwrap_or_else(|| ":".to_string());
        let mut prev_prev_word = if tokens.len() >= 2 { tokens[tokens.len() - 2].clone() } else { "".to_string() };

        for _ in 0..max_words {
            self.perceive_word(&prev_word);
            if !prev_prev_word.is_empty() {
                let prev_id = self.vocab.token_to_id(&prev_prev_word);
                if let Some(assembly) = self.assemblies.get_assembly(prev_id) {
                    for &id in &assembly.monad_ids {
                        if let Some(m) = self.monads.get_mut(id) {
                            m.receive(0.9);
                        }
                    }
                }
            }
            self.tick();
            self.tick();

            let mut best_id = 3;
            let mut best_score = f32::NEG_INFINITY;

            let user_token_id = self.vocab.token_to_id("user");
            let ultron_token_id = self.vocab.token_to_id("ultron");
            let colon_token_id = self.vocab.token_to_id(":");

            for w_idx in 3..self.vocab.size() {
                if w_idx == user_token_id || w_idx == ultron_token_id || w_idx == colon_token_id {
                    continue;
                }
                let m_id = self.motor_start + w_idx;
                let mut score = self.monads[m_id].potential;

                let word_str = self.vocab.id_to_token(w_idx);
                let recent_count = generated_words.iter().rev().take(3).filter(|&w| w == word_str).count();
                if recent_count > 0 {
                    score -= repetition_penalty * (recent_count as f32);
                }

                if score > best_score {
                    best_score = score;
                    best_id = w_idx;
                }
            }

            for m_id in self.motor_start..self.total_monads {
                self.monads[m_id].potential = 0.0;
            }

            let word = self.vocab.id_to_token(best_id).to_string();

            if word == "<eos>" || ([".", "!", "?"].contains(&word.as_str()) && generated_words.len() >= 4) {
                if [".", "!", "?"].contains(&word.as_str()) {
                    generated_words.push(word);
                }
                break;
            }

            prev_prev_word = prev_word;
            prev_word = word.clone();
            generated_words.push(word);
        }

        // Format into a natural English sentence
        let mut formatted = String::new();
        for (i, w) in generated_words.iter().enumerate() {
            if [".", ",", "!", "?", ":", ";"].contains(&w.as_str()) {
                formatted.push_str(w);
            } else {
                if i > 0 {
                    formatted.push(' ');
                }
                formatted.push_str(w);
            }
        }

        if formatted.is_empty() {
            "I am listening. How can I assist you?".to_string()
        } else {
            formatted
        }
    }

    /// Save brain synaptic wiring with fast BufWriter
    pub fn save_brain(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "# OpenUltron v3 Brain State")?;
        writeln!(writer, "total_monads:{}", self.total_monads)?;

        for m in &self.monads {
            for syn in &m.synapses {
                writeln!(writer, "{} {} {:.6} {}", m.id, syn.target_id, syn.weight, syn.delay)?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    /// Load brain synaptic wiring from disk
    pub fn load_brain(&mut self, path: &str) -> std::io::Result<usize> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut loaded = 0;

        for m in &mut self.monads {
            m.synapses.clear();
        }

        for line in reader.lines() {
            let line = line?;
            if line.starts_with('#') || line.starts_with("total_monads:") || line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 4 {
                if let (Ok(src), Ok(dst), Ok(w), Ok(d)) = (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<usize>(),
                    parts[2].parse::<f32>(),
                    parts[3].parse::<u8>(),
                ) {
                    if src < self.monads.len() && dst < self.monads.len() {
                        self.monads[src].synapses.push(Synapse::new(dst, w, d));
                        loaded += 1;
                    }
                }
            }
        }

        Ok(loaded)
    }

    pub fn get_telemetry(&self) -> BrainTelemetryV3 {
        let total_synapses: usize = self.monads.iter().map(|m| m.synapses.len()).sum();
        let total_spikes: u64 = self.monads.iter().map(|m| m.total_spikes).sum();
        let avg_weight: f32 = if total_synapses > 0 {
            self.monads.iter().flat_map(|m| m.synapses.iter().map(|s| s.weight)).sum::<f32>()
                / total_synapses as f32
        } else {
            0.0
        };

        BrainTelemetryV3 {
            total_monads: self.total_monads,
            vocabulary_size: self.vocab.size(),
            assembly_monads: self.assemblies.total_assembly_monads(),
            reservoir_monads: self.config.reservoir_size,
            motor_monads: self.vocab.size(),
            total_synapses,
            total_spikes,
            average_synaptic_weight: avg_weight,
            current_simulation_tick: self.current_time,
        }
    }
}

pub struct BrainTelemetryV3 {
    pub total_monads: usize,
    pub vocabulary_size: usize,
    pub assembly_monads: usize,
    pub reservoir_monads: usize,
    pub motor_monads: usize,
    pub total_synapses: usize,
    pub total_spikes: u64,
    pub average_synaptic_weight: f32,
    pub current_simulation_tick: u64,
}
