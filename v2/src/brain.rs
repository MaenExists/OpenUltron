//! OpenUltron v2 Cognitive Brain Architecture.
//!
//! Orchestrates the tripartite nervous system:
//! 1. Sensory Eyes (Input Receptive Field)
//! 2. Liquid Reservoir (Recurrent Small-World Monad Mesh)
//! 3. Motor Voice (Output Articulators)
//!
//! Features online associative STDP learning and closed-loop conversational generation.

use crate::config::BrainConfig;
use crate::monad::{Monad, MonadId, Synapse};
use crate::motor::MotorVoice;
use crate::sensory::{char_to_index, index_to_char, SensoryEyes, VOCAB_SIZE};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

#[derive(Debug, Clone)]
struct InFlightPulse {
    arrival_time: u64,
    #[allow(dead_code)]
    source_id: MonadId,
    target_id: MonadId,
    weight: f32,
}

pub struct OpenUltronBrain {
    pub config: BrainConfig,
    pub eyes: SensoryEyes,
    pub voice: MotorVoice,
    pub monads: Vec<Monad>,
    pub current_time: u64,
    pending_pulses: VecDeque<InFlightPulse>,

    // Segment index boundaries
    sensory_start: usize,
    reservoir_start: usize,
    motor_start: usize,
    total_monads: usize,
}

impl OpenUltronBrain {
    pub fn new(config: BrainConfig) -> Self {
        let eyes = SensoryEyes::new(config.sensory_per_char);
        let voice = MotorVoice::new();

        let sensory_start = 0;
        let reservoir_start = eyes.total_sensory_monads;
        let motor_start = reservoir_start + config.reservoir_size;
        let total_monads = motor_start + voice.total_motor_monads;

        let mut monads = Vec::with_capacity(total_monads);
        for id in 0..total_monads {
            monads.push(Monad::new(id, &config));
        }

        let mut brain = Self {
            config,
            eyes,
            voice,
            monads,
            current_time: 0,
            pending_pulses: VecDeque::new(),
            sensory_start,
            reservoir_start,
            motor_start,
            total_monads,
        };

        brain.wire_architecture();
        brain
    }

    /// Wire the tripartite brain architecture
    fn wire_architecture(&mut self) {
        let mut seed: u64 = 88888888;
        let mut rand_u32 = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };

        let res_len = self.config.reservoir_size;

        // 1. Wire Sensory -> Reservoir (Clustered Ensemble Projections)
        for c_idx in 0..VOCAB_SIZE {
            let cluster_start = (c_idx * 45) % res_len;
            let sensory_field = self.eyes.receptive_field(index_to_char(c_idx));
            for &s_id in &sensory_field {
                for offset in 0..12 {
                    let target_id = self.reservoir_start + ((cluster_start + offset) % res_len);
                    let weight = 0.90 + 0.30 * ((rand_u32() % 1000) as f32 / 1000.0);
                    self.monads[s_id].synapses.push(Synapse::new(target_id, weight, 1));
                }
            }
        }

        // 2. Wire Reservoir Internal Mesh (Small-World recurrent liquid)
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
                let weight = if is_inhibitory { -w_mag * 0.3 } else { w_mag * 0.7 };
                let delay = 1 + (rand_u32() % 2) as u8;

                self.monads[r_id].synapses.push(Synapse::new(target_id, weight, delay));
            }

            // 3. Wire Reservoir -> Motor Voice Projections
            // Each reservoir cell projects to 4 random motor articulators with initial neutral weights
            for _ in 0..4 {
                let motor_offset = rand_u32() as usize % VOCAB_SIZE;
                let target_id = self.motor_start + motor_offset;
                let weight = 0.05 * ((rand_u32() % 1000) as f32 / 1000.0);
                self.monads[r_id].synapses.push(Synapse::new(target_id, weight, 1));
            }
        }
    }

    /// Inject sensory stimulus for a character
    pub fn perceive_character(&mut self, c: char) {
        let field = self.eyes.receptive_field(c);
        for id in field {
            if let Some(m) = self.monads.get_mut(id) {
                m.receive(1.5); // Strong sensory driving current
            }
        }
    }

    /// Advance the brain by 1 discrete micro-tick
    pub fn tick(&mut self) -> Vec<MonadId> {
        self.current_time += 1;

        // 1. Deliver pending synaptic pulses
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

        // 2. Step integration on all Monads and collect firings
        let mut fired = Vec::new();
        for id in 0..self.total_monads {
            let did_fire = self.monads[id].step_integration(self.current_time, &self.config);
            if did_fire {
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
                    source_id: src_id,
                    target_id: syn.target_id,
                    weight: syn.weight,
                });
            }
        }

        fired
    }

    /// Read motor activations across the vocabulary
    pub fn get_motor_activations(&self) -> Vec<f32> {
        let mut activations = Vec::with_capacity(VOCAB_SIZE);
        for offset in 0..VOCAB_SIZE {
            let m_id = self.motor_start + offset;
            activations.push(self.monads[m_id].potential);
        }
        activations
    }

    /// Train online on a stream of text using predictive STDP and associative reward
    pub fn train_text(&mut self, text: &str) -> (usize, f32) {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 2 {
            return (0, 0.0);
        }

        let mut correct_predictions = 0;
        let mut total_steps = 0;

        for window in chars.windows(2) {
            let current_char = window[0];
            let next_target_char = window[1];

            // 1. Perceive current character through the eyes
            self.perceive_character(current_char);
            let sensory_field = self.eyes.receptive_field(current_char);

            // 2. Propagate for 2 micro-ticks to let waves traverse the reservoir
            let mut recent_active_spikes: Vec<MonadId> = sensory_field;
            for _ in 0..2 {
                let fired = self.tick();
                for &id in &fired {
                    if id >= self.reservoir_start && id < self.motor_start {
                        recent_active_spikes.push(id);
                    }
                }
            }

            // 3. Check what the motor voice predicts
            let motor_activations = self.get_motor_activations();
            let predicted_char = self.voice.decode(&motor_activations, 0.0);

            if predicted_char == next_target_char {
                correct_predictions += 1;
            }
            total_steps += 1;

            // 4. Associative Plasticity (Dual-Pathway: Sensory & Reservoir -> Motor)
            if let Some(target_idx) = char_to_index(next_target_char) {
                let target_motor_id = self.motor_start + target_idx;

                for &src_id in &recent_active_spikes {
                    let mut found = false;
                    for syn in &mut self.monads[src_id].synapses {
                        if syn.target_id == target_motor_id {
                            syn.weight = (syn.weight + self.config.lr_ltp).min(self.config.max_weight);
                            found = true;
                            break;
                        }
                    }
                    if !found && self.monads[src_id].synapses.len() < 35 {
                        self.monads[src_id].synapses.push(Synapse::new(target_motor_id, self.config.lr_ltp, 1));
                    }
                }

                if let Some(pred_idx) = char_to_index(predicted_char) {
                    if pred_idx != target_idx {
                        let wrong_motor_id = self.motor_start + pred_idx;
                        for &src_id in &recent_active_spikes {
                            for syn in &mut self.monads[src_id].synapses {
                                if syn.target_id == wrong_motor_id {
                                    syn.weight = (syn.weight - self.config.lr_ltd).max(self.config.min_weight);
                                }
                            }
                        }
                    }
                }
            }

            // Reset motor potentials after each character step
            for m_id in self.motor_start..self.total_monads {
                self.monads[m_id].potential = 0.0;
            }
        }

        let accuracy = if total_steps > 0 {
            correct_predictions as f32 / total_steps as f32
        } else {
            0.0
        };

        (total_steps, accuracy)
    }

    /// Closed-loop conversational response generation
    pub fn generate_response(&mut self, prompt: &str, max_chars: usize, temperature: f32) -> String {
        let full_prompt = if prompt.starts_with("User: ") || prompt.contains("\nUltron: ") {
            prompt.to_string()
        } else {
            format!("User: {}\nUltron: ", prompt)
        };

        // 1. Prime the network with the prompt (matching the 2-tick training phase)
        for c in full_prompt.chars() {
            self.perceive_character(c);
            self.tick();
            self.tick();
        }

        let mut output = String::new();
        let mut recent_chars: Vec<char> = Vec::new();
        let mut last_char = full_prompt.chars().last().unwrap_or(' ');

        for _ in 0..max_chars {
            self.perceive_character(last_char);
            self.tick();
            self.tick();

            let motor_activations = self.get_motor_activations();
            let next_c = self.voice.decode_with_penalty(&motor_activations, &recent_chars, 0.40, temperature);

            // Reset chosen motor monad to avoid lingering excitation
            if let Some(idx) = char_to_index(next_c) {
                self.monads[self.motor_start + idx].potential = 0.0;
            }

            // Stop condition on newline or sequence end
            if next_c == '\n' && output.len() >= 3 {
                break;
            }

            output.push(next_c);
            recent_chars.push(next_c);
            if recent_chars.len() > 4 {
                recent_chars.remove(0);
            }
            last_char = next_c;
        }

        output
    }

    /// Save brain synaptic wiring to a file using fast buffered writing
    pub fn save_brain(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "# OpenUltron v2 Brain State")?;
        writeln!(writer, "total_monads:{}", self.total_monads)?;

        for m in &self.monads {
            for syn in &m.synapses {
                writeln!(writer, "{} {} {:.6} {}", m.id, syn.target_id, syn.weight, syn.delay)?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    /// Load brain synaptic wiring from a file
    pub fn load_brain(&mut self, path: &str) -> std::io::Result<usize> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut loaded_synapses = 0;

        // Clear existing synapses
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
                        loaded_synapses += 1;
                    }
                }
            }
        }

        Ok(loaded_synapses)
    }

    /// Live telemetry statistics
    pub fn get_telemetry(&self) -> BrainTelemetry {
        let total_synapses: usize = self.monads.iter().map(|m| m.synapses.len()).sum();
        let total_spikes: u64 = self.monads.iter().map(|m| m.total_spikes).sum();
        let avg_energy: f32 = self.monads.iter().map(|m| m.energy).sum::<f32>() / self.total_monads as f32;
        let avg_weight: f32 = if total_synapses > 0 {
            self.monads.iter().flat_map(|m| m.synapses.iter().map(|s| s.weight)).sum::<f32>()
                / total_synapses as f32
        } else {
            0.0
        };

        BrainTelemetry {
            total_monads: self.total_monads,
            reservoir_monads: self.config.reservoir_size,
            total_synapses,
            total_spikes,
            average_energy: avg_energy,
            average_synaptic_weight: avg_weight,
            current_simulation_tick: self.current_time,
        }
    }
}

pub struct BrainTelemetry {
    pub total_monads: usize,
    pub reservoir_monads: usize,
    pub total_synapses: usize,
    pub total_spikes: u64,
    pub average_energy: f32,
    pub average_synaptic_weight: f32,
    pub current_simulation_tick: u64,
}
