//! OpenUltron v5 Holistic Neuromorphic Organism.
//!
//! Integrates 5 biological organ subsystems into a unified living cognitive loop:
//! 1. Sensory Thalamus (Afferent gateway & surprise detection)
//! 2. Hippocampal Formation (Episodic memory & fast cross-turn binding)
//! 3. Neocortical Substrate (Spiking Monad assemblies & multi-order STDP)
//! 4. Neuromodulatory Core (Free energy, noradrenaline, dopamine, ATP)
//! 5. Vocal-Motor Organ (Efferent articulation & cadence closure)

use crate::organs::*;
use crate::vocab::Vocabulary;

pub struct OpenUltronOrganism {
    pub vocab: Vocabulary,
    pub sensory: SensoryThalamus,
    pub hippocampus: Hippocampus,
    pub cortex: Neocortex,
    pub neuromodulator: Neuromodulator,
    pub vocal: VocalOrgan,

    // Homeostatic constraints
    pub max_synapse_capacity: usize,
}

impl OpenUltronOrganism {
    pub fn new(assembly_size: usize) -> Self {
        let vocab = Vocabulary::new();
        let cortex = Neocortex::new(vocab.size(), assembly_size);
        let sensory = SensoryThalamus::new();
        let hippocampus = Hippocampus::new(6); // 6 turns of episodic depth
        let neuromodulator = Neuromodulator::new();
        let vocal = VocalOrgan::new(999999999);

        Self {
            vocab,
            sensory,
            hippocampus,
            cortex,
            neuromodulator,
            vocal,
            max_synapse_capacity: 50_000,
        }
    }

    /// Ingest and train on a multi-line developmental corpus.
    pub fn train_corpus(&mut self, corpus: &str) {
        for line in corpus.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let raw_tokens = Vocabulary::tokenize(trimmed);
            if raw_tokens.is_empty() {
                continue;
            }

            let mut token_ids = Vec::with_capacity(raw_tokens.len());
            for t in &raw_tokens {
                let id = self.vocab.get_or_insert(t);
                self.cortex.ensure_assembly(id);
                token_ids.push(id);
            }

            self.sensory.habituate_sequence(&token_ids);

            let ultron_id = self.vocab.token_to_id("ultron");
            let colon_id = self.vocab.token_to_id(":");

            // Check if formatted as: user : ... . ultron : ... .
            let mut split_idx = None;
            for i in 0..token_ids.len() {
                if token_ids[i] == ultron_id && i + 1 < token_ids.len() && token_ids[i + 1] == colon_id {
                    split_idx = Some(i + 2);
                    break;
                }
            }

            if let Some(s_idx) = split_idx {
                let prompt = &token_ids[..s_idx.saturating_sub(2)];
                let response = &token_ids[s_idx..];
                self.cortex.learn_sequence(prompt, response, &self.vocab, 0.6);
            } else {
                self.cortex.learn_sequence(&[], &token_ids, &self.vocab, 0.5);
            }
        }
    }

    /// Primary living interaction loop: processes sensory input, retrieves episodic context,
    /// articulates response, fires biological spikes, and rewires synapses in real time.
    pub fn interact(&mut self, user_input: &str) -> (String, OrganismBiometrics) {
        // 1. Sensory Thalamic Processing
        let burst = self.sensory.process_input(user_input, &mut self.vocab);
        for &id in &burst.token_ids {
            self.cortex.ensure_assembly(id);
        }

        // 2. Hippocampal Episodic Attractor Field
        let vocab_size = self.vocab.size();
        let episodic_attractors = self.hippocampus.get_episodic_attractors(vocab_size);

        // 3. Efferent Vocal-Motor Articulation (emergent speech + LIF spikes)
        let (utterance, emitted_tokens, spikes_fired) = self.vocal.articulate(
            &burst.token_ids,
            &mut self.cortex,
            &self.neuromodulator,
            &episodic_attractors,
            &self.vocab,
            16,
        );

        // 4. Online Synaptic Rewiring & Episodic Memory
        let communication_success = if emitted_tokens.len() >= 4 { 0.85 } else { 0.40 };
        self.neuromodulator.update_state(burst.sensory_surprise, communication_success);
        self.neuromodulator.consume_spike_energy(spikes_fired);

        // Wire user prompt to emitted response live via Cortical STDP
        self.cortex.learn_sequence(
            &burst.token_ids,
            &emitted_tokens,
            &self.vocab,
            self.neuromodulator.dopamine * 0.4,
        );

        let avg_salience = if !burst.salience_weights.is_empty() {
            burst.salience_weights.iter().sum::<f32>() / (burst.salience_weights.len() as f32)
        } else {
            1.0
        };

        // Filter out low-IDF stop words from Hippocampal episodic encoding
        let filtered_user: Vec<usize> = burst
            .token_ids
            .iter()
            .copied()
            .filter(|&id| self.vocab.idf(id) >= 1.4 && !self.vocab.is_punctuation(id))
            .collect();
        let filtered_emitted: Vec<usize> = emitted_tokens
            .iter()
            .copied()
            .filter(|&id| self.vocab.idf(id) >= 1.4 && !self.vocab.is_punctuation(id))
            .collect();

        if !filtered_user.is_empty() {
            self.hippocampus.encode_turn(&filtered_user, avg_salience);
        }
        if !filtered_emitted.is_empty() {
            self.hippocampus.encode_turn(&filtered_emitted, 1.2);
        }

        // Memory consolidation: hippocampal sharp-wave ripples into neocortex
        let ripples = self.hippocampus.consolidate_ripples();
        self.cortex.integrate_hippocampal_ripples(&ripples);

        // 5. Homeostatic Metabolic Synaptic Pruning
        if self.neuromodulator.should_prune_synapses(self.cortex.total_synapses(), self.max_synapse_capacity) {
            self.cortex.prune_synapses(0.35);
        }

        let biometrics = self.biometrics();
        (utterance, biometrics)
    }

    /// Extract complete scientific and biological telemetry of the organism.
    pub fn biometrics(&self) -> OrganismBiometrics {
        OrganismBiometrics {
            free_energy: self.neuromodulator.free_energy,
            noradrenaline: self.neuromodulator.noradrenaline,
            dopamine: self.neuromodulator.dopamine,
            metabolic_atp: self.neuromodulator.metabolic_atp,
            total_spikes: self.cortex.total_spikes,
            synaptic_density: self.cortex.total_synapses(),
            episodic_depth: self.hippocampus.episodic_buffer.len(),
            vocabulary_size: self.vocab.size(),
        }
    }
}
