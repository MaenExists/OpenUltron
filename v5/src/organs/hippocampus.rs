//! Hippocampal Formation Organ for OpenUltron v5.
//!
//! Functions:
//! - Episodic Working Memory Buffer: Retains multi-turn conversational context with exponential decay.
//! - Fast-Plasticity Binding (STDP_fast): Rapidly links concepts spoken across recent dialogue turns.
//! - Context Resonance: Provides temporal attractor gravity across multiple conversational turns.
//! - Sharp-Wave Ripple Consolidation: Periodically consolidates short-term episodic traces into long-term cortical matrices.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EpisodicTrace {
    pub token_ids: Vec<usize>,
    pub salience: f32,
    pub turn_age: usize,
}

pub struct Hippocampus {
    pub episodic_buffer: Vec<EpisodicTrace>,
    pub max_episodic_depth: usize,
    pub fast_synapses: HashMap<(usize, usize), f32>,
    pub decay_rate: f32,
}

impl Hippocampus {
    pub fn new(max_depth: usize) -> Self {
        Self {
            episodic_buffer: Vec::new(),
            max_episodic_depth: max_depth,
            fast_synapses: HashMap::new(),
            decay_rate: 0.75, // Exponential decay per conversational turn
        }
    }

    /// Ingest a newly spoken turn into episodic memory and update fast-binding synapses.
    pub fn encode_turn(&mut self, token_ids: &[usize], salience: f32) {
        // Age existing episodic traces
        for trace in self.episodic_buffer.iter_mut() {
            trace.turn_age += 1;
        }

        // Evict traces older than max_episodic_depth
        self.episodic_buffer.retain(|t| t.turn_age < self.max_episodic_depth);

        // Record new episodic trace
        self.episodic_buffer.push(EpisodicTrace {
            token_ids: token_ids.to_vec(),
            salience,
            turn_age: 0,
        });

        // Fast STDP cross-binding between tokens in this turn
        let window = 4;
        for i in 0..token_ids.len() {
            let u = token_ids[i];
            for j in (i + 1)..((i + window).min(token_ids.len())) {
                let v = token_ids[j];
                *self.fast_synapses.entry((u, v)).or_insert(0.0) += 0.8;
                *self.fast_synapses.entry((v, u)).or_insert(0.0) += 0.8;
            }
        }

        // Also bind current turn tokens to previous turn tokens (cross-turn episodic link)
        if self.episodic_buffer.len() >= 2 {
            let prev_trace = &self.episodic_buffer[self.episodic_buffer.len() - 2];
            for &prev_tok in &prev_trace.token_ids {
                for &curr_tok in token_ids {
                    *self.fast_synapses.entry((prev_tok, curr_tok)).or_insert(0.0) += 0.5;
                }
            }
        }
    }

    /// Extract decaying episodic context attractor field across all stored turns.
    pub fn get_episodic_attractors(&self, vocab_size: usize) -> Vec<f32> {
        let mut context_field = vec![0.0f32; vocab_size];

        for trace in &self.episodic_buffer {
            let weight = self.decay_rate.powi(trace.turn_age as i32) * trace.salience;
            for &tok in &trace.token_ids {
                if tok < vocab_size {
                    context_field[tok] += weight;

                    // Spread through fast hippocampal synapses
                    for target in 4..vocab_size {
                        if let Some(&syn) = self.fast_synapses.get(&(tok, target)) {
                            context_field[target] += syn * weight * 0.4;
                        }
                    }
                }
            }
        }

        context_field
    }

    /// Sharp-wave ripples: return pairs of (token_u, token_v, weight) that should be consolidated into neocortex.
    pub fn consolidate_ripples(&mut self) -> Vec<(usize, usize, f32)> {
        let mut ripples = Vec::new();
        for (&(u, v), &w) in &self.fast_synapses {
            if w >= 0.5 {
                ripples.push((u, v, w * 0.3));
            }
        }
        // Slightly decay fast synapses after consolidation
        for w in self.fast_synapses.values_mut() {
            *w *= 0.90;
        }
        ripples
    }

    pub fn reset_session(&mut self) {
        self.episodic_buffer.clear();
        self.fast_synapses.clear();
    }
}
