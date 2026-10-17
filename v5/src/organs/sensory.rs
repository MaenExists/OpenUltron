//! Sensory Thalamus Organ for OpenUltron v5.
//!
//! Functions:
//! - Afferent Gateway: Ingests external sensory text stream.
//! - Novelty & Surprise Detection: Evaluates sensory prediction error against transition expectations.
//! - Thalamic Reticular Filtering: Dampens repetitive stop tokens and amplifies high-salience concepts.
//! - Generates structured afferent spike-train bursts for downstream Cortical and Hippocampal organs.

use crate::vocab::Vocabulary;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SensoryBurst {
    pub token_ids: Vec<usize>,
    pub salience_weights: Vec<f32>,
    pub sensory_surprise: f32, // [0.0, 1.0] Novelty / prediction error
}

pub struct SensoryThalamus {
    sensory_expectations: HashMap<(usize, usize), f32>,
}

impl SensoryThalamus {
    pub fn new() -> Self {
        Self {
            sensory_expectations: HashMap::new(),
        }
    }

    /// Process raw text input from user/environment, dynamically registering new tokens,
    /// evaluating sensory surprise, and producing an afferent spike packet.
    pub fn process_input(&mut self, text: &str, vocab: &mut Vocabulary) -> SensoryBurst {
        let raw_tokens = Vocabulary::tokenize(text);
        if raw_tokens.is_empty() {
            return SensoryBurst {
                token_ids: Vec::new(),
                salience_weights: Vec::new(),
                sensory_surprise: 0.5,
            };
        }

        let mut token_ids = Vec::with_capacity(raw_tokens.len());
        for t in &raw_tokens {
            let id = vocab.get_or_insert(t);
            token_ids.push(id);
        }

        // Calculate sensory prediction error / surprise
        let mut expected_transitions = 0.0;
        let mut transition_count = 0.0;

        for i in 1..token_ids.len() {
            let u = token_ids[i - 1];
            let v = token_ids[i];
            if let Some(&w) = self.sensory_expectations.get(&(u, v)) {
                expected_transitions += (w / 8.0).min(1.0);
            }
            transition_count += 1.0;

            // Incrementally reinforce sensory expectations (habituation)
            *self.sensory_expectations.entry((u, v)).or_insert(0.0) += 0.5;
        }

        let sensory_surprise = if transition_count > 0.0 {
            (1.0 - (expected_transitions / transition_count)).clamp(0.05, 0.98)
        } else {
            0.6
        };

        // Compute thalamic salience weights using Shannon IDF
        let mut salience_weights = Vec::with_capacity(token_ids.len());
        for &id in &token_ids {
            let idf = vocab.idf(id);
            // Non-punctuation and rare words get higher salience gating
            let is_punct = vocab.is_punctuation(id);
            let salience = if is_punct {
                0.2
            } else {
                (idf * 1.2).clamp(0.8, 5.0)
            };
            salience_weights.push(salience);
        }

        SensoryBurst {
            token_ids,
            salience_weights,
            sensory_surprise,
        }
    }

    /// Habituate on training sequences.
    pub fn habituate_sequence(&mut self, token_ids: &[usize]) {
        for i in 1..token_ids.len() {
            let u = token_ids[i - 1];
            let v = token_ids[i];
            *self.sensory_expectations.entry((u, v)).or_insert(0.0) += 1.0;
        }
    }
}
