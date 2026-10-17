//! Motor Articulation ("The Voice"): Spike-to-Text Decoding.
//!
//! Translates the collective firings and potentials of output Motor Monads
//! into discrete generated linguistic characters.

use crate::sensory::{index_to_char, VOCAB_SIZE};

pub struct MotorVoice {
    pub total_motor_monads: usize,
}

impl MotorVoice {
    pub fn new() -> Self {
        Self {
            total_motor_monads: VOCAB_SIZE,
        }
    }

    /// Decode motor activity into the most activated character.
    /// Supports temperature-based Boltzmann sampling for natural linguistic variety.
    pub fn decode(&self, activations: &[f32], temperature: f32) -> char {
        if activations.is_empty() {
            return ' ';
        }

        if temperature <= 0.05 {
            // Greedy Argmax
            let mut best_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &val) in activations.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    best_idx = i;
                }
            }
            return index_to_char(best_idx);
        }

        // Softmax temperature distribution
        let max_val = activations.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_weights = Vec::with_capacity(activations.len());
        let mut sum_exp = 0.0;

        for &val in activations {
            let exp_v = ((val - max_val) / temperature).exp();
            exp_weights.push(exp_v);
            sum_exp += exp_v;
        }

        if sum_exp <= 0.0 {
            return ' ';
        }

        // Pseudo-random sampling
        let mut seed = (activations.len() as u64)
            .wrapping_mul(48271)
            .wrapping_add((max_val.abs() * 10000.0) as u64);
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let sample_point = ((seed >> 32) as f32 / u32::MAX as f32) * sum_exp;

        let mut cum_sum = 0.0;
        for (i, &exp_v) in exp_weights.iter().enumerate() {
            cum_sum += exp_v;
            if cum_sum >= sample_point {
                return index_to_char(i);
            }
        }

        index_to_char(activations.len() - 1)
    }
}
