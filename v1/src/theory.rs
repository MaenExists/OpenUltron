//! Mathematical Theory of Emergence for OpenUltron v1.
//!
//! Provides high-end statistical mechanics, non-linear dynamics, and complex systems
//! mathematics to quantify and predict emergent collective phenomena:
//!
//! 1. Branching Parameter (sigma) - Phase transition to the Edge of Chaos.
//! 2. Self-Organized Criticality (SOC) - Scale-free avalanche distribution P(s) ~ s^(-alpha).
//! 3. Kuramoto Order Parameter (R) - Meta-stable phase synchronization.
//! 4. Population Entropy H(S) - Information-processing capacity.

use std::collections::HashMap;

/// Quantitative assessment of collective emergence in the Monad network.
#[derive(Debug, Clone)]
pub struct EmergenceMetrics {
    /// Average branching ratio: sigma = <S_{t+1} / S_t>. Critical point is sigma ~ 1.0.
    pub branching_ratio: f64,

    /// Classification of dynamical regime based on sigma.
    pub dynamical_regime: DynamicalRegime,

    /// Total number of avalanches detected during the observation window.
    pub total_avalanches: usize,

    /// Maximum Likelihood Estimator (MLE) of the power-law exponent alpha in P(s) ~ s^(-alpha).
    /// Theoretical critical cortex exponent is alpha ~ 1.50.
    pub avalanche_alpha: f64,

    /// Mean Kuramoto phase synchronization order parameter R in [0, 1].
    pub mean_kuramoto_order: f64,

    /// Variance of Kuramoto order parameter (susceptibility). Peaked at phase transitions.
    pub kuramoto_susceptibility: f64,

    /// Shannon entropy of the population activity distribution in bits.
    pub population_entropy_bits: f64,

    /// Overall Emergence Index in [0.0, 1.0] measuring proximity to optimal critical cognition.
    pub emergence_index: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicalRegime {
    SubcriticalQuiescent,
    CriticalEdgeOfChaos,
    SupercriticalEpileptic,
}

/// Avalanche analyzer for tracking spatio-temporal cascades.
pub struct TheoryEngine;

impl TheoryEngine {
    /// Calculate the average branching ratio sigma = < S_{t+1} / S_t >.
    /// Excludes zero-activity denominators to avoid division by zero.
    pub fn compute_branching_ratio(spike_counts: &[usize]) -> f64 {
        if spike_counts.len() < 2 {
            return 0.0;
        }

        let mut ratio_sum = 0.0;
        let mut valid_pairs = 0;

        for w in spike_counts.windows(2) {
            let s_t = w[0];
            let s_t_next = w[1];
            if s_t > 0 {
                ratio_sum += s_t_next as f64 / s_t as f64;
                valid_pairs += 1;
            }
        }

        if valid_pairs == 0 {
            0.0
        } else {
            ratio_sum / valid_pairs as f64
        }
    }

    /// Detect discrete avalanches of activity separated by periods of quiescence (spikes = 0).
    /// Returns vector of avalanche sizes (total spikes per cascade).
    pub fn extract_avalanches(spike_counts: &[usize]) -> Vec<usize> {
        let mut avalanches = Vec::new();
        let mut current_avalanche_size = 0;
        let mut in_avalanche = false;

        for &count in spike_counts {
            if count > 0 {
                current_avalanche_size += count;
                in_avalanche = true;
            } else if in_avalanche {
                avalanches.push(current_avalanche_size);
                current_avalanche_size = 0;
                in_avalanche = false;
            }
        }

        if in_avalanche && current_avalanche_size > 0 {
            avalanches.push(current_avalanche_size);
        }

        avalanches
    }

    /// Calculate Maximum Likelihood Estimate (MLE) for power-law exponent alpha:
    /// alpha_hat = 1 + n * [ sum_{i=1}^n ln( s_i / (s_min - 0.5) ) ]^(-1)
    /// (Clauset, Shalizi, Newman, 2009 for discrete power laws).
    pub fn compute_power_law_alpha(avalanche_sizes: &[usize], s_min: usize) -> f64 {
        let valid_samples: Vec<f64> = avalanche_sizes
            .iter()
            .filter(|&&s| s >= s_min)
            .map(|&s| s as f64)
            .collect();

        let n = valid_samples.len();
        if n < 10 {
            return 0.0; // Insufficient statistics
        }

        let s_min_adj = s_min as f64 - 0.5;
        let log_sum: f64 = valid_samples.iter().map(|&s| (s / s_min_adj).ln()).sum();

        if log_sum <= 0.0 {
            return 0.0;
        }

        1.0 + (n as f64) / log_sum
    }

    /// Compute Kuramoto Phase Order Parameter R(t) = 1/N * | sum_{j=1}^N exp(i * phi_j(t)) |.
    ///
    /// For each Monad, phase is estimated from time elapsed since last spike relative
    /// to its expected firing interval: phi_j(t) = 2 * pi * (dt / T_expected).
    pub fn compute_kuramoto_series(
        monad_spike_histories: &[Option<u64>],
        total_monads: usize,
        current_time: u64,
        expected_interval: f64,
    ) -> f64 {
        if total_monads == 0 {
            return 0.0;
        }

        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;

        for &last_spike in monad_spike_histories {
            let phi = match last_spike {
                Some(t_spike) => {
                    let dt = (current_time.saturating_sub(t_spike)) as f64;
                    (2.0 * std::f64::consts::PI * (dt / expected_interval)) % (2.0 * std::f64::consts::PI)
                }
                None => 0.0,
            };
            sum_cos += phi.cos();
            sum_sin += phi.sin();
        }

        let n = total_monads as f64;
        let r = ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt();
        r.clamp(0.0, 1.0)
    }

    /// Compute Shannon Population Entropy H(S) = -sum p(k) * log2(p(k))
    pub fn compute_population_entropy(spike_counts: &[usize]) -> f64 {
        if spike_counts.is_empty() {
            return 0.0;
        }

        let total_timesteps = spike_counts.len() as f64;
        let mut freq_map: HashMap<usize, usize> = HashMap::new();

        for &count in spike_counts {
            *freq_map.entry(count).or_insert(0) += 1;
        }

        let mut entropy = 0.0;
        for &freq in freq_map.values() {
            let p = freq as f64 / total_timesteps;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Evaluate full emergence metrics from simulation observation data.
    pub fn evaluate_emergence(
        spike_counts: &[usize],
        kuramoto_history: &[f64],
    ) -> EmergenceMetrics {
        let sigma = Self::compute_branching_ratio(spike_counts);

        let regime = if sigma < 0.85 {
            DynamicalRegime::SubcriticalQuiescent
        } else if sigma > 1.25 {
            DynamicalRegime::SupercriticalEpileptic
        } else {
            DynamicalRegime::CriticalEdgeOfChaos
        };

        let avalanches = Self::extract_avalanches(spike_counts);
        let alpha = Self::compute_power_law_alpha(&avalanches, 2);

        let mean_r = if kuramoto_history.is_empty() {
            0.0
        } else {
            kuramoto_history.iter().sum::<f64>() / kuramoto_history.len() as f64
        };

        let kuramoto_variance = if kuramoto_history.is_empty() {
            0.0
        } else {
            let var: f64 = kuramoto_history
                .iter()
                .map(|&r| (r - mean_r).powi(2))
                .sum();
            var / kuramoto_history.len() as f64
        };

        let entropy = Self::compute_population_entropy(spike_counts);

        // Mathematical Emergence Index:
        // Proximity to sigma = 1.0, proximity to alpha = 1.5, and meta-stable Kuramoto (0.2 - 0.6)
        let sigma_score = (-3.0 * (sigma - 1.0).abs()).exp();
        let alpha_score = if alpha > 0.0 {
            (-2.0 * (alpha - 1.50).abs()).exp()
        } else {
            0.1
        };
        let kuramoto_score = if (0.15..=0.65).contains(&mean_r) {
            1.0
        } else {
            (-4.0 * (mean_r - 0.40).abs()).exp()
        };

        let emergence_index = (0.4 * sigma_score + 0.4 * alpha_score + 0.2 * kuramoto_score).clamp(0.0, 1.0);

        EmergenceMetrics {
            branching_ratio: sigma,
            dynamical_regime: regime,
            total_avalanches: avalanches.len(),
            avalanche_alpha: alpha,
            mean_kuramoto_order: mean_r,
            kuramoto_susceptibility: kuramoto_variance,
            population_entropy_bits: entropy,
            emergence_index,
        }
    }
}
