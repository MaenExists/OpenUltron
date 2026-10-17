//! Hyperparameters for OpenUltron v3 Semantic Assembly Brain.

#[derive(Debug, Clone)]
pub struct BrainConfig {
    /// Number of internal cognitive reservoir Monads.
    pub reservoir_size: usize,

    /// Number of Monads dedicated to each word's Cell Assembly.
    pub assembly_size: usize,

    /// Local connections per reservoir Monad.
    pub connections_per_monad: usize,

    /// Small-world long-range shortcut probability.
    pub small_world_rewire_prob: f32,

    /// Membrane potential leak rate (0.0 to 1.0).
    pub potential_leak: f32,

    /// Base resting threshold.
    pub v_threshold_resting: f32,

    /// Homeostatic threshold adaptation rate.
    pub eta_homeostasis: f32,

    /// Target firing rate for homeostasis.
    pub target_firing_rate: f32,

    /// Causal learning rate (Long-Term Potentiation).
    pub lr_ltp: f32,

    /// Anti-causal learning rate (Long-Term Depression).
    pub lr_ltd: f32,

    /// Maximum vocabulary size for word assemblies.
    pub max_vocab_size: usize,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 6_000,
            assembly_size: 6,
            connections_per_monad: 14,
            small_world_rewire_prob: 0.15,
            potential_leak: 0.85,
            v_threshold_resting: 0.90,
            eta_homeostasis: 0.015,
            target_firing_rate: 0.05,
            lr_ltp: 0.18,
            lr_ltd: 0.04,
            max_vocab_size: 1_200,
        }
    }
}
