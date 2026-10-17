//! Configuration for OpenUltron v2 Sensorimotor Language Brain.

#[derive(Debug, Clone)]
pub struct BrainConfig {
    /// Number of internal cognitive reservoir Monads.
    pub reservoir_size: usize,

    /// Number of sensory Monads allocated per distinct character.
    pub sensory_per_char: usize,

    /// Number of local outgoing connections per reservoir Monad.
    pub connections_per_monad: usize,

    /// Small-world long-range shortcut probability.
    pub small_world_rewire_prob: f32,

    /// Membrane potential leak rate (0.0 to 1.0).
    pub potential_leak: f32,

    /// Base resting firing threshold.
    pub v_threshold_resting: f32,

    /// Homeostatic threshold adaptation rate.
    pub eta_homeostasis: f32,

    /// Target firing rate for homeostasis.
    pub target_firing_rate: f32,

    /// Causal learning rate (Long-Term Potentiation).
    pub lr_ltp: f32,

    /// Anti-causal learning rate (Long-Term Depression).
    pub lr_ltd: f32,

    /// Minimum synaptic weight.
    pub min_weight: f32,

    /// Maximum synaptic weight.
    pub max_weight: f32,

    /// Energy cost per spike.
    pub spike_cost: f32,

    /// Synapse pruning threshold.
    pub prune_threshold: f32,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 5_000, // Balanced for fast interactive typing and rich dynamics
            sensory_per_char: 4,
            connections_per_monad: 12,
            small_world_rewire_prob: 0.12,
            potential_leak: 0.85,
            v_threshold_resting: 0.90,
            eta_homeostasis: 0.015,
            target_firing_rate: 0.05,
            lr_ltp: 0.08,
            lr_ltd: 0.04,
            min_weight: -1.0,
            max_weight: 1.0,
            spike_cost: 0.01,
            prune_threshold: 0.005,
        }
    }
}
