//! Configuration and hyperparameters for the OpenUltron v1 Monad and Network.
//!
//! Every parameter is mathematically grounded in biophysical dynamical systems
//! and Self-Organized Criticality (SOC) theory.

#[derive(Debug, Clone)]
pub struct MonadConfig {
    /// Leak factor for membrane potential in [0, 1].
    /// V(t) = V(t-1) * potential_leak + inputs
    pub potential_leak: f32,

    /// Potential reset value after a spike event.
    pub v_reset: f32,

    /// Initial resting potential threshold for firing.
    pub v_threshold_resting: f32,

    /// Target firing rate for homeostatic regulation (e.g., 0.05 = 5% of timesteps).
    pub target_firing_rate: f32,

    /// Homeostatic plasticity learning rate for threshold adaptation (eta_theta).
    pub eta_homeostasis: f32,

    /// Absolute refractory period in timesteps during which the monad cannot fire.
    pub refractory_period: u32,

    /// Positive learning amplitude for causal STDP (pre before post).
    pub stdp_a_plus: f32,

    /// Negative learning amplitude for anti-causal STDP (post before pre).
    pub stdp_a_minus: f32,

    /// Temporal decay window for positive STDP (tau_plus) in timesteps.
    pub stdp_tau_plus: f32,

    /// Temporal decay window for negative STDP (tau_minus) in timesteps.
    pub stdp_tau_minus: f32,

    /// Minimum allowable synaptic weight.
    pub min_weight: f32,

    /// Maximum allowable synaptic weight.
    pub max_weight: f32,

    /// Metabolic energy consumed per fired spike.
    pub spike_energy_cost: f32,

    /// Baseline energy recovery per timestep.
    pub baseline_energy_recovery: f32,

    /// Synapse pruning threshold: weights with absolute value below this are pruned.
    pub prune_threshold: f32,
}

impl Default for MonadConfig {
    fn default() -> Self {
        Self {
            potential_leak: 0.90,
            v_reset: 0.0,
            v_threshold_resting: 1.0,
            target_firing_rate: 0.05,
            eta_homeostasis: 0.02,
            refractory_period: 2,
            stdp_a_plus: 0.05,
            stdp_a_minus: 0.06, // Slightly asymmetric (A- > A+) to prevent runaway excitation
            stdp_tau_plus: 15.0,
            stdp_tau_minus: 20.0,
            min_weight: -1.0,
            max_weight: 1.0,
            spike_energy_cost: 0.02,
            baseline_energy_recovery: 0.005,
            prune_threshold: 0.01,
        }
    }
}
