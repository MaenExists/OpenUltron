//! The Monad: The atomic micro-agent of OpenUltron.
//!
//! A Monad is an autonomous, lightweight unit that follows 4 deterministic local rules:
//! 1. Leaky Integration & Discrete Event Spiking
//! 2. Homeostatic Excitability / Dynamic Threshold Regulation
//! 3. Spike-Timing-Dependent Plasticity (STDP) for Causal Adaptation
//! 4. Metabolic Energy Conservation & Synaptic Pruning

use crate::config::MonadConfig;

pub type MonadId = usize;

/// A directed synaptic connection to a target Monad.
#[derive(Debug, Clone)]
pub struct Synapse {
    pub target_id: MonadId,
    pub weight: f32,
    pub delay: u32,
    pub last_transmission_time: Option<u64>,
}

impl Synapse {
    pub fn new(target_id: MonadId, weight: f32, delay: u32) -> Self {
        Self {
            target_id,
            weight,
            delay,
            last_transmission_time: None,
        }
    }
}

/// The atomic Monad state.
#[derive(Debug, Clone)]
pub struct Monad {
    pub id: MonadId,
    pub potential: f32,
    pub threshold: f32,
    pub energy: f32,
    pub refractory_timer: u32,
    pub last_spike_time: Option<u64>,
    pub total_spikes: u64,
    pub synapses: Vec<Synapse>,
    pub accumulated_input: f32,
}

impl Monad {
    pub fn new(id: MonadId, config: &MonadConfig) -> Self {
        Self {
            id,
            potential: config.v_reset,
            threshold: config.v_threshold_resting,
            energy: 1.0, // Fully fueled at initialization
            refractory_timer: 0,
            last_spike_time: None,
            total_spikes: 0,
            synapses: Vec::new(),
            accumulated_input: 0.0,
        }
    }

    /// Add an outgoing synapse.
    pub fn add_synapse(&mut self, target_id: MonadId, weight: f32, delay: u32) {
        self.synapses.push(Synapse::new(target_id, weight, delay));
    }

    /// Rule 1a: Receive incoming signal from a presynaptic spike or external stimulus.
    pub fn receive_signal(&mut self, signal_strength: f32) {
        self.accumulated_input += signal_strength;
    }

    /// Rule 1b: Integrate potential, apply leak, and evaluate firing condition.
    /// Returns true if the Monad fired a spike at this timestep.
    pub fn step_integration(&mut self, current_time: u64, config: &MonadConfig) -> bool {
        // Refractory cooldown
        if self.refractory_timer > 0 {
            self.refractory_timer -= 1;
            self.accumulated_input = 0.0;
            self.potential = config.v_reset;
            return false;
        }

        // Leaky integration of inputs
        self.potential = (self.potential * config.potential_leak) + self.accumulated_input;
        self.accumulated_input = 0.0;

        // Firing check: Must exceed threshold AND have metabolic energy
        let can_fire = self.potential >= self.threshold && self.energy > config.spike_energy_cost;

        if can_fire {
            // Fired an action potential!
            self.potential = config.v_reset;
            self.refractory_timer = config.refractory_period;
            self.last_spike_time = Some(current_time);
            self.total_spikes += 1;
            true
        } else {
            false
        }
    }

    /// Rule 2: Homeostatic Threshold Adaptation.
    ///
    /// Dynamically shifts threshold:
    /// - If the cell fires, threshold jumps up to prevent epileptic cascade.
    /// - If the cell is silent, threshold slowly relaxes toward resting baseline to prevent quiescence.
    pub fn step_homeostasis(&mut self, fired: bool, config: &MonadConfig) {
        let spike_val = if fired { 1.0 } else { 0.0 };
        // Homeostatic gradient: delta_theta = eta * (actual_rate - target_rate)
        let delta_theta = config.eta_homeostasis * (spike_val - config.target_firing_rate);
        self.threshold += delta_theta;

        // Prevent threshold from collapsing to <= 0 (would cause infinite spontaneous firing)
        let min_threshold = config.v_threshold_resting * 0.2;
        if self.threshold < min_threshold {
            self.threshold = min_threshold;
        }
    }

    /// Rule 3: Spike-Timing-Dependent Plasticity (STDP) for Local Causal Learning.
    ///
    /// Called when either the presynaptic or postsynaptic cell fires.
    /// delta_t = t_post - t_pre
    /// If delta_t > 0: Causal (presynaptic contributed to postsynaptic firing) -> Long-Term Potentiation (LTP)
    /// If delta_t < 0: Anti-causal (postsynaptic fired before presynaptic arrived) -> Long-Term Depression (LTD)
    pub fn update_synapse_stdp(
        synapse: &mut Synapse,
        t_pre: u64,
        t_post: u64,
        config: &MonadConfig,
    ) {
        if t_post > t_pre {
            let dt = (t_post - t_pre) as f32;
            let dw = config.stdp_a_plus * (-dt / config.stdp_tau_plus).exp();
            synapse.weight = (synapse.weight + dw).min(config.max_weight);
        } else if t_post < t_pre {
            let dt = (t_pre - t_post) as f32;
            let dw = config.stdp_a_minus * (-dt / config.stdp_tau_minus).exp();
            synapse.weight = (synapse.weight - dw).max(config.min_weight);
        }
    }

    /// Rule 4: Metabolic Conservation & Structural Synaptic Pruning.
    ///
    /// - Burning energy when firing
    /// - Harvesting environmental reward or baseline recharge
    /// - Pruning near-zero synapses
    pub fn step_metabolism(&mut self, fired: bool, reward: f32, config: &MonadConfig) {
        // Spike cost
        if fired {
            self.energy = (self.energy - config.spike_energy_cost).max(0.0);
        }

        // Energy replenishment: baseline plus external reward
        self.energy = (self.energy + config.baseline_energy_recovery + reward).min(1.0);

        // Synaptic maintenance / Pruning
        self.synapses.retain(|syn| syn.weight.abs() >= config.prune_threshold);
    }
}
