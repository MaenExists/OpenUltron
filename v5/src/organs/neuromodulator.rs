//! Neuromodulatory & Homeostatic Core Organ for OpenUltron v5.
//!
//! Models the neurochemical and thermodynamic equilibrium:
//! - Variational Free Energy F(t): Surprise, prediction error, thermodynamic confusion.
//! - Noradrenaline (Arousal): Modulates stochastic thermal chaos and exploration.
//! - Dopamine (Reward & Bonding): Signals prediction success and strengthens synaptic plasticity.
//! - Metabolic ATP Budget: Enforces homeostatic energy constraints and triggers synaptic pruning.

#[derive(Debug, Clone)]
pub struct OrganismBiometrics {
    pub free_energy: f32,       // Variational confusion [0.0, 1.0]
    pub noradrenaline: f32,     // Arousal / Temperature [0.0, 1.0]
    pub dopamine: f32,          // Motivation / Reward [0.0, 1.0]
    pub metabolic_atp: f32,     // Systemic energy level [0.0, 1.0]
    pub total_spikes: u64,      // Lifetime action potentials
    pub synaptic_density: usize,// Total physical connections
    pub episodic_depth: usize,  // Active traces in hippocampus
    pub vocabulary_size: usize, // Known concepts
}

pub struct Neuromodulator {
    pub free_energy: f32,
    pub noradrenaline: f32,
    pub dopamine: f32,
    pub metabolic_atp: f32,
    pub total_ticks: u64,
}

impl Neuromodulator {
    pub fn new() -> Self {
        Self {
            free_energy: 0.95,   // High infant confusion on awakening
            noradrenaline: 0.90, // High infant arousal
            dopamine: 0.40,      // Baseline exploratory dopamine
            metabolic_atp: 1.00, // Full initial battery
            total_ticks: 0,
        }
    }

    /// Update internal neurochemistry based on sensory prediction error and conversational feedback.
    pub fn update_state(&mut self, sensory_surprise: f32, communication_success: f32) {
        self.total_ticks += 1;

        // Smooth update of Variational Free Energy
        let target_fe = (sensory_surprise * 0.7 + (1.0 - communication_success) * 0.3).clamp(0.15, 0.95);
        self.free_energy = (self.free_energy * 0.85) + (target_fe * 0.15);

        // Noradrenaline tracks Free Energy / surprise
        self.noradrenaline = (self.noradrenaline * 0.80) + (self.free_energy * 0.20);

        // Dopamine surges when surprise drops or communication succeeds
        let dopamine_target = (communication_success * 0.8 + (1.0 - sensory_surprise) * 0.2).clamp(0.1, 1.0);
        self.dopamine = (self.dopamine * 0.82) + (dopamine_target * 0.18);

        // Regenerate systemic metabolic ATP
        self.metabolic_atp = (self.metabolic_atp + 0.08).min(1.0);
    }

    /// Deduct metabolic cost of firing spikes.
    pub fn consume_spike_energy(&mut self, spike_count: u64) {
        let energy_cost = (spike_count as f32) * 0.0003;
        self.metabolic_atp = (self.metabolic_atp - energy_cost).max(0.05);
    }

    /// Check if homeostatic pruning is required to maintain metabolic equilibrium.
    pub fn should_prune_synapses(&self, current_synapses: usize, max_capacity: usize) -> bool {
        current_synapses > max_capacity || self.metabolic_atp < 0.25
    }

    /// Compute dynamic sampling temperature for thought generation.
    pub fn dynamic_temperature(&self) -> f32 {
        0.45 + (self.noradrenaline * 0.40)
    }

    /// Compute thermal chaos noise amplitude.
    pub fn thermal_noise_amplitude(&self) -> f32 {
        0.10 + (self.free_energy * 0.28)
    }
}
