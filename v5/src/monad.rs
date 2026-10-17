//! Spiking Monad Neural Substrate & Cell Assemblies for OpenUltron v5.
//!
//! Grounded in physical and thermodynamic constraints:
//! - Leaky Integrate-and-Fire (LIF) electrodynamics.
//! - Ion-channel thermal fluctuations (Brownian stochastic resonance).
//! - Absolute and relative refractory dynamics.
//! - Metabolic ATP consumption per action potential.
//! - Donald Hebb's Cell Assembly phase synchronization.

#[derive(Debug, Clone)]
pub struct SpikingMonad {
    pub id: usize,
    pub potential: f32,
    pub threshold: f32,
    pub rest_potential: f32,
    pub leak_factor: f32,
    pub spikes_fired: u64,
    pub refractory_ticks: usize,
    pub atp_level: f32, // Metabolic energy reserve [0.0, 1.0]
}

impl SpikingMonad {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            potential: 0.0,
            threshold: 1.0,
            rest_potential: 0.0,
            leak_factor: 0.88,
            spikes_fired: 0,
            refractory_ticks: 0,
            atp_level: 1.0,
        }
    }

    /// Integrate incoming synaptic current with metabolic leak, thermal noise, and ATP constraint.
    pub fn integrate(&mut self, current: f32, noise: f32) -> bool {
        // Regenerate metabolic ATP continuously towards 1.0
        self.atp_level = (self.atp_level + 0.04).min(1.0);

        if self.refractory_ticks > 0 {
            self.refractory_ticks -= 1;
            return false;
        }

        // Available energy modulates integration efficiency
        let effective_current = current * (0.5 + 0.5 * self.atp_level);
        self.potential = (self.potential * self.leak_factor) + effective_current + noise;

        if self.potential >= self.threshold && self.atp_level > 0.15 {
            self.potential = self.rest_potential;
            self.spikes_fired += 1;
            self.refractory_ticks = 1;
            self.atp_level = (self.atp_level - 0.12).max(0.0); // Spike metabolic cost
            true
        } else {
            if self.potential >= self.threshold {
                // Metabolic exhaustion prevents full action potential
                self.potential = self.threshold * 0.7;
            }
            false
        }
    }

    pub fn reset(&mut self) {
        self.potential = self.rest_potential;
        self.refractory_ticks = 0;
    }
}

/// A synchronized Cell Assembly of Spiking Monads representing a concept/word.
#[derive(Debug, Clone)]
pub struct ConceptAssembly {
    pub word_id: usize,
    pub monads: Vec<SpikingMonad>,
}

impl ConceptAssembly {
    pub fn new(word_id: usize, size: usize) -> Self {
        let monads = (0..size)
            .map(|i| SpikingMonad::new(word_id * size + i))
            .collect();
        Self { word_id, monads }
    }

    /// Excite the assembly with input current, returning total action potentials produced.
    pub fn excite(&mut self, current: f32, thermal_chaos: f32) -> u64 {
        let mut spikes = 0;
        for (i, monad) in self.monads.iter_mut().enumerate() {
            // Distribute phase-shifted current across the ensemble
            let phase_offset = ((i as f32) * 0.2).sin() * 0.18;
            let net_current = current * (0.85 + phase_offset);
            if monad.integrate(net_current, thermal_chaos) {
                spikes += 1;
            }
        }
        spikes
    }

    pub fn total_spikes(&self) -> u64 {
        self.monads.iter().map(|m| m.spikes_fired).sum()
    }

    pub fn average_atp(&self) -> f32 {
        if self.monads.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.monads.iter().map(|m| m.atp_level).sum();
        sum / (self.monads.len() as f32)
    }
}
