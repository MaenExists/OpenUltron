//! Spiking Monad Neural Substrate for OpenUltron v4.
//!
//! Models Leaky Integrate-and-Fire (LIF) units operating with
//! dynamic membrane potentials, metabolic leaks, and stochastic chaos.

#[derive(Debug, Clone)]
pub struct SpikingMonad {
    pub id: usize,
    pub potential: f32,
    pub threshold: f32,
    pub rest_potential: f32,
    pub leak_factor: f32,
    pub spikes_fired: u64,
    pub refractory_ticks: usize,
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
        }
    }

    /// Integrate incoming synaptic current with metabolic leak and thermal noise.
    pub fn integrate(&mut self, current: f32, noise: f32) -> bool {
        if self.refractory_ticks > 0 {
            self.refractory_ticks -= 1;
            return false;
        }

        self.potential = (self.potential * self.leak_factor) + current + noise;

        if self.potential >= self.threshold {
            self.potential = self.rest_potential;
            self.spikes_fired += 1;
            self.refractory_ticks = 1;
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.potential = self.rest_potential;
        self.refractory_ticks = 0;
    }
}

/// A synchronized Cell Assembly of Spiking Monads representing a single word/concept.
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

    /// Excite the assembly with input current, returning the total spikes produced.
    pub fn excite(&mut self, current: f32, thermal_chaos: f32) -> u64 {
        let mut spikes = 0;
        for (i, monad) in self.monads.iter_mut().enumerate() {
            // Distribute current across the assembly with slight phase variance
            let phase_offset = ((i as f32) * 0.15).sin() * 0.2;
            let net_current = current * (0.8 + phase_offset);
            if monad.integrate(net_current, thermal_chaos) {
                spikes += 1;
            }
        }
        spikes
    }

    pub fn total_spikes(&self) -> u64 {
        self.monads.iter().map(|m| m.spikes_fired).sum()
    }
}
