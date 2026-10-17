//! High-performance Monad micro-agent for OpenUltron v3.

use crate::config::BrainConfig;

pub type MonadId = usize;

#[derive(Debug, Clone)]
pub struct Synapse {
    pub target_id: MonadId,
    pub weight: f32,
    pub delay: u8,
}

impl Synapse {
    #[inline(always)]
    pub fn new(target_id: MonadId, weight: f32, delay: u8) -> Self {
        Self {
            target_id,
            weight,
            delay,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Monad {
    pub id: MonadId,
    pub potential: f32,
    pub threshold: f32,
    pub energy: f32,
    pub refractory: u8,
    pub last_spike_time: Option<u64>,
    pub total_spikes: u64,
    pub input_accumulator: f32,
    pub synapses: Vec<Synapse>,
}

impl Monad {
    pub fn new(id: MonadId, config: &BrainConfig) -> Self {
        Self {
            id,
            potential: 0.0,
            threshold: config.v_threshold_resting,
            energy: 1.0,
            refractory: 0,
            last_spike_time: None,
            total_spikes: 0,
            input_accumulator: 0.0,
            synapses: Vec::with_capacity(config.connections_per_monad),
        }
    }

    #[inline(always)]
    pub fn receive(&mut self, current: f32) {
        self.input_accumulator += current;
    }

    #[inline(always)]
    pub fn step_integration(&mut self, current_time: u64, config: &BrainConfig) -> bool {
        if self.refractory > 0 {
            self.refractory -= 1;
            self.input_accumulator = 0.0;
            self.potential = 0.0;
            return false;
        }

        self.potential = (self.potential * config.potential_leak) + self.input_accumulator;
        self.input_accumulator = 0.0;

        if self.potential >= self.threshold {
            self.potential = 0.0;
            self.refractory = 2;
            self.last_spike_time = Some(current_time);
            self.total_spikes += 1;
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn step_homeostasis(&mut self, fired: bool, config: &BrainConfig) {
        let actual = if fired { 1.0 } else { 0.0 };
        self.threshold += config.eta_homeostasis * (actual - config.target_firing_rate);
        let min_t = config.v_threshold_resting * 0.3;
        if self.threshold < min_t {
            self.threshold = min_t;
        }
    }
}
