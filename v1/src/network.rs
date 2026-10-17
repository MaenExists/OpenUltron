//! The MonadNetwork: Substrate connecting collections of Monads.
//!
//! Orchestrates asynchronous event routing, small-world graph wiring,
//! external stimulus delivery, and real-time recording for mathematical analysis.

use crate::config::MonadConfig;
use crate::monad::{Monad, MonadId};
use std::collections::VecDeque;

/// A spike event traveling along a synapse with conduction delay.
#[derive(Debug, Clone)]
pub struct PendingPulse {
    pub arrival_time: u64,
    pub source_id: MonadId,
    pub target_id: MonadId,
    pub signal_weight: f32,
}

pub struct MonadNetwork {
    pub monads: Vec<Monad>,
    pub config: MonadConfig,
    pub pending_pulses: VecDeque<PendingPulse>,
    pub current_time: u64,
}

impl MonadNetwork {
    /// Initialize an empty network with N Monads.
    pub fn new(size: usize, config: MonadConfig) -> Self {
        let monads = (0..size).map(|id| Monad::new(id, &config)).collect();
        Self {
            monads,
            config,
            pending_pulses: VecDeque::new(),
            current_time: 0,
        }
    }

    /// Wire the network with a Small-World (Watts-Strogatz inspired) topology:
    /// - `k_neighbors`: Number of nearest-neighbor local connections per Monad.
    /// - `rewire_prob`: Probability of rewiring a local connection to a distant random Monad.
    /// - `initial_weight`: Initial weight range [-initial_weight, initial_weight].
    pub fn wire_small_world(&mut self, k_neighbors: usize, rewire_prob: f32, initial_weight: f32) {
        let n = self.monads.len();
        if n < 2 {
            return;
        }

        // Deterministic pseudo-random sequence for reproducible emergence testing
        let mut seed: u64 = 42424242;
        let mut next_rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };

        for i in 0..n {
            for offset in 1..=(k_neighbors / 2) {
                // Forward neighbor
                let mut target = (i + offset) % n;
                // Rewire with probability rewire_prob
                let rand_val = (next_rand() % 10000) as f32 / 10000.0;
                if rand_val < rewire_prob {
                    let random_target = (next_rand() as usize) % n;
                    if random_target != i {
                        target = random_target;
                    }
                }

                let is_inhibitory = (next_rand() % 100) < 20;
                let w_mag = 0.4 + 0.6 * ((next_rand() % 10000) as f32 / 10000.0);
                let weight = if is_inhibitory {
                    -w_mag * initial_weight * 0.4
                } else {
                    w_mag * initial_weight
                };
                let delay = 1 + (next_rand() % 3); // Latency 1 to 3 timesteps

                self.monads[i].add_synapse(target, weight, delay);
            }
        }
    }

    /// Step the entire network forward by 1 discrete timestep:
    /// 1. Deliver pending synaptic pulses arriving at current_time.
    /// 2. Inject external stimuli.
    /// 3. Step leaky integration on all monads and collect firing events.
    /// 4. Apply STDP (Rule 3) between firing pairs.
    /// 5. Apply Homeostasis (Rule 2) and Metabolism (Rule 4).
    /// 6. Schedule outgoing pulses for firing cells.
    /// Returns: Count of monads that fired at this timestep.
    pub fn step(&mut self, external_stimuli: &[(MonadId, f32)], reward: f32) -> usize {
        self.current_time += 1;

        // 1. Deliver pending pulses
        let mut remaining_pulses = VecDeque::new();
        while let Some(pulse) = self.pending_pulses.pop_front() {
            if pulse.arrival_time <= self.current_time {
                if let Some(target) = self.monads.get_mut(pulse.target_id) {
                    target.receive_signal(pulse.signal_weight);
                }
            } else {
                remaining_pulses.push_back(pulse);
            }
        }
        self.pending_pulses = remaining_pulses;

        // 2. Inject external stimuli
        for &(monad_id, strength) in external_stimuli {
            if let Some(target) = self.monads.get_mut(monad_id) {
                target.receive_signal(strength);
            }
        }

        // 3. Step integration and collect firings
        let mut fired_monads: Vec<MonadId> = Vec::new();
        for monad in &mut self.monads {
            let fired = monad.step_integration(self.current_time, &self.config);
            if fired {
                fired_monads.push(monad.id);
            }
        }

        // 4. Local Causal Learning (STDP)
        // For each firing monad (postsynaptic), check when its presynaptic inputs fired
        // and adjust synaptic weights locally.
        for &post_id in &fired_monads {
            let t_post = self.current_time;
            for pre_monad in &mut self.monads {
                if let Some(t_pre) = pre_monad.last_spike_time {
                    for synapse in &mut pre_monad.synapses {
                        if synapse.target_id == post_id {
                            Monad::update_synapse_stdp(synapse, t_pre, t_post, &self.config);
                        }
                    }
                }
            }
        }

        // 5. Apply Homeostasis (Rule 2) and Metabolism (Rule 4)
        for monad in &mut self.monads {
            let fired = monad.last_spike_time == Some(self.current_time);
            monad.step_homeostasis(fired, &self.config);
            monad.step_metabolism(fired, reward, &self.config);
        }

        // 6. Schedule outgoing pulses for firing cells
        for &src_id in &fired_monads {
            let src_monad = &self.monads[src_id];
            for syn in &src_monad.synapses {
                self.pending_pulses.push_back(PendingPulse {
                    arrival_time: self.current_time + syn.delay as u64,
                    source_id: src_id,
                    target_id: syn.target_id,
                    signal_weight: syn.weight,
                });
            }
        }

        fired_monads.len()
    }

    /// Extract spike timestamps for Kuramoto order parameter calculation.
    pub fn get_last_spike_times(&self) -> Vec<Option<u64>> {
        self.monads.iter().map(|m| m.last_spike_time).collect()
    }
}
