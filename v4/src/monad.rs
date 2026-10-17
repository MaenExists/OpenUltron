//! Lightweight Spiking Monad substrate for OpenUltron v4.

#[derive(Debug, Clone)]
pub struct Monad {
    pub id: usize,
    pub potential: f32,
    pub threshold: f32,
    pub total_spikes: u64,
}

impl Monad {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            potential: 0.0,
            threshold: 0.90,
            total_spikes: 0,
        }
    }

    #[inline(always)]
    pub fn excite(&mut self, current: f32) -> bool {
        self.potential += current;
        if self.potential >= self.threshold {
            self.potential = 0.0;
            self.total_spikes += 1;
            true
        } else {
            false
        }
    }
}
