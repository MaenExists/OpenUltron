//! Subsystem Organs of the OpenUltron v5 Neuromorphic Organism.

pub mod cortex;
pub mod hippocampus;
pub mod neuromodulator;
pub mod sensory;
pub mod vocal;

pub use cortex::Neocortex;
pub use hippocampus::Hippocampus;
pub use neuromodulator::{Neuromodulator, OrganismBiometrics};
pub use sensory::{SensoryBurst, SensoryThalamus};
pub use vocal::VocalOrgan;
