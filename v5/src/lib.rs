//! OpenUltron v5: The Synthetic Neuromorphic Organism.
//!
//! An emergent artificial intelligence constructed from Donald Hebb's Cell Assembly Theory,
//! Karl Friston's Variational Free Energy Principle, and 5 biological organ subsystems:
//! 1. Sensory Thalamus
//! 2. Hippocampal Formation
//! 3. Neocortical Substrate
//! 4. Neuromodulatory Core
//! 5. Vocal-Motor Organ

pub mod monad;
pub mod organism;
pub mod organs;
pub mod vocab;

pub use monad::{ConceptAssembly, SpikingMonad};
pub use organism::OpenUltronOrganism;
pub use organs::OrganismBiometrics;
pub use vocab::Vocabulary;
