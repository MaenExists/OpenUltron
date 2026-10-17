//! OpenUltron v4: The Alien Neural Mind.
//!
//! A complex adaptive spiking neural organism:
//! - Real Leaky Integrate-and-Fire (LIF) Monads in synchronized assemblies.
//! - Causal STDP, second-order n-gram transitions, and associative attractor basins.
//! - Online lifelong synaptic plasticity adapting in real time from conversation.
//! - Autoregressive token-by-token neural speech generation without if-else templates.

pub mod monad;
pub mod network;
pub mod vocab;

pub use monad::{ConceptAssembly, SpikingMonad};
pub use network::{AlienBrain, AlienMetrics};
pub use vocab::Vocabulary;
