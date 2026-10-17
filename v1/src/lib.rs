//! OpenUltron v1 Library
//!
//! Grounded in Systems Theory, Cybernetics, and Self-Organized Criticality.

pub mod config;
pub mod monad;
pub mod network;
pub mod theory;

pub use config::MonadConfig;
pub use monad::{Monad, MonadId, Synapse};
pub use network::MonadNetwork;
pub use theory::{DynamicalRegime, EmergenceMetrics, TheoryEngine};
