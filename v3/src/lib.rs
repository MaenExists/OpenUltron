//! OpenUltron v3: Semantic Assemblies & Lexical Attractors Library.

pub mod assembly;
pub mod brain;
pub mod config;
pub mod monad;
pub mod vocab;

pub use assembly::{AssemblyManager, WordAssembly};
pub use brain::{BrainTelemetryV3, OpenUltronBrainV3};
pub use config::BrainConfig;
pub use monad::{Monad, MonadId, Synapse};
pub use vocab::Vocabulary;
