//! OpenUltron v2: Sensorimotor Language Brain Library.

pub mod brain;
pub mod config;
pub mod monad;
pub mod motor;
pub mod sensory;

pub use brain::{BrainTelemetry, OpenUltronBrain};
pub use config::BrainConfig;
pub use monad::{Monad, MonadId, Synapse};
pub use motor::MotorVoice;
pub use sensory::{char_to_index, index_to_char, SensoryEyes, VOCAB_SIZE};
