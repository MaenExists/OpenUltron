//! Sensory Perception ("The Eyes"): Text-to-Spike Encoding.
//!
//! Maps discrete linguistic characters into spatial-temporal spike bursts
//! injected directly into the sensory receptive fields of the OpenUltron brain.

pub const VOCAB_SIZE: usize = 97;

/// Convert a character to a compact vocabulary index [0, 96].
pub fn char_to_index(c: char) -> Option<usize> {
    match c {
        '\n' => Some(0),
        ' '..='~' => Some(1 + (c as usize - ' ' as usize)),
        _ => Some(1), // Fallback to space for unrecognized characters
    }
}

/// Convert a vocabulary index [0, 96] back to its character.
pub fn index_to_char(idx: usize) -> char {
    match idx {
        0 => '\n',
        1..=95 => ((' ' as u8) + (idx - 1) as u8) as char,
        _ => '?',
    }
}

/// The Sensory Receptive Field ("The Eyes")
pub struct SensoryEyes {
    pub sensory_per_char: usize,
    pub total_sensory_monads: usize,
}

impl SensoryEyes {
    pub fn new(sensory_per_char: usize) -> Self {
        let total_sensory_monads = VOCAB_SIZE * sensory_per_char;
        Self {
            sensory_per_char,
            total_sensory_monads,
        }
    }

    /// Returns the global Monad IDs associated with a specific character's receptive field.
    pub fn receptive_field(&self, c: char) -> Vec<usize> {
        if let Some(idx) = char_to_index(c) {
            let start = idx * self.sensory_per_char;
            (start..start + self.sensory_per_char).collect()
        } else {
            Vec::new()
        }
    }
}
