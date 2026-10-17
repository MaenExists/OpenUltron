//! Donald Hebb's Cell Assembly Engine for OpenUltron v3.
//!
//! A Cell Assembly is a tightly-coupled cluster of Monads that ignite in resonance
//! to represent a specific word, concept, or cognitive symbol.

use crate::monad::MonadId;

#[derive(Debug, Clone)]
pub struct WordAssembly {
    pub word_id: usize,
    pub monad_ids: Vec<MonadId>,
}

impl WordAssembly {
    pub fn new(word_id: usize, monad_ids: Vec<MonadId>) -> Self {
        Self {
            word_id,
            monad_ids,
        }
    }
}

pub struct AssemblyManager {
    pub assembly_size: usize,
    pub assemblies: Vec<WordAssembly>,
}

impl AssemblyManager {
    pub fn new(total_vocab: usize, assembly_size: usize, start_monad_id: usize) -> Self {
        let mut assemblies = Vec::with_capacity(total_vocab);
        for word_id in 0..total_vocab {
            let start = start_monad_id + (word_id * assembly_size);
            let monad_ids = (start..start + assembly_size).collect();
            assemblies.push(WordAssembly::new(word_id, monad_ids));
        }

        Self {
            assembly_size,
            assemblies,
        }
    }

    #[inline(always)]
    pub fn get_assembly(&self, word_id: usize) -> Option<&WordAssembly> {
        self.assemblies.get(word_id)
    }

    pub fn total_assembly_monads(&self) -> usize {
        self.assemblies.len() * self.assembly_size
    }
}
