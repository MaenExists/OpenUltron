//! Neocortical Assembly Substrate Organ for OpenUltron v5.
//!
//! Functions:
//! - Physical Cell Assembly Substrate: Spiking Monad ensembles per concept.
//! - Multi-Order Causal STDP: Bigram and trigram sequential manifolds governing grammar and syntax.
//! - Lateral Associative Resonance: Bidirectional Hebbian semantic networks.
//! - Cross-Turn Conditioning: Synaptic bridges connecting sensory prompt assemblies to response assemblies.
//! - Homeostatic Synaptic Pruning: Eliminates weak, disused synapses under metabolic pressure.

use crate::monad::ConceptAssembly;
use crate::vocab::Vocabulary;
use std::collections::HashMap;

pub struct Neocortex {
    pub assemblies: Vec<ConceptAssembly>,
    pub assembly_size: usize,

    // Synaptic weight manifolds
    pub w_trans: HashMap<(usize, usize), f32>,
    pub w_tri: HashMap<(usize, usize, usize), f32>,
    pub w_assoc: HashMap<(usize, usize), f32>,
    pub w_cross: HashMap<(usize, usize), f32>,
    pub w_starter: HashMap<usize, f32>,
    pub w_prompt_starter: HashMap<(usize, usize), f32>,
    pub w_prompt_bi_starter: HashMap<(usize, usize, usize), f32>,
    pub w_term: HashMap<usize, f32>,

    pub total_spikes: u64,
}

impl Neocortex {
    pub fn new(initial_vocab_size: usize, assembly_size: usize) -> Self {
        let mut assemblies = Vec::with_capacity(initial_vocab_size);
        for id in 0..initial_vocab_size {
            assemblies.push(ConceptAssembly::new(id, assembly_size));
        }

        Self {
            assemblies,
            assembly_size,
            w_trans: HashMap::new(),
            w_tri: HashMap::new(),
            w_assoc: HashMap::new(),
            w_cross: HashMap::new(),
            w_starter: HashMap::new(),
            w_prompt_starter: HashMap::new(),
            w_prompt_bi_starter: HashMap::new(),
            w_term: HashMap::new(),
            total_spikes: 0,
        }
    }

    pub fn ensure_assembly(&mut self, token_id: usize) {
        while self.assemblies.len() <= token_id {
            let id = self.assemblies.len();
            self.assemblies.push(ConceptAssembly::new(id, self.assembly_size));
        }
    }

    /// Excite a specific concept assembly, firing biological spikes and updating spike metrics.
    pub fn excite_concept(&mut self, token_id: usize, current: f32, noise: f32) -> u64 {
        self.ensure_assembly(token_id);
        if let Some(assembly) = self.assemblies.get_mut(token_id) {
            let spikes = assembly.excite(current, noise);
            self.total_spikes += spikes;
            spikes
        } else {
            0
        }
    }

    /// STDP update from sequential speech stream.
    pub fn learn_sequence(
        &mut self,
        prompt_tokens: &[usize],
        response_tokens: &[usize],
        vocab: &Vocabulary,
        dopamine: f32,
    ) {
        let period_id = vocab.token_to_id(".");
        let qmark_id = vocab.token_to_id("?");
        let colon_id = vocab.token_to_id(":");
        let user_id = vocab.token_to_id("user");
        let ultron_id = vocab.token_to_id("ultron");

        let plasticity_rate = (0.7 + 0.6 * dopamine).clamp(0.5, 2.0);

        // 1. Starter conditioning
        if let Some(&first_tok) = response_tokens.first() {
            if first_tok != period_id && first_tok != qmark_id && first_tok != colon_id {
                *self.w_starter.entry(first_tok).or_insert(0.0) += 2.0 * plasticity_rate;
                for &u in prompt_tokens {
                    if u != user_id && u != colon_id && u != period_id {
                        *self.w_prompt_starter.entry((u, first_tok)).or_insert(0.0) += 3.0 * plasticity_rate;
                    }
                }
                for i in 1..prompt_tokens.len() {
                    let u1 = prompt_tokens[i - 1];
                    let u2 = prompt_tokens[i];
                    if u1 != user_id && u2 != user_id && u1 != colon_id && u2 != colon_id && u1 != period_id && u2 != period_id {
                        *self.w_prompt_bi_starter.entry((u1, u2, first_tok)).or_insert(0.0) += 4.5 * plasticity_rate;
                    }
                }
            }
        }

        // 2. Cross-Turn Associations (Prompt -> Response)
        for &u in prompt_tokens {
            if u == user_id || u == colon_id || u == period_id {
                continue;
            }
            for &v in response_tokens {
                if v == colon_id || v == period_id || v == qmark_id || v == ultron_id {
                    continue;
                }
                *self.w_cross.entry((u, v)).or_insert(0.0) += 0.9 * plasticity_rate;
            }
        }

        // 3. Sequential Causal Manifold (Bigrams & Trigrams)
        for i in 1..response_tokens.len() {
            let u = response_tokens[i - 1];
            let v = response_tokens[i];
            if u != colon_id && v != colon_id {
                *self.w_trans.entry((u, v)).or_insert(0.0) += 2.2 * plasticity_rate;

                if i >= 2 {
                    let w = response_tokens[i - 2];
                    if w != colon_id {
                        *self.w_tri.entry((w, u, v)).or_insert(0.0) += 2.8 * plasticity_rate;
                    }
                }
            }
        }

        // 4. Syntactic Termination Conditioning
        if let Some(&last_tok) = response_tokens.last() {
            if vocab.is_terminal_punct(last_tok) && response_tokens.len() >= 2 {
                let pen_ult = response_tokens[response_tokens.len() - 2];
                *self.w_term.entry(pen_ult).or_insert(0.0) += 1.5 * plasticity_rate;
            }
        }

        // 5. Symmetric Associative Resonance (Hebbian co-activation)
        let window = 4;
        for i in 0..response_tokens.len() {
            let u = response_tokens[i];
            for j in (i + 1)..((i + window).min(response_tokens.len())) {
                let v = response_tokens[j];
                *self.w_assoc.entry((u, v)).or_insert(0.0) += 0.45 * plasticity_rate;
                *self.w_assoc.entry((v, u)).or_insert(0.0) += 0.45 * plasticity_rate;
            }
        }
    }

    /// Incorporate consolidated memory traces from Hippocampus.
    pub fn integrate_hippocampal_ripples(&mut self, ripples: &[(usize, usize, f32)]) {
        for &(u, v, w) in ripples {
            *self.w_assoc.entry((u, v)).or_insert(0.0) += w;
            *self.w_assoc.entry((v, u)).or_insert(0.0) += w;
        }
    }

    /// Metabolic Synaptic Pruning (Oja's Rule Homeostatic Balancing).
    /// Removes synapses below threshold to prevent synaptic weight explosion and preserve lightweightness.
    pub fn prune_synapses(&mut self, threshold: f32) -> usize {
        let before_count = self.total_synapses();
        self.w_trans.retain(|_, &mut w| w >= threshold);
        self.w_tri.retain(|_, &mut w| w >= threshold);
        self.w_assoc.retain(|_, &mut w| w >= threshold);
        self.w_cross.retain(|_, &mut w| w >= threshold);
        self.w_prompt_starter.retain(|_, &mut w| w >= threshold);
        self.w_prompt_bi_starter.retain(|_, &mut w| w >= threshold);
        let after_count = self.total_synapses();
        before_count.saturating_sub(after_count)
    }

    pub fn total_synapses(&self) -> usize {
        self.w_trans.len()
            + self.w_tri.len()
            + self.w_assoc.len()
            + self.w_cross.len()
            + self.w_prompt_starter.len()
            + self.w_prompt_bi_starter.len()
    }
}
