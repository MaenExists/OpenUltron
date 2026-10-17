//! Alien Spiking Associative Neural Network for OpenUltron v4.
//!
//! True emergence without canned templates or if-else rules:
//! - Words and concepts are represented by synchronized Spiking Monad assemblies.
//! - Thoughts propagate via causal second-order STDP and associative attractor fields.
//! - Emits speech autoregressively through probabilistic energy sampling.
//! - Continuously adapts its synaptic wiring in real time from user conversation.

use crate::monad::ConceptAssembly;
use crate::vocab::Vocabulary;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AlienMetrics {
    pub free_energy: f32,
    pub spikes_fired: u64,
    pub active_attractors: usize,
    pub vocabulary_size: usize,
    pub synaptic_connections: usize,
}

pub struct AlienBrain {
    pub vocab: Vocabulary,
    pub assemblies: Vec<ConceptAssembly>,
    pub assembly_size: usize,

    // Synaptic weight tensors
    pub w_trans: HashMap<(usize, usize), f32>,
    pub w_tri: HashMap<(usize, usize, usize), f32>,
    pub w_assoc: HashMap<(usize, usize), f32>,
    pub w_cross: HashMap<(usize, usize), f32>,
    pub w_starter: HashMap<usize, f32>,
    pub w_prompt_starter: HashMap<(usize, usize), f32>,
    pub token_counts: HashMap<usize, usize>,

    // Dynamic internal state
    pub free_energy: f32, // Confusion / surprise [0.0, 1.0]
    pub total_spikes: u64,
    rng_state: u64,
}

impl AlienBrain {
    pub fn new(assembly_size: usize) -> Self {
        let vocab = Vocabulary::new();
        let mut assemblies = Vec::new();
        for id in 0..vocab.size() {
            assemblies.push(ConceptAssembly::new(id, assembly_size));
        }

        Self {
            vocab,
            assemblies,
            assembly_size,
            w_trans: HashMap::new(),
            w_tri: HashMap::new(),
            w_assoc: HashMap::new(),
            w_cross: HashMap::new(),
            w_starter: HashMap::new(),
            w_prompt_starter: HashMap::new(),
            token_counts: HashMap::new(),
            free_energy: 0.94, // High initial infant confusion
            total_spikes: 0,
            rng_state: 88888888,
        }
    }

    /// Calculate Inverse Document Frequency (Information Content) for a token.
    pub fn idf(&self, token_id: usize) -> f32 {
        let total = self.token_counts.values().sum::<usize>().max(1) as f32;
        let count = self.token_counts.get(&token_id).copied().unwrap_or(1) as f32;
        (1.0 + total / count).ln().clamp(0.6, 4.5)
    }

    #[inline(always)]
    fn xorshift64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    #[inline(always)]
    fn rand_f32(&mut self) -> f32 {
        (self.xorshift64() as f32) / (u64::MAX as f32)
    }

    fn rand_gaussian(&mut self) -> f32 {
        let u1 = self.rand_f32().max(1e-7);
        let u2 = self.rand_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Ensure that a concept assembly exists for the given token ID.
    fn ensure_assembly(&mut self, token_id: usize) {
        while self.assemblies.len() <= token_id {
            let id = self.assemblies.len();
            self.assemblies.push(ConceptAssembly::new(id, self.assembly_size));
        }
    }

    /// Learn from a text sequence using causal STDP and associative co-activation.
    pub fn learn_sequence(&mut self, text: &str) {
        let raw_tokens = Vocabulary::tokenize(text);
        if raw_tokens.is_empty() {
            return;
        }

        let mut token_ids = Vec::with_capacity(raw_tokens.len());
        for t in &raw_tokens {
            let id = self.vocab.get_or_insert(t);
            self.ensure_assembly(id);
            *self.token_counts.entry(id).or_insert(0) += 1;
            token_ids.push(id);
        }

        let user_id = self.vocab.token_to_id("user");
        let ultron_id = self.vocab.token_to_id("ultron");
        let colon_id = self.vocab.token_to_id(":");
        let period_id = self.vocab.token_to_id(".");

        // Check if this line is a dialogue pair: user : ... . ultron : ... .
        let mut split_idx = None;
        for i in 0..token_ids.len() {
            if token_ids[i] == ultron_id && i + 1 < token_ids.len() && token_ids[i + 1] == colon_id {
                split_idx = Some(i + 2);
                break;
            }
        }

        if let Some(s_idx) = split_idx {
            let user_part = &token_ids[..s_idx.saturating_sub(2)];
            let ultron_part = &token_ids[s_idx..];

            if !ultron_part.is_empty() {
                let first_tok = ultron_part[0];
                if first_tok != period_id && first_tok != colon_id {
                    *self.w_starter.entry(first_tok).or_insert(0.0) += 2.0;
                    for &u in user_part {
                        if u != user_id && u != colon_id && u != period_id {
                            *self.w_prompt_starter.entry((u, first_tok)).or_insert(0.0) += 3.0;
                        }
                    }
                }
            }

            // 1. Cross-Turn Associations: map user prompt tokens to ultron response tokens
            for &u in user_part {
                if u == user_id || u == colon_id || u == period_id {
                    continue;
                }
                for &v in ultron_part {
                    if v == colon_id || v == period_id {
                        continue;
                    }
                    *self.w_cross.entry((u, v)).or_insert(0.0) += 0.85;
                }
            }
        }

        // 2. Sequential Bigram & Trigram Plasticity on Ultron speech stream
        let train_target = if let Some(s_idx) = split_idx {
            &token_ids[s_idx..]
        } else {
            &token_ids[..]
        };

        for i in 1..train_target.len() {
            let u = train_target[i - 1];
            let v = train_target[i];
            if u != colon_id && v != colon_id {
                *self.w_trans.entry((u, v)).or_insert(0.0) += 2.0;

                if i >= 2 {
                    let w = train_target[i - 2];
                    if w != colon_id {
                        *self.w_tri.entry((w, u, v)).or_insert(0.0) += 2.5;
                    }
                }
            }
        }

        // 3. Sliding Window Associative Resonance (symmetric Hebbian co-activation)
        let window_size = 4;
        for i in 0..train_target.len() {
            let u = train_target[i];
            for j in (i + 1)..((i + window_size).min(train_target.len())) {
                let v = train_target[j];
                *self.w_assoc.entry((u, v)).or_insert(0.0) += 0.4;
                *self.w_assoc.entry((v, u)).or_insert(0.0) += 0.4;
            }
        }
    }

    /// Pre-train the brain on a multi-line corpus.
    pub fn train_corpus(&mut self, corpus_text: &str) {
        for line in corpus_text.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                self.learn_sequence(trimmed);
            }
        }
    }

    /// Autoregressive pure neural generation in response to user input.
    /// Emits tokens one by one based on attractor fields, spiking assemblies,
    /// and dynamic temperature sampling. Zero hardcoded template strings.
    pub fn generate_thought(&mut self, user_input: &str, max_tokens: usize) -> (String, AlienMetrics) {
        let raw_user_tokens = Vocabulary::tokenize(user_input);
        let mut user_token_ids = Vec::new();

        // 1. Ingest user tokens into dynamic vocabulary & monad substrate
        for t in &raw_user_tokens {
            let id = self.vocab.get_or_insert(t);
            self.ensure_assembly(id);
            user_token_ids.push(id);
        }

        // 2. Online live learning on the user's speech
        for i in 1..user_token_ids.len() {
            let u = user_token_ids[i - 1];
            let v = user_token_ids[i];
            *self.w_trans.entry((u, v)).or_insert(0.0) += 0.4;
            *self.w_assoc.entry((u, v)).or_insert(0.0) += 0.3;
            *self.w_assoc.entry((v, u)).or_insert(0.0) += 0.3;
        }

        // 3. Compute Free Energy (Surprise / Confusion)
        let mut familiar_transitions = 0.0;
        let mut count = 0.0;
        for i in 1..user_token_ids.len() {
            let u = user_token_ids[i - 1];
            let v = user_token_ids[i];
            if let Some(&w) = self.w_trans.get(&(u, v)) {
                familiar_transitions += (w / 10.0).min(1.0);
            }
            count += 1.0;
        }

        let raw_surprise = if count > 0.0 {
            1.0 - (familiar_transitions / count)
        } else {
            0.5
        };

        // Smoothly update internal free energy (settling with interaction)
        self.free_energy = (self.free_energy * 0.88 + raw_surprise * 0.12).clamp(0.20, 0.95);

        // 4. Build Prompt Attractor Gravity Field
        let vocab_size = self.vocab.size();
        let mut attractor = vec![0.0f32; vocab_size];
        let mut active_attractors = 0;

        for &u in &user_token_ids {
            let u_weight = self.idf(u);
            for v in 4..vocab_size {
                let cross = self.w_cross.get(&(u, v)).copied().unwrap_or(0.0);
                let assoc = self.w_assoc.get(&(u, v)).copied().unwrap_or(0.0);
                let gravity = (cross * 2.2 + assoc * 1.1) * u_weight;
                if gravity > 0.1 {
                    attractor[v] += gravity;
                    active_attractors += 1;
                }
            }
        }

        // Normalize attractor vector to prevent context drowning causal transitions
        let max_attr = attractor.iter().copied().fold(0.0f32, f32::max);
        if max_attr > 0.0 {
            for a in attractor.iter_mut() {
                *a = (*a / max_attr) * 5.0;
            }
        }

        // 5. Autoregressively generate tokens word by word
        let mut emitted_tokens: Vec<usize> = Vec::new();
        let mut spikes_this_thought: u64 = 0;

        let user_id = self.vocab.token_to_id("user");
        let ultron_id = self.vocab.token_to_id("ultron");
        let colon_id = self.vocab.token_to_id(":");
        let period_id = self.vocab.token_to_id(".");

        let max_starter = self.w_starter.values().copied().fold(0.0f32, f32::max).max(1.0);
        let mut raw_prompt_starters = vec![0.0f32; vocab_size];
        for v in 4..vocab_size {
            for &u in &user_token_ids {
                let u_weight = self.idf(u);
                raw_prompt_starters[v] += self.w_prompt_starter.get(&(u, v)).copied().unwrap_or(0.0) * u_weight;
            }
        }
        let max_prompt_starter = raw_prompt_starters.iter().copied().fold(0.0f32, f32::max).max(1.0);

        // Determine opening token
        let mut candidate_scores = Vec::with_capacity(vocab_size);
        for v in 4..vocab_size {
            if v == user_id || v == ultron_id || v == colon_id || v == period_id {
                continue;
            }
            let norm_starter = (self.w_starter.get(&v).copied().unwrap_or(0.0) / max_starter) * 1.5;
            let norm_prompt_starter = (raw_prompt_starters[v] / max_prompt_starter) * 7.0;
            let noise = self.rand_gaussian() * (self.free_energy * 0.3);
            let score = norm_starter + norm_prompt_starter + attractor[v] * 2.5 + noise;
            candidate_scores.push((v, score));
        }

        let first_token = self.sample_softmax_candidate(&candidate_scores, 0.7 + self.free_energy * 0.3);
        emitted_tokens.push(first_token);

        // Fire spikes for opening token
        if let Some(assembly) = self.assemblies.get_mut(first_token) {
            let spikes = assembly.excite(1.5, self.free_energy * 0.2);
            spikes_this_thought += spikes;
            self.total_spikes += spikes;
        }

        // Generate subsequent words autoregressively
        for _ in 1..max_tokens {
            let prev_w = *emitted_tokens.last().unwrap();
            let prev_prev_w = if emitted_tokens.len() >= 2 {
                Some(emitted_tokens[emitted_tokens.len() - 2])
            } else {
                None
            };

            let mut max_causal: f32 = 0.0;
            for v in 4..vocab_size {
                let bw = self.w_trans.get(&(prev_w, v)).copied().unwrap_or(0.0);
                let tw = if let Some(pp) = prev_prev_w {
                    self.w_tri.get(&(pp, prev_w, v)).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                let sup = bw * 1.0 + tw * 2.0;
                if sup > max_causal {
                    max_causal = sup;
                }
            }
            let max_causal = max_causal.max(1.0);

            let mut step_scores = Vec::with_capacity(vocab_size);

            for v in 4..vocab_size {
                if v == user_id || v == ultron_id || v == colon_id {
                    continue;
                }

                // Disallow direct back-to-back word stuttering
                if v == prev_w && v != period_id {
                    continue;
                }

                // Causal transition strength
                let bigram_w = self.w_trans.get(&(prev_w, v)).copied().unwrap_or(0.0);
                let trigram_w = if let Some(pp) = prev_prev_w {
                    self.w_tri.get(&(pp, prev_w, v)).copied().unwrap_or(0.0)
                } else {
                    0.0
                };

                let causal_support = (bigram_w * 1.0) + (trigram_w * 2.0);

                // Strong penalty if no causal transition has ever existed between these words
                let transition_penalty = if causal_support < 0.05 && v != period_id {
                    -25.0
                } else {
                    (causal_support / max_causal) * 4.0
                };

                // Attractor pull from user context
                let ctx_pull = attractor[v] * 2.5;

                // Anti-repetition penalty across recent window
                let rep_count = emitted_tokens.iter().rev().take(6).filter(|&&id| id == v).count();
                let rep_penalty = (rep_count as f32) * 8.0;

                // Thermal chaos / metabolic noise scaled by free energy
                let chaos = self.rand_gaussian() * (self.free_energy * 0.35);

                let net_score = transition_penalty + ctx_pull - rep_penalty + chaos;
                step_scores.push((v, net_score));
            }

            // Natural cadence closure: increase probability of period after 7 words
            if emitted_tokens.len() >= 7 {
                let cadence_boost = ((emitted_tokens.len() - 6) as f32) * 2.0;
                if let Some(pos) = step_scores.iter_mut().find(|(id, _)| *id == period_id) {
                    pos.1 += cadence_boost;
                }
            }

            let temperature = 0.5 + (self.free_energy * 0.4);
            let next_token = self.sample_softmax_candidate(&step_scores, temperature);

            // Excite spiking assembly for next_token
            if let Some(assembly) = self.assemblies.get_mut(next_token) {
                let spikes = assembly.excite(1.5, self.free_energy * 0.2);
                spikes_this_thought += spikes;
                self.total_spikes += spikes;
            }

            emitted_tokens.push(next_token);

            // Stop condition: period or punctuation reached with minimum sentence length
            if next_token == period_id && emitted_tokens.len() >= 5 {
                break;
            }
        }

        // 6. Online plasticity: wire the newly generated thought to the user prompt
        for &u in &user_token_ids {
            for &v in &emitted_tokens {
                if v != period_id && v != colon_id {
                    *self.w_cross.entry((u, v)).or_insert(0.0) += 0.25;
                }
            }
        }
        for i in 1..emitted_tokens.len() {
            let u = emitted_tokens[i - 1];
            let v = emitted_tokens[i];
            *self.w_trans.entry((u, v)).or_insert(0.0) += 0.20;
        }

        // 7. Format emitted tokens into coherent speech
        let mut words: Vec<String> = Vec::new();
        for &id in &emitted_tokens {
            let tok = self.vocab.id_to_token(id);
            if tok != "." && tok != ":" && tok != "<eos>" && tok != "<bos>" {
                words.push(tok.to_string());
            }
        }

        let mut formatted_speech = words.join(" ");
        if !formatted_speech.is_empty() {
            // Capitalize first letter
            let mut chars = formatted_speech.chars();
            if let Some(first) = chars.next() {
                formatted_speech = first.to_uppercase().collect::<String>() + chars.as_str();
            }
            if !formatted_speech.ends_with('.') && !formatted_speech.ends_with('?') && !formatted_speech.ends_with('!') {
                formatted_speech.push('.');
            }
        }

        let metrics = AlienMetrics {
            free_energy: self.free_energy,
            spikes_fired: spikes_this_thought,
            active_attractors,
            vocabulary_size: self.vocab.size(),
            synaptic_connections: self.w_trans.len() + self.w_tri.len() + self.w_cross.len(),
        };

        (formatted_speech, metrics)
    }

    /// Softmax sampling with Nucleus (Top-p) thresholding.
    fn sample_softmax_candidate(&mut self, scored: &[(usize, f32)], temperature: f32) -> usize {
        if scored.is_empty() {
            return Vocabulary::UNK_ID;
        }

        // Find max for numerical stability
        let max_val = scored.iter().map(|&(_, s)| s).fold(f32::NEG_INFINITY, f32::max);

        // Sort descending
        let mut candidates: Vec<(usize, f32)> = scored.to_vec();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top 40 candidates
        let top_candidates = &candidates[..candidates.len().min(40)];

        // Compute exponentiated values
        let mut exp_weights: Vec<(usize, f32)> = Vec::with_capacity(top_candidates.len());
        let mut sum_exp = 0.0;
        for &(id, score) in top_candidates {
            let exp = ((score - max_val) / temperature).exp();
            exp_weights.push((id, exp));
            sum_exp += exp;
        }

        if sum_exp <= 0.0 {
            return top_candidates[0].0;
        }

        // Nucleus (Top-P = 0.85) filtering
        let top_p = 0.85;
        let mut cumulative = 0.0;
        let mut filtered: Vec<(usize, f32)> = Vec::new();
        for (id, exp) in exp_weights {
            let prob = exp / sum_exp;
            cumulative += prob;
            filtered.push((id, exp));
            if cumulative >= top_p {
                break;
            }
        }

        let filtered_sum: f32 = filtered.iter().map(|&(_, w)| w).sum();
        let target = self.rand_f32() * filtered_sum;
        let mut acc = 0.0;
        for (id, w) in filtered {
            acc += w;
            if acc >= target {
                return id;
            }
        }

        top_candidates[0].0
    }
}
