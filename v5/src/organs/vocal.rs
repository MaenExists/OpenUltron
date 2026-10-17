//! Efferent Vocal-Motor Production Organ for OpenUltron v5.
//!
//! Functions:
//! - Attractor Basin Decoding: Synthesizes cortical attractor fields, hippocampal context, and causal manifolds.
//! - Autoregressive Motor Action Selection: Probabilistic Boltzmann & Nucleus sampling.
//! - Syntactic Cadence & Closure: Resolves phrases into complete thoughts using punctuation attractors.
//! - Efferent Spike Triggering: Stimulates the physical Neocortical monads during speech emission.

use crate::organs::cortex::Neocortex;
use crate::organs::neuromodulator::Neuromodulator;
use crate::vocab::Vocabulary;

pub struct VocalOrgan {
    rng_state: u64,
}

impl VocalOrgan {
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: seed.max(123456789),
        }
    }

    #[inline(always)]
    pub fn is_dangling(tok: &str) -> bool {
        matches!(
            tok,
            "a" | "an" | "the" | "my" | "your" | "its" | "our" | "their" | "to" | "of" | "in"
                | "with" | "for" | "and" | "or" | "is" | "are" | "was" | "were" | ","
        )
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

    /// Sample candidate from scores using temperature softmax and top-p filtering.
    fn sample_candidate(&mut self, candidates: &[(usize, f32)], temperature: f32) -> usize {
        if candidates.is_empty() {
            return Vocabulary::UNK_ID;
        }

        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = sorted.len().min(40);
        let active = &sorted[..top_k];

        let max_val = active[0].1;
        let temp = temperature.clamp(0.2, 2.0);

        let exp_scores: Vec<f32> = active
            .iter()
            .map(|(_, score)| ((score - max_val) / temp).exp())
            .collect();

        let sum_exp: f32 = exp_scores.iter().sum();
        if sum_exp <= 0.0 || sum_exp.is_nan() {
            return active[0].0;
        }

        let probs: Vec<f32> = exp_scores.iter().map(|s| s / sum_exp).collect();

        // Top-p nucleus filtering (p = 0.88)
        let top_p = 0.88;
        let mut cumulative = 0.0;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= top_p {
                cutoff = i + 1;
                break;
            }
        }

        let filtered_probs = &probs[..cutoff];
        let filtered_sum: f32 = filtered_probs.iter().sum();
        let r = self.rand_f32() * filtered_sum;

        let mut acc = 0.0;
        for (i, &p) in filtered_probs.iter().enumerate() {
            acc += p;
            if r <= acc {
                return active[i].0;
            }
        }

        active[0].0
    }

    /// Generate an emergent spoken response through collective organ interaction.
    pub fn articulate(
        &mut self,
        sensory_tokens: &[usize],
        cortex: &mut Neocortex,
        neuromodulator: &Neuromodulator,
        hippocampal_attractors: &[f32],
        vocab: &Vocabulary,
        max_words: usize,
    ) -> (String, Vec<usize>, u64) {
        let vocab_size = vocab.size();
        let period_id = vocab.token_to_id(".");
        let qmark_id = vocab.token_to_id("?");
        let colon_id = vocab.token_to_id(":");
        let user_id = vocab.token_to_id("user");
        let ultron_id = vocab.token_to_id("ultron");

        // 1. Build Unified Sensory + Hippocampal Attractor Field
        let mut attractor = vec![0.0f32; vocab_size];

        for &u in sensory_tokens {
            let u_weight = vocab.idf(u);
            let mut total_cross = 0.0f32;
            let mut total_assoc = 0.0f32;
            for v in 4..vocab_size {
                total_cross += cortex.w_cross.get(&(u, v)).copied().unwrap_or(0.0);
                total_assoc += cortex.w_assoc.get(&(u, v)).copied().unwrap_or(0.0);
            }
            let norm_cross = total_cross.max(1.0);
            let norm_assoc = total_assoc.max(1.0);

            for v in 4..vocab_size {
                let cross = cortex.w_cross.get(&(u, v)).copied().unwrap_or(0.0) / norm_cross;
                let assoc = cortex.w_assoc.get(&(u, v)).copied().unwrap_or(0.0) / norm_assoc;
                attractor[v] += (cross * 3.5 + assoc * 1.5) * u_weight;
            }
        }

        // Add hippocampal episodic gravity
        for (v, &hip_w) in hippocampal_attractors.iter().enumerate() {
            if v < vocab_size {
                attractor[v] += hip_w * 1.5;
            }
        }

        // Normalize attractor field to prevent context from overpowering causal transitions
        let max_attr = attractor.iter().copied().fold(0.0f32, f32::max);
        if max_attr > 0.0 {
            for a in attractor.iter_mut() {
                *a = (*a / max_attr) * 4.0;
            }
        }

        // 2. Determine Starter Token with Bayesian conditional specificity
        let mut starter_scores = vec![0.0f32; vocab_size];

        // Unigram prompt starter specificity
        for &u in sensory_tokens {
            let u_weight = vocab.idf(u);
            let mut total_starter = 0.0f32;
            for v in 4..vocab_size {
                total_starter += cortex.w_prompt_starter.get(&(u, v)).copied().unwrap_or(0.0);
            }
            if total_starter > 0.0 {
                for v in 4..vocab_size {
                    let w = cortex.w_prompt_starter.get(&(u, v)).copied().unwrap_or(0.0);
                    starter_scores[v] += (w / total_starter) * u_weight * 5.0;
                }
            }
        }

        // Bigram prompt starter specificity (second-order precision)
        for i in 1..sensory_tokens.len() {
            let u1 = sensory_tokens[i - 1];
            let u2 = sensory_tokens[i];
            let u2_weight = vocab.idf(u2);
            let mut total_bi = 0.0f32;
            for v in 4..vocab_size {
                total_bi += cortex.w_prompt_bi_starter.get(&(u1, u2, v)).copied().unwrap_or(0.0);
            }
            if total_bi > 0.0 {
                for v in 4..vocab_size {
                    let w = cortex.w_prompt_bi_starter.get(&(u1, u2, v)).copied().unwrap_or(0.0);
                    starter_scores[v] += (w / total_bi) * u2_weight * 8.0;
                }
            }
        }

        let max_gen_starter = cortex.w_starter.values().copied().fold(0.0f32, f32::max).max(1.0);

        let mut starter_candidates = Vec::with_capacity(vocab_size);
        for v in 4..vocab_size {
            if vocab.is_punctuation(v) || v == user_id || v == ultron_id || v == colon_id {
                continue;
            }
            let gen_starter = (cortex.w_starter.get(&v).copied().unwrap_or(0.0) / max_gen_starter) * 1.2;
            let prompt_starter = starter_scores[v];
            let noise = self.rand_gaussian() * neuromodulator.thermal_noise_amplitude();
            let score = gen_starter + prompt_starter + (attractor[v] * 2.2) + noise;
            starter_candidates.push((v, score));
        }

        let first_token = self.sample_candidate(&starter_candidates, neuromodulator.dynamic_temperature());
        let mut emitted_tokens = vec![first_token];

        // Fire opening spikes in neocortex
        let mut spikes_fired: u64 = cortex.excite_concept(
            first_token,
            1.6,
            neuromodulator.thermal_noise_amplitude(),
        );

        // 3. Autoregressively emit subsequent motor tokens
        for _ in 1..max_words {
            let prev_w = *emitted_tokens.last().unwrap();
            let prev_tok = vocab.id_to_token(prev_w);
            let prev_is_dangling = Self::is_dangling(prev_tok);

            let prev_prev_w = if emitted_tokens.len() >= 2 {
                Some(emitted_tokens[emitted_tokens.len() - 2])
            } else {
                None
            };

            // Calculate max causal transition support
            let mut max_causal: f32 = 0.0;
            for v in 4..vocab_size {
                let bw = cortex.w_trans.get(&(prev_w, v)).copied().unwrap_or(0.0);
                let tw = if let Some(pp) = prev_prev_w {
                    cortex.w_tri.get(&(pp, prev_w, v)).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                let sup = bw * 1.0 + tw * 2.2;
                if sup > max_causal {
                    max_causal = sup;
                }
            }
            let max_causal = max_causal.max(1.0);

            let mut candidates = Vec::with_capacity(vocab_size);

            for v in 4..vocab_size {
                if v == user_id || v == ultron_id || v == colon_id {
                    continue;
                }

                // Prevent immediate repetition
                if v == prev_w && !vocab.is_punctuation(v) {
                    continue;
                }

                // Disallow terminal punctuation before minimum utterance length
                if emitted_tokens.len() < 4 && vocab.is_terminal_punct(v) {
                    continue;
                }

                // Prevent dangling prepositions/determiners from being followed by punctuation
                if prev_is_dangling && vocab.is_punctuation(v) {
                    continue;
                }

                let bigram_w = cortex.w_trans.get(&(prev_w, v)).copied().unwrap_or(0.0);
                let trigram_w = if let Some(pp) = prev_prev_w {
                    cortex.w_tri.get(&(pp, prev_w, v)).copied().unwrap_or(0.0)
                } else {
                    0.0
                };

                let causal_support = (bigram_w * 1.0) + (trigram_w * 2.2);

                let transition_score = if causal_support < 0.02 && !vocab.is_punctuation(v) {
                    -28.0 // Steep penalty for physically non-existent transitions
                } else {
                    (causal_support / max_causal) * 4.2
                };

                let attractor_pull = attractor[v] * 2.4;

                // Synaptic vesicle depression (strict anti-cycling inhibition)
                let rep_count = emitted_tokens.iter().filter(|&&id| id == v).count();
                let rep_penalty = if !vocab.is_punctuation(v) && rep_count > 0 {
                    (rep_count as f32) * 20.0
                } else {
                    0.0
                };

                // Termination preference
                let term_bias = if vocab.is_terminal_punct(v) {
                    cortex.w_term.get(&prev_w).copied().unwrap_or(0.0) * 0.5
                } else {
                    0.0
                };

                let noise = self.rand_gaussian() * neuromodulator.thermal_noise_amplitude();
                let score = transition_score + attractor_pull - rep_penalty + term_bias + noise;
                candidates.push((v, score));
            }

            // Syntactic Cadence Boost: Natural closure pressure after 6-8 words
            // Only if current word is NOT a dangling connector
            if emitted_tokens.len() >= 6 && !prev_is_dangling {
                let cadence_boost = ((emitted_tokens.len() - 5) as f32) * 2.5;
                if let Some(pos) = candidates.iter_mut().find(|(id, _)| *id == period_id) {
                    pos.1 += cadence_boost;
                }
                if let Some(pos) = candidates.iter_mut().find(|(id, _)| *id == qmark_id) {
                    pos.1 += cadence_boost * 0.8;
                }
            }

            let next_token = self.sample_candidate(&candidates, neuromodulator.dynamic_temperature());

            // Fire action potential in Neocortex
            let spikes = cortex.excite_concept(
                next_token,
                1.5,
                neuromodulator.thermal_noise_amplitude(),
            );
            spikes_fired += spikes;

            emitted_tokens.push(next_token);

            // Natural stopping condition: terminal punctuation reached
            if vocab.is_terminal_punct(next_token) {
                break;
            }
        }

        // 4. Decode emitted tokens into formatted speech
        let mut words: Vec<String> = Vec::new();
        let mut punctuation_suffix = ".";

        for &id in &emitted_tokens {
            let tok = vocab.id_to_token(id);
            if tok == "." || tok == "?" || tok == "!" {
                punctuation_suffix = tok;
            } else if tok == "," {
                if let Some(last) = words.last_mut() {
                    last.push(',');
                }
            } else if tok != ":" && tok != "<eos>" && tok != "<bos>" && tok != "<pad>" && tok != "<unk>" {
                words.push(tok.to_string());
            }
        }

        // If ends with a dangling preposition/determiner, trim it before punctuation
        while let Some(last) = words.last() {
            let clean_last = last.trim_end_matches(',').to_lowercase();
            if Self::is_dangling(&clean_last) && words.len() > 3 {
                words.pop();
            } else {
                break;
            }
        }

        let mut utterance = words.join(" ");
        if !utterance.is_empty() {
            let mut chars = utterance.chars();
            if let Some(first) = chars.next() {
                utterance = first.to_uppercase().collect::<String>() + chars.as_str();
            }
            if !utterance.ends_with('.') && !utterance.ends_with('?') && !utterance.ends_with('!') {
                utterance.push_str(punctuation_suffix);
            }
        }

        (utterance, emitted_tokens, spikes_fired)
    }
}
