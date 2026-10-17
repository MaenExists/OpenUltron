use crate::model::NanoUltron;
use crate::vocab::Tokenizer;
use std::time::Instant;

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x853c49e6748fea9b } else { seed };
        Self { state: s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub stop_token_ids: Vec<usize>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 48,
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.15,
            seed: 42,
            stop_token_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub generated_tokens: usize,
    pub elapsed_ms: f64,
    pub tokens_per_sec: f64,
}

pub struct Generator;

impl Generator {
    pub fn generate(
        model: &NanoUltron,
        tokenizer: &Tokenizer,
        prompt: &str,
        config: &GenerationConfig,
    ) -> GenerationResult {
        let mut rng = SimpleRng::new(
            if config.seed == 0 {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64
            } else {
                config.seed
            }
        );

        let mut tokens = tokenizer.encode(prompt);
        if tokens.is_empty() {
            tokens.push(tokenizer.vocab.bos_id);
        }

        let start_len = tokens.len();
        let max_ctx = model.config.max_seq_len;
        let start_time = Instant::now();

        for _ in 0..config.max_new_tokens {
            let context_slice = if tokens.len() > max_ctx {
                &tokens[tokens.len() - max_ctx..]
            } else {
                &tokens[..]
            };

            let out = model.forward(context_slice);
            let seq_len = context_slice.len();
            let vocab_size = model.config.vocab_size;

            // Logits of the last token position
            let last_logits_offset = (seq_len - 1) * vocab_size;
            let mut logits = out.logits[last_logits_offset..last_logits_offset + vocab_size].to_vec();

            // Repetition penalty
            if config.repetition_penalty > 1.0 {
                let recent_start = tokens.len().saturating_sub(32);
                for &past_token in &tokens[recent_start..] {
                    if past_token < logits.len() {
                        if logits[past_token] > 0.0 {
                            logits[past_token] /= config.repetition_penalty;
                        } else {
                            logits[past_token] *= config.repetition_penalty;
                        }
                    }
                }
            }

            // Temperature scaling
            let temp = config.temperature.max(1e-4);
            for val in logits.iter_mut() {
                *val /= temp;
            }

            // Convert to probabilities with Softmax
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for val in logits.iter_mut() {
                *val = (*val - max_logit).exp();
                sum_exp += *val;
            }
            for val in logits.iter_mut() {
                *val /= sum_exp;
            }

            // Nucleus (Top-p) filtering & sampling
            let next_token = Self::sample_top_p(&logits, config.top_p, &mut rng);

            if next_token == tokenizer.vocab.eos_id || config.stop_token_ids.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
        }

        let elapsed = start_time.elapsed();
        let generated_token_ids = &tokens[start_len..];
        let text = tokenizer.decode(generated_token_ids);
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
            generated_token_ids.len() as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        GenerationResult {
            text,
            generated_tokens: generated_token_ids.len(),
            elapsed_ms,
            tokens_per_sec,
        }
    }

    fn sample_top_p(probs: &[f32], top_p: f32, rng: &mut SimpleRng) -> usize {
        if top_p <= 0.0 || top_p >= 1.0 {
            // Greedy argmax
            return probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
        }

        // Create indexed list and sort descending
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find cutoff
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        let filtered = &indexed[..cutoff_idx];
        let total_p: f32 = filtered.iter().map(|&(_, p)| p).sum();
        let r = rng.next_f32() * total_p;

        let mut acc = 0.0f32;
        for &(token, p) in filtered {
            acc += p;
            if acc >= r {
                return token;
            }
        }

        filtered.last().map(|&(t, _)| t).unwrap_or(0)
    }
}
