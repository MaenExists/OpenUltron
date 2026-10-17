//! NanoUltron: Pure Neural Transformer Language Model for OpenUltron v6.
//!
//! A real, mathematically complete causal decoder Transformer:
//! - Multi-Head Causal Self-Attention with Scaled Dot-Product
//! - Pre-Layer RMSNorm and Residual Skip Connections
//! - Two-layer Feedforward MLP with GELU activations
//! - Full analytical backpropagation training engine with AdamW optimizer
//! - Zero n-gram tables, zero canned templates, 100% genuine tensor computation

use crate::tensor::TensorOps;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 512,
            d_model: 64,
            n_layers: 3,
            n_heads: 4,
            d_ff: 256,
            max_seq_len: 64,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TransformerLayer {
    // Attention weights
    pub wq: Vec<f32>, // (d_model x d_model)
    pub wk: Vec<f32>, // (d_model x d_model)
    pub wv: Vec<f32>, // (d_model x d_model)
    pub wo: Vec<f32>, // (d_model x d_model)
    pub ln1_g: Vec<f32>, // (d_model)

    // MLP weights
    pub w1: Vec<f32>, // (d_ff x d_model) -> projects d_model up to d_ff
    pub w2: Vec<f32>, // (d_model x d_ff) -> projects d_ff down to d_model
    pub ln2_g: Vec<f32>, // (d_model)

    // AdamW momentum & velocity buffers
    #[serde(skip)]
    pub mwq: Vec<f32>,
    #[serde(skip)]
    pub vwq: Vec<f32>,
    #[serde(skip)]
    pub mwk: Vec<f32>,
    #[serde(skip)]
    pub vwk: Vec<f32>,
    #[serde(skip)]
    pub mwv: Vec<f32>,
    #[serde(skip)]
    pub vwv: Vec<f32>,
    #[serde(skip)]
    pub mwo: Vec<f32>,
    #[serde(skip)]
    pub vwo: Vec<f32>,
    #[serde(skip)]
    pub mln1: Vec<f32>,
    #[serde(skip)]
    pub vln1: Vec<f32>,
    #[serde(skip)]
    pub mw1: Vec<f32>,
    #[serde(skip)]
    pub vw1: Vec<f32>,
    #[serde(skip)]
    pub mw2: Vec<f32>,
    #[serde(skip)]
    pub v2: Vec<f32>,
    #[serde(skip)]
    pub mln2: Vec<f32>,
    #[serde(skip)]
    pub vln2: Vec<f32>,
}

impl TransformerLayer {
    pub fn new(d_model: usize, d_ff: usize, rng: &mut u64) -> Self {
        let std_attn = (2.0 / (d_model as f32)).sqrt();
        let std_ff = (2.0 / (d_ff as f32)).sqrt();

        Self {
            wq: Self::rand_matrix(d_model * d_model, std_attn, rng),
            wk: Self::rand_matrix(d_model * d_model, std_attn, rng),
            wv: Self::rand_matrix(d_model * d_model, std_attn, rng),
            wo: Self::rand_matrix(d_model * d_model, std_attn, rng),
            ln1_g: vec![1.0; d_model],

            w1: Self::rand_matrix(d_ff * d_model, std_ff, rng),
            w2: Self::rand_matrix(d_model * d_ff, std_attn, rng),
            ln2_g: vec![1.0; d_model],

            mwq: vec![0.0; d_model * d_model],
            vwq: vec![0.0; d_model * d_model],
            mwk: vec![0.0; d_model * d_model],
            vwk: vec![0.0; d_model * d_model],
            mwv: vec![0.0; d_model * d_model],
            vwv: vec![0.0; d_model * d_model],
            mwo: vec![0.0; d_model * d_model],
            vwo: vec![0.0; d_model * d_model],
            mln1: vec![0.0; d_model],
            vln1: vec![0.0; d_model],

            mw1: vec![0.0; d_ff * d_model],
            vw1: vec![0.0; d_ff * d_model],
            mw2: vec![0.0; d_model * d_ff],
            v2: vec![0.0; d_model * d_ff],
            mln2: vec![0.0; d_model],
            vln2: vec![0.0; d_model],
        }
    }

    fn rand_matrix(len: usize, std: f32, rng: &mut u64) -> Vec<f32> {
        (0..len)
            .map(|_| {
                // Box-Muller gaussian generator
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                let u1 = ((*rng as f32) / (u64::MAX as f32)).max(1e-7);
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                let u2 = (*rng as f32) / (u64::MAX as f32);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std
            })
            .collect()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NanoUltron {
    pub config: ModelConfig,
    pub wte: Vec<f32>,      // (vocab_size x d_model) Token Embeddings
    pub wpe: Vec<f32>,      // (max_seq_len x d_model) Positional Embeddings
    pub layers: Vec<TransformerLayer>,
    pub ln_f_g: Vec<f32>,   // (d_model) Final RMSNorm
    pub lm_head: Vec<f32>,  // (vocab_size x d_model) Language Model Head

    #[serde(skip)]
    pub mwte: Vec<f32>,
    #[serde(skip)]
    pub vwte: Vec<f32>,
    #[serde(skip)]
    pub mwpe: Vec<f32>,
    #[serde(skip)]
    pub vwpe: Vec<f32>,
    #[serde(skip)]
    pub mln_f: Vec<f32>,
    #[serde(skip)]
    pub vln_f: Vec<f32>,
    #[serde(skip)]
    pub mlm_head: Vec<f32>,
    #[serde(skip)]
    pub vlm_head: Vec<f32>,

    pub optimizer_step: usize,
}

pub struct LayerCache {
    pub x_in: Vec<f32>,
    pub ln1_out: Vec<f32>,
    pub ln1_rms: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_probs: Vec<f32>, // (n_heads x T x T)
    pub attn_out: Vec<f32>,
    pub x_post_attn: Vec<f32>,
    pub ln2_out: Vec<f32>,
    pub ln2_rms: Vec<f32>,
    pub mlp_up: Vec<f32>,
    pub mlp_act: Vec<f32>,
}

pub struct ForwardCache {
    pub tokens: Vec<usize>,
    pub x0: Vec<f32>,
    pub layer_caches: Vec<LayerCache>,
    pub x_final: Vec<f32>,
    pub final_rms: Vec<f32>,
    pub logits: Vec<f32>,
}

impl NanoUltron {
    pub fn new(config: ModelConfig) -> Self {
        let mut rng = 4242424242u64;
        let std_emb = (1.0 / (config.d_model as f32)).sqrt();

        let wte = TransformerLayer::rand_matrix(config.vocab_size * config.d_model, std_emb, &mut rng);
        let wpe = TransformerLayer::rand_matrix(config.max_seq_len * config.d_model, std_emb, &mut rng);
        let layers = (0..config.n_layers)
            .map(|_| TransformerLayer::new(config.d_model, config.d_ff, &mut rng))
            .collect();
        let ln_f_g = vec![1.0; config.d_model];
        let lm_head = TransformerLayer::rand_matrix(config.vocab_size * config.d_model, std_emb, &mut rng);

        Self {
            wte,
            wpe,
            layers,
            ln_f_g,
            lm_head,
            mwte: vec![0.0; config.vocab_size * config.d_model],
            vwte: vec![0.0; config.vocab_size * config.d_model],
            mwpe: vec![0.0; config.max_seq_len * config.d_model],
            vwpe: vec![0.0; config.max_seq_len * config.d_model],
            mln_f: vec![0.0; config.d_model],
            vln_f: vec![0.0; config.d_model],
            mlm_head: vec![0.0; config.vocab_size * config.d_model],
            vlm_head: vec![0.0; config.vocab_size * config.d_model],
            config,
            optimizer_step: 0,
        }
    }

    /// Forward pass through the entire Transformer Decoder
    pub fn forward(&self, tokens: &[usize]) -> ForwardCache {
        let t = tokens.len();
        let d = self.config.d_model;
        let h = self.config.n_heads;
        let dk = d / h;
        let d_ff = self.config.d_ff;
        let scale = 1.0 / (dk as f32).sqrt();

        // 1. Embedding lookup + Positional Encoding: x0 = WTE[tokens] + WPE[0..t]
        let mut x = vec![0.0f32; t * d];
        for i in 0..t {
            let tok = tokens[i];
            let tok_emb = &self.wte[tok * d..(tok + 1) * d];
            let pos_emb = &self.wpe[i * d..(i + 1) * d];
            let x_row = &mut x[i * d..(i + 1) * d];
            for j in 0..d {
                x_row[j] = tok_emb[j] + pos_emb[j];
            }
        }
        let x0 = x.clone();

        let mut layer_caches = Vec::with_capacity(self.config.n_layers);

        // 2. Transformer Decoder Layers
        for layer in &self.layers {
            let x_in = x.clone();

            // Pre-LN 1: RMSNorm(x)
            let (ln1_out, ln1_rms) = TensorOps::rmsnorm_forward(&x, &layer.ln1_g, t, d, 1e-5);

            // Q = ln1 * WQ^T, K = ln1 * WK^T, V = ln1 * WV^T
            let q = TensorOps::matmul_trans_b(&ln1_out, &layer.wq, t, d, d);
            let k = TensorOps::matmul_trans_b(&ln1_out, &layer.wk, t, d, d);
            let v = TensorOps::matmul_trans_b(&ln1_out, &layer.wv, t, d, d);

            // Multi-Head Attention with Causal Mask
            let mut attn_probs = vec![0.0f32; h * t * t];
            let mut head_outputs = vec![0.0f32; t * d];

            for head in 0..h {
                let probs_offset = head * t * t;
                for i in 0..t {
                    let q_offset = i * d + head * dk;
                    let q_vec = &q[q_offset..q_offset + dk];

                    let scores_offset = probs_offset + i * t;
                    let scores = &mut attn_probs[scores_offset..scores_offset + t];

                    for j in 0..=i {
                        let k_offset = j * d + head * dk;
                        let k_vec = &k[k_offset..k_offset + dk];
                        let mut dot = 0.0f32;
                        for p in 0..dk {
                            dot += q_vec[p] * k_vec[p];
                        }
                        scores[j] = dot * scale;
                    }
                    // Causal mask: future tokens get -inf
                    for j in (i + 1)..t {
                        scores[j] = f32::NEG_INFINITY;
                    }

                    // Softmax across visible tokens 0..=i
                    TensorOps::softmax_inplace(&mut scores[0..=i], 1.0);

                    // Head output: O_h[i] = sum_j P[i, j] * V_h[j]
                    let out_offset = i * d + head * dk;
                    let out_slice = &mut head_outputs[out_offset..out_offset + dk];
                    for j in 0..=i {
                        let p_val = scores[j];
                        let v_offset = j * d + head * dk;
                        let v_vec = &v[v_offset..v_offset + dk];
                        for p in 0..dk {
                            out_slice[p] += p_val * v_vec[p];
                        }
                    }
                }
            }

            // Output projection: attn_out = head_outputs * WO^T
            let attn_out = TensorOps::matmul_trans_b(&head_outputs, &layer.wo, t, d, d);

            // Residual 1: x = x_in + attn_out
            for (xi, ai) in x.iter_mut().zip(attn_out.iter()) {
                *xi += ai;
            }
            let x_post_attn = x.clone();

            // Pre-LN 2: RMSNorm(x_post_attn)
            let (ln2_out, ln2_rms) = TensorOps::rmsnorm_forward(&x, &layer.ln2_g, t, d, 1e-5);

            // MLP Up: W1 * ln2_out^T -> (t x d_ff)
            let mlp_up = TensorOps::matmul_trans_b(&ln2_out, &layer.w1, t, d, d_ff);

            // GELU Activation
            let mut mlp_act = mlp_up.clone();
            for val in mlp_act.iter_mut() {
                *val = TensorOps::gelu(*val);
            }

            // MLP Down: W2 * mlp_act^T -> (t x d)
            let mlp_out = TensorOps::matmul_trans_b(&mlp_act, &layer.w2, t, d_ff, d);

            // Residual 2: x = x_post_attn + mlp_out
            for (xi, mi) in x.iter_mut().zip(mlp_out.iter()) {
                *xi += mi;
            }

            layer_caches.push(LayerCache {
                x_in,
                ln1_out,
                ln1_rms,
                q,
                k,
                v,
                attn_probs,
                attn_out,
                x_post_attn,
                ln2_out,
                ln2_rms,
                mlp_up,
                mlp_act,
            });
        }

        // 3. Final RMSNorm
        let (x_final, final_rms) = TensorOps::rmsnorm_forward(&x, &self.ln_f_g, t, d, 1e-5);

        // 4. LM Head: logits = x_final * lm_head^T -> (t x vocab_size)
        let logits = TensorOps::matmul_trans_b(&x_final, &self.lm_head, t, d, self.config.vocab_size);

        ForwardCache {
            tokens: tokens.to_vec(),
            x0,
            layer_caches,
            x_final,
            final_rms,
            logits,
        }
    }

    /// Full analytical backpropagation training step with AdamW optimizer
    pub fn train_step(&mut self, tokens: &[usize], targets: &[usize], lr: f32, weight_decay: f32) -> f32 {
        let t = tokens.len();
        let d = self.config.d_model;
        let h = self.config.n_heads;
        let dk = d / h;
        let d_ff = self.config.d_ff;
        let v_size = self.config.vocab_size;
        let scale = 1.0 / (dk as f32).sqrt();

        // 1. Forward Pass
        let cache = self.forward(tokens);

        // 2. Cross-Entropy Loss & dLogits
        let (loss, dlogits) = TensorOps::cross_entropy(&cache.logits, targets, t, v_size);

        // 3. Backward through LM Head: dlm_head = dlogits^T * x_final
        let dlm_head = TensorOps::matmul_trans_a(&dlogits, &cache.x_final, t, v_size, d);
        // dx_final = dlogits * lm_head
        let dx_final = TensorOps::matmul(&dlogits, &self.lm_head, t, v_size, d);

        // 4. Backward through Final RMSNorm
        let (mut dx, dln_f_g) = TensorOps::rmsnorm_backward(
            &dx_final,
            &cache.layer_caches.last().unwrap().x_post_attn,
            &self.ln_f_g,
            &cache.final_rms,
            t,
            d,
        );

        // Accumulators for layer gradients
        let mut layer_grads = Vec::with_capacity(self.config.n_layers);

        // 5. Backward through Transformer Layers in Reverse
        for (l_idx, layer) in self.layers.iter().enumerate().rev() {
            let lc = &cache.layer_caches[l_idx];

            // --- Backward through MLP ---
            // dmlp_out = dx (from residual)
            let dmlp_out = dx.clone();

            // dw2 = dmlp_out^T * mlp_act: (d x d_ff)
            let dw2 = TensorOps::matmul_trans_a(&dmlp_out, &lc.mlp_act, t, d, d_ff);

            // dmlp_act = dmlp_out * w2: (t x d_ff)
            let dmlp_act = TensorOps::matmul(&dmlp_out, &layer.w2, t, d, d_ff);

            // Backward through GELU: dmlp_up = dmlp_act * dgelu(mlp_up)
            let mut dmlp_up = vec![0.0f32; t * d_ff];
            for (du, (&da, &u)) in dmlp_up.iter_mut().zip(dmlp_act.iter().zip(lc.mlp_up.iter())) {
                *du = da * TensorOps::dgelu(u);
            }

            // dw1 = dmlp_up^T * ln2_out: (d_ff x d)
            let dw1 = TensorOps::matmul_trans_a(&dmlp_up, &lc.ln2_out, t, d_ff, d);

            // dln2_out = dmlp_up * w1: (t x d)
            let dln2_out = TensorOps::matmul(&dmlp_up, &layer.w1, t, d_ff, d);

            // Backward through LN2
            let (dx_ln2, dln2_g) = TensorOps::rmsnorm_backward(
                &dln2_out,
                &lc.x_post_attn,
                &layer.ln2_g,
                &lc.ln2_rms,
                t,
                d,
            );

            // Accumulate residual before attention: dx = dx + dx_ln2
            for (xi, dxi) in dx.iter_mut().zip(dx_ln2.iter()) {
                *xi += dxi;
            }

            // --- Backward through Attention ---
            // dattn_out = dx (residual branch)
            let dattn_out = dx.clone();

            // dwo = dattn_out^T * head_outputs: (d x d)
            // Recompute head outputs from cache
            let mut head_outputs = vec![0.0f32; t * d];
            for head in 0..h {
                let probs_offset = head * t * t;
                for i in 0..t {
                    let scores_offset = probs_offset + i * t;
                    let scores = &lc.attn_probs[scores_offset..scores_offset + t];
                    let out_offset = i * d + head * dk;
                    let out_slice = &mut head_outputs[out_offset..out_offset + dk];
                    for j in 0..=i {
                        let p_val = scores[j];
                        let v_offset = j * d + head * dk;
                        let v_vec = &lc.v[v_offset..v_offset + dk];
                        for p in 0..dk {
                            out_slice[p] += p_val * v_vec[p];
                        }
                    }
                }
            }

            let dwo = TensorOps::matmul_trans_a(&dattn_out, &head_outputs, t, d, d);

            // dhead_outputs = dattn_out * wo: (t x d)
            let dhead_outputs = TensorOps::matmul(&dattn_out, &layer.wo, t, d, d);

            let mut dq = vec![0.0f32; t * d];
            let mut dk_mat = vec![0.0f32; t * d];
            let mut dv = vec![0.0f32; t * d];

            for head in 0..h {
                let probs_offset = head * t * t;
                for i in 0..t {
                    let dhead_offset = i * d + head * dk;
                    let dhead_slice = &dhead_outputs[dhead_offset..dhead_offset + dk];

                    let scores_offset = probs_offset + i * t;
                    let scores = &lc.attn_probs[scores_offset..scores_offset + t];

                    // dv[j] += scores[j] * dhead_slice
                    for j in 0..=i {
                        let p_val = scores[j];
                        let v_offset = j * d + head * dk;
                        let dv_slice = &mut dv[v_offset..v_offset + dk];
                        for p in 0..dk {
                            dv_slice[p] += p_val * dhead_slice[p];
                        }
                    }

                    // dP[j] = dhead_slice . V[j]
                    let mut dp = vec![0.0f32; i + 1];
                    for j in 0..=i {
                        let v_offset = j * d + head * dk;
                        let v_vec = &lc.v[v_offset..v_offset + dk];
                        let mut dot = 0.0f32;
                        for p in 0..dk {
                            dot += dhead_slice[p] * v_vec[p];
                        }
                        dp[j] = dot;
                    }

                    // Softmax backward: dS[j] = P[j] * (dP[j] - sum_k dP[k]*P[k])
                    let mut sum_dp_p = 0.0f32;
                    for j in 0..=i {
                        sum_dp_p += dp[j] * scores[j];
                    }

                    let q_offset = i * d + head * dk;
                    let dq_slice = &mut dq[q_offset..q_offset + dk];

                    for j in 0..=i {
                        let ds = scores[j] * (dp[j] - sum_dp_p) * scale;

                        // dq[i] += ds * K[j]
                        let k_offset = j * d + head * dk;
                        let k_vec = &lc.k[k_offset..k_offset + dk];
                        for p in 0..dk {
                            dq_slice[p] += ds * k_vec[p];
                        }

                        // dk[j] += ds * Q[i]
                        let q_vec = &lc.q[q_offset..q_offset + dk];
                        let dk_slice = &mut dk_mat[k_offset..k_offset + dk];
                        for p in 0..dk {
                            dk_slice[p] += ds * q_vec[p];
                        }
                    }
                }
            }

            // dwq = dq^T * ln1_out, dwk = dk^T * ln1_out, dwv = dv^T * ln1_out
            let dwq = TensorOps::matmul_trans_a(&dq, &lc.ln1_out, t, d, d);
            let dwk = TensorOps::matmul_trans_a(&dk_mat, &lc.ln1_out, t, d, d);
            let dwv = TensorOps::matmul_trans_a(&dv, &lc.ln1_out, t, d, d);

            // dln1_out = dq*wq + dk*wk + dv*wv
            let dln1_q = TensorOps::matmul(&dq, &layer.wq, t, d, d);
            let dln1_k = TensorOps::matmul(&dk_mat, &layer.wk, t, d, d);
            let dln1_v = TensorOps::matmul(&dv, &layer.wv, t, d, d);

            let mut dln1_out = dln1_q;
            for (d1, (d2, d3)) in dln1_out.iter_mut().zip(dln1_k.iter().zip(dln1_v.iter())) {
                *d1 += d2 + d3;
            }

            // Backward through LN1
            let (dx_ln1, dln1_g) = TensorOps::rmsnorm_backward(
                &dln1_out,
                &lc.x_in,
                &layer.ln1_g,
                &lc.ln1_rms,
                t,
                d,
            );

            // Accumulate residual before layer
            for (xi, dxi) in dx.iter_mut().zip(dx_ln1.iter()) {
                *xi += dxi;
            }

            layer_grads.push((dwq, dwk, dwv, dwo, dln1_g, dw1, dw2, dln2_g));
        }

        layer_grads.reverse();

        // 6. Backward into Token Embeddings & Positional Embeddings
        let mut dwte = vec![0.0f32; v_size * d];
        let mut dwpe = vec![0.0f32; self.config.max_seq_len * d];

        for i in 0..t {
            let tok = tokens[i];
            let dx_row = &dx[i * d..(i + 1) * d];

            let dwte_row = &mut dwte[tok * d..(tok + 1) * d];
            for j in 0..d {
                dwte_row[j] += dx_row[j];
            }

            let dwpe_row = &mut dwpe[i * d..(i + 1) * d];
            for j in 0..d {
                dwpe_row[j] += dx_row[j];
            }
        }

        // 7. AdamW Optimizer Step
        self.optimizer_step += 1;
        let step = self.optimizer_step;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        // Update Embeddings & LM Head
        TensorOps::adamw_step(&mut self.wte, &dwte, &mut self.mwte, &mut self.vwte, lr, beta1, beta2, eps, weight_decay, step);
        TensorOps::adamw_step(&mut self.wpe, &dwpe, &mut self.mwpe, &mut self.vwpe, lr, beta1, beta2, eps, weight_decay, step);
        TensorOps::adamw_step(&mut self.ln_f_g, &dln_f_g, &mut self.mln_f, &mut self.vln_f, lr, beta1, beta2, eps, 0.0, step);
        TensorOps::adamw_step(&mut self.lm_head, &dlm_head, &mut self.mlm_head, &mut self.vlm_head, lr, beta1, beta2, eps, weight_decay, step);

        // Update Transformer Layers
        for (layer, (dwq, dwk, dwv, dwo, dln1_g, dw1, dw2, dln2_g)) in self.layers.iter_mut().zip(layer_grads) {
            TensorOps::adamw_step(&mut layer.wq, &dwq, &mut layer.mwq, &mut layer.vwq, lr, beta1, beta2, eps, weight_decay, step);
            TensorOps::adamw_step(&mut layer.wk, &dwk, &mut layer.mwk, &mut layer.vwk, lr, beta1, beta2, eps, weight_decay, step);
            TensorOps::adamw_step(&mut layer.wv, &dwv, &mut layer.mwv, &mut layer.vwv, lr, beta1, beta2, eps, weight_decay, step);
            TensorOps::adamw_step(&mut layer.wo, &dwo, &mut layer.mwo, &mut layer.vwo, lr, beta1, beta2, eps, weight_decay, step);
            TensorOps::adamw_step(&mut layer.ln1_g, &dln1_g, &mut layer.mln1, &mut layer.vln1, lr, beta1, beta2, eps, 0.0, step);

            TensorOps::adamw_step(&mut layer.w1, &dw1, &mut layer.mw1, &mut layer.vw1, lr, beta1, beta2, eps, weight_decay, step);
            TensorOps::adamw_step(&mut layer.w2, &dw2, &mut layer.mw2, &mut layer.v2, lr, beta1, beta2, eps, weight_decay, step);
            TensorOps::adamw_step(&mut layer.ln2_g, &dln2_g, &mut layer.mln2, &mut layer.vln2, lr, beta1, beta2, eps, 0.0, step);
        }

        loss
    }

    /// Count total learnable parameters
    pub fn param_count(&self) -> usize {
        let mut total = self.wte.len() + self.wpe.len() + self.ln_f_g.len() + self.lm_head.len();
        for l in &self.layers {
            total += l.wq.len() + l.wk.len() + l.wv.len() + l.wo.len() + l.ln1_g.len();
            total += l.w1.len() + l.w2.len() + l.ln2_g.len();
        }
        total
    }
}
