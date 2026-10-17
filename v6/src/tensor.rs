//! High-Performance Vectorized & Multi-Threaded Tensor Engine for OpenUltron v6.
//!
//! Grounded in pure linear algebra, SIMD cache-friendly layouts, and Rayon parallelization.
//! Provides forward operations, analytical backward passes, and AdamW optimization.

use rayon::prelude::*;
use std::f32::consts::PI;

pub struct TensorOps;

impl TensorOps {
    /// Cache-friendly parallel Matrix Multiplication: C = A * B
    /// A is (M x K), B is (K x N), C is (M x N)
    pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            let a_row = &a[i * k..(i + 1) * k];
            for (p, &a_val) in a_row.iter().enumerate() {
                if a_val != 0.0 {
                    let b_row = &b[p * n..(p + 1) * n];
                    for (j, b_val) in b_row.iter().enumerate() {
                        c_row[j] += a_val * b_val;
                    }
                }
            }
        });
        c
    }

    /// Fast Matrix Multiplication with Transposed B: C = A * B^T
    /// A is (M x K), B is (N x K), C is (M x N)
    /// Both matrices read sequentially along rows for maximum CPU cache utilization.
    pub fn matmul_trans_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            let a_row = &a[i * k..(i + 1) * k];
            for (j, c_val) in c_row.iter_mut().enumerate() {
                let b_row = &b[j * k..(j + 1) * k];
                let mut dot = 0.0f32;
                for p in 0..k {
                    dot += a_row[p] * b_row[p];
                }
                *c_val = dot;
            }
        });
        c
    }

    /// Fast Matrix Multiplication with Transposed A: C = A^T * B
    /// A is (M x K), B is (M x N), C is (K x N)
    /// Used heavily in backpropagation (computing weight gradients dW = X^T * dY).
    pub fn matmul_trans_a(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; k * n];
        // Outer-product accumulation over m
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let b_row = &b[i * n..(i + 1) * n];
            for p in 0..k {
                let a_val = a_row[p];
                let c_row = &mut c[p * n..(p + 1) * n];
                for j in 0..n {
                    c_row[j] += a_val * b_row[j];
                }
            }
        }
        c
    }

    /// Root Mean Square Layer Normalization (RMSNorm) forward pass
    /// x: (tokens x dim), gain: (dim)
    /// Returns: (normalized_output, rms_factors)
    pub fn rmsnorm_forward(x: &[f32], gain: &[f32], tokens: usize, dim: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
        let mut out = vec![0.0f32; tokens * dim];
        let mut rms = vec![0.0f32; tokens];

        out.par_chunks_mut(dim)
            .zip(rms.par_iter_mut())
            .enumerate()
            .for_each(|(i, (out_row, rms_val))| {
                let x_row = &x[i * dim..(i + 1) * dim];
                let mut sum_sq = 0.0f32;
                for &val in x_row {
                    sum_sq += val * val;
                }
                let r = 1.0 / (sum_sq / (dim as f32) + eps).sqrt();
                *rms_val = r;

                for j in 0..dim {
                    out_row[j] = x_row[j] * r * gain[j];
                }
            });

        (out, rms)
    }

    /// RMSNorm backward pass
    pub fn rmsnorm_backward(
        dout: &[f32],
        x: &[f32],
        gain: &[f32],
        rms: &[f32],
        tokens: usize,
        dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut dx = vec![0.0f32; tokens * dim];
        let mut dgain = vec![0.0f32; dim];

        for i in 0..tokens {
            let x_row = &x[i * dim..(i + 1) * dim];
            let dout_row = &dout[i * dim..(i + 1) * dim];
            let r = rms[i];

            let mut sum_dout_gain_x = 0.0f32;
            for j in 0..dim {
                dgain[j] += dout_row[j] * x_row[j] * r;
                sum_dout_gain_x += dout_row[j] * gain[j] * x_row[j];
            }

            let coeff = sum_dout_gain_x * r * r * r / (dim as f32);
            let dx_row = &mut dx[i * dim..(i + 1) * dim];
            for j in 0..dim {
                dx_row[j] = dout_row[j] * gain[j] * r - x_row[j] * coeff;
            }
        }

        (dx, dgain)
    }

    /// GELU activation forward pass: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #[inline(always)]
    pub fn gelu(x: f32) -> f32 {
        let sqrt_2_over_pi = (2.0 / PI).sqrt();
        let cube = 0.044715 * x * x * x;
        let inner = sqrt_2_over_pi * (x + cube);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// GELU activation derivative
    #[inline(always)]
    pub fn dgelu(x: f32) -> f32 {
        let sqrt_2_over_pi = (2.0 / PI).sqrt();
        let x2 = x * x;
        let x3 = x2 * x;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
        let tanh_inner = inner.tanh();
        let sech2 = 1.0 - tanh_inner * tanh_inner;
        0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x2)
    }

    /// In-place numerically stable Softmax over slice with temperature scaling
    pub fn softmax_inplace(x: &mut [f32], temperature: f32) {
        if x.is_empty() {
            return;
        }
        let inv_temp = 1.0 / temperature.max(1e-4);
        let mut max_val = f32::NEG_INFINITY;
        for &v in x.iter() {
            let scaled = v * inv_temp;
            if scaled > max_val {
                max_val = scaled;
            }
        }

        let mut sum_exp = 0.0f32;
        for v in x.iter_mut() {
            *v = ((*v * inv_temp) - max_val).exp();
            sum_exp += *v;
        }

        let inv_sum = 1.0 / sum_exp.max(1e-12);
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }

    /// Cross-Entropy Loss with Softmax logits and analytical gradients
    /// logits: (tokens x vocab_size), targets: (tokens)
    /// Returns: (average_loss, dlogits)
    pub fn cross_entropy(logits: &[f32], targets: &[usize], tokens: usize, vocab_size: usize) -> (f32, Vec<f32>) {
        let mut dlogits = logits.to_vec();
        let mut total_loss = 0.0f32;

        for (i, &target) in targets.iter().enumerate() {
            let row = &mut dlogits[i * vocab_size..(i + 1) * vocab_size];
            Self::softmax_inplace(row, 1.0);

            let prob = row[target].max(1e-12);
            total_loss -= prob.ln();

            // Gradient of cross-entropy with softmax: dL/dz = prob - 1 (for target class)
            row[target] -= 1.0;
        }

        // Scale gradients by 1 / tokens
        let scale = 1.0 / (tokens as f32);
        for g in dlogits.iter_mut() {
            *g *= scale;
        }

        (total_loss * scale, dlogits)
    }

    /// AdamW Parameter Optimizer Step
    pub fn adamw_step(
        param: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: usize,
    ) {
        let beta1_pow = beta1.powi(step as i32);
        let beta2_pow = beta2.powi(step as i32);
        let alpha = lr * (1.0 - beta2_pow).sqrt() / (1.0 - beta1_pow);

        param
            .par_iter_mut()
            .zip(grad.par_iter())
            .zip(m.par_iter_mut())
            .zip(v.par_iter_mut())
            .for_each(|(((p, &g), m_val), v_val)| {
                // Apply decoupled weight decay
                *p -= lr * weight_decay * *p;

                // Update biased first and second moment estimates
                *m_val = beta1 * *m_val + (1.0 - beta1) * g;
                *v_val = beta2 * *v_val + (1.0 - beta2) * g * g;

                // Compute update
                *p -= alpha * *m_val / (v_val.sqrt() + eps);
            });
    }
}
