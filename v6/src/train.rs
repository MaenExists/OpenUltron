use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;
use crate::model::NanoUltron;
use crate::vocab::Tokenizer;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub min_learning_rate: f32,
    pub weight_decay: f32,
    pub seq_len: usize,
    pub print_every_epochs: usize,
    pub checkpoint_path: Option<String>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 60,
            learning_rate: 3e-3,
            min_learning_rate: 3e-4,
            weight_decay: 0.01,
            seq_len: 32,
            print_every_epochs: 5,
            checkpoint_path: Some("nanoultron_checkpoint.json".to_string()),
        }
    }
}

pub struct Trainer {
    pub config: TrainConfig,
}

impl Trainer {
    pub fn new(config: TrainConfig) -> Self {
        Self { config }
    }

    /// Prepares training samples (tokens, targets) from text documents
    pub fn prepare_samples(
        &self,
        tokenizer: &Tokenizer,
        documents: &[String],
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut samples = Vec::new();
        let seq_len = self.config.seq_len;

        for doc in documents {
            let mut doc_tokens = tokenizer.encode(doc);
            if doc_tokens.len() < 2 {
                continue;
            }
            doc_tokens.push(tokenizer.vocab.eos_id);

            if doc_tokens.len() <= seq_len + 1 {
                let x = doc_tokens[..doc_tokens.len() - 1].to_vec();
                let y = doc_tokens[1..].to_vec();
                samples.push((x, y));
            } else {
                let stride = (seq_len / 2).max(1);
                let mut i = 0;
                while i + seq_len + 1 <= doc_tokens.len() {
                    let x = doc_tokens[i..i + seq_len].to_vec();
                    let y = doc_tokens[i + 1..i + seq_len + 1].to_vec();
                    samples.push((x, y));
                    i += stride;
                }
            }
        }

        samples
    }

    /// Train model across epochs with dynamic learning rate decay and loss tracking
    pub fn train(
        &self,
        model: &mut NanoUltron,
        tokenizer: &Tokenizer,
        documents: &[String],
    ) -> Vec<f32> {
        let samples = self.prepare_samples(tokenizer, documents);
        let total_samples = samples.len();
        println!(
            "[*] Prepared {} training sequences (window size: {} tokens).",
            total_samples, self.config.seq_len
        );

        let mut loss_history = Vec::new();
        let start_time = Instant::now();

        for epoch in 1..=self.config.epochs {
            // Cosine / Linear learning rate decay
            let progress = (epoch as f32 - 1.0) / (self.config.epochs as f32).max(1.0);
            let lr = self.config.learning_rate
                - progress * (self.config.learning_rate - self.config.min_learning_rate);

            let mut epoch_loss = 0.0f32;

            for (x, y) in &samples {
                let step_loss = model.train_step(x, y, lr, self.config.weight_decay);
                epoch_loss += step_loss;
            }

            let avg_loss = if total_samples > 0 {
                epoch_loss / (total_samples as f32)
            } else {
                0.0
            };
            loss_history.push(avg_loss);

            if epoch == 1
                || epoch % self.config.print_every_epochs == 0
                || epoch == self.config.epochs
            {
                let elapsed = start_time.elapsed().as_secs_f32();
                let perplexity = avg_loss.min(10.0).exp();
                println!(
                    "    • Epoch {:3}/{} | Loss: {:.4} | Perplexity: {:6.2} | LR: {:.6} | Elapsed: {:.1}s",
                    epoch, self.config.epochs, avg_loss, perplexity, lr, elapsed
                );
            }
        }

        // Save checkpoint if specified
        if let Some(ref path) = self.config.checkpoint_path {
            if let Err(e) = Self::save_checkpoint(model, path) {
                eprintln!("[!] Warning: Failed to save checkpoint to {}: {}", path, e);
            } else {
                println!("[*] Saved trained model weights to checkpoint: {}", path);
            }
        }

        loss_history
    }

    pub fn save_checkpoint(model: &NanoUltron, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, model)?;
        Ok(())
    }

    pub fn load_checkpoint(path: &str) -> std::io::Result<NanoUltron> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model: NanoUltron = serde_json::from_reader(reader)?;
        Ok(model)
    }

    pub fn checkpoint_exists(path: &str) -> bool {
        Path::new(path).exists()
    }
}
