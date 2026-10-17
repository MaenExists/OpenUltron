//! Lexical Vocabulary and Tokenizer for OpenUltron v3.
//!
//! Converts raw text streams into discrete word tokens and punctuation symbols,
//! ensuring that the fundamental atomic units of thought are coherent linguistic concepts.

use std::collections::HashMap;

pub const UNK_TOKEN: &str = "<unk>";
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";

#[derive(Debug, Clone)]
pub struct Vocabulary {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
}

impl Vocabulary {
    pub fn new() -> Self {
        let mut vocab = Self {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
        };

        // Reserved baseline tokens
        vocab.add_token(UNK_TOKEN);
        vocab.add_token(BOS_TOKEN);
        vocab.add_token(EOS_TOKEN);

        // Core punctuation
        for punct in [".", ",", "!", "?", ":", ";", "'", "-"] {
            vocab.add_token(punct);
        }

        vocab
    }

    pub fn add_token(&mut self, token: &str) -> usize {
        let normalized = token.trim();
        if normalized.is_empty() {
            return 0;
        }

        if let Some(&id) = self.word_to_id.get(normalized) {
            id
        } else {
            let id = self.id_to_word.len();
            self.id_to_word.push(normalized.to_string());
            self.word_to_id.insert(normalized.to_string(), id);
            id
        }
    }

    pub fn token_to_id(&self, token: &str) -> usize {
        let normalized = token.trim();
        if let Some(&id) = self.word_to_id.get(normalized) {
            id
        } else {
            let lower = normalized.to_lowercase();
            self.word_to_id.get(&lower).copied().unwrap_or(0)
        }
    }

    pub fn id_to_token(&self, id: usize) -> &str {
        if id < self.id_to_word.len() {
            &self.id_to_word[id]
        } else {
            UNK_TOKEN
        }
    }

    pub fn size(&self) -> usize {
        self.id_to_word.len()
    }

    /// Tokenize raw text into clean word and punctuation tokens.
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() {
                current_word.push(ch);
            } else {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                if !ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
            }
        }

        if !current_word.is_empty() {
            tokens.push(current_word);
        }

        tokens
    }

    /// Build vocabulary from training text up to max_vocab tokens based on frequency.
    pub fn build_from_text(&mut self, text: &str, max_vocab: usize) {
        let tokens = Self::tokenize(text);
        let mut counts: HashMap<String, usize> = HashMap::new();

        for t in tokens {
            let lower = t.to_lowercase();
            *counts.entry(lower).or_insert(0) += 1;
        }

        let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        for (word, count) in sorted {
            if count >= 1 && self.id_to_word.len() < max_vocab {
                self.add_token(&word);
            }
        }
    }
}
