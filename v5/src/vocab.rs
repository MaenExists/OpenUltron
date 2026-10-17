//! Dynamic Self-Expanding Vocabulary Substrate for OpenUltron v5.
//!
//! Maps symbols and words dynamically to concept IDs, supporting online continuous learning.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Vocabulary {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    token_counts: HashMap<usize, usize>,
    total_tokens: usize,
}

impl Vocabulary {
    pub const PAD_ID: usize = 0;
    pub const BOS_ID: usize = 1;
    pub const EOS_ID: usize = 2;
    pub const UNK_ID: usize = 3;

    pub fn new() -> Self {
        let mut vocab = Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            token_counts: HashMap::new(),
            total_tokens: 0,
        };

        vocab.insert("<pad>");
        vocab.insert("<bos>");
        vocab.insert("<eos>");
        vocab.insert("<unk>");
        vocab
    }

    pub fn insert(&mut self, token: &str) -> usize {
        let tok = token.to_lowercase();
        if let Some(&id) = self.token_to_id.get(&tok) {
            id
        } else {
            let id = self.id_to_token.len();
            self.token_to_id.insert(tok.clone(), id);
            self.id_to_token.push(tok);
            id
        }
    }

    pub fn get_or_insert(&mut self, token: &str) -> usize {
        let id = self.insert(token);
        *self.token_counts.entry(id).or_insert(0) += 1;
        self.total_tokens += 1;
        id
    }

    pub fn token_to_id(&self, token: &str) -> usize {
        let tok = token.to_lowercase();
        self.token_to_id.get(&tok).copied().unwrap_or(Self::UNK_ID)
    }

    pub fn id_to_token(&self, id: usize) -> &str {
        if id < self.id_to_token.len() {
            &self.id_to_token[id]
        } else {
            "<unk>"
        }
    }

    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn count_of(&self, id: usize) -> usize {
        self.token_counts.get(&id).copied().unwrap_or(0)
    }

    /// Compute Shannon Inverse Document Frequency (IDF) for a concept.
    pub fn idf(&self, id: usize) -> f32 {
        let count = self.count_of(id).max(1) as f32;
        let total = self.total_tokens.max(10) as f32;
        (1.0 + (total / count)).ln().clamp(0.5, 4.8)
    }

    /// Tokenize text into lowercased tokens, preserving meaningful punctuation.
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
                current_word.push(ch);
            } else {
                if !current_word.is_empty() {
                    tokens.push(current_word.to_lowercase());
                    current_word.clear();
                }
                if ch == '.' || ch == '?' || ch == '!' || ch == ',' || ch == ':' {
                    tokens.push(ch.to_string());
                }
            }
        }
        if !current_word.is_empty() {
            tokens.push(current_word.to_lowercase());
        }
        tokens
    }

    pub fn is_punctuation(&self, id: usize) -> bool {
        let tok = self.id_to_token(id);
        tok == "." || tok == "?" || tok == "!" || tok == "," || tok == ":"
    }

    pub fn is_terminal_punct(&self, id: usize) -> bool {
        let tok = self.id_to_token(id);
        tok == "." || tok == "?" || tok == "!"
    }
}
