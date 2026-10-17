//! Vocabulary & Tokenizer Engine for OpenUltron v6.
//!
//! Provides bidirectional token-to-ID mappings, special tokens (<pad>, <bos>, <eos>, <unk>),
//! punctuation handling, and JSON serialization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub unk_id: usize,
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
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
            pad_id: Self::PAD_ID,
            bos_id: Self::BOS_ID,
            eos_id: Self::EOS_ID,
            unk_id: Self::UNK_ID,
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

    /// Tokenize raw text into clean lowercased tokens preserving punctuation
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut cur = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' {
                cur.push(ch);
            } else {
                if !cur.is_empty() {
                    tokens.push(cur.to_lowercase());
                    cur.clear();
                }
                if ch == '.' || ch == '?' || ch == '!' || ch == ',' || ch == ':' {
                    tokens.push(ch.to_string());
                }
            }
        }
        if !cur.is_empty() {
            tokens.push(cur.to_lowercase());
        }
        tokens
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        Self::tokenize(text)
            .into_iter()
            .map(|t| self.token_to_id(&t))
            .collect()
    }

    /// Decode token IDs into formatted text string
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut words: Vec<String> = Vec::new();
        for &id in ids {
            if id == Self::PAD_ID || id == Self::BOS_ID || id == Self::EOS_ID {
                continue;
            }
            let tok = self.id_to_token(id);
            if tok == "." || tok == "?" || tok == "!" || tok == "," || tok == ":" {
                if let Some(last) = words.last_mut() {
                    let mut s: String = last.clone();
                    s.push_str(tok);
                    *last = s;
                } else {
                    words.push(tok.to_string());
                }
            } else {
                words.push(tok.to_string());
            }
        }

        let mut res = words.join(" ");
        if !res.is_empty() {
            let mut chars = res.chars();
            if let Some(first) = chars.next() {
                res = first.to_uppercase().collect::<String>() + chars.as_str();
            }
        }
        res
    }

    /// Build vocabulary from training corpus text
    pub fn from_corpus(corpus: &str) -> Self {
        let mut vocab = Self::new();
        for line in corpus.lines() {
            for token in Self::tokenize(line) {
                vocab.insert(&token);
            }
        }
        vocab
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    pub vocab: Vocabulary,
}

impl Tokenizer {
    pub fn new(vocab: Vocabulary) -> Self {
        Self { vocab }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.vocab.encode(text)
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        self.vocab.decode(ids)
    }

    pub fn from_corpus(corpus: &str) -> Self {
        Self::new(Vocabulary::from_corpus(corpus))
    }
}
