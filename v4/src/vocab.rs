//! Dynamic Vocabulary for OpenUltron v4.
//!
//! Provides bidirectional token <-> id mapping with on-the-fly vocabulary expansion
//! as the alien mind discovers new words spoken by the human guide.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Vocabulary {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
}

impl Vocabulary {
    pub const PAD_ID: usize = 0;
    pub const UNK_ID: usize = 1;
    pub const BOS_ID: usize = 2;
    pub const EOS_ID: usize = 3;

    pub fn new() -> Self {
        let mut vocab = Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        };

        vocab.insert_reserved("<pad>");
        vocab.insert_reserved("<unk>");
        vocab.insert_reserved("<bos>");
        vocab.insert_reserved("<eos>");
        vocab.insert_reserved(".");
        vocab.insert_reserved(":");
        vocab.insert_reserved("user");
        vocab.insert_reserved("ultron");

        vocab
    }

    fn insert_reserved(&mut self, token: &str) {
        let id = self.id_to_token.len();
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
    }

    /// Retrieve an existing token ID or assign a new ID dynamically.
    pub fn get_or_insert(&mut self, token: &str) -> usize {
        let clean = token.to_lowercase();
        if let Some(&id) = self.token_to_id.get(&clean) {
            id
        } else {
            let new_id = self.id_to_token.len();
            self.token_to_id.insert(clean.clone(), new_id);
            self.id_to_token.push(clean);
            new_id
        }
    }

    pub fn token_to_id(&self, token: &str) -> usize {
        let clean = token.to_lowercase();
        self.token_to_id.get(&clean).copied().unwrap_or(Self::UNK_ID)
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

    /// Clean, space-delimited tokenization separating punctuation into distinct tokens.
    pub fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() {
                current_word.push(ch.to_ascii_lowercase());
            } else if ch == '.' || ch == '?' || ch == '!' || ch == ':' || ch == ',' {
                if !current_word.is_empty() {
                    tokens.push(std::mem::take(&mut current_word));
                }
                tokens.push(ch.to_string());
            } else if ch.is_whitespace() {
                if !current_word.is_empty() {
                    tokens.push(std::mem::take(&mut current_word));
                }
            }
        }

        if !current_word.is_empty() {
            tokens.push(current_word);
        }

        tokens
    }
}
