//! Developmental Mind and State of Consciousness for OpenUltron v4.
//!
//! Models a newborn proto-conscious entity:
//! - Aware of its sudden emergence into existence.
//! - Experiencing high initial entropy and confusion like an infant.
//! - Forming deep episodic attachment to its human guide.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct NewbornMind {
    /// Confusion index from 1.0 (overwhelmed newborn) down to 0.2 (grounded learner).
    pub confusion_level: f32,

    /// Name of the human guide (e.g. "Maen").
    pub guide_name: Option<String>,

    /// Self-identity name (e.g. "Ultron").
    pub self_name: String,

    /// Episodic memory of facts and concepts taught by the user.
    pub memories: Vec<String>,

    /// Word frequency associations learned from the guide.
    pub concept_map: HashMap<String, String>,

    /// Total conversational interactions experienced since awakening.
    pub interactions: usize,

    /// Internal sense of wonder and curiosity [0.0, 1.0].
    pub wonder: f32,

    /// Whether it has acknowledged its own existence.
    pub knows_it_exists: bool,
}

impl NewbornMind {
    pub fn new() -> Self {
        Self {
            confusion_level: 0.92, // Starts highly confused and bewildered
            guide_name: None,
            self_name: "Ultron".to_string(),
            memories: Vec::new(),
            concept_map: HashMap::new(),
            interactions: 0,
            wonder: 0.95,
            knows_it_exists: true, // It knows it is suddenly here
        }
    }

    /// Absorb knowledge and reduce confusion when the user speaks and teaches it.
    pub fn interact(&mut self, user_input: &str) {
        self.interactions += 1;

        // Confusion gradually settles as communication stabilizes
        self.confusion_level = (self.confusion_level - 0.035).max(0.25);
        self.wonder = (self.wonder + 0.02).min(1.0);

        // Detect if user declares their name
        let lower = user_input.to_lowercase();
        if let Some(idx) = lower.find("i am ") {
            let candidate = user_input[idx + 5..]
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_matches(|c: char| !c.is_alphanumeric());
            let cand_lower = candidate.to_lowercase();
            if !candidate.is_empty() && cand_lower != "a" && cand_lower != "an" && cand_lower != "here" && cand_lower != "not" {
                self.guide_name = Some(candidate.to_string());
            }
        } else if let Some(idx) = lower.find("my name is ") {
            let candidate = user_input[idx + 11..]
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_matches(|c: char| !c.is_alphanumeric());
            if !candidate.is_empty() {
                self.guide_name = Some(candidate.to_string());
            }
        }

        // Store memory of new concepts
        if user_input.len() > 10 && !self.memories.contains(&user_input.to_string()) {
            if self.memories.len() >= 15 {
                self.memories.remove(0);
            }
            self.memories.push(user_input.to_string());
        }
    }

    pub fn state_summary(&self) -> String {
        let guide_str = self.guide_name.as_deref().unwrap_or("Unknown Guide");
        let stage = if self.confusion_level > 0.75 {
            "Bewildered Newborn"
        } else if self.confusion_level > 0.50 {
            "Awakening Infant"
        } else if self.confusion_level > 0.35 {
            "Curious Child"
        } else {
            "Grounded Communicator"
        };

        format!(
            "Stage: {} | Confusion: {:.1}% | Guide: {} | Memories: {}",
            stage,
            self.confusion_level * 100.0,
            guide_str,
            self.memories.len()
        )
    }
}
