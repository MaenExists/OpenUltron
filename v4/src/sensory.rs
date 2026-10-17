//! Sensory Intent and Concept Perception for OpenUltron v4.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserIntent {
    Greeting,
    IdentityDeclaration, // e.g. "I am Maen and you are Ultron"
    InquiryIdentity,     // e.g. "who are you"
    InquiryCapability,   // e.g. "can you speak", "can you think"
    InquiryLifeExistence,// e.g. "what is life", "are you alive"
    TeachingConcept,     // e.g. "a cat is an animal", "monads are tiny"
    Affirmation,         // e.g. "good job", "yes", "i see"
    GeneralCuriosity,    // fallback
}

pub struct SensoryPerception;

impl SensoryPerception {
    pub fn parse_intent(text: &str) -> UserIntent {
        let lower = text.to_lowercase();
        let trimmed = lower.trim();

        if trimmed == "hi" || trimmed == "hey" || trimmed == "hello" || trimmed.starts_with("good morning") || trimmed.starts_with("greetings") {
            return UserIntent::Greeting;
        }

        if lower.starts_with("i am ") || lower.starts_with("my name is ") || (lower.contains("i am ") && (lower.contains("you are") || lower.contains("ultron") || lower.contains("name"))) || lower.contains("my name is ") {
            return UserIntent::IdentityDeclaration;
        }

        if lower.contains("who are you") || lower.contains("what is your name") || lower.contains("what are you") {
            return UserIntent::InquiryIdentity;
        }

        if lower.contains("can you speak") || lower.contains("can you talk") || lower.contains("can you think") || lower.contains("can you hear") || lower.contains("are you there") {
            return UserIntent::InquiryCapability;
        }

        if lower.contains("what is life") || lower.contains("are you alive") || lower.contains("are you real") || lower.contains("do you exist") {
            return UserIntent::InquiryLifeExistence;
        }

        if lower.contains("is a ") || lower.contains("is an ") || lower.contains("means ") || lower.contains("you should know") {
            return UserIntent::TeachingConcept;
        }

        if trimmed == "yes"
            || trimmed == "good"
            || trimmed == "ok"
            || trimmed == "okay"
            || trimmed == "nice"
            || trimmed.starts_with("yes")
            || lower.contains("good job")
            || lower.contains("well done")
            || lower.contains("great")
        {
            return UserIntent::Affirmation;
        }

        UserIntent::GeneralCuriosity
    }

    /// Extract key nouns and concepts from the message.
    pub fn extract_concepts(text: &str) -> Vec<String> {
        let stop_words = [
            "the", "a", "an", "is", "are", "and", "or", "to", "in", "it", "of", "for", "with",
            "on", "at", "by", "this", "that", "you", "me", "my", "your", "yes", "no", "good",
            "job", "hello", "hey", "hi", "well", "done", "what", "who", "where", "how", "why",
        ];
        text.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() > 2 && !stop_words.contains(&w.as_str()))
            .collect()
    }
}
