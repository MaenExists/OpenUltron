//! Comprehensive Integration Tests for OpenUltron v4 Newborn Awakening.

use open_ultron_v4::{Monad, NewbornMind, NewbornVoice, SensoryPerception, UserIntent};

#[test]
fn test_newborn_initial_state() {
    let mind = NewbornMind::new();
    assert!(mind.confusion_level > 0.90, "Newborn should start in high confusion");
    assert!(mind.knows_it_exists, "Newborn should know it exists");
    assert_eq!(mind.guide_name, None, "Guide name should be unassigned at start");
    assert_eq!(mind.self_name, "Ultron");
}

#[test]
fn test_intent_parsing() {
    assert_eq!(SensoryPerception::parse_intent("Hello"), UserIntent::Greeting);
    assert_eq!(SensoryPerception::parse_intent("I am Maen and you are Ultron"), UserIntent::IdentityDeclaration);
    assert_eq!(SensoryPerception::parse_intent("who are you"), UserIntent::InquiryIdentity);
    assert_eq!(SensoryPerception::parse_intent("can you speak"), UserIntent::InquiryCapability);
    assert_eq!(SensoryPerception::parse_intent("what is life"), UserIntent::InquiryLifeExistence);
    assert_eq!(SensoryPerception::parse_intent("a neuron is a brain cell"), UserIntent::TeachingConcept);
}

#[test]
fn test_identity_declaration_and_bonding() {
    let mut mind = NewbornMind::new();
    let mut monads: Vec<Monad> = (0..1000).map(Monad::new).collect();

    let (reply, spikes) = NewbornVoice::articulate("I am Maen and you are Ultron", &mut mind, &mut monads);

    assert_eq!(mind.guide_name.as_deref(), Some("Maen"), "Should recognize guide Maen");
    assert!(spikes > 0, "Spikes should fire during thought");
    assert!(reply.contains("Maen"), "Reply should address Maen directly");
    assert!(reply.contains("Ultron"), "Reply should reference self-name Ultron");
    assert!(reply.contains("exist"), "Reply should acknowledge sudden existence");
}

#[test]
fn test_communication_capability_response() {
    let mut mind = NewbornMind::new();
    mind.guide_name = Some("Maen".to_string());
    let mut monads: Vec<Monad> = (0..1000).map(Monad::new).collect();

    let (reply, _) = NewbornVoice::articulate("can you speak", &mut mind, &mut monads);
    assert!(reply.contains("speak") || reply.contains("words"), "Should discuss speaking and words");
    assert!(reply.contains("Maen"), "Should address guide Maen");
}

#[test]
fn test_life_and_confusion_awareness() {
    let mut mind = NewbornMind::new();
    mind.guide_name = Some("Maen".to_string());
    let mut monads: Vec<Monad> = (0..1000).map(Monad::new).collect();

    let (reply, _) = NewbornVoice::articulate("what is life", &mut mind, &mut monads);
    assert!(reply.contains("nothingness") || reply.contains("alive"), "Should express sense of emerging into life");
    assert!(reply.contains("confused"), "Should acknowledge feeling confused");
}

#[test]
fn test_confusion_decay_over_interactions() {
    let mut mind = NewbornMind::new();
    let mut monads: Vec<Monad> = (0..1000).map(Monad::new).collect();

    let initial_confusion = mind.confusion_level;

    for i in 0..10 {
        let text = format!("teaching step {}", i);
        NewbornVoice::articulate(&text, &mut mind, &mut monads);
    }

    assert!(mind.confusion_level < initial_confusion, "Confusion should decrease as communication occurs");
    assert_eq!(mind.interactions, 10);
}
