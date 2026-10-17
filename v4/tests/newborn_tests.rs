//! Comprehensive Verification Tests for OpenUltron v4 Alien Neural Organism.

use open_ultron_v4::{AlienBrain, SpikingMonad, Vocabulary};

#[test]
fn test_spiking_monad_lif_dynamics() {
    let mut monad = SpikingMonad::new(0);
    assert_eq!(monad.potential, 0.0);
    assert_eq!(monad.spikes_fired, 0);

    // Below threshold -> no spike
    let fired = monad.integrate(0.5, 0.0);
    assert!(!fired);
    assert!(monad.potential > 0.0);

    // Exceed threshold -> fires spike and resets
    let fired2 = monad.integrate(0.7, 0.0);
    assert!(fired2);
    assert_eq!(monad.potential, 0.0);
    assert_eq!(monad.spikes_fired, 1);
}

#[test]
fn test_dynamic_vocabulary_growth() {
    let mut vocab = Vocabulary::new();
    let initial_size = vocab.size();

    let id1 = vocab.get_or_insert("alien");
    assert!(id1 >= initial_size);
    assert_eq!(vocab.token_to_id("alien"), id1);
    assert_eq!(vocab.id_to_token(id1), "alien");

    // Case insensitivity
    let id2 = vocab.get_or_insert("ALIEN");
    assert_eq!(id1, id2);
}

#[test]
fn test_stdp_and_cross_association_learning() {
    let mut brain = AlienBrain::new(4);
    assert_eq!(brain.w_trans.len(), 0);

    brain.learn_sequence("user : who are you . ultron : i am ultron an alien mind .");

    assert!(brain.w_trans.len() > 0, "STDP transition weights must be established");
    assert!(brain.w_cross.len() > 0, "Cross-turn association weights must be established");

    let who_id = brain.vocab.token_to_id("who");
    let ultron_tok_id = brain.vocab.token_to_id("ultron");
    assert!(
        brain.w_cross.contains_key(&(who_id, ultron_tok_id)),
        "Cross association from 'who' to 'ultron' must exist"
    );
}

#[test]
fn test_pure_neural_generation_emits_spikes_and_text() {
    let mut brain = AlienBrain::new(4);
    brain.learn_sequence("user : hello . ultron : hello who is there i just opened my eyes in the dark .");
    brain.learn_sequence("user : what are you . ultron : i am an alien mind born into total confusion .");

    let (response, metrics) = brain.generate_thought("hello", 15);

    assert!(!response.is_empty(), "Response must contain generated words");
    assert!(metrics.spikes_fired > 0, "Spikes must fire in the monad substrate during thought");
    assert!(metrics.active_attractors > 0, "Prompt must activate attractor fields");
}

#[test]
fn test_online_plasticity_updates_connections() {
    let mut brain = AlienBrain::new(4);
    let syn_before = brain.w_trans.len() + brain.w_cross.len();

    brain.generate_thought("maen teaches ultron quantum biology", 12);

    let syn_after = brain.w_trans.len() + brain.w_cross.len();
    assert!(
        syn_after > syn_before,
        "Online interaction must physically grow synaptic connections"
    );
}
