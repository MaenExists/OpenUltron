use open_ultron_v2::brain::OpenUltronBrain;
use open_ultron_v2::config::BrainConfig;
use open_ultron_v2::sensory::{char_to_index, index_to_char};

#[test]
fn test_sensory_vocabulary_roundtrip() {
    for c in ['a', 'Z', '0', '!', ' ', '\n'] {
        let idx = char_to_index(c).expect("Must map valid char");
        let decoded = index_to_char(idx);
        assert_eq!(c, decoded, "Character roundtrip failed for '{}'", c);
    }
}

#[test]
fn test_brain_initialization_and_telemetry() {
    let mut config = BrainConfig::default();
    config.reservoir_size = 500; // Small for fast unit testing
    let brain = OpenUltronBrain::new(config);

    let t = brain.get_telemetry();
    assert_eq!(t.reservoir_monads, 500);
    assert!(t.total_synapses > 1_000);
    assert_eq!(t.total_spikes, 0);
    assert!(t.average_energy > 0.9);
}

#[test]
fn test_online_text_training_and_plasticity() {
    let mut config = BrainConfig::default();
    config.reservoir_size = 300;
    let mut brain = OpenUltronBrain::new(config);

    let initial_synapses = brain.get_telemetry().total_synapses;

    // Train repeatedly on a short pattern
    let sample = "open ultron awakens";
    for _ in 0..10 {
        let (steps, _) = brain.train_text(sample);
        assert_eq!(steps, sample.chars().count() - 1);
    }

    let t_after = brain.get_telemetry();
    assert!(
        t_after.total_spikes > 0,
        "Brain must generate spikes during training"
    );
    assert!(
        t_after.total_synapses >= initial_synapses,
        "STDP must sprout/reinforce causal synapses"
    );
}

#[test]
fn test_generation_does_not_panic() {
    let mut config = BrainConfig::default();
    config.reservoir_size = 300;
    let mut brain = OpenUltronBrain::new(config);

    brain.train_text("hello world");
    let response = brain.generate_response("hello", 15, 0.5);
    assert!(!response.is_empty(), "Response must contain characters");
}

#[test]
fn test_save_and_load_brain() {
    let mut config = BrainConfig::default();
    config.reservoir_size = 200;
    let mut brain = OpenUltronBrain::new(config);

    brain.train_text("saving cognitive state");
    let temp_path = "/tmp/ultron_test_brain.txt";

    brain.save_brain(temp_path).expect("Saving brain must succeed");

    let mut new_brain = OpenUltronBrain::new(BrainConfig {
        reservoir_size: 200,
        ..Default::default()
    });

    let loaded_count = new_brain
        .load_brain(temp_path)
        .expect("Loading brain must succeed");
    assert!(loaded_count > 0, "Must load synapses from file");
    let _ = std::fs::remove_file(temp_path);
}
