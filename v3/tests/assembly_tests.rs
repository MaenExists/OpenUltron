use open_ultron_v3::brain::OpenUltronBrainV3;
use open_ultron_v3::config::BrainConfig;
use open_ultron_v3::vocab::Vocabulary;

#[test]
fn test_vocabulary_tokenization_and_retrieval() {
    let mut vocab = Vocabulary::new();
    let text = "Hello, world! I am OpenUltron.";
    vocab.build_from_text(text, 100);

    let tokens = Vocabulary::tokenize(text);
    assert!(tokens.contains(&"Hello".to_string()));
    assert!(tokens.contains(&"OpenUltron".to_string()));
    assert!(tokens.contains(&".".to_string()));

    let id = vocab.token_to_id("hello");
    assert_ne!(id, 0, "Known word must not map to UNK");
}

#[test]
fn test_word_assembly_initialization() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("thought");
    vocab.add_token("emergence");

    let config = BrainConfig {
        reservoir_size: 500,
        assembly_size: 5,
        ..Default::default()
    };

    let brain = OpenUltronBrainV3::new(config, vocab);
    let t = brain.get_telemetry();

    assert!(t.assembly_monads > 0);
    assert_eq!(t.reservoir_monads, 500);
    assert!(t.total_synapses > 500);
}

#[test]
fn test_word_sequence_learning_and_generation() {
    let mut vocab = Vocabulary::new();
    let text = "who are you ? I am OpenUltron .";
    vocab.build_from_text(text, 50);

    let config = BrainConfig {
        reservoir_size: 300,
        assembly_size: 4,
        ..Default::default()
    };

    let mut brain = OpenUltronBrainV3::new(config, vocab);
    let tokens = Vocabulary::tokenize(text);

    // Train on sentence
    for _ in 0..10 {
        brain.train_tokens(&tokens);
    }

    let response = brain.generate_response("who are you", 6, 0.3);
    assert!(!response.is_empty(), "Must produce non-empty word sequence");

    // Output must consist of valid English words, never raw ASCII nonsense
    for word in response.split_whitespace() {
        assert!(
            word.chars().all(|c| c.is_alphanumeric() || ['.', ',', '!', '?', ':', ';', '\'', '-'].contains(&c)),
            "Output must be valid words, got: '{}'",
            word
        );
    }
}
