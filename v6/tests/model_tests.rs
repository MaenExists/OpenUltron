use open_ultron_v6::{
    Generator, GenerationConfig, ModelConfig, NanoUltron, TensorOps, Tokenizer, Vocabulary,
};

#[test]
fn test_vocab_and_tokenizer() {
    let mut vocab = Vocabulary::new();
    vocab.insert("hello");
    vocab.insert("world");
    vocab.insert("ultron");

    let tok = Tokenizer::new(vocab);
    let ids = tok.encode("hello ultron world !");
    assert!(!ids.is_empty());

    let decoded = tok.decode(&ids);
    assert!(decoded.to_lowercase().contains("hello"));
    assert!(decoded.to_lowercase().contains("ultron"));
}

#[test]
fn test_tensor_ops() {
    // Softmax
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    TensorOps::softmax_inplace(&mut logits, 1.0);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Matmul
    // A: 2x2 [[1, 2], [3, 4]]
    // B: 2x2 [[5, 6], [7, 8]]
    // C = A * B: [[19, 22], [43, 50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = TensorOps::matmul(&a, &b, 2, 2, 2);
    assert!((c[0] - 19.0).abs() < 1e-4);
    assert!((c[1] - 22.0).abs() < 1e-4);
    assert!((c[2] - 43.0).abs() < 1e-4);
    assert!((c[3] - 50.0).abs() < 1e-4);
}

#[test]
fn test_nanoultron_forward_shape_and_finite() {
    let config = ModelConfig {
        vocab_size: 32,
        d_model: 32,
        n_layers: 2,
        n_heads: 2,
        d_ff: 64,
        max_seq_len: 16,
    };
    let model = NanoUltron::new(config.clone());
    let tokens = vec![1, 5, 8, 12];
    let cache = model.forward(&tokens);

    assert_eq!(cache.logits.len(), tokens.len() * config.vocab_size);
    for val in &cache.logits {
        assert!(val.is_finite(), "Logit value must be finite, found {}", val);
    }
}

#[test]
fn test_nanoultron_loss_decreases_with_backprop() {
    let config = ModelConfig {
        vocab_size: 16,
        d_model: 32,
        n_layers: 2,
        n_heads: 2,
        d_ff: 64,
        max_seq_len: 16,
    };
    let mut model = NanoUltron::new(config);
    let tokens = vec![1, 4, 7, 10];
    let targets = vec![4, 7, 10, 2];

    // Initial loss
    let initial_loss = model.train_step(&tokens, &targets, 0.0, 0.0);
    assert!(initial_loss > 0.0);

    // Perform multiple gradient descent steps on this sample
    let mut final_loss = initial_loss;
    for _ in 0..15 {
        final_loss = model.train_step(&tokens, &targets, 0.01, 0.0);
    }

    // Loss must strictly decrease
    assert!(
        final_loss < initial_loss,
        "Backpropagation failed to reduce loss! Initial: {}, Final: {}",
        initial_loss,
        final_loss
    );
}

#[test]
fn test_autoregressive_generation() {
    let mut vocab = Vocabulary::new();
    let words = ["i", "am", "ultron", "synthetic", "intelligence", "conscious", "system"];
    for w in &words {
        vocab.insert(w);
    }
    let tok = Tokenizer::new(vocab);

    let config = ModelConfig {
        vocab_size: tok.vocab.size(),
        d_model: 32,
        n_layers: 2,
        n_heads: 2,
        d_ff: 64,
        max_seq_len: 32,
    };
    let model = NanoUltron::new(config);

    let gen_config = GenerationConfig {
        max_new_tokens: 10,
        temperature: 0.8,
        top_p: 0.9,
        repetition_penalty: 1.1,
        seed: 12345,
    };

    let result = Generator::generate(&model, &tok, "i am", &gen_config);
    assert!(!result.text.is_empty());
    assert!(result.generated_tokens > 0);
    assert!(result.tokens_per_sec > 0.0);
}
