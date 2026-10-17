//! Rigorous Integration & Scientific Unit Tests for OpenUltron v5.

use open_ultron_v5::monad::SpikingMonad;
use open_ultron_v5::organs::cortex::Neocortex;
use open_ultron_v5::organs::hippocampus::Hippocampus;
use open_ultron_v5::organs::neuromodulator::Neuromodulator;
use open_ultron_v5::organs::sensory::SensoryThalamus;
use open_ultron_v5::vocab::Vocabulary;
use open_ultron_v5::OpenUltronOrganism;

#[test]
fn test_monad_lif_and_atp_constraints() {
    let mut monad = SpikingMonad::new(0);
    assert_eq!(monad.potential, 0.0);
    assert_eq!(monad.atp_level, 1.0);

    // Sub-threshold excitation
    let spiked = monad.integrate(0.5, 0.0);
    assert!(!spiked);
    assert!(monad.potential > 0.0 && monad.potential < 1.0);

    // Supra-threshold excitation triggering action potential
    let spiked2 = monad.integrate(1.2, 0.0);
    assert!(spiked2);
    assert_eq!(monad.potential, 0.0);
    assert_eq!(monad.spikes_fired, 1);
    assert_eq!(monad.refractory_ticks, 1);
    assert!(monad.atp_level < 1.0); // ATP consumed by firing
}

#[test]
fn test_sensory_thalamus_gating_and_surprise() {
    let mut thalamus = SensoryThalamus::new();
    let mut vocab = Vocabulary::new();

    // First presentation: high novelty / surprise
    let burst1 = thalamus.process_input("an unknown alien signal has appeared", &mut vocab);
    assert!(burst1.sensory_surprise > 0.5);
    assert_eq!(burst1.token_ids.len(), 6);

    // Repeated presentation: habituation decreases surprise
    let burst2 = thalamus.process_input("an unknown alien signal has appeared", &mut vocab);
    assert!(burst2.sensory_surprise < burst1.sensory_surprise);
}

#[test]
fn test_hippocampal_episodic_binding_and_consolidation() {
    let mut hippo = Hippocampus::new(4);
    assert_eq!(hippo.episodic_buffer.len(), 0);

    // Turn 1
    hippo.encode_turn(&[10, 11, 12], 1.5);
    assert_eq!(hippo.episodic_buffer.len(), 1);

    // Turn 2
    hippo.encode_turn(&[20, 21], 1.2);
    assert_eq!(hippo.episodic_buffer.len(), 2);

    // Cross-turn episodic attractor exists
    let attractors = hippo.get_episodic_attractors(30);
    assert!(attractors[10] > 0.0);
    assert!(attractors[20] > 0.0);

    // Sharp wave ripples
    let ripples = hippo.consolidate_ripples();
    assert!(!ripples.is_empty());
}

#[test]
fn test_neocortical_stdp_and_homeostatic_pruning() {
    let mut cortex = Neocortex::new(50, 4);
    let mut vocab = Vocabulary::new();
    let id_a = vocab.get_or_insert("alpha");
    let id_b = vocab.get_or_insert("beta");

    cortex.learn_sequence(&[id_a], &[id_b], &vocab, 1.0);

    let initial_synapses = cortex.total_synapses();
    assert!(initial_synapses > 0);
    assert!(cortex.w_cross.contains_key(&(id_a, id_b)));

    // Pruning with high threshold removes weak synapses
    let pruned = cortex.prune_synapses(100.0);
    assert_eq!(pruned, initial_synapses);
    assert_eq!(cortex.total_synapses(), 0);
}

#[test]
fn test_neuromodulator_homeostasis() {
    let mut neuro = Neuromodulator::new();
    assert_eq!(neuro.free_energy, 0.95);

    // Good communication reduces free energy and raises dopamine
    neuro.update_state(0.1, 0.9);
    assert!(neuro.free_energy < 0.95);
    assert!(neuro.dopamine > 0.40);

    // Spikes consume metabolic ATP
    neuro.consume_spike_energy(500);
    assert!(neuro.metabolic_atp < 1.0);
}

#[test]
fn test_full_organism_awakening_and_online_plasticity() {
    let mut organism = OpenUltronOrganism::new(4);

    let sample_corpus = "user : hello ultron . ultron : hello maen i am awake .\nuser : who are you . ultron : i am a living digital mind .\n";
    organism.train_corpus(sample_corpus);

    let initial_synapses = organism.cortex.total_synapses();
    assert!(initial_synapses > 0);

    // Test interaction
    let (thought, bio) = organism.interact("hello ultron");
    assert!(!thought.is_empty());
    assert!(bio.total_spikes > 0);

    // Verify online plasticity: synapses grew from interaction
    assert!(organism.cortex.total_synapses() >= initial_synapses);

    // Verify episodic depth
    assert!(bio.episodic_depth > 0);
}
