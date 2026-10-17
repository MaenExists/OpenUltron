use open_ultron_v1::config::MonadConfig;
use open_ultron_v1::monad::{Monad, Synapse};
use open_ultron_v1::network::MonadNetwork;
use open_ultron_v1::theory::TheoryEngine;

#[test]
fn test_single_monad_leaky_integration_and_fire() {
    let config = MonadConfig::default();
    let mut monad = Monad::new(0, &config);

    // Initial state
    assert_eq!(monad.potential, 0.0);
    assert_eq!(monad.threshold, config.v_threshold_resting);

    // Subthreshold input
    monad.receive_signal(0.4);
    let fired = monad.step_integration(1, &config);
    assert!(!fired);
    assert!((monad.potential - 0.4).abs() < 1e-4);

    // Second input pushes over threshold
    monad.receive_signal(0.8); // 0.4 * 0.90 + 0.8 = 1.16 >= 1.0
    let fired = monad.step_integration(2, &config);
    assert!(fired);
    assert_eq!(monad.potential, config.v_reset);
    assert_eq!(monad.total_spikes, 1);
    assert_eq!(monad.refractory_timer, config.refractory_period);

    // During refractory period, inputs are ignored
    monad.receive_signal(2.0);
    let fired_during_refractory = monad.step_integration(3, &config);
    assert!(!fired_during_refractory);
    assert_eq!(monad.potential, config.v_reset);
}

#[test]
fn test_homeostatic_threshold_adaptation() {
    let config = MonadConfig::default();
    let mut monad = Monad::new(0, &config);
    let initial_thresh = monad.threshold;

    // Simulate over-firing (seizure pressure)
    for _ in 0..10 {
        monad.step_homeostasis(true, &config);
    }
    // Threshold should have increased to suppress runaway firing
    assert!(
        monad.threshold > initial_thresh,
        "Threshold must increase after firing: {} > {}",
        monad.threshold,
        initial_thresh
    );

    // Simulate prolonged silence (quiescence)
    let elevated_thresh = monad.threshold;
    for _ in 0..50 {
        monad.step_homeostasis(false, &config);
    }
    // Threshold should relax downward toward resting sensitivity
    assert!(
        monad.threshold < elevated_thresh,
        "Threshold must decrease during silence: {} < {}",
        monad.threshold,
        elevated_thresh
    );
}

#[test]
fn test_local_causal_stdp_learning() {
    let config = MonadConfig::default();
    let mut syn_causal = Synapse::new(1, 0.3, 1);

    // Causal event: Pre fired at t=10, Post fired at t=12 (dt = +2)
    Monad::update_synapse_stdp(&mut syn_causal, 10, 12, &config);
    assert!(
        syn_causal.weight > 0.3,
        "Causal timing must strengthen synapse (LTP): new weight = {}",
        syn_causal.weight
    );

    // Anti-causal event: Pre fired at t=15, Post fired at t=13 (dt = -2)
    let mut syn_anticausal = Synapse::new(1, 0.5, 1);
    Monad::update_synapse_stdp(&mut syn_anticausal, 15, 13, &config);
    assert!(
        syn_anticausal.weight < 0.5,
        "Anti-causal timing must depress synapse (LTD): new weight = {}",
        syn_anticausal.weight
    );
}

#[test]
fn test_metabolic_decay_and_synapse_pruning() {
    let config = MonadConfig::default();
    let mut monad = Monad::new(0, &config);

    // Attach two synapses: one strong, one weak
    monad.add_synapse(1, 0.45, 1);
    monad.add_synapse(2, 0.005, 1); // Below prune_threshold = 0.01

    let initial_energy = monad.energy;
    monad.step_metabolism(true, 0.0, &config);

    // Energy should decrease after firing
    assert!(
        monad.energy < initial_energy,
        "Energy must decrease on spike: {} < {}",
        monad.energy,
        initial_energy
    );

    // Weak synapse below pruning threshold must be removed
    assert_eq!(
        monad.synapses.len(),
        1,
        "Sub-threshold synapse must be pruned"
    );
    assert_eq!(monad.synapses[0].target_id, 1);
}

#[test]
fn test_mathematical_theory_engine() {
    // 1. Branching ratio calculation
    // Spike sequence: [10, 10, 10] -> sigma = 1.0 (Critical)
    let critical_spikes = vec![10, 10, 10, 10];
    let sigma_crit = TheoryEngine::compute_branching_ratio(&critical_spikes);
    assert!((sigma_crit - 1.0).abs() < 1e-4);

    // 2. Avalanche extraction
    let activity = vec![0, 3, 5, 2, 0, 1, 4, 0, 0, 8, 0];
    let avalanches = TheoryEngine::extract_avalanches(&activity);
    // Avalanche 1: 3 + 5 + 2 = 10
    // Avalanche 2: 1 + 4 = 5
    // Avalanche 3: 8
    assert_eq!(avalanches, vec![10, 5, 8]);

    // 3. Power-law exponent estimation
    let synthetic_avalanche_sample = vec![2, 2, 3, 3, 4, 5, 7, 10, 15, 25, 40, 70, 120];
    let alpha = TheoryEngine::compute_power_law_alpha(&synthetic_avalanche_sample, 2);
    assert!(alpha > 1.0 && alpha < 3.0, "Alpha must be physical: alpha = {}", alpha);
}

#[test]
fn test_network_self_organized_criticality() {
    let mut config = MonadConfig::default();
    config.eta_homeostasis = 0.03;
    config.target_firing_rate = 0.04;

    let network_size = 200;
    let mut network = MonadNetwork::new(network_size, config);

    // Wire small-world connectivity: 10 neighbors, 15% long-range rewiring, initial weight 0.75
    network.wire_small_world(10, 0.15, 0.75);

    let total_steps = 600;
    let mut spike_counts = Vec::with_capacity(total_steps);
    let mut kuramoto_history = Vec::with_capacity(total_steps);

    for t in 0..total_steps {
        // Sparse external stimulus: pulse 2 random monads every 10 steps to seed cascades
        let stimuli = if t % 10 == 0 {
            let seed_id = (t * 7) % network_size;
            vec![(seed_id, 1.2), ((seed_id + 37) % network_size, 1.2)]
        } else {
            vec![]
        };

        let fired_count = network.step(&stimuli, 0.001);
        spike_counts.push(fired_count);

        let k_order = TheoryEngine::compute_kuramoto_series(
            &network.get_last_spike_times(),
            network_size,
            network.current_time,
            20.0,
        );
        kuramoto_history.push(k_order);
    }

    let metrics = TheoryEngine::evaluate_emergence(&spike_counts, &kuramoto_history);

    println!("\n=== EMERGENCE REPORT: 200 MONAD TEST ===");
    println!("Branching ratio (sigma): {:.4}", metrics.branching_ratio);
    println!("Dynamical regime: {:?}", metrics.dynamical_regime);
    println!("Total avalanches: {}", metrics.total_avalanches);
    println!("Power-law exponent alpha: {:.4}", metrics.avalanche_alpha);
    println!("Mean Kuramoto order parameter R: {:.4}", metrics.mean_kuramoto_order);
    println!("Population entropy: {:.4} bits", metrics.population_entropy_bits);
    println!("Emergence Index: {:.4}", metrics.emergence_index);

    // Network must avoid runaway explosion (sigma > 2.0) and death (sigma = 0)
    assert!(
        metrics.branching_ratio > 0.6 && metrics.branching_ratio < 1.4,
        "Network must self-tune near critical branching ratio ~ 1.0. Actual: {}",
        metrics.branching_ratio
    );
    assert!(metrics.total_avalanches > 10, "Must exhibit recurrent avalanches");
    assert!(
        metrics.mean_kuramoto_order < 0.90,
        "Must avoid pathological seizure lockstep"
    );
}
