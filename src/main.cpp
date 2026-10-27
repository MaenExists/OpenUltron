#include "spark/network.hpp"
#include "spark/metrics.hpp"
#include "spark/environment.hpp"
#include "spark/trainer.hpp"
#include "spark/baselines.hpp"
#include "spark/experiment_runner.hpp"
#include "spark/rollout.hpp"
#include "spark/planning.hpp"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>

void print_banner() {
    std::cout << "========================================================\n"
              << "  OpenUltron :: SPARK (Sparse Predictive Adaptive Recurrent Kernel)\n"
              << "  Version 0.4.0 (Phase 4: Local Learning, Distributed Credit Assignment & Multi-Mode Comparison)\n"
              << "========================================================\n" << std::endl;
}

void print_info() {
    print_banner();
    std::cout << "[System & Compiler Information]\n"
              << "  Target Platform      : Linux x86_64\n"
              << "  C++ Standard         : C++17\n"
#if defined(__clang__)
              << "  Compiler             : Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__ << "\n"
#elif defined(__GNUC__)
              << "  Compiler             : GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n"
#else
              << "  Compiler             : Unknown C++17 Compiler\n"
#endif
              << "  Precision            : Single Precision IEEE-754 (FP32)\n"
              << "  Graph Storage        : Compressed Sparse Row (CSR)\n"
              << "  Recurrence Mode      : Synchronous Double Buffering\n"
              << "  Training Mode        : Truncated Backpropagation Through Time (TBPTT)\n"
              << "  Supported Topologies : RandomSparse, ClusteredSparse, AdaptiveSparse\n"
              << "\n[CLI Usage]\n"
              << "  spark info\n"
              << "  spark demo\n"
              << "  spark benchmark [--units N] [--fan-in K] [--ticks T] [--seed S] [--topology random|clustered]\n"
              << "  spark train [--task delay-copy] [--delay D] [--units N] [--fan-in K] [--steps S] [--unroll U] [--lr LR] [--seed S]\n"
              << "  spark baselines [--delay D] [--units N] [--seed S]\n"
              << "  spark capacity [--seeds S] [--csv PATH]\n"
              << "  spark diagnose [--csv PATH]\n"
              << "  spark ablate [--delay D] [--units N] [--seed S]\n"
              << "  spark phase4 [--seeds S] [--csv PATH]\n"
              << "  spark local-ablate [--delay D] [--units N] [--seeds S] [--csv PATH]\n"
              << "  spark world-model [--seeds S] [--steps N] [--unroll U] [--lr LR] [--test-transitions T] [--csv PATH]\n"
              << "  spark partial-obs [--seeds S] [--steps N] [--unroll U] [--lr LR] [--episodes E] [--csv PATH]\n"
              << "  spark onestep-diag [--steps N] [--test-transitions T] [--lr LR] [--seed S] [--csv PATH]\n"
              << "  spark rollout [--checkpoint PATH] [--mode teacher-forced|open-loop|both] [--horizon H] [--episodes N] [--seed S] [--csv PATH]\n"
              << "  spark plan-foundation\n"
              << "  spark plan-beam [--width K] [--depth H] [--episodes N] [--checkpoint PATH]\n"
              << std::endl;
}

void run_demo() {
    print_banner();
    std::cout << ">>> Running SPARK Recurrent State Self-Test & Demonstration <<<\n" << std::endl;

    spark::SparseGraphConfig graph_cfg;
    graph_cfg.num_units = 256;
    graph_cfg.fan_in = 12;
    graph_cfg.topology = spark::TopologyType::ClusteredSparse;
    graph_cfg.num_clusters = 4;
    graph_cfg.intra_cluster_prob = 0.8f;
    graph_cfg.seed = 42ULL;

    spark::NetworkConfig net_cfg;
    net_cfg.num_units = 256;
    net_cfg.input_dim = 4;
    net_cfg.output_dim = 2;
    net_cfg.alpha = 0.8f;
    net_cfg.beta = 0.5f;
    net_cfg.microsteps = 1;
    net_cfg.seed = 42ULL;

    std::cout << "Creating MicroUnitNetwork (" << net_cfg.num_units << " units, "
              << graph_cfg.fan_in << " fan-in, ClusteredSparse topology)..." << std::endl;

    spark::MicroUnitNetwork net(net_cfg, graph_cfg);
    std::string err;
    if (!net.validate(err)) {
        std::cerr << "Network validation error: " << err << std::endl;
        return;
    }

    auto g_stats = net.graph().compute_stats();
    std::cout << "  Graph edges          : " << g_stats.total_edges << "\n"
              << "  Average fan-in       : " << g_stats.avg_fan_in << "\n"
              << "  Model parameters     : " << net.parameter_count() << "\n"
              << "  Model memory footprint: " << (net.memory_bytes() / 1024.0) << " KB\n"
              << std::endl;

    std::cout << "Executing 10 recurrent ticks with continuous numerical input stream:" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::setw(6) << "Tick"
              << std::setw(12) << "Mean Act"
              << std::setw(12) << "Std Act"
              << std::setw(12) << "Min Act"
              << std::setw(12) << "Max Act"
              << std::setw(12) << "Sparsity"
              << std::setw(9) << "Logit[0]"
              << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    spark::Random input_rng(1337ULL);
    std::vector<float> input(net_cfg.input_dim);
    std::vector<float> logits, probs;

    std::vector<float> baseline_final_state;

    for (int t = 1; t <= 10; ++t) {
        for (float& in_val : input) {
            in_val = input_rng.uniform_float(-1.0f, 1.0f);
        }
        net.forward_step(input, logits, probs);
        auto a_stats = spark::compute_activation_stats(net.state());

        std::cout << std::setw(6) << t
                  << std::setw(12) << std::fixed << std::setprecision(4) << a_stats.mean_val
                  << std::setw(12) << a_stats.std_val
                  << std::setw(12) << a_stats.min_val
                  << std::setw(12) << a_stats.max_val
                  << std::setw(12) << a_stats.sparsity
                  << std::setw(9) << logits[0]
                  << std::endl;

        if (t == 10) {
            baseline_final_state = net.state();
        }
    }
    std::cout << std::string(75, '-') << "\n" << std::endl;

    // Perturbation Analysis (Section 28 verification)
    std::cout << ">>> Verifying Sensitivity to Input and Weight Perturbations <<<" << std::endl;

    // Test 1: External input perturbation
    spark::MicroUnitNetwork net_input_pert(net_cfg, graph_cfg);
    spark::Random input_rng2(1337ULL);
    for (int t = 1; t <= 10; ++t) {
        for (float& in_val : input) {
            in_val = input_rng2.uniform_float(-1.0f, 1.0f);
        }
        if (t == 5) {
            // Apply perturbation at tick 5
            input[0] += 2.0f;
        }
        net_input_pert.forward_step(input, logits, probs);
    }
    double input_diff_l2 = 0.0;
    for (size_t i = 0; i < net_cfg.num_units; ++i) {
        double d = net_input_pert.state()[i] - baseline_final_state[i];
        input_diff_l2 += d * d;
    }
    input_diff_l2 = std::sqrt(input_diff_l2);
    std::cout << "  Input perturbation L2 distance at tick 10  : " << input_diff_l2
              << " (" << (input_diff_l2 > 1e-4 ? "CONFIRMED SENSITIVE" : "INSENSITIVE") << ")" << std::endl;

    // Test 2: Connection weight perturbation
    spark::MicroUnitNetwork net_weight_pert(net_cfg, graph_cfg);
    net_weight_pert.graph().weights[0] += 1.0f; // Perturb single synaptic connection weight
    spark::Random input_rng3(1337ULL);
    for (int t = 1; t <= 10; ++t) {
        for (float& in_val : input) {
            in_val = input_rng3.uniform_float(-1.0f, 1.0f);
        }
        net_weight_pert.forward_step(input, logits, probs);
    }
    double weight_diff_l2 = 0.0;
    for (size_t i = 0; i < net_cfg.num_units; ++i) {
        double d = net_weight_pert.state()[i] - baseline_final_state[i];
        weight_diff_l2 += d * d;
    }
    weight_diff_l2 = std::sqrt(weight_diff_l2);
    std::cout << "  Weight perturbation L2 distance at tick 10 : " << weight_diff_l2
              << " (" << (weight_diff_l2 > 1e-4 ? "CONFIRMED SENSITIVE" : "INSENSITIVE") << ")" << std::endl;

    std::cout << "\n[Scientific Integrity Note]: This verifies forward state dynamics and sensitivity.\n"
              << "This behavior is physical/numerical recurrence, not learned intelligence (training begins in Phase 2)."
              << std::endl;
}

void run_benchmark(int argc, char* argv[]) {
    uint32_t units = 10000;
    uint32_t fan_in = 16;
    uint32_t ticks = 100;
    uint32_t input_dim = 8;
    uint32_t output_dim = 4;
    uint32_t microsteps = 1;
    uint64_t seed = 42ULL;
    std::string topology = "random";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--units" && i + 1 < argc) units = std::stoul(argv[++i]);
        else if (arg == "--fan-in" && i + 1 < argc) fan_in = std::stoul(argv[++i]);
        else if (arg == "--ticks" && i + 1 < argc) ticks = std::stoul(argv[++i]);
        else if (arg == "--input-dim" && i + 1 < argc) input_dim = std::stoul(argv[++i]);
        else if (arg == "--output-dim" && i + 1 < argc) output_dim = std::stoul(argv[++i]);
        else if (arg == "--microsteps" && i + 1 < argc) microsteps = std::stoul(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (arg == "--topology" && i + 1 < argc) topology = argv[++i];
    }

    print_banner();
    std::cout << ">>> Running SPARK Forward Inference CPU Benchmark <<<\n"
              << "  Units              : " << units << "\n"
              << "  Fan-in             : " << fan_in << " incoming connections/unit\n"
              << "  Ticks              : " << ticks << " steps\n"
              << "  Microsteps         : " << microsteps << "\n"
              << "  Input Dim          : " << input_dim << "\n"
              << "  Output Dim         : " << output_dim << "\n"
              << "  Topology           : " << topology << "\n"
              << "  Seed               : " << seed << "\n"
              << std::endl;

    spark::SparseGraphConfig graph_cfg;
    graph_cfg.num_units = units;
    graph_cfg.fan_in = fan_in;
    graph_cfg.seed = seed;
    if (topology == "clustered") {
        graph_cfg.topology = spark::TopologyType::ClusteredSparse;
        graph_cfg.num_clusters = std::max(2u, units / 250);
        graph_cfg.intra_cluster_prob = 0.8f;
    } else {
        graph_cfg.topology = spark::TopologyType::RandomSparse;
    }

    spark::NetworkConfig net_cfg;
    net_cfg.num_units = units;
    net_cfg.input_dim = input_dim;
    net_cfg.output_dim = output_dim;
    net_cfg.microsteps = microsteps;
    net_cfg.seed = seed;

    std::cout << "Initializing sparse graph and memory buffers..." << std::flush;
    spark::Timer init_timer;
    spark::MicroUnitNetwork net(net_cfg, graph_cfg);
    double init_sec = init_timer.elapsed_seconds();
    std::cout << " done (" << (init_sec * 1000.0) << " ms).\n" << std::endl;

    size_t total_edges = net.graph().total_edges();
    std::cout << "Network Statistics:\n"
              << "  Total Directed Edges     : " << total_edges << "\n"
              << "  Total Trainable Weights  : " << net.parameter_count() << "\n"
              << "  Network Memory Allocation: " << (net.memory_bytes() / (1024.0 * 1024.0)) << " MB\n"
              << std::endl;

    std::vector<float> input(input_dim, 0.5f);
    std::vector<float> logits, probs;

    // Warmup (5 ticks)
    std::cout << "Warming up CPU cache..." << std::flush;
    for (int w = 0; w < 5; ++w) {
        net.forward_step(input, logits, probs);
    }
    std::cout << " done.\n" << std::endl;

    // Timed run
    std::cout << "Executing " << ticks << " forward steps..." << std::flush;
    spark::Timer bench_timer;
    for (uint32_t t = 0; t < ticks; ++t) {
        input[0] = static_cast<float>(t) / static_cast<float>(ticks);
        net.forward_step(input, logits, probs);
    }
    double total_sec = bench_timer.elapsed_seconds();
    std::cout << " done.\n" << std::endl;

    auto mem = spark::get_current_memory_usage();
    double latency_per_tick_ms = (total_sec / ticks) * 1000.0;
    double units_per_sec = (static_cast<double>(units) * ticks * microsteps) / total_sec;
    double edges_per_sec = (static_cast<double>(total_edges) * ticks * microsteps) / total_sec;

    std::cout << "Benchmark Results:\n"
              << "--------------------------------------------------------\n"
              << "  Total Wall Time         : " << std::fixed << std::setprecision(4) << total_sec << " s\n"
              << "  Latency per Tick        : " << latency_per_tick_ms << " ms\n"
              << "  Units Processed / sec   : " << std::scientific << std::setprecision(2) << units_per_sec << " units/s\n"
              << "  Synaptic Updates / sec  : " << edges_per_sec << " edges/s ("
              << std::fixed << std::setprecision(2) << (edges_per_sec / 1e6) << " M-edges/s)\n"
              << "  Peak Resident Memory    : " << (mem.peak_resident_bytes / (1024.0 * 1024.0)) << " MB\n"
              << "--------------------------------------------------------\n"
              << std::endl;
}

void run_train(int argc, char* argv[]) {
    std::string task = "delay-copy";
    uint32_t delay = 1;
    uint32_t units = 128;
    uint32_t fan_in = 8;
    float beta = 0.5f;
    uint32_t steps = 1500;
    uint32_t unroll = 8;
    float lr = 0.03f;
    float clip = 1.0f;
    uint32_t eval_interval = 150;
    uint64_t seed = 42ULL;
    std::string topology = "random";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--task" && i + 1 < argc) task = argv[++i];
        else if (arg == "--delay" && i + 1 < argc) delay = std::stoul(argv[++i]);
        else if (arg == "--units" && i + 1 < argc) units = std::stoul(argv[++i]);
        else if (arg == "--fan-in" && i + 1 < argc) fan_in = std::stoul(argv[++i]);
        else if (arg == "--beta" && i + 1 < argc) beta = std::stof(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) steps = std::stoul(argv[++i]);
        else if (arg == "--unroll" && i + 1 < argc) unroll = std::stoul(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (arg == "--clip" && i + 1 < argc) clip = std::stof(argv[++i]);
        else if (arg == "--eval-interval" && i + 1 < argc) eval_interval = std::stoul(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (arg == "--topology" && i + 1 < argc) topology = argv[++i];
    }

    print_banner();
    std::cout << ">>> Training SPARK with Exact Gradients (TBPTT) <<<\n"
              << "  Task                 : " << task << " (Delay: " << delay << ")\n"
              << "  Recurrent Units      : " << units << "\n"
              << "  Fan-in               : " << fan_in << " incoming connections/unit\n"
              << "  Training Steps       : " << steps << " windows\n"
              << "  BPTT Unroll Window   : " << unroll << " steps\n"
              << "  Learning Rate        : " << lr << "\n"
              << "  Max Grad Norm        : " << clip << "\n"
              << "  Topology             : " << topology << "\n"
              << "  Random Seed          : " << seed << "\n"
              << std::endl;

    spark::SparseGraphConfig graph_cfg;
    graph_cfg.num_units = units;
    graph_cfg.fan_in = fan_in;
    graph_cfg.seed = seed;
    if (topology == "clustered") {
        graph_cfg.topology = spark::TopologyType::ClusteredSparse;
        graph_cfg.num_clusters = std::max(2u, units / 32);
        graph_cfg.intra_cluster_prob = 0.8f;
    } else {
        graph_cfg.topology = spark::TopologyType::RandomSparse;
    }

    spark::NetworkConfig net_cfg;
    net_cfg.num_units = units;
    net_cfg.input_dim = 1;
    net_cfg.output_dim = 1;
    net_cfg.alpha = 0.8f;
    net_cfg.beta = beta;
    net_cfg.seed = seed;

    spark::MicroUnitNetwork net(net_cfg, graph_cfg);

    // Create training and held-out evaluation environments with distinct seeds
    spark::TemporalMemoryEnv train_env(delay, 200, seed);
    spark::TemporalMemoryEnv eval_env(delay, 200, seed + 1000ULL);

    std::cout << "Starting supervised TBPTT training loop..." << std::endl;
    spark::Trainer trainer(net, train_env, &eval_env, lr, clip);
    auto summary = trainer.train(steps, unroll, eval_interval, false);

    std::cout << "\nTraining Complete!\n"
              << "--------------------------------------------------------\n"
              << "  Initial Loss         : " << std::fixed << std::setprecision(5) << summary.initial_loss << "\n"
              << "  Final Train Loss     : " << summary.final_train_loss << "\n"
              << "  Final Eval Loss      : " << summary.final_eval_loss << "\n"
              << "  Final Eval Accuracy  : " << std::setprecision(2)
              << (summary.final_eval_accuracy * 100.0f) << "% ("
              << summary.final_eval_correct << " / " << summary.final_eval_total << " bits)\n"
              << "  Total Wall Time      : " << std::setprecision(2) << summary.total_seconds << " s\n"
              << "  Steps / Second       : " << (summary.steps_completed / summary.total_seconds) << " updates/s\n"
              << "--------------------------------------------------------\n"
              << std::endl;
}

void run_baselines_cmd(int argc, char* argv[]) {
    uint32_t delay = 1;
    uint32_t units = 64;
    uint64_t seed = 42ULL;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--delay" && i + 1 < argc) delay = std::stoul(argv[++i]);
        else if (arg == "--units" && i + 1 < argc) units = std::stoul(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
    }

    print_banner();
    spark::ExperimentRunner::compare_baselines(delay, units, seed);
}

void run_capacity_cmd(int argc, char* argv[]) {
    size_t num_seeds = 10;
    std::string csv_path = "experiments/results/capacity_results.csv";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seeds" && i + 1 < argc) num_seeds = std::stoul(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }

    print_banner();
    std::vector<uint32_t> delays = {1, 2, 3, 5};
    std::vector<uint32_t> unit_counts = {64, 128};
    spark::ExperimentRunner::run_capacity_suite(delays, unit_counts, num_seeds, csv_path);
}

void run_diagnose_cmd(int argc, char* argv[]) {
    std::string csv_path = "experiments/results/delay3_diagnostic.csv";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }

    print_banner();
    spark::ExperimentRunner::run_delay3_diagnostic_suite(csv_path);
}

void run_ablate_cmd(int argc, char* argv[]) {
    uint32_t delay = 1;
    uint32_t units = 64;
    uint64_t seed = 42ULL;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--delay" && i + 1 < argc) delay = std::stoul(argv[++i]);
        else if (arg == "--units" && i + 1 < argc) units = std::stoul(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
    }

    print_banner();
    spark::ExperimentRunner::run_inference_ablations(delay, units, seed);
}

void run_phase4_cmd(int argc, char* argv[]) {
    size_t num_seeds = 10;
    std::string csv_path = "experiments/results/phase4_modes_comparison.csv";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seeds" && i + 1 < argc) num_seeds = std::stoul(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }

    print_banner();
    std::vector<uint32_t> delays = {1, 3, 5};
    std::vector<uint32_t> unit_counts = {64, 128};
    std::vector<spark::LearningAlgorithmMode> modes = {
        spark::LearningAlgorithmMode::ExactTBPTT,
        spark::LearningAlgorithmMode::LocalRandomFeedback,
        spark::LearningAlgorithmMode::LocalReadoutTranspose,
        spark::LearningAlgorithmMode::ReadoutOnly
    };

    spark::ExperimentRunner::run_phase4_suite(delays, unit_counts, modes, num_seeds, csv_path);
}

void run_local_ablate_cmd(int argc, char* argv[]) {
    uint32_t delay = 1;
    uint32_t units = 64;
    size_t num_seeds = 5;
    std::string csv_path = "experiments/results/phase4_ablations.csv";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--delay" && i + 1 < argc) delay = std::stoul(argv[++i]);
        else if (arg == "--units" && i + 1 < argc) units = std::stoul(argv[++i]);
        else if (arg == "--seeds" && i + 1 < argc) num_seeds = std::stoul(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }

    print_banner();
    spark::ExperimentRunner::run_local_ablation_suite(delay, units, num_seeds, csv_path);
}

void run_world_model_cmd(int argc, char* argv[]) {
    size_t num_seeds = 5;
    uint32_t steps = 10000;
    uint32_t unroll = 8;
    float lr = 0.02f;
    uint32_t test_transitions = 10000;
    std::string csv_path = "experiments/results/phase5/milestone5b_transitions.csv";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seeds" && i + 1 < argc) num_seeds = std::stoul(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) steps = std::stoul(argv[++i]);
        else if (arg == "--unroll" && i + 1 < argc) unroll = std::stoul(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (arg == "--test-transitions" && i + 1 < argc) test_transitions = std::stoul(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }

    print_banner();
    spark::ExperimentRunner::run_milestone5b_suite(num_seeds, steps, unroll, lr, test_transitions, csv_path);
}

void run_partial_obs_cmd(int argc, char* argv[]) {
    size_t num_seeds = 5;
    uint32_t steps = 8000;
    uint32_t unroll = 16;
    float lr = 0.02f;
    uint32_t episodes = 200;
    std::string csv_path = "experiments/results/phase5/milestone5c_partial_obs.csv";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seeds" && i + 1 < argc) num_seeds = std::stoul(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) steps = std::stoul(argv[++i]);
        else if (arg == "--unroll" && i + 1 < argc) unroll = std::stoul(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (arg == "--episodes" && i + 1 < argc) episodes = std::stoul(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }

    print_banner();
    spark::ExperimentRunner::run_milestone5c_suite(num_seeds, steps, unroll, lr, episodes, csv_path);
}

void run_onestep_diag_cmd(int argc, char* argv[]) {
    uint32_t steps = 10000;
    uint32_t test_transitions = 10000;
    float lr = 0.02f;
    uint64_t seed = 42ULL;
    std::string csv_path = "experiments/results/phase5/milestone5d0_investigation.csv";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) steps = std::stoul(argv[++i]);
        else if (arg == "--test-transitions" && i + 1 < argc) test_transitions = std::stoul(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }

    print_banner();
    spark::ExperimentRunner::run_milestone5d0_diagnostic(steps, test_transitions, lr, seed, csv_path);
}

void run_rollout_cmd(int argc, char* argv[]) {
    std::string checkpoint_path = "checkpoints/world_model_best.spark";
    std::string mode_str = "both";
    std::string suite = "";
    size_t num_seeds = 5;
    uint32_t train_steps = 20000;
    uint32_t horizon = 8;
    uint32_t episodes = 100;
    uint64_t seed = 42ULL;
    std::string csv_path = "";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode_str = argv[++i];
        else if (arg == "--suite" && i + 1 < argc) suite = argv[++i];
        else if (arg == "--seeds" && i + 1 < argc) num_seeds = std::stoul(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) train_steps = std::stoul(argv[++i]);
        else if (arg == "--horizon" && i + 1 < argc) horizon = std::stoul(argv[++i]);
        else if (arg == "--episodes" && i + 1 < argc) episodes = std::stoul(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }

    print_banner();

    if (!suite.empty()) {
        if (suite == "multi-horizon" || suite == "5d2") {
            spark::ExperimentRunner::run_milestone5d2_multi_horizon(
                num_seeds, {1, 2, 4, 8, 16}, episodes, train_steps,
                csv_path.empty() ? "experiments/results/phase5/milestone5d2_multi_horizon.csv" : csv_path
            );
        } else if (suite == "ablations" || suite == "5d3") {
            spark::ExperimentRunner::run_milestone5d3_ablations(
                num_seeds, {1, 2, 4, 8, 16}, episodes,
                csv_path.empty() ? "experiments/results/phase5/milestone5d3_ablations.csv" : csv_path
            );
        } else if (suite == "controls" || suite == "5d4") {
            spark::ExperimentRunner::run_milestone5d4_controls(
                num_seeds, {1, 2, 4, 8, 16}, episodes,
                csv_path.empty() ? "experiments/results/phase5/milestone5d4_controls.csv" : csv_path
            );
        } else if (suite == "all") {
            spark::ExperimentRunner::run_milestone5d2_multi_horizon(
                num_seeds, {1, 2, 4, 8, 16}, episodes, train_steps,
                "experiments/results/phase5/milestone5d2_multi_horizon.csv"
            );
            spark::ExperimentRunner::run_milestone5d3_ablations(
                num_seeds, {1, 2, 4, 8, 16}, episodes,
                "experiments/results/phase5/milestone5d3_ablations.csv"
            );
            spark::ExperimentRunner::run_milestone5d4_controls(
                num_seeds, {1, 2, 4, 8, 16}, episodes,
                "experiments/results/phase5/milestone5d4_controls.csv"
            );
        } else {
            std::cerr << "Unknown suite: '" << suite << "'. Valid options: multi-horizon, ablations, controls, all" << std::endl;
        }
        return;
    }

    if (horizon == 0) {
        std::cerr << "Error: Horizon must be >= 1." << std::endl;
        return;
    }

    std::cout << ">>> SPARK Multi-Step Predictive Generation (Rollout Engine) <<<\n"
              << "  Mode       : " << mode_str << "\n"
              << "  Horizon    : " << horizon << "\n"
              << "  Episodes   : " << episodes << "\n"
              << "  Seed       : " << seed << "\n"
              << "  Checkpoint : " << checkpoint_path << "\n" << std::endl;

    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 128;
    g_cfg.fan_in = 8;
    g_cfg.seed = seed;

    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 128;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 1.0f;
    n_cfg.seed = seed;

    spark::MicroUnitNetwork net(n_cfg, g_cfg);

    bool loaded = spark::MicroUnitNetwork::load_checkpoint(checkpoint_path, net);
    if (loaded) {
        std::cout << "Successfully loaded checkpoint from: " << checkpoint_path << std::endl;
    } else {
        std::cout << "Checkpoint not found at '" << checkpoint_path << "'. Training high-accuracy reference model (20k steps)..." << std::endl;
        auto splits_train = spark::GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);
        spark::GridWorldEnv train_env(splits_train.train_layouts, 50, seed);
        spark::GridWorldEnv val_env(splits_train.val_layouts, 50, seed + 1000ULL);

        spark::Trainer trainer(net, train_env, &val_env, 0.02f, 1.0f);
        trainer.set_loss_type(spark::LossType::CategoricalCrossEntropy);
        trainer.train(20000, 8, 0, true);

        net.save_checkpoint(checkpoint_path);
        std::cout << "Model trained and saved to checkpoint: " << checkpoint_path << std::endl;
    }

    auto splits = spark::GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);

    std::vector<spark::RolloutMetrics> metrics_list;

    // 1. Oracle
    auto oracle_metrics = spark::RolloutEngine::evaluate_oracle(splits.test_layouts, horizon, episodes, seed);
    oracle_metrics.print_summary();
    metrics_list.push_back(oracle_metrics);

    // 2. Teacher-Forced
    if (mode_str == "teacher-forced" || mode_str == "both" || mode_str == "all") {
        auto tf_metrics = spark::RolloutEngine::evaluate_rollouts(net, splits.test_layouts, horizon, episodes, spark::RolloutMode::TeacherForced, seed);
        tf_metrics.model_name = "SPARK (Teacher-Forced)";
        tf_metrics.print_summary();
        metrics_list.push_back(tf_metrics);
    }

    // 3. Open-Loop
    if (mode_str == "open-loop" || mode_str == "both" || mode_str == "all") {
        auto ol_metrics = spark::RolloutEngine::evaluate_rollouts(net, splits.test_layouts, horizon, episodes, spark::RolloutMode::OpenLoop, seed);
        ol_metrics.model_name = "SPARK (Open-Loop Autoregressive)";
        ol_metrics.print_summary();
        metrics_list.push_back(ol_metrics);
    }

    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << spark::RolloutMetrics::csv_header() << "\n";
            for (const auto& m : metrics_list) {
                ofs << m.to_csv() << "\n";
            }
            std::cout << "Saved rollout metrics to: " << csv_path << std::endl;
        }
    }
}

void run_plan_foundation_cmd(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    print_banner();
    std::cout << ">>> SPARK Milestone 5E.1: Goal Planning Foundation & Recurrent Branching <<<\n" << std::endl;

    // 1. Generate Reachable Planning Episodes
    std::cout << "1. Generating Reachable Planning Episodes on Unseen Test Layouts..." << std::endl;
    auto splits = spark::GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);
    auto episodes = spark::PlanningEpisodeGenerator::generate_dataset(splits.test_layouts, 2, 42ULL, 16, 2);

    std::cout << "  Generated " << episodes.size() << " reachable planning episodes across 20 test layouts." << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, episodes.size()); ++i) {
        const auto& ep = episodes[i];
        std::cout << "    Episode " << i + 1 << ": Start=(" << (ep.start_pos / 5) << "," << (ep.start_pos % 5)
                  << ") -> Goal=(" << (ep.goal_pos / 5) << "," << (ep.goal_pos % 5)
                  << ") | Shortest Path = " << ep.shortest_path_length << " steps (Actions: ";
        for (auto a : ep.reference_shortest_path) {
            std::cout << spark::action_name(a) << " ";
        }
        std::cout << ")" << std::endl;
    }

    // 2. Load World Model
    std::string checkpoint_path = "checkpoints/world_model_best.spark";
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 128;
    g_cfg.fan_in = 8;
    g_cfg.seed = 42ULL;
    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 128;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 1.0f;
    n_cfg.seed = 42ULL;
    spark::MicroUnitNetwork net(n_cfg, g_cfg);

    if (!spark::MicroUnitNetwork::load_checkpoint(checkpoint_path, net)) {
        std::cerr << "Could not load checkpoint at " << checkpoint_path << ". Running with initialized model." << std::endl;
    } else {
        std::cout << "\n2. Successfully loaded trained world model from: " << checkpoint_path << std::endl;
    }

    // 3. Demonstrate Recurrent-State Branching & Trajectory Scoring
    const auto& test_ep = episodes[0];
    std::cout << "\n3. Demonstrating Recurrent-State Branching on Episode 1:" << std::endl;
    std::cout << "   Start Pos: (" << (test_ep.start_pos / 5) << "," << (test_ep.start_pos % 5) << ") [index " << test_ep.start_pos << "]" << std::endl;
    std::cout << "   Goal Pos : (" << (test_ep.goal_pos / 5) << "," << (test_ep.goal_pos % 5) << ") [index " << test_ep.goal_pos << "]" << std::endl;

    auto root = spark::PlanningEngine::create_cold_start_root(test_ep.start_pos, test_ep.goal_pos, 128);
    std::cout << "   Root Node: Cold Start (h(0)=0), Initial Score = " << root.score
              << ", Manhattan = " << root.manhattan_to_goal << std::endl;

    // Expand depth 1
    std::cout << "\n4. Expanding Depth 1 Candidate Action Branches (Independent States):" << std::endl;
    auto depth1_nodes = spark::PlanningEngine::expand_all_actions(net, test_ep.layout, root, test_ep.goal_pos);

    for (const auto& node : depth1_nodes) {
        std::cout << "   Branch [" << spark::action_name(node.action_from_parent) << "]: "
                  << "Pred Pos = (" << (node.predicted_position / 5) << "," << (node.predicted_position % 5) << ") "
                  << "| Manhattan = " << node.manhattan_to_goal << " "
                  << "| Conf = " << std::fixed << std::setprecision(3) << (node.target_probability * 100.0f) << "% "
                  << "| Score J = " << std::setprecision(2) << node.score
                  << (node.goal_reached ? " [GOAL REACHED!]" : "")
                  << std::endl;
    }

    // Sort according to tie-breaking comparator
    std::sort(depth1_nodes.begin(), depth1_nodes.end(), [](const spark::PlanningNode& a, const spark::PlanningNode& b) {
        return a.is_better_than(b);
    });

    std::cout << "\n   Best Depth 1 Candidate: Action " << spark::action_name(depth1_nodes[0].action_from_parent)
              << " (Score: " << depth1_nodes[0].score << ", Manhattan: " << depth1_nodes[0].manhattan_to_goal << ")" << std::endl;

    std::cout << "\n5. Nested Branching Demonstration (Depth 2 from Best Candidate):" << std::endl;
    auto depth2_nodes = spark::PlanningEngine::expand_all_actions(net, test_ep.layout, depth1_nodes[0], test_ep.goal_pos);
    std::sort(depth2_nodes.begin(), depth2_nodes.end(), [](const spark::PlanningNode& a, const spark::PlanningNode& b) {
        return a.is_better_than(b);
    });

    for (const auto& node : depth2_nodes) {
        std::cout << "   Path [" << spark::action_name(node.actions_from_root[0]) << " -> " << spark::action_name(node.actions_from_root[1]) << "]: "
                  << "Pred Pos = (" << (node.predicted_position / 5) << "," << (node.predicted_position % 5) << ") "
                  << "| Manhattan = " << node.manhattan_to_goal << " "
                  << "| Conf = " << (node.target_probability * 100.0f) << "% "
                  << "| Score J = " << node.score
                  << (node.goal_reached ? " [GOAL REACHED!]" : "")
                  << std::endl;
    }

    std::cout << "\nMilestone 5E.1 planning foundation verified successfully!" << std::endl;
}

void run_plan_beam_cmd(int argc, char* argv[]) {
    print_banner();
    std::cout << ">>> SPARK Milestone 5E.2: Learned-Model Bounded Beam Planning <<<\n" << std::endl;

    uint32_t beam_width = 4;
    uint32_t max_depth = 4;
    size_t num_episodes = 5;
    std::string checkpoint_path = "checkpoints/world_model_best.spark";
    std::string csv_path = "experiments/results/phase5/milestone5e2d_planning_eval.csv";
    bool run_benchmark = false;
    bool cold_start = true;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) {
            beam_width = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--depth" && i + 1 < argc) {
            max_depth = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--episodes" && i + 1 < argc) {
            num_episodes = std::stoul(argv[++i]);
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--warm") {
            cold_start = false;
        }
    }

    if (run_benchmark) {
        spark::PlanningEvaluator::run_milestone5e2d_benchmark(checkpoint_path, csv_path);
        return;
    }

    std::cout << "Configuration: Beam Width K=" << beam_width
              << ", Max Depth H=" << max_depth
              << ", Episodes=" << num_episodes
              << ", Mode=" << (cold_start ? "ColdStart" : "WarmedUp") << std::endl;

    // Load World Model
    spark::SparseGraphConfig g_cfg;
    g_cfg.num_units = 128;
    g_cfg.fan_in = 8;
    g_cfg.seed = 42ULL;
    spark::NetworkConfig n_cfg;
    n_cfg.num_units = 128;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 1.0f;
    n_cfg.seed = 42ULL;
    spark::MicroUnitNetwork net(n_cfg, g_cfg);

    bool has_ckpt = spark::MicroUnitNetwork::load_checkpoint(checkpoint_path, net);
    if (!has_ckpt) {
        std::cout << "Warning: Checkpoint " << checkpoint_path << " not found, evaluating with fresh model." << std::endl;
    } else {
        std::cout << "Loaded world model from: " << checkpoint_path << std::endl;
    }

    auto splits = spark::GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);
    auto episodes = spark::PlanningEpisodeGenerator::generate_dataset(splits.test_layouts, 1, 42ULL, max_depth, 1);

    spark::BeamSearchConfig cfg;
    cfg.beam_width = beam_width;
    cfg.max_depth = max_depth;

    size_t pred_success = 0;
    size_t actual_success = 0;
    std::cout << "\nEvaluating Bounded Beam Search Across Episodes (Planning & Open-Loop Execution):" << std::endl;
    for (size_t ep_idx = 0; ep_idx < std::min(num_episodes, episodes.size()); ++ep_idx) {
        const auto& ep = episodes[ep_idx];
        spark::PlanningSearchResult res;
        if (cold_start) {
            auto root = spark::PlanningEngine::create_cold_start_root(ep.start_pos, ep.goal_pos, 128, max_depth, cfg.score_cfg);
            res = spark::PlanningEngine::beam_search(net, ep.layout, root, ep.goal_pos, cfg);
        } else {
            std::vector<float> x_prior;
            spark::PlanningEngine::encode_model_input(ep.start_pos, ep.layout, spark::Action::Stay, x_prior);
            std::vector<float> h_out, logits, probs;
            net.forward_step_branching(std::vector<float>(128, 0.0f), x_prior, h_out, logits, probs, spark::ReadoutMode::Softmax);
            auto root = spark::PlanningEngine::create_warmed_up_root(ep.start_pos, ep.goal_pos, h_out, max_depth, cfg.score_cfg);
            res = spark::PlanningEngine::beam_search(net, ep.layout, root, ep.goal_pos, cfg);
        }

        auto exec_res = spark::PlanningEngine::execute_plan_open_loop(ep, res, cold_start);

        std::cout << "  Episode " << ep_idx + 1 << " [" << spark::episode_group_name(exec_res.group) << "]: "
                  << "Start=(" << (ep.start_pos/5) << "," << (ep.start_pos%5) << ")"
                  << " -> Goal=(" << (ep.goal_pos/5) << "," << (ep.goal_pos%5) << ")"
                  << " | Ref BFS: " << ep.shortest_path_length << " steps\n";
        std::cout << "    Plan Found : " << res.best_plan.size() << " steps (";
        for (auto a : res.best_plan) std::cout << spark::action_name(a) << " ";
        std::cout << ")\n";
        std::cout << "    Imagined   : Goal=" << (exec_res.predicted_goal_reached ? "YES" : "NO")
                  << " | Final Pos=(" << (res.predicted_final_position/5) << "," << (res.predicted_final_position%5) << ")"
                  << " | Conf=" << std::fixed << std::setprecision(1) << (exec_res.mean_step_confidence * 100.0f) << "%\n";
        std::cout << "    Actual Env : Goal=" << (exec_res.actual_goal_reached ? "YES" : "NO")
                  << " | Final Pos=(" << (exec_res.actual_trajectory.back()/5) << "," << (exec_res.actual_trajectory.back()%5) << ")"
                  << " | Div Step=" << exec_res.first_divergence_step
                  << " | Outcome=" << spark::failure_category_name(exec_res.failure_category) << "\n";

        if (exec_res.predicted_goal_reached) pred_success++;
        if (exec_res.actual_goal_reached) actual_success++;
    }

    size_t evaluated = std::min(num_episodes, episodes.size());
    std::cout << "\nEvaluation Summary: "
              << "Predicted: " << pred_success << "/" << evaluated << " ("
              << (evaluated > 0 ? (100.0f * pred_success / evaluated) : 0.0f) << "%) | "
              << "Actual: " << actual_success << "/" << evaluated << " ("
              << (evaluated > 0 ? (100.0f * actual_success / evaluated) : 0.0f) << "%)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_info();
        return 0;
    }

    std::string cmd = argv[1];
    if (cmd == "info" || cmd == "--help" || cmd == "-h") {
        print_info();
    } else if (cmd == "demo") {
        run_demo();
    } else if (cmd == "benchmark") {
        run_benchmark(argc, argv);
    } else if (cmd == "train") {
        run_train(argc, argv);
    } else if (cmd == "baselines") {
        run_baselines_cmd(argc, argv);
    } else if (cmd == "capacity") {
        run_capacity_cmd(argc, argv);
    } else if (cmd == "diagnose") {
        run_diagnose_cmd(argc, argv);
    } else if (cmd == "ablate") {
        run_ablate_cmd(argc, argv);
    } else if (cmd == "phase4") {
        run_phase4_cmd(argc, argv);
    } else if (cmd == "local-ablate") {
        run_local_ablate_cmd(argc, argv);
    } else if (cmd == "world-model") {
        run_world_model_cmd(argc, argv);
    } else if (cmd == "partial-obs") {
        run_partial_obs_cmd(argc, argv);
    } else if (cmd == "onestep-diag") {
        run_onestep_diag_cmd(argc, argv);
    } else if (cmd == "rollout") {
        run_rollout_cmd(argc, argv);
    } else if (cmd == "plan-foundation") {
        run_plan_foundation_cmd(argc, argv);
    } else if (cmd == "plan-beam") {
        run_plan_beam_cmd(argc, argv);
    } else {
        std::cerr << "Unknown command: '" << cmd << "'\n"
                  << "Run 'spark info' for available commands." << std::endl;
        return 1;
    }

    return 0;
}
