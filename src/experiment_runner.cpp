#include "spark/experiment_runner.hpp"
#include "spark/metrics.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace spark {

std::string ExperimentRecord::csv_header() {
    return "run_id,task,mode_name,delay,seed,units,fan_in,alpha,beta,unroll,learning_rate,lambda,train_steps,train_examples,eval_examples,train_loss,validation_loss,test_loss,test_accuracy,training_seconds,model_bytes,training_memory_bytes";
}

std::string ExperimentRecord::to_csv() const {
    std::ostringstream ss;
    ss << run_id << ","
       << task << ","
       << mode_name << ","
       << delay << ","
       << seed << ","
       << units << ","
       << fan_in << ","
       << alpha << ","
       << beta << ","
       << unroll << ","
       << learning_rate << ","
       << lambda << ","
       << train_steps << ","
       << train_examples << ","
       << eval_examples << ","
       << std::fixed << std::setprecision(5) << train_loss << ","
       << validation_loss << ","
       << test_loss << ","
       << std::setprecision(4) << test_accuracy << ","
       << std::setprecision(4) << training_seconds << ","
       << model_bytes << ","
       << training_memory_bytes;
    return ss.str();
}

AggregatedStats ExperimentRunner::compute_aggregate(const std::vector<ExperimentRecord>& records, const std::string& name) {
    AggregatedStats stats;
    stats.config_name = name;
    stats.num_runs = records.size();
    if (records.empty()) return stats;

    stats.mode_name = records[0].mode_name;
    stats.delay = records[0].delay;
    stats.units = records[0].units;
    stats.min_accuracy = records[0].test_accuracy;
    stats.max_accuracy = records[0].test_accuracy;

    double acc_sum = 0.0;
    double loss_sum = 0.0;
    double time_sum = 0.0;
    double mem_sum = 0.0;

    for (const auto& r : records) {
        acc_sum += r.test_accuracy;
        loss_sum += r.test_loss;
        time_sum += r.training_seconds;
        mem_sum += static_cast<double>(r.training_memory_bytes) / 1024.0;
        if (r.test_accuracy < stats.min_accuracy) stats.min_accuracy = r.test_accuracy;
        if (r.test_accuracy > stats.max_accuracy) stats.max_accuracy = r.test_accuracy;
    }

    stats.mean_accuracy = acc_sum / stats.num_runs;
    stats.mean_eval_loss = loss_sum / stats.num_runs;
    stats.mean_train_time = time_sum / stats.num_runs;
    stats.mean_training_memory_kb = mem_sum / stats.num_runs;

    double acc_var = 0.0;
    for (const auto& r : records) {
        double diff = r.test_accuracy - stats.mean_accuracy;
        acc_var += diff * diff;
    }
    stats.std_accuracy = std::sqrt(acc_var / stats.num_runs);

    return stats;
}

void ExperimentRunner::compare_baselines(uint32_t delay, uint32_t units, uint64_t seed) {
    std::cout << "\n========================================================"
              << "\n  SPARK Baseline Comparison (Delay: " << delay << ", Units: " << units << ", Seed: " << seed << ")"
              << "\n========================================================\n" << std::endl;

    TemporalMemoryEnv train_env(delay, 200, seed);
    TemporalMemoryEnv eval_env(delay, 200, seed + 1000ULL);

    // 1. Shift-Register Oracle
    ShiftRegisterOracle oracle(delay);
    auto res_oracle = BaselineEvaluator::evaluate_oracle(oracle, eval_env);

    // 2. Constant Predictor
    ConstantPredictor const_pred;
    const_pred.fit(train_env, 1000);
    auto res_const = BaselineEvaluator::evaluate_constant(const_pred, eval_env);

    // 3. Memoryless Predictor
    MemorylessPredictor mem_pred;
    mem_pred.fit(train_env, 2000, 0.05f);
    auto res_memless = BaselineEvaluator::evaluate_memoryless(mem_pred, eval_env);

    // 4. Untrained SPARK
    SparseGraphConfig g_cfg;
    g_cfg.num_units = units;
    g_cfg.fan_in = 8;
    g_cfg.seed = seed;
    NetworkConfig n_cfg;
    n_cfg.num_units = units;
    n_cfg.input_dim = 1;
    n_cfg.output_dim = 1;
    n_cfg.seed = seed;

    MicroUnitNetwork untrained_net(n_cfg, g_cfg);
    auto res_untrained = BaselineEvaluator::evaluate_untrained_spark(untrained_net, eval_env);

    // 5. Trained SPARK (TBPTT)
    MicroUnitNetwork trained_net(n_cfg, g_cfg);
    Trainer trainer(trained_net, train_env, &eval_env, 0.03f, 1.0f);
    trainer.train(delay == 1 ? 1200 : 2000, 8, 200, true);
    auto res_trained = BaselineEvaluator::evaluate_trained_spark(trained_net, eval_env);

    std::vector<BaselineResult> results = {
        res_oracle,
        res_const,
        res_memless,
        res_untrained,
        res_trained
    };

    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Model / Predictor"
              << std::right << std::setw(12) << "Accuracy"
              << std::setw(14) << "Eval Loss"
              << std::setw(20) << "Correct / Total"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& r : results) {
        std::string counts = std::to_string(r.correct_steps) + " / " + std::to_string(r.evaluated_steps);
        std::cout << std::left << std::setw(30) << r.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(11) << (r.accuracy * 100.0f) << "%"
                  << std::setprecision(5) << std::setw(14) << r.loss
                  << std::setw(20) << counts
                  << std::endl;
    }
    std::cout << std::string(80, '-') << std::endl;
}

std::vector<ExperimentRecord> ExperimentRunner::run_capacity_suite(
    const std::vector<uint32_t>& delays,
    const std::vector<uint32_t>& unit_counts,
    size_t num_seeds,
    const std::string& csv_path
) {
    std::vector<ExperimentRecord> all_records;
    std::vector<AggregatedStats> all_stats;

    std::cout << "\n========================================================"
              << "\n  SPARK Temporal Memory Capacity Suite (" << num_seeds << " seeds/config)"
              << "\n========================================================\n" << std::endl;

    for (uint32_t delay : delays) {
        for (uint32_t units : unit_counts) {
            std::vector<ExperimentRecord> config_records;
            std::string config_name = "D=" + std::to_string(delay) + ", N=" + std::to_string(units);
            std::cout << "Testing Configuration: " << config_name << " across " << num_seeds << " seeds..." << std::flush;

            uint32_t steps = (delay <= 2) ? 1500 : 2500;
            float lr = (delay <= 2) ? 0.03f : 0.04f;

            for (size_t s_idx = 0; s_idx < num_seeds; ++s_idx) {
                uint64_t seed = 42ULL + s_idx * 17ULL;

                SparseGraphConfig g_cfg;
                g_cfg.num_units = units;
                g_cfg.fan_in = 8;
                g_cfg.seed = seed;

                NetworkConfig n_cfg;
                n_cfg.num_units = units;
                n_cfg.input_dim = 1;
                n_cfg.output_dim = 1;
                n_cfg.alpha = 0.8f;
                n_cfg.beta = 0.5f;
                n_cfg.seed = seed;

                MicroUnitNetwork net(n_cfg, g_cfg);
                TemporalMemoryEnv train_env(delay, 200, seed);
                TemporalMemoryEnv eval_env(delay, 200, seed + 10000ULL);

                Trainer trainer(net, train_env, &eval_env, lr, 1.0f);
                TrainSummary summary = trainer.train(steps, 8, 200, true);

                ExperimentRecord rec;
                rec.run_id = "cap_d" + std::to_string(delay) + "_u" + std::to_string(units) + "_s" + std::to_string(seed);
                rec.task = "delay-copy";
                rec.delay = delay;
                rec.seed = seed;
                rec.units = units;
                rec.fan_in = 8;
                rec.alpha = 0.8f;
                rec.beta = 0.5f;
                rec.unroll = 8;
                rec.learning_rate = lr;
                rec.train_steps = steps;
                rec.train_examples = static_cast<size_t>(steps) * 8;
                rec.eval_examples = 200 - delay;
                rec.train_loss = summary.final_train_loss;
                rec.validation_loss = summary.final_eval_loss;
                rec.test_loss = summary.final_eval_loss;
                rec.test_accuracy = summary.final_eval_accuracy;
                rec.training_seconds = summary.total_seconds;
                rec.model_bytes = net.memory_bytes();

                config_records.push_back(rec);
                all_records.push_back(rec);
            }

            auto agg = compute_aggregate(config_records, config_name);
            all_stats.push_back(agg);

            std::cout << " done! Mean Acc: " << std::fixed << std::setprecision(2)
                      << (agg.mean_accuracy * 100.0) << "% (+/- " << (agg.std_accuracy * 100.0) << "%)\n";
        }
    }

    // Print summary report
    std::cout << "\n" << std::string(85, '=') << "\n"
              << "  CAPACITY EXPERIMENT AGGREGATE SUMMARY\n"
              << std::string(85, '=') << "\n";
    std::cout << std::left << std::setw(16) << "Config"
              << std::right << std::setw(10) << "Runs"
              << std::setw(14) << "Mean Acc"
              << std::setw(12) << "Std Acc"
              << std::setw(12) << "Min Acc"
              << std::setw(12) << "Max Acc"
              << std::setw(16) << "Mean Loss"
              << std::setw(14) << "Mean Time"
              << "\n" << std::string(85, '-') << "\n";

    for (const auto& a : all_stats) {
        std::cout << std::left << std::setw(16) << a.config_name
                  << std::right << std::setw(10) << a.num_runs
                  << std::fixed << std::setprecision(2)
                  << std::setw(13) << (a.mean_accuracy * 100.0) << "%"
                  << std::setw(11) << (a.std_accuracy * 100.0) << "%"
                  << std::setw(11) << (a.min_accuracy * 100.0) << "%"
                  << std::setw(11) << (a.max_accuracy * 100.0) << "%"
                  << std::setprecision(4) << std::setw(16) << a.mean_eval_loss
                  << std::setprecision(2) << std::setw(13) << a.mean_train_time << "s"
                  << std::endl;
    }
    std::cout << std::string(85, '=') << std::endl;

    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << ExperimentRecord::csv_header() << "\n";
            for (const auto& r : all_records) {
                ofs << r.to_csv() << "\n";
            }
            std::cout << "Exported " << all_records.size() << " experimental runs to " << csv_path << std::endl;
        }
    }

    return all_records;
}

std::vector<ExperimentRecord> ExperimentRunner::run_delay3_diagnostic_suite(const std::string& csv_path) {
    std::cout << "\n========================================================"
              << "\n  SPARK Delay-3 Diagnostic Factorial Suite"
              << "\n========================================================\n" << std::endl;

    std::vector<ExperimentRecord> records;

    struct Variation {
        std::string factor;
        std::string label;
        uint32_t units = 128;
        uint32_t fan_in = 8;
        float alpha = 0.8f;
        float beta = 0.5f;
        uint32_t unroll = 8;
        float lr = 0.03f;
        uint32_t steps = 2000;
    };

    std::vector<Variation> trials = {
        // Baseline
        {"Baseline", "Baseline (N=128, K=8, a=0.8, b=0.5, T=8, lr=0.03, S=2000)", 128, 8, 0.8f, 0.5f, 8, 0.03f, 2000},

        // 1. Alpha retention
        {"Alpha", "Alpha = 0.5", 128, 8, 0.5f, 0.5f, 8, 0.03f, 2000},
        {"Alpha", "Alpha = 0.9", 128, 8, 0.9f, 0.5f, 8, 0.03f, 2000},
        {"Alpha", "Alpha = 0.95", 128, 8, 0.95f, 0.5f, 8, 0.03f, 2000},

        // 2. Beta scale
        {"Beta", "Beta = 0.2", 128, 8, 0.8f, 0.2f, 8, 0.03f, 2000},
        {"Beta", "Beta = 1.0", 128, 8, 0.8f, 1.0f, 8, 0.03f, 2000},
        {"Beta", "Beta = 1.5", 128, 8, 0.8f, 1.5f, 8, 0.03f, 2000},

        // 3. Unroll window T
        {"Unroll", "Unroll T = 4", 128, 8, 0.8f, 0.5f, 4, 0.03f, 2000},
        {"Unroll", "Unroll T = 12", 128, 8, 0.8f, 0.5f, 12, 0.03f, 2000},
        {"Unroll", "Unroll T = 16", 128, 8, 0.8f, 0.5f, 16, 0.03f, 2000},

        // 4. Learning rate
        {"LearningRate", "LR = 0.01", 128, 8, 0.8f, 0.5f, 8, 0.01f, 2000},
        {"LearningRate", "LR = 0.05", 128, 8, 0.8f, 0.5f, 8, 0.05f, 2000},
        {"LearningRate", "LR = 0.10", 128, 8, 0.8f, 0.5f, 8, 0.10f, 2000},

        // 5. Units
        {"Units", "Units N = 64", 64, 8, 0.8f, 0.5f, 8, 0.03f, 2000},
        {"Units", "Units N = 256", 256, 8, 0.8f, 0.5f, 8, 0.03f, 2000},

        // 6. Fan-in
        {"FanIn", "Fan-in K = 4", 128, 4, 0.8f, 0.5f, 8, 0.03f, 2000},
        {"FanIn", "Fan-in K = 16", 128, 16, 0.8f, 0.5f, 8, 0.03f, 2000},

        // 7. Training Steps
        {"Steps", "Steps S = 1000", 128, 8, 0.8f, 0.5f, 8, 0.03f, 1000},
        {"Steps", "Steps S = 4000", 128, 8, 0.8f, 0.5f, 8, 0.03f, 4000},
        {"Steps", "Steps S = 6000", 128, 8, 0.8f, 0.5f, 8, 0.03f, 6000}
    };

    std::cout << std::string(85, '-') << "\n"
              << std::left << std::setw(15) << "Factor"
              << std::setw(38) << "Condition"
              << std::right << std::setw(14) << "Test Acc"
              << std::setw(14) << "Test Loss"
              << "\n" << std::string(85, '-') << "\n";

    for (const auto& tr : trials) {
        uint64_t seed = 42ULL;
        SparseGraphConfig g_cfg;
        g_cfg.num_units = tr.units;
        g_cfg.fan_in = tr.fan_in;
        g_cfg.seed = seed;

        NetworkConfig n_cfg;
        n_cfg.num_units = tr.units;
        n_cfg.input_dim = 1;
        n_cfg.output_dim = 1;
        n_cfg.alpha = tr.alpha;
        n_cfg.beta = tr.beta;
        n_cfg.seed = seed;

        MicroUnitNetwork net(n_cfg, g_cfg);
        TemporalMemoryEnv train_env(3, 200, seed);
        TemporalMemoryEnv eval_env(3, 200, seed + 1000ULL);

        Trainer trainer(net, train_env, &eval_env, tr.lr, 1.0f);
        TrainSummary summary = trainer.train(tr.steps, tr.unroll, 200, true);

        ExperimentRecord rec;
        rec.run_id = "diag_d3_" + tr.factor + "_" + std::to_string(tr.steps);
        rec.task = "delay-copy";
        rec.delay = 3;
        rec.seed = seed;
        rec.units = tr.units;
        rec.fan_in = tr.fan_in;
        rec.alpha = tr.alpha;
        rec.beta = tr.beta;
        rec.unroll = tr.unroll;
        rec.learning_rate = tr.lr;
        rec.train_steps = tr.steps;
        rec.train_examples = static_cast<size_t>(tr.steps) * tr.unroll;
        rec.eval_examples = 197;
        rec.train_loss = summary.final_train_loss;
        rec.validation_loss = summary.final_eval_loss;
        rec.test_loss = summary.final_eval_loss;
        rec.test_accuracy = summary.final_eval_accuracy;
        rec.training_seconds = summary.total_seconds;
        rec.model_bytes = net.memory_bytes();
        records.push_back(rec);

        std::cout << std::left << std::setw(15) << tr.factor
                  << std::setw(38) << tr.label
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(13) << (rec.test_accuracy * 100.0f) << "%"
                  << std::setprecision(5) << std::setw(14) << rec.test_loss
                  << std::endl;
    }
    std::cout << std::string(85, '-') << std::endl;

    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << ExperimentRecord::csv_header() << "\n";
            for (const auto& r : records) {
                ofs << r.to_csv() << "\n";
            }
            std::cout << "Saved diagnostic records to " << csv_path << std::endl;
        }
    }

    return records;
}

void ExperimentRunner::run_inference_ablations(uint32_t delay, uint32_t units, uint64_t seed) {
    std::cout << "\n========================================================"
              << "\n  SPARK Inference-Time Ablation Suite (Delay: " << delay << ")"
              << "\n========================================================\n" << std::endl;

    SparseGraphConfig g_cfg;
    g_cfg.num_units = units;
    g_cfg.fan_in = 8;
    g_cfg.seed = seed;

    NetworkConfig n_cfg;
    n_cfg.num_units = units;
    n_cfg.input_dim = 1;
    n_cfg.output_dim = 1;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 0.5f;
    n_cfg.seed = seed;

    MicroUnitNetwork base_net(n_cfg, g_cfg);
    TemporalMemoryEnv train_env(delay, 200, seed);
    TemporalMemoryEnv eval_env(delay, 200, seed + 1000ULL);

    std::cout << "Training reference network..." << std::flush;
    Trainer trainer(base_net, train_env, &eval_env, 0.03f, 1.0f);
    trainer.train(delay == 1 ? 1200 : 2500, 8, 200, true);
    std::cout << " done.\n" << std::endl;

    auto eval_model = [&](MicroUnitNetwork& net, bool reset_each_step, const std::string& name) {
        eval_env.reset();
        net.reset_state();

        size_t valid_count = 0;
        size_t correct_bits = 0;
        double total_loss = 0.0;
        std::vector<float> logits, probs;

        for (uint32_t s = 0; s < 500 && !eval_env.is_done(); ++s) {
            if (reset_each_step) {
                net.reset_state(); // Sever historical memory across steps
            }
            Observation obs = eval_env.get_observation();
            Target target = eval_env.get_target();

            net.forward_step(obs.values, logits, probs, ReadoutMode::Sigmoid);
            eval_env.step();

            if (target.valid) {
                valid_count++;
                float pred_bit = (probs[0] >= 0.5f) ? 1.0f : 0.0f;
                if (std::abs(pred_bit - target.values[0]) < 1e-3f) {
                    correct_bits++;
                }
                total_loss += TBPTTTrainer::bce_loss(logits[0], target.values[0]);
            }
        }

        float acc = (valid_count > 0) ? (static_cast<float>(correct_bits) / valid_count) : 0.0f;
        float loss = (valid_count > 0) ? static_cast<float>(total_loss / valid_count) : 0.0f;

        std::cout << std::left << std::setw(42) << name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << (acc * 100.0f) << "%"
                  << std::setprecision(5) << std::setw(14) << loss
                  << std::endl;
    };

    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::left << std::setw(42) << "Ablation Condition"
              << std::right << std::setw(12) << "Accuracy"
              << std::setw(14) << "Eval Loss"
              << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    // 1. Intact Trained Model
    eval_model(base_net, false, "1. Intact Model (Baseline)");

    // 2. Hidden-state reset before every tick
    eval_model(base_net, true, "2. State Reset Every Step");

    // 3. Zero recurrent connection weights (W = 0)
    {
        MicroUnitNetwork net_zero_w = base_net;
        std::fill(net_zero_w.graph().weights.begin(), net_zero_w.graph().weights.end(), 0.0f);
        eval_model(net_zero_w, false, "3. Zero Recurrent Graph (W = 0)");
    }

    // 4. Disable recurrent state retention (alpha = 0)
    // Note: alpha is in config, so let's zero prev_state effect by setting alpha in config or copy
    // We can test alpha = 0 by creating a network with alpha=0 and copying weights!
    {
        NetworkConfig n_cfg_no_alpha = n_cfg;
        n_cfg_no_alpha.alpha = 0.0f;
        MicroUnitNetwork net_no_alpha(n_cfg_no_alpha, g_cfg);
        net_no_alpha.graph().weights = base_net.graph().weights;
        net_no_alpha.input_weights() = base_net.input_weights();
        net_no_alpha.output_weights() = base_net.output_weights();
        net_no_alpha.biases() = base_net.biases();
        net_no_alpha.output_biases() = base_net.output_biases();
        eval_model(net_no_alpha, false, "4. Zero Self-Retention (alpha = 0)");
    }

    // 5. Zero both recurrent weights and retention (Pure Feedforward)
    {
        NetworkConfig n_cfg_pure_ff = n_cfg;
        n_cfg_pure_ff.alpha = 0.0f;
        MicroUnitNetwork net_pure_ff(n_cfg_pure_ff, g_cfg);
        std::fill(net_pure_ff.graph().weights.begin(), net_pure_ff.graph().weights.end(), 0.0f);
        net_pure_ff.input_weights() = base_net.input_weights();
        net_pure_ff.output_weights() = base_net.output_weights();
        net_pure_ff.biases() = base_net.biases();
        net_pure_ff.output_biases() = base_net.output_biases();
        eval_model(net_pure_ff, false, "5. Pure Feedforward (W = 0, alpha = 0)");
    }

    // 6. Randomized Recurrent Weights (Shuffled / Scrambled)
    {
        MicroUnitNetwork net_rand_w = base_net;
        Random rng(999ULL);
        for (float& w : net_rand_w.graph().weights) {
            w = rng.normal_float(0.0f, 0.5f);
        }
        eval_model(net_rand_w, false, "6. Scrambled Recurrent Weights");
    }

    std::cout << std::string(75, '-') << std::endl;
}

std::vector<ExperimentRecord> ExperimentRunner::run_phase4_suite(
    const std::vector<uint32_t>& delays,
    const std::vector<uint32_t>& unit_counts,
    const std::vector<LearningAlgorithmMode>& modes,
    size_t num_seeds,
    const std::string& csv_path
) {
    std::vector<ExperimentRecord> all_records;
    std::vector<AggregatedStats> all_stats;

    std::cout << "\n================================================================================"
              << "\n  SPARK Phase 4: Multi-Mode Learning Comparison (" << num_seeds << " seeds/config)"
              << "\n================================================================================\n" << std::endl;

    const uint32_t total_budget_bits = 12000;
    const uint32_t eval_bits = 10000;

    for (auto mode : modes) {
        std::string mode_str = learning_algorithm_name(mode);

        for (uint32_t delay : delays) {
            for (uint32_t units : unit_counts) {
                std::vector<ExperimentRecord> config_records;
                std::string config_name = mode_str + " (D=" + std::to_string(delay) + ", N=" + std::to_string(units) + ")";
                std::cout << "Testing: " << std::left << std::setw(38) << config_name
                          << " across " << num_seeds << " seeds..." << std::flush;

                for (size_t s_idx = 0; s_idx < num_seeds; ++s_idx) {
                    uint64_t seed = 42ULL + s_idx * 17ULL;

                    SparseGraphConfig g_cfg;
                    g_cfg.num_units = units;
                    g_cfg.fan_in = 8;
                    g_cfg.seed = seed;

                    NetworkConfig n_cfg;
                    n_cfg.num_units = units;
                    n_cfg.input_dim = 1;
                    n_cfg.output_dim = 1;
                    n_cfg.alpha = 0.8f;
                    n_cfg.beta = 1.2f;
                    n_cfg.seed = seed;

                    MicroUnitNetwork net(n_cfg, g_cfg);

                    TemporalMemoryEnv train_env(delay, 200, seed);
                    TemporalMemoryEnv eval_env(delay, eval_bits + delay, seed + 100000ULL);

                    ExperimentRecord rec;
                    rec.run_id = "p4_" + mode_str + "_d" + std::to_string(delay) + "_u" + std::to_string(units) + "_s" + std::to_string(seed);
                    rec.task = "delay-copy";
                    rec.mode_name = mode_str;
                    rec.delay = delay;
                    rec.seed = seed;
                    rec.units = units;
                    rec.fan_in = 8;
                    rec.alpha = 0.8f;
                    rec.beta = 1.2f;
                    rec.train_examples = total_budget_bits;
                    rec.learning_rate = 0.03f;
                    rec.lambda = 0.8f;
                    rec.model_bytes = net.memory_bytes();

                    if (mode == LearningAlgorithmMode::ExactTBPTT) {
                        uint32_t unroll = 8;
                        uint32_t steps = total_budget_bits / unroll;
                        rec.unroll = unroll;
                        rec.train_steps = steps;

                        Trainer trainer(net, train_env, nullptr, 0.03f, 1.0f);
                        TrainSummary summary = trainer.train(steps, unroll, 0, true);

                        EvalMetrics ev = trainer.evaluate(eval_env, eval_env.sequence_length());
                        rec.eval_examples = ev.evaluated_steps;
                        rec.train_loss = summary.final_train_loss;
                        rec.validation_loss = summary.final_eval_loss;
                        rec.test_loss = ev.loss;
                        rec.test_accuracy = ev.accuracy;
                        rec.training_seconds = summary.total_seconds;

                        size_t grads_bytes = sizeof(Gradients) +
                            (net.graph().weights.size() + net.input_weights().size() +
                             net.biases().size() + net.output_weights().size() +
                             net.output_biases().size()) * sizeof(float);
                        size_t unroll_bytes = unroll * (sizeof(StepRecord) + units * 4 * sizeof(float));
                        rec.training_memory_bytes = net.memory_bytes() + grads_bytes + unroll_bytes;
                    } else {
                        rec.unroll = 1;
                        rec.train_steps = total_budget_bits;

                        LocalLearningConfig l_cfg;
                        l_cfg.lr = 0.03f;
                        l_cfg.lambda = 0.8f;
                        l_cfg.feedback_scale = 1.0f;
                        l_cfg.feedback_seed = seed + 50000ULL;

                        if (mode == LearningAlgorithmMode::LocalRandomFeedback) {
                            l_cfg.feedback_mode = FeedbackMode::RandomFeedback;
                            l_cfg.train_recurrent = true;
                            l_cfg.train_input = true;
                            l_cfg.train_readout = true;
                        } else if (mode == LearningAlgorithmMode::LocalReadoutTranspose) {
                            l_cfg.feedback_mode = FeedbackMode::ReadoutTranspose;
                            l_cfg.train_recurrent = true;
                            l_cfg.train_input = true;
                            l_cfg.train_readout = true;
                        } else { // ReadoutOnly
                            l_cfg.feedback_mode = FeedbackMode::ReadoutTranspose;
                            l_cfg.train_recurrent = false;
                            l_cfg.train_input = false;
                            l_cfg.train_readout = true;
                        }

                        LocalLearner learner(net, l_cfg);
                        TrainSummary summary = learner.train(train_env, nullptr, total_budget_bits, 0, true);

                        Trainer eval_runner(net, train_env);
                        EvalMetrics ev = eval_runner.evaluate(eval_env, eval_env.sequence_length());

                        rec.eval_examples = ev.evaluated_steps;
                        rec.train_loss = summary.final_train_loss;
                        rec.validation_loss = summary.final_eval_loss;
                        rec.test_loss = ev.loss;
                        rec.test_accuracy = ev.accuracy;
                        rec.training_seconds = summary.total_seconds;
                        rec.training_memory_bytes = net.memory_bytes() + learner.training_memory_bytes();
                    }

                    config_records.push_back(rec);
                    all_records.push_back(rec);
                }

                auto agg = compute_aggregate(config_records, config_name);
                all_stats.push_back(agg);

                std::cout << " done! Mean Acc: " << std::fixed << std::setprecision(2)
                          << (agg.mean_accuracy * 100.0) << "% (+/- " << (agg.std_accuracy * 100.0)
                          << "%), Time: " << std::setprecision(2) << agg.mean_train_time << "s\n";
            }
        }
    }

    std::cout << "\n" << std::string(115, '=') << "\n"
              << "  PHASE 4 MULTI-MODE AGGREGATE RESULTS\n"
              << std::string(115, '=') << "\n";
    std::cout << std::left << std::setw(36) << "Configuration"
              << std::right << std::setw(6) << "Runs"
              << std::setw(12) << "Mean Acc"
              << std::setw(11) << "Std Acc"
              << std::setw(11) << "Min Acc"
              << std::setw(11) << "Max Acc"
              << std::setw(13) << "Mean Loss"
              << std::setw(11) << "Time (s)"
              << std::setw(11) << "Mem (KB)"
              << "\n" << std::string(115, '-') << "\n";

    for (const auto& a : all_stats) {
        std::cout << std::left << std::setw(36) << a.config_name
                  << std::right << std::setw(6) << a.num_runs
                  << std::fixed << std::setprecision(2)
                  << std::setw(11) << (a.mean_accuracy * 100.0) << "%"
                  << std::setw(10) << (a.std_accuracy * 100.0) << "%"
                  << std::setw(10) << (a.min_accuracy * 100.0) << "%"
                  << std::setw(10) << (a.max_accuracy * 100.0) << "%"
                  << std::setprecision(4) << std::setw(13) << a.mean_eval_loss
                  << std::setprecision(3) << std::setw(11) << a.mean_train_time
                  << std::setprecision(1) << std::setw(11) << a.mean_training_memory_kb
                  << std::endl;
    }
    std::cout << std::string(115, '=') << std::endl;

    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << ExperimentRecord::csv_header() << "\n";
            for (const auto& r : all_records) {
                ofs << r.to_csv() << "\n";
            }
            std::cout << "Exported " << all_records.size() << " experimental runs to " << csv_path << std::endl;
        }
    }

    return all_records;
}

std::vector<ExperimentRecord> ExperimentRunner::run_local_ablation_suite(
    uint32_t delay,
    uint32_t units,
    size_t num_seeds,
    const std::string& csv_path
) {
    struct AblationConfig {
        std::string name;
        FeedbackMode feedback_mode;
        bool train_recurrent;
        bool train_input;
        bool train_readout;
        float lambda;
    };

    std::vector<AblationConfig> ablations = {
        {"1. Intact Local (Mode C)", FeedbackMode::ReadoutTranspose, true, true, true, 0.8f},
        {"2. Freeze Recurrent W", FeedbackMode::ReadoutTranspose, false, true, true, 0.8f},
        {"3. Freeze Input Proj U", FeedbackMode::ReadoutTranspose, true, false, true, 0.8f},
        {"4. No Traces (lambda=0)", FeedbackMode::ReadoutTranspose, true, true, true, 0.0f},
        {"5. Readout Only (Mode D)", FeedbackMode::ReadoutTranspose, false, false, true, 0.8f},
        {"6. Random Feedback (Mode B)", FeedbackMode::RandomFeedback, true, true, true, 0.8f}
    };

    std::vector<ExperimentRecord> all_records;
    std::vector<AggregatedStats> all_stats;

    std::cout << "\n================================================================================"
              << "\n  SPARK Phase 4: Local Representation & Freezing Ablations"
              << "\n  (Delay=" << delay << ", Units=" << units << ", " << num_seeds << " seeds/condition)"
              << "\n================================================================================\n" << std::endl;

    const uint32_t total_budget_bits = 12000;
    const uint32_t eval_bits = 10000;

    for (const auto& ab : ablations) {
        std::vector<ExperimentRecord> cond_records;
        std::cout << "Testing: " << std::left << std::setw(32) << ab.name
                  << " across " << num_seeds << " seeds..." << std::flush;

        for (size_t s_idx = 0; s_idx < num_seeds; ++s_idx) {
            uint64_t seed = 42ULL + s_idx * 17ULL;

            SparseGraphConfig g_cfg;
            g_cfg.num_units = units;
            g_cfg.fan_in = 8;
            g_cfg.seed = seed;

            NetworkConfig n_cfg;
            n_cfg.num_units = units;
            n_cfg.input_dim = 1;
            n_cfg.output_dim = 1;
            n_cfg.alpha = 0.8f;
            n_cfg.beta = 1.2f;
            n_cfg.seed = seed;

            MicroUnitNetwork net(n_cfg, g_cfg);

            TemporalMemoryEnv train_env(delay, 200, seed);
            TemporalMemoryEnv eval_env(delay, eval_bits + delay, seed + 100000ULL);

            LocalLearningConfig l_cfg;
            l_cfg.lr = 0.03f;
            l_cfg.lambda = ab.lambda;
            l_cfg.feedback_mode = ab.feedback_mode;
            l_cfg.feedback_scale = 1.0f;
            l_cfg.feedback_seed = seed + 50000ULL;
            l_cfg.train_recurrent = ab.train_recurrent;
            l_cfg.train_input = ab.train_input;
            l_cfg.train_readout = ab.train_readout;

            LocalLearner learner(net, l_cfg);
            TrainSummary summary = learner.train(train_env, nullptr, total_budget_bits, 0, true);

            Trainer eval_runner(net, train_env);
            EvalMetrics ev = eval_runner.evaluate(eval_env, eval_env.sequence_length());

            ExperimentRecord rec;
            rec.run_id = "ablate_" + ab.name + "_s" + std::to_string(seed);
            rec.task = "delay-copy";
            rec.mode_name = ab.name;
            rec.delay = delay;
            rec.seed = seed;
            rec.units = units;
            rec.fan_in = 8;
            rec.alpha = 0.8f;
            rec.beta = 1.2f;
            rec.unroll = 1;
            rec.learning_rate = 0.03f;
            rec.lambda = ab.lambda;
            rec.train_steps = total_budget_bits;
            rec.train_examples = total_budget_bits;
            rec.eval_examples = ev.evaluated_steps;
            rec.train_loss = summary.final_train_loss;
            rec.validation_loss = summary.final_eval_loss;
            rec.test_loss = ev.loss;
            rec.test_accuracy = ev.accuracy;
            rec.training_seconds = summary.total_seconds;
            rec.model_bytes = net.memory_bytes();
            rec.training_memory_bytes = net.memory_bytes() + learner.training_memory_bytes();

            cond_records.push_back(rec);
            all_records.push_back(rec);
        }

        auto agg = compute_aggregate(cond_records, ab.name);
        all_stats.push_back(agg);

        std::cout << " done! Mean Acc: " << std::fixed << std::setprecision(2)
                  << (agg.mean_accuracy * 100.0) << "% (+/- " << (agg.std_accuracy * 100.0) << "%)\n";
    }

    std::cout << "\n" << std::string(115, '=') << "\n"
              << "  PHASE 4 LOCAL ABLATIONS AGGREGATE RESULTS\n"
              << std::string(115, '=') << "\n";
    std::cout << std::left << std::setw(32) << "Ablation Condition"
              << std::right << std::setw(6) << "Runs"
              << std::setw(12) << "Mean Acc"
              << std::setw(11) << "Std Acc"
              << std::setw(11) << "Min Acc"
              << std::setw(11) << "Max Acc"
              << std::setw(13) << "Mean Loss"
              << std::setw(11) << "Time (s)"
              << std::setw(11) << "Mem (KB)"
              << "\n" << std::string(115, '-') << "\n";

    for (const auto& a : all_stats) {
        std::cout << std::left << std::setw(32) << a.config_name
                  << std::right << std::setw(6) << a.num_runs
                  << std::fixed << std::setprecision(2)
                  << std::setw(11) << (a.mean_accuracy * 100.0) << "%"
                  << std::setw(10) << (a.std_accuracy * 100.0) << "%"
                  << std::setw(10) << (a.min_accuracy * 100.0) << "%"
                  << std::setw(10) << (a.max_accuracy * 100.0) << "%"
                  << std::setprecision(4) << std::setw(13) << a.mean_eval_loss
                  << std::setprecision(3) << std::setw(11) << a.mean_train_time
                  << std::setprecision(1) << std::setw(11) << a.mean_training_memory_kb
                  << std::endl;
    }
    std::cout << std::string(115, '=') << std::endl;

    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << ExperimentRecord::csv_header() << "\n";
            for (const auto& r : all_records) {
                ofs << r.to_csv() << "\n";
            }
            std::cout << "Exported " << all_records.size() << " experimental runs to " << csv_path << std::endl;
        }
    }

    return all_records;
}

std::vector<Milestone5bRecord> ExperimentRunner::run_milestone5b_suite(
    size_t num_seeds,
    uint32_t train_steps,
    uint32_t unroll_steps,
    float lr,
    uint32_t test_transitions,
    const std::string& csv_path
) {
    std::vector<Milestone5bRecord> all_records;

    std::cout << "\n================================================================================"
              << "\n  SPARK Phase 5 Milestone 5B: World-Model Transition Learning Suite"
              << "\n  (100 Train Layouts, 20 Val Layouts, 20 Unseen Test Layouts)"
              << "\n  (" << num_seeds << " Seeds, " << test_transitions << " Held-Out Test Transitions / Seed)"
              << "\n================================================================================\n" << std::endl;

    // 1. Generate strictly disjoint layout dataset splits
    std::cout << "Generating dataset splits with 0-overlap layout deduplication..." << std::flush;
    auto splits = GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 42ULL);
    std::cout << " done! (" << splits.train_layouts.size() << " train, "
              << splits.val_layouts.size() << " val, "
              << splits.test_layouts.size() << " test)\n" << std::endl;

    // Compute average obstacle density on test set
    double test_obs_sum = 0.0;
    for (const auto& l : splits.test_layouts) {
        test_obs_sum += static_cast<double>(l.obstacle_count()) / GridWorldEnv::NUM_CELLS;
    }
    float avg_test_density = static_cast<float>(test_obs_sum / splits.test_layouts.size());

    // 2. Evaluate Non-Learning Baselines on Unseen Test Layouts
    std::cout << ">>> Evaluating Non-Learning Baselines on Unseen Test Layouts (" << test_transitions << " transitions) <<<" << std::endl;
    GridWorldEnv baseline_env(splits.test_layouts, 50, 42ULL);
    auto baselines = evaluate_all_baselines(baseline_env, test_transitions, 42ULL);

    std::cout << std::string(105, '-') << "\n"
              << std::left << std::setw(30) << "Model / Predictor"
              << std::right << std::setw(12) << "Overall Acc"
              << std::setw(12) << "Succ Move"
              << std::setw(12) << "Blocked"
              << std::setw(12) << "Boundary"
              << std::setw(12) << "Obstacle"
              << std::setw(12) << "Mean Loss"
              << "\n" << std::string(105, '-') << std::endl;

    baselines.random_predictor.print_summary();
    baselines.copy_current_predictor.print_summary();
    baselines.naive_movement_predictor.print_summary();
    baselines.ground_truth_oracle.print_summary();

    // Record baselines
    auto record_baseline = [&](const WorldModelMetrics& m, const std::string& name) {
        Milestone5bRecord rec;
        rec.experiment_id = "baseline_" + name;
        rec.seed = 42;
        rec.task = "gridworld-transition";
        rec.trainer = "none";
        rec.units = 0;
        rec.fan_in = 0;
        rec.training_examples = 0;
        rec.test_examples = test_transitions;
        rec.obstacle_density = avg_test_density;
        rec.visibility_mode = "fully_observable";
        rec.rollout_horizon = 1;
        rec.train_loss = 0.0f;
        rec.validation_loss = 0.0f;
        rec.test_loss = m.mean_loss();
        rec.next_state_accuracy = m.overall_accuracy();
        rec.successful_move_acc = m.successful_move_accuracy();
        rec.blocked_move_acc = m.blocked_move_accuracy();
        rec.boundary_collision_acc = m.boundary_block_accuracy();
        rec.obstacle_collision_acc = m.obstacle_block_accuracy();
        rec.training_seconds = 0.0;
        rec.inference_seconds = m.inference_seconds;
        rec.model_bytes = 0;
        all_records.push_back(rec);
    };

    record_baseline(baselines.random_predictor, "random");
    record_baseline(baselines.copy_current_predictor, "copy_current");
    record_baseline(baselines.naive_movement_predictor, "naive_movement");
    record_baseline(baselines.ground_truth_oracle, "ground_truth_oracle");

    std::cout << std::string(105, '-') << "\n" << std::endl;

    // 3. Train and Evaluate Readout-Only SPARK (Baseline 5)
    std::cout << ">>> Training Readout-Only SPARK (Frozen Recurrent Graph & Input Weights) <<<" << std::endl;
    {
        uint64_t seed = 42ULL;
        SparseGraphConfig g_cfg;
        g_cfg.num_units = 128;
        g_cfg.fan_in = 8;
        g_cfg.seed = seed;

        NetworkConfig n_cfg;
        n_cfg.num_units = 128;
        n_cfg.input_dim = 55;
        n_cfg.output_dim = 25;
        n_cfg.alpha = 0.8f;
        n_cfg.beta = 1.0f;
        n_cfg.seed = seed;

        MicroUnitNetwork net(n_cfg, g_cfg);
        GridWorldEnv train_env(splits.train_layouts, 50, seed);
        GridWorldEnv val_env(splits.val_layouts, 50, seed + 1000ULL);
        GridWorldEnv test_env(splits.test_layouts, 50, seed + 2000ULL);

        Trainer trainer(net, train_env, &val_env, lr, 1.0f);
        trainer.set_loss_type(LossType::CategoricalCrossEntropy);
        trainer.set_readout_only(true);

        TrainSummary summary = trainer.train(train_steps, unroll_steps, 0, true);
        WorldModelMetrics m = evaluate_world_model(net, test_env, test_transitions);
        m.model_name = "5. Readout-Only SPARK";
        m.print_summary();

        Milestone5bRecord rec;
        rec.experiment_id = "readout_only_s" + std::to_string(seed);
        rec.seed = seed;
        rec.task = "gridworld-transition";
        rec.trainer = "readout_only";
        rec.units = 128;
        rec.fan_in = 8;
        rec.training_examples = static_cast<size_t>(train_steps) * unroll_steps;
        rec.test_examples = test_transitions;
        rec.obstacle_density = avg_test_density;
        rec.visibility_mode = "fully_observable";
        rec.rollout_horizon = 1;
        rec.train_loss = summary.final_train_loss;
        rec.validation_loss = summary.final_eval_loss;
        rec.test_loss = m.mean_loss();
        rec.next_state_accuracy = m.overall_accuracy();
        rec.successful_move_acc = m.successful_move_accuracy();
        rec.blocked_move_acc = m.blocked_move_accuracy();
        rec.boundary_collision_acc = m.boundary_block_accuracy();
        rec.obstacle_collision_acc = m.obstacle_block_accuracy();
        rec.training_seconds = summary.total_seconds;
        rec.inference_seconds = m.inference_seconds;
        rec.model_bytes = net.memory_bytes();
        all_records.push_back(rec);
    }
    std::cout << std::string(105, '-') << "\n" << std::endl;

    // 4. Train and Evaluate SPARK with Exact TBPTT Across Multiple Seeds
    std::cout << ">>> Training SPARK World Model (Exact TBPTT) Across " << num_seeds << " Seeds <<<" << std::endl;
    std::cout << std::string(105, '-') << "\n"
              << std::left << std::setw(30) << "Seed / Run ID"
              << std::right << std::setw(12) << "Overall Acc"
              << std::setw(12) << "Succ Move"
              << std::setw(12) << "Blocked"
              << std::setw(12) << "Boundary"
              << std::setw(12) << "Obstacle"
              << std::setw(12) << "Test Loss"
              << "\n" << std::string(105, '-') << std::endl;

    double acc_sum = 0.0;
    double succ_sum = 0.0;
    double block_sum = 0.0;
    double bound_sum = 0.0;
    double obs_sum = 0.0;
    double loss_sum = 0.0;

    for (size_t s = 0; s < num_seeds; ++s) {
        uint64_t seed = 42ULL + s * 17ULL;

        SparseGraphConfig g_cfg;
        g_cfg.num_units = 128;
        g_cfg.fan_in = 8;
        g_cfg.seed = seed;

        NetworkConfig n_cfg;
        n_cfg.num_units = 128;
        n_cfg.input_dim = 55;
        n_cfg.output_dim = 25;
        n_cfg.alpha = 0.8f;
        n_cfg.beta = 1.0f;
        n_cfg.seed = seed;

        MicroUnitNetwork net(n_cfg, g_cfg);
        GridWorldEnv train_env(splits.train_layouts, 50, seed);
        GridWorldEnv val_env(splits.val_layouts, 50, seed + 1000ULL);
        GridWorldEnv test_env(splits.test_layouts, 50, seed + 2000ULL);

        Trainer trainer(net, train_env, &val_env, lr, 1.0f);
        trainer.set_loss_type(LossType::CategoricalCrossEntropy);

        TrainSummary summary = trainer.train(train_steps, unroll_steps, 0, true);
        WorldModelMetrics m = evaluate_world_model(net, test_env, test_transitions);
        m.model_name = "SPARK_TBPTT (Seed " + std::to_string(seed) + ")";
        m.print_summary();

        acc_sum += m.overall_accuracy();
        succ_sum += m.successful_move_accuracy();
        block_sum += m.blocked_move_accuracy();
        bound_sum += m.boundary_block_accuracy();
        obs_sum += m.obstacle_block_accuracy();
        loss_sum += m.mean_loss();

        Milestone5bRecord rec;
        rec.experiment_id = "spark_tbptt_s" + std::to_string(seed);
        rec.seed = seed;
        rec.task = "gridworld-transition";
        rec.trainer = "exact_tbptt";
        rec.units = 128;
        rec.fan_in = 8;
        rec.training_examples = static_cast<size_t>(train_steps) * unroll_steps;
        rec.test_examples = test_transitions;
        rec.obstacle_density = avg_test_density;
        rec.visibility_mode = "fully_observable";
        rec.rollout_horizon = 1;
        rec.train_loss = summary.final_train_loss;
        rec.validation_loss = summary.final_eval_loss;
        rec.test_loss = m.mean_loss();
        rec.next_state_accuracy = m.overall_accuracy();
        rec.successful_move_acc = m.successful_move_accuracy();
        rec.blocked_move_acc = m.blocked_move_accuracy();
        rec.boundary_collision_acc = m.boundary_block_accuracy();
        rec.obstacle_collision_acc = m.obstacle_block_accuracy();
        rec.training_seconds = summary.total_seconds;
        rec.inference_seconds = m.inference_seconds;
        rec.model_bytes = net.memory_bytes();
        all_records.push_back(rec);
    }

    std::cout << std::string(105, '=') << "\n"
              << std::left << std::setw(30) << "SPARK Mean (" + std::to_string(num_seeds) + " seeds)"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(12) << ((acc_sum / num_seeds) * 100.0f) << "%"
              << std::setw(12) << ((succ_sum / num_seeds) * 100.0f) << "%"
              << std::setw(12) << ((block_sum / num_seeds) * 100.0f) << "%"
              << std::setw(12) << ((bound_sum / num_seeds) * 100.0f) << "%"
              << std::setw(12) << ((obs_sum / num_seeds) * 100.0f) << "%"
              << std::setprecision(4) << std::setw(12) << (loss_sum / num_seeds)
              << "\n" << std::string(105, '=') << "\n" << std::endl;

    // Export CSV
    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << Milestone5bRecord::csv_header() << "\n";
            for (const auto& r : all_records) {
                ofs << r.to_csv() << "\n";
            }
            std::cout << "Exported " << all_records.size() << " experimental runs to " << csv_path << std::endl;
        }
    }

    return all_records;
}

std::vector<Milestone5cRecord> ExperimentRunner::run_milestone5c_suite(
    size_t num_seeds,
    uint32_t train_steps,
    uint32_t unroll_steps,
    float lr,
    uint32_t num_eval_episodes,
    const std::string& csv_path
) {
    std::cout << "\n=========================================================================================\n"
              << "  SPARK Milestone 5C: Partial Observability, Temporal State Estimation & Ablations\n"
              << "=========================================================================================\n"
              << "  Evaluation Horizons: {2, 4, 8, 16} steps\n"
              << "  Episodes per Eval: " << num_eval_episodes << " on 20 unseen test layouts\n"
              << "  Training: " << train_steps << " steps (unroll=" << unroll_steps << ", lr=" << lr << ")\n"
              << "=========================================================================================\n"
              << std::endl;

    std::vector<Milestone5cRecord> all_records;

    // 1. Generate strictly partitioned layouts
    auto splits = GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 5000ULL);
    std::cout << "Generated layout splits with 0 overlap: "
              << splits.train_layouts.size() << " train, "
              << splits.val_layouts.size() << " val, "
              << splits.test_layouts.size() << " test.\n" << std::endl;

    GridWorldEnv test_env(splits.test_layouts, 16, 8888ULL, GridObservationMode::PartiallyObservable);

    // Print header for horizon 16 table
    auto print_h16_header = []() {
        std::cout << std::string(105, '-') << "\n"
                  << std::left << std::setw(28) << "Condition / Run (H=16)"
                  << std::right << std::setw(12) << "Overall Acc"
                  << std::setw(10) << "Step 1"
                  << std::setw(10) << "Step 2"
                  << std::setw(10) << "Step 4"
                  << std::setw(10) << "Step 8"
                  << std::setw(10) << "Step 16"
                  << std::setw(12) << "Test Loss"
                  << "\n" << std::string(105, '-') << std::endl;
    };

    auto record_eval = [&](
        MicroUnitNetwork& net,
        const std::string& model_type,
        uint64_t seed,
        double training_time,
        PartialObsAblation ablation
    ) {
        for (uint32_t H : {2, 4, 8, 16}) {
            PartialObsMetrics m = evaluate_partial_observability(net, test_env, num_eval_episodes, H, ablation);
            Milestone5cRecord rec;
            rec.experiment_id = model_type + "_H" + std::to_string(H) + "_s" + std::to_string(seed);
            rec.seed = seed;
            rec.model_type = model_type;
            rec.units = net.config().num_units;
            rec.fan_in = 8;
            rec.horizon = H;
            rec.evaluated_episodes = num_eval_episodes;
            rec.overall_accuracy = m.overall_accuracy();
            rec.step1_acc = m.accuracy_at_step(0);
            rec.step2_acc = (H >= 2) ? m.accuracy_at_step(1) : 0.0f;
            rec.step4_acc = (H >= 4) ? m.accuracy_at_step(3) : 0.0f;
            rec.step8_acc = (H >= 8) ? m.accuracy_at_step(7) : 0.0f;
            rec.step16_acc = (H >= 16) ? m.accuracy_at_step(15) : 0.0f;
            rec.mean_loss = m.mean_loss();
            rec.elapsed_seconds = training_time + m.inference_seconds;
            all_records.push_back(rec);

            if (H == 16) {
                std::cout << std::left << std::setw(28) << (model_type + " (H=16)")
                          << std::right << std::fixed << std::setprecision(2)
                          << std::setw(12) << (rec.overall_accuracy * 100.0f) << "%"
                          << std::setw(10) << (rec.step1_acc * 100.0f) << "%"
                          << std::setw(10) << (rec.step2_acc * 100.0f) << "%"
                          << std::setw(10) << (rec.step4_acc * 100.0f) << "%"
                          << std::setw(10) << (rec.step8_acc * 100.0f) << "%"
                          << std::setw(10) << (rec.step16_acc * 100.0f) << "%"
                          << std::setprecision(4) << std::setw(12) << rec.mean_loss
                          << std::endl;
            }
        }
    };

    // 2. Baselines (Oracle & Random)
    std::cout << ">>> Evaluating Control Baselines Under Partial Observability <<<" << std::endl;
    print_h16_header();

    // Oracle
    for (uint32_t H : {2, 4, 8, 16}) {
        PartialObsMetrics m = evaluate_oracle_partial_obs(test_env, num_eval_episodes, H);
        Milestone5cRecord rec;
        rec.experiment_id = "oracle_H" + std::to_string(H);
        rec.seed = 42;
        rec.model_type = "ground_truth_oracle";
        rec.units = 0;
        rec.fan_in = 0;
        rec.horizon = H;
        rec.evaluated_episodes = num_eval_episodes;
        rec.overall_accuracy = m.overall_accuracy();
        rec.step1_acc = m.accuracy_at_step(0);
        rec.step2_acc = (H >= 2) ? m.accuracy_at_step(1) : 0.0f;
        rec.step4_acc = (H >= 4) ? m.accuracy_at_step(3) : 0.0f;
        rec.step8_acc = (H >= 8) ? m.accuracy_at_step(7) : 0.0f;
        rec.step16_acc = (H >= 16) ? m.accuracy_at_step(15) : 0.0f;
        rec.mean_loss = 0.0f;
        rec.elapsed_seconds = m.inference_seconds;
        all_records.push_back(rec);

        if (H == 16) {
            std::cout << std::left << std::setw(28) << "Ground-Truth Oracle"
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(12) << (rec.overall_accuracy * 100.0f) << "%"
                      << std::setw(10) << (rec.step1_acc * 100.0f) << "%"
                      << std::setw(10) << (rec.step2_acc * 100.0f) << "%"
                      << std::setw(10) << (rec.step4_acc * 100.0f) << "%"
                      << std::setw(10) << (rec.step8_acc * 100.0f) << "%"
                      << std::setw(10) << (rec.step16_acc * 100.0f) << "%"
                      << std::setprecision(4) << std::setw(12) << rec.mean_loss
                      << std::endl;
        }
    }

    // Random
    for (uint32_t H : {2, 4, 8, 16}) {
        PartialObsMetrics m = evaluate_random_partial_obs(test_env, num_eval_episodes, H, 42ULL);
        Milestone5cRecord rec;
        rec.experiment_id = "random_H" + std::to_string(H);
        rec.seed = 42;
        rec.model_type = "random_predictor";
        rec.units = 0;
        rec.fan_in = 0;
        rec.horizon = H;
        rec.evaluated_episodes = num_eval_episodes;
        rec.overall_accuracy = m.overall_accuracy();
        rec.step1_acc = m.accuracy_at_step(0);
        rec.step2_acc = (H >= 2) ? m.accuracy_at_step(1) : 0.0f;
        rec.step4_acc = (H >= 4) ? m.accuracy_at_step(3) : 0.0f;
        rec.step8_acc = (H >= 8) ? m.accuracy_at_step(7) : 0.0f;
        rec.step16_acc = (H >= 16) ? m.accuracy_at_step(15) : 0.0f;
        rec.mean_loss = m.mean_loss();
        rec.elapsed_seconds = m.inference_seconds;
        all_records.push_back(rec);

        if (H == 16) {
            std::cout << std::left << std::setw(28) << "Random Predictor"
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(12) << (rec.overall_accuracy * 100.0f) << "%"
                      << std::setw(10) << (rec.step1_acc * 100.0f) << "%"
                      << std::setw(10) << (rec.step2_acc * 100.0f) << "%"
                      << std::setw(10) << (rec.step4_acc * 100.0f) << "%"
                      << std::setw(10) << (rec.step8_acc * 100.0f) << "%"
                      << std::setw(10) << (rec.step16_acc * 100.0f) << "%"
                      << std::setprecision(4) << std::setw(12) << rec.mean_loss
                      << std::endl;
        }
    }
    std::cout << std::string(105, '-') << "\n" << std::endl;

    // 3. Training Ablations (Readout-Only & Frozen Recurrent)
    std::cout << ">>> Training & Evaluating Structural Training Ablations <<<" << std::endl;
    print_h16_header();

    // Readout-Only
    {
        uint64_t seed = 42ULL;
        SparseGraphConfig g_cfg;
        g_cfg.num_units = 128;
        g_cfg.fan_in = 8;
        g_cfg.seed = seed;

        NetworkConfig n_cfg;
        n_cfg.num_units = 128;
        n_cfg.input_dim = GridWorldEnv::PARTIAL_INPUT_DIM;
        n_cfg.output_dim = GridWorldEnv::OUTPUT_DIM;
        n_cfg.alpha = 0.8f;
        n_cfg.beta = 1.0f;
        n_cfg.seed = seed;

        MicroUnitNetwork net(n_cfg, g_cfg);
        GridWorldEnv train_env(splits.train_layouts, 16, seed, GridObservationMode::PartiallyObservable);
        GridWorldEnv val_env(splits.val_layouts, 16, seed + 1000ULL, GridObservationMode::PartiallyObservable);

        Trainer trainer(net, train_env, &val_env, lr, 1.0f);
        trainer.set_loss_type(LossType::CategoricalCrossEntropy);
        trainer.set_readout_only(true);

        TrainSummary s = trainer.train(train_steps, unroll_steps, 0, true);
        record_eval(net, "readout_only", seed, s.total_seconds, PartialObsAblation::None);
    }

    // Frozen Recurrent (train input, biases, readout, keep W frozen)
    {
        uint64_t seed = 42ULL;
        SparseGraphConfig g_cfg;
        g_cfg.num_units = 128;
        g_cfg.fan_in = 8;
        g_cfg.seed = seed;

        NetworkConfig n_cfg;
        n_cfg.num_units = 128;
        n_cfg.input_dim = GridWorldEnv::PARTIAL_INPUT_DIM;
        n_cfg.output_dim = GridWorldEnv::OUTPUT_DIM;
        n_cfg.alpha = 0.8f;
        n_cfg.beta = 1.0f;
        n_cfg.seed = seed;

        MicroUnitNetwork net(n_cfg, g_cfg);
        GridWorldEnv train_env(splits.train_layouts, 16, seed, GridObservationMode::PartiallyObservable);
        GridWorldEnv val_env(splits.val_layouts, 16, seed + 1000ULL, GridObservationMode::PartiallyObservable);

        Trainer trainer(net, train_env, &val_env, lr, 1.0f);
        trainer.set_loss_type(LossType::CategoricalCrossEntropy);
        trainer.set_train_recurrent(false);

        TrainSummary s = trainer.train(train_steps, unroll_steps, 0, true);
        record_eval(net, "frozen_recurrent", seed, s.total_seconds, PartialObsAblation::None);
    }
    std::cout << std::string(105, '-') << "\n" << std::endl;

    // 4. Fully Trained SPARK across multiple seeds + Inference Ablations
    std::cout << ">>> Training SPARK Partial Observability Models Across " << num_seeds << " Seeds <<<" << std::endl;
    print_h16_header();

    double intact_h16_acc_sum = 0.0;
    double reset_h16_acc_sum = 0.0;
    double zero_h16_acc_sum = 0.0;

    for (size_t s_idx = 0; s_idx < num_seeds; ++s_idx) {
        uint64_t seed = 42ULL + s_idx * 17ULL;

        SparseGraphConfig g_cfg;
        g_cfg.num_units = 128;
        g_cfg.fan_in = 8;
        g_cfg.seed = seed;

        NetworkConfig n_cfg;
        n_cfg.num_units = 128;
        n_cfg.input_dim = GridWorldEnv::PARTIAL_INPUT_DIM;
        n_cfg.output_dim = GridWorldEnv::OUTPUT_DIM;
        n_cfg.alpha = 0.8f;
        n_cfg.beta = 1.0f;
        n_cfg.seed = seed;

        MicroUnitNetwork net(n_cfg, g_cfg);
        GridWorldEnv train_env(splits.train_layouts, 16, seed, GridObservationMode::PartiallyObservable);
        GridWorldEnv val_env(splits.val_layouts, 16, seed + 1000ULL, GridObservationMode::PartiallyObservable);

        Trainer trainer(net, train_env, &val_env, lr, 1.0f);
        trainer.set_loss_type(LossType::CategoricalCrossEntropy);

        TrainSummary s = trainer.train(train_steps, unroll_steps, 500, false);

        // a) Intact Model
        record_eval(net, "trained_spark_s" + std::to_string(seed), seed, s.total_seconds, PartialObsAblation::None);
        intact_h16_acc_sum += all_records.back().overall_accuracy;

        // b) Reset State Each Step Ablation
        record_eval(net, "reset_state_s" + std::to_string(seed), seed, 0.0, PartialObsAblation::ResetStateEachStep);
        reset_h16_acc_sum += all_records.back().overall_accuracy;

        // c) Zero Recurrent Weights Ablation
        record_eval(net, "zero_weights_s" + std::to_string(seed), seed, 0.0, PartialObsAblation::ZeroRecurrentWeights);
        zero_h16_acc_sum += all_records.back().overall_accuracy;
    }

    std::cout << std::string(105, '=') << "\n"
              << std::left << std::setw(28) << "SPARK Mean (Intact, H=16)"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(12) << ((intact_h16_acc_sum / num_seeds) * 100.0f) << "%\n"
              << std::left << std::setw(28) << "SPARK Mean (Reset State, H=16)"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(12) << ((reset_h16_acc_sum / num_seeds) * 100.0f) << "%\n"
              << std::left << std::setw(28) << "SPARK Mean (Zero Weights, H=16)"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(12) << ((zero_h16_acc_sum / num_seeds) * 100.0f) << "%\n"
              << std::string(105, '=') << "\n" << std::endl;

    // Export CSV
    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << Milestone5cRecord::csv_header() << "\n";
            for (const auto& r : all_records) {
                ofs << r.to_csv() << "\n";
            }
            std::cout << "Exported " << all_records.size() << " experimental runs to " << csv_path << std::endl;
        }
    }

    return all_records;
}

void ExperimentRunner::run_milestone5d0_diagnostic(
    uint32_t train_steps,
    uint32_t test_transitions,
    float lr,
    uint64_t seed,
    const std::string& csv_path
) {
    std::cout << "\n=========================================================================================\n"
              << "  SPARK Milestone 5D.0: One-Step Transition Accuracy & Weakness Investigation\n"
              << "=========================================================================================\n"
              << "  Evaluating: Train Layouts vs Unseen Test Layouts (10,000 transitions each)\n"
              << "  Investigating: Free moves, Boundary collisions, Obstacle collisions, Actions, Densities\n"
              << "  Comparing: SPARK TBPTT vs Feedforward Baseline vs Non-learning Controls vs Oracle\n"
              << "=========================================================================================\n"
              << std::endl;

    auto splits = GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);

    GridWorldEnv train_env(splits.train_layouts, 50, seed);
    GridWorldEnv val_env(splits.val_layouts, 50, seed + 1000ULL);
    GridWorldEnv test_env(splits.test_layouts, 50, seed + 2000ULL);

    // 1. Non-Learning Controls on Test Layouts
    std::cout << ">>> 1. Non-Learning Controls & Simulator Oracle (Unseen Test Layouts) <<<" << std::endl;
    auto base_diag = evaluate_baseline_diagnostics(test_env, test_transitions, seed);
    base_diag.ground_truth_oracle.print_report();
    base_diag.random_predictor.print_report();
    base_diag.copy_current_predictor.print_report();
    base_diag.naive_movement_predictor.print_report();

    // 2. Feedforward Baseline (same parameter count: 128 units, W=0, no recurrence)
    std::cout << ">>> 2. Training Feedforward Baseline (128 units, 10,393 params, W=0) <<<" << std::endl;
    SparseGraphConfig g_cfg_ff;
    g_cfg_ff.num_units = 128;
    g_cfg_ff.fan_in = 8;
    g_cfg_ff.seed = seed;
    NetworkConfig n_cfg_ff;
    n_cfg_ff.num_units = 128;
    n_cfg_ff.input_dim = 55;
    n_cfg_ff.output_dim = 25;
    n_cfg_ff.alpha = 0.0f;
    n_cfg_ff.beta = 0.0f;
    n_cfg_ff.seed = seed;

    MicroUnitNetwork ff_net(n_cfg_ff, g_cfg_ff);
    std::fill(ff_net.graph().weights.begin(), ff_net.graph().weights.end(), 0.0f);

    Trainer ff_trainer(ff_net, train_env, &val_env, lr, 1.0f);
    ff_trainer.set_loss_type(LossType::CategoricalCrossEntropy);
    ff_trainer.set_train_recurrent(false);
    ff_trainer.train(train_steps, 8, 0, true);

    auto ff_train_m = evaluate_onestep_diagnostic(ff_net, train_env, test_transitions);
    ff_train_m.model_name = "Feedforward Baseline (128 units, Train Layouts)";
    ff_train_m.print_report();

    auto ff_test_m = evaluate_onestep_diagnostic(ff_net, test_env, test_transitions);
    ff_test_m.model_name = "Feedforward Baseline (128 units, Unseen Test Layouts)";
    ff_test_m.print_report();

    // 3. SPARK Recurrent Model (128 units, fan-in 8, TBPTT unroll 8)
    std::cout << ">>> 3. Training SPARK Recurrent Architecture (128 units, fan-in 8, TBPTT) <<<" << std::endl;
    SparseGraphConfig g_cfg;
    g_cfg.num_units = 128;
    g_cfg.fan_in = 8;
    g_cfg.seed = seed;
    NetworkConfig n_cfg;
    n_cfg.num_units = 128;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = 0.8f;
    n_cfg.beta = 1.0f;
    n_cfg.seed = seed;

    MicroUnitNetwork spark_net(n_cfg, g_cfg);
    Trainer spark_trainer(spark_net, train_env, &val_env, lr, 1.0f);
    spark_trainer.set_loss_type(LossType::CategoricalCrossEntropy);
    spark_trainer.train(train_steps, 8, 0, true);

    auto spark_train_m = evaluate_onestep_diagnostic(spark_net, train_env, test_transitions);
    spark_train_m.model_name = "SPARK TBPTT (128 units, Train Layouts)";
    spark_train_m.print_report();

    auto spark_val_m = evaluate_onestep_diagnostic(spark_net, val_env, test_transitions);
    spark_val_m.model_name = "SPARK TBPTT (128 units, Val Layouts)";
    spark_val_m.print_report();

    auto spark_test_m = evaluate_onestep_diagnostic(spark_net, test_env, test_transitions);
    spark_test_m.model_name = "SPARK TBPTT (128 units, Unseen Test Layouts)";
    spark_test_m.print_report();

    // 4. Extended Optimization Investigation (20,000 steps)
    std::cout << ">>> 4. Extended Optimization Investigation (20,000 steps) <<<" << std::endl;
    MicroUnitNetwork spark_20k_net(n_cfg, g_cfg);
    Trainer spark_20k_trainer(spark_20k_net, train_env, &val_env, lr, 1.0f);
    spark_20k_trainer.set_loss_type(LossType::CategoricalCrossEntropy);
    spark_20k_trainer.train(20000, 8, 0, true);

    auto spark_20k_test_m = evaluate_onestep_diagnostic(spark_20k_net, test_env, test_transitions);
    spark_20k_test_m.model_name = "SPARK TBPTT 20k steps (Unseen Test Layouts)";
    spark_20k_test_m.print_report();

    // 5. Consolidated Summary Table
    std::cout << "\n=========================================================================================\n"
              << "  CONSOLIDATED 5D.0 DIAGNOSTIC COMPARISON (Unseen Test Layouts, 10,000 transitions)\n"
              << "=========================================================================================\n"
              << std::left << std::setw(32) << "Model / Condition"
              << std::right << std::setw(12) << "Overall Acc"
              << std::setw(12) << "Free Move"
              << std::setw(12) << "Boundary"
              << std::setw(12) << "Obstacle"
              << std::setw(12) << "Stay"
              << std::setw(10) << "Loss"
              << "\n" << std::string(102, '-') << std::endl;

    auto print_row = [](const std::string& name, const OneStepDiagnosticMetrics& m) {
        std::cout << std::left << std::setw(32) << name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(11) << (m.overall_acc() * 100.0f) << "%"
                  << std::setw(11) << (m.succ_acc() * 100.0f) << "%"
                  << std::setw(11) << (m.bound_acc() * 100.0f) << "%"
                  << std::setw(11) << (m.obs_acc() * 100.0f) << "%"
                  << std::setw(11) << (m.stay_acc() * 100.0f) << "%"
                  << std::setprecision(4) << std::setw(10) << m.mean_loss()
                  << std::endl;
    };

    print_row("1. Random Predictor", base_diag.random_predictor);
    print_row("2. Copy Current Position", base_diag.copy_current_predictor);
    print_row("3. Naive Movement (Blind)", base_diag.naive_movement_predictor);
    print_row("4. Ground-Truth Oracle", base_diag.ground_truth_oracle);
    print_row("5. Feedforward MLP (Train)", ff_train_m);
    print_row("6. Feedforward MLP (Test)", ff_test_m);
    print_row("7. SPARK TBPTT (Train)", spark_train_m);
    print_row("8. SPARK TBPTT (Val)", spark_val_m);
    print_row("9. SPARK TBPTT (Test, 10k)", spark_test_m);
    print_row("10. SPARK TBPTT (Test, 20k)", spark_20k_test_m);
    std::cout << std::string(102, '=') << "\n" << std::endl;

    // Export CSV
    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << "model_name,dataset,total_transitions,overall_acc,succ_acc,bound_acc,obs_acc,stay_acc,"
                << "up_acc,down_acc,left_acc,right_acc,stay_action_acc,low_dens_acc,med_dens_acc,high_dens_acc,mean_loss\n";
            auto write_csv_row = [&](const std::string& ds, const OneStepDiagnosticMetrics& m) {
                ofs << m.model_name << "," << ds << "," << m.total_transitions << ","
                    << std::fixed << std::setprecision(4) << m.overall_acc() << ","
                    << m.succ_acc() << "," << m.bound_acc() << "," << m.obs_acc() << "," << m.stay_acc() << ","
                    << m.action_acc(0) << "," << m.action_acc(1) << "," << m.action_acc(2) << "," << m.action_acc(3) << "," << m.action_acc(4) << ","
                    << m.density_acc(0) << "," << m.density_acc(1) << "," << m.density_acc(2) << ","
                    << std::setprecision(5) << m.mean_loss() << "\n";
            };
            write_csv_row("test", base_diag.random_predictor);
            write_csv_row("test", base_diag.copy_current_predictor);
            write_csv_row("test", base_diag.naive_movement_predictor);
            write_csv_row("test", base_diag.ground_truth_oracle);
            write_csv_row("train", ff_train_m);
            write_csv_row("test", ff_test_m);
            write_csv_row("train", spark_train_m);
            write_csv_row("val", spark_val_m);
            write_csv_row("test", spark_test_m);
            write_csv_row("test", spark_20k_test_m);
            std::cout << "Saved diagnostic records to " << csv_path << std::endl;
        }
    }
}

static MicroUnitNetwork get_or_train_world_model(
    uint64_t seed,
    uint32_t train_steps,
    const LayoutSplits& splits,
    bool feedforward = false
) {
    SparseGraphConfig g_cfg;
    g_cfg.num_units = 128;
    g_cfg.fan_in = 8;
    g_cfg.seed = seed;

    NetworkConfig n_cfg;
    n_cfg.num_units = 128;
    n_cfg.input_dim = 55;
    n_cfg.output_dim = 25;
    n_cfg.alpha = feedforward ? 0.0f : 0.8f;
    n_cfg.beta = feedforward ? 0.0f : 1.0f;
    n_cfg.seed = seed;

    MicroUnitNetwork net(n_cfg, g_cfg);
    if (feedforward) {
        std::fill(net.graph().weights.begin(), net.graph().weights.end(), 0.0f);
    }

    std::string prefix = feedforward ? "checkpoints/ff_mlp_seed_" : "checkpoints/world_model_seed_";
    std::string ckpt_path = prefix + std::to_string(seed) + ".spark";

    if (MicroUnitNetwork::load_checkpoint(ckpt_path, net)) {
        return net;
    }

    std::cout << "Training checkpoint " << ckpt_path << " (" << (feedforward ? "Feedforward" : "SPARK TBPTT")
              << ", " << train_steps << " steps)..." << std::flush;
    GridWorldEnv train_env(splits.train_layouts, 50, seed);
    GridWorldEnv val_env(splits.val_layouts, 50, seed + 1000ULL);

    Trainer trainer(net, train_env, &val_env, 0.02f, 1.0f);
    trainer.set_loss_type(LossType::CategoricalCrossEntropy);
    if (feedforward) {
        trainer.set_train_recurrent(false);
    }
    trainer.train(train_steps, 8, 0, true);

    net.save_checkpoint(ckpt_path);
    std::cout << " saved." << std::endl;
    return net;
}

std::vector<RolloutMetrics> ExperimentRunner::run_milestone5d2_multi_horizon(
    size_t num_seeds,
    const std::vector<uint32_t>& horizons,
    uint32_t episodes_per_seed,
    uint32_t train_steps,
    const std::string& csv_path
) {
    std::cout << "\n=========================================================================================\n"
              << "  SPARK Milestone 5D.2: Multi-Horizon Statistical Rollout Experiments\n"
              << "  (" << num_seeds << " independent seeds, " << episodes_per_seed << " episodes/seed, 20 unseen test layouts)\n"
              << "=========================================================================================\n" << std::endl;

    auto splits = GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);
    std::vector<RolloutMetrics> all_metrics;

    // Train or load models for all seeds
    std::vector<MicroUnitNetwork> models;
    models.reserve(num_seeds);
    for (size_t s = 0; s < num_seeds; ++s) {
        uint64_t seed = 42ULL + s * 17ULL;
        models.push_back(get_or_train_world_model(seed, train_steps, splits, false));
    }

    std::cout << "\nEvaluating Teacher-Forced (Mode A) vs Autoregressive Open-Loop (Mode B)...\n" << std::endl;

    // Print table header
    std::cout << std::left << std::setw(6)  << "H"
              << std::right << std::setw(14) << "TF StepAcc"
              << std::setw(14) << "OL StepAcc"
              << std::setw(14) << "Divergence"
              << std::setw(14) << "OL TrajAcc"
              << std::setw(14) << "OL FinalAcc"
              << std::setw(14) << "OL ManhErr"
              << std::setw(12) << "OL CE Loss"
              << "\n" << std::string(102, '-') << std::endl;

    for (uint32_t H : horizons) {
        double tf_acc_sum = 0.0, tf_acc_sq = 0.0;
        double ol_acc_sum = 0.0, ol_acc_sq = 0.0;
        double ol_traj_sum = 0.0;
        double ol_final_sum = 0.0;
        double ol_manh_sum = 0.0;
        double ol_ce_sum = 0.0;

        for (size_t s = 0; s < num_seeds; ++s) {
            uint64_t seed = 42ULL + s * 17ULL;
            auto& net = models[s];

            // Mode A: Teacher-Forced
            auto tf_m = RolloutEngine::evaluate_rollouts(
                net, splits.test_layouts, H, episodes_per_seed,
                RolloutMode::TeacherForced, seed + 10000ULL
            );
            tf_m.seed = seed;
            tf_m.model_name = "SPARK";
            all_metrics.push_back(tf_m);

            // Mode B: Autoregressive Open-Loop
            auto ol_m = RolloutEngine::evaluate_rollouts(
                net, splits.test_layouts, H, episodes_per_seed,
                RolloutMode::OpenLoop, seed + 10000ULL
            );
            ol_m.seed = seed;
            ol_m.model_name = "SPARK";
            all_metrics.push_back(ol_m);

            // Accumulate statistics
            float tf_acc = tf_m.overall_step_accuracy();
            float ol_acc = ol_m.overall_step_accuracy();
            tf_acc_sum += tf_acc;
            tf_acc_sq += tf_acc * tf_acc;
            ol_acc_sum += ol_acc;
            ol_acc_sq += ol_acc * ol_acc;
            ol_traj_sum += ol_m.complete_trajectory_accuracy();
            ol_final_sum += ol_m.final_position_accuracy();
            ol_manh_sum += ol_m.mean_manhattan_error();
            ol_ce_sum += ol_m.mean_cross_entropy();
        }

        double tf_acc_mean = tf_acc_sum / num_seeds;
        double tf_acc_std = std::sqrt(std::max(0.0, (tf_acc_sq / num_seeds) - (tf_acc_mean * tf_acc_mean)));

        double ol_acc_mean = ol_acc_sum / num_seeds;
        double ol_acc_std = std::sqrt(std::max(0.0, (ol_acc_sq / num_seeds) - (ol_acc_mean * ol_acc_mean)));

        double divergence = tf_acc_mean - ol_acc_mean;
        double ol_traj_mean = ol_traj_sum / num_seeds;
        double ol_final_mean = ol_final_sum / num_seeds;
        double ol_manh_mean = ol_manh_sum / num_seeds;
        double ol_ce_mean = ol_ce_sum / num_seeds;

        std::ostringstream ss_tf, ss_ol, ss_div;
        ss_tf << std::fixed << std::setprecision(1) << (tf_acc_mean * 100.0) << "+-" << (tf_acc_std * 100.0) << "%";
        ss_ol << std::fixed << std::setprecision(1) << (ol_acc_mean * 100.0) << "+-" << (ol_acc_std * 100.0) << "%";
        ss_div << std::fixed << std::setprecision(2) << (divergence * 100.0) << "%";

        std::cout << std::left << "H=" << std::setw(4) << H
                  << std::right << std::setw(14) << ss_tf.str()
                  << std::setw(14) << ss_ol.str()
                  << std::setw(14) << ss_div.str()
                  << std::fixed << std::setprecision(1)
                  << std::setw(13) << (ol_traj_mean * 100.0) << "%"
                  << std::setw(13) << (ol_final_mean * 100.0) << "%"
                  << std::setprecision(3)
                  << std::setw(14) << ol_manh_mean
                  << std::setprecision(4)
                  << std::setw(12) << ol_ce_mean
                  << std::endl;
    }
    std::cout << std::string(102, '=') << "\n" << std::endl;

    // Export CSV
    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << RolloutMetrics::csv_header() << "\n";
            for (const auto& m : all_metrics) {
                ofs << m.to_csv() << "\n";
            }
            std::cout << "Saved multi-horizon rollout records to: " << csv_path << std::endl;
        }
    }

    return all_metrics;
}

std::vector<RolloutMetrics> ExperimentRunner::run_milestone5d3_ablations(
    size_t num_seeds,
    const std::vector<uint32_t>& horizons,
    uint32_t episodes_per_seed,
    const std::string& csv_path
) {
    std::cout << "\n=========================================================================================\n"
              << "  SPARK Milestone 5D.3: Recurrence Memory Ablations Under Open-Loop Generation\n"
              << "  (" << num_seeds << " independent seeds, " << episodes_per_seed << " episodes/seed, 20 unseen test layouts)\n"
              << "=========================================================================================\n" << std::endl;

    auto splits = GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);
    std::vector<RolloutMetrics> all_metrics;

    // Load or train intact models
    std::vector<MicroUnitNetwork> models;
    models.reserve(num_seeds);
    for (size_t s = 0; s < num_seeds; ++s) {
        uint64_t seed = 42ULL + s * 17ULL;
        models.push_back(get_or_train_world_model(seed, 20000, splits, false));
    }

    std::cout << std::left << std::setw(6)  << "H"
              << std::right << std::setw(22) << "1. Intact Recurrent"
              << std::setw(22) << "2. State Reset"
              << std::setw(22) << "3. Zero Weights (W=0)"
              << std::setw(22) << "4. Scrambled Weights"
              << "\n" << std::string(96, '-') << std::endl;

    for (uint32_t H : horizons) {
        double intact_acc_sum = 0.0, intact_traj_sum = 0.0;
        double reset_acc_sum = 0.0, reset_traj_sum = 0.0;
        double zero_acc_sum = 0.0, zero_traj_sum = 0.0;
        double scram_acc_sum = 0.0, scram_traj_sum = 0.0;

        for (size_t s = 0; s < num_seeds; ++s) {
            uint64_t seed = 42ULL + s * 17ULL;
            auto& base_net = models[s];

            // 1. Intact
            auto m_intact = RolloutEngine::evaluate_rollouts(
                base_net, splits.test_layouts, H, episodes_per_seed,
                RolloutMode::OpenLoop, seed + 10000ULL, RolloutAblation::None
            );
            m_intact.seed = seed;
            m_intact.model_name = "SPARK_Intact";
            m_intact.ablation = "none";
            all_metrics.push_back(m_intact);
            intact_acc_sum += m_intact.overall_step_accuracy();
            intact_traj_sum += m_intact.complete_trajectory_accuracy();

            // 2. State Reset
            auto m_reset = RolloutEngine::evaluate_rollouts(
                base_net, splits.test_layouts, H, episodes_per_seed,
                RolloutMode::OpenLoop, seed + 10000ULL, RolloutAblation::StateReset
            );
            m_reset.seed = seed;
            m_reset.model_name = "SPARK_StateReset";
            m_reset.ablation = "state-reset";
            all_metrics.push_back(m_reset);
            reset_acc_sum += m_reset.overall_step_accuracy();
            reset_traj_sum += m_reset.complete_trajectory_accuracy();

            // 3. Zero Recurrent
            MicroUnitNetwork net_zero = base_net;
            std::fill(net_zero.graph().weights.begin(), net_zero.graph().weights.end(), 0.0f);
            auto m_zero = RolloutEngine::evaluate_rollouts(
                net_zero, splits.test_layouts, H, episodes_per_seed,
                RolloutMode::OpenLoop, seed + 10000ULL, RolloutAblation::None
            );
            m_zero.seed = seed;
            m_zero.model_name = "SPARK_ZeroWeights";
            m_zero.ablation = "zero-weights";
            all_metrics.push_back(m_zero);
            zero_acc_sum += m_zero.overall_step_accuracy();
            zero_traj_sum += m_zero.complete_trajectory_accuracy();

            // 4. Scrambled Recurrent
            MicroUnitNetwork net_scram = base_net;
            Random rng(seed + 999ULL);
            for (float& w : net_scram.graph().weights) {
                w = rng.normal_float(0.0f, 0.5f);
            }
            auto m_scram = RolloutEngine::evaluate_rollouts(
                net_scram, splits.test_layouts, H, episodes_per_seed,
                RolloutMode::OpenLoop, seed + 10000ULL, RolloutAblation::None
            );
            m_scram.seed = seed;
            m_scram.model_name = "SPARK_ScrambledWeights";
            m_scram.ablation = "scrambled-weights";
            all_metrics.push_back(m_scram);
            scram_acc_sum += m_scram.overall_step_accuracy();
            scram_traj_sum += m_scram.complete_trajectory_accuracy();
        }

        auto fmt_entry = [](double acc, double traj) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << (acc * 100.0) << "% (" << (traj * 100.0) << "% traj)";
            return ss.str();
        };

        std::cout << std::left << "H=" << std::setw(4) << H
                  << std::right << std::setw(22) << fmt_entry(intact_acc_sum / num_seeds, intact_traj_sum / num_seeds)
                  << std::setw(22) << fmt_entry(reset_acc_sum / num_seeds, reset_traj_sum / num_seeds)
                  << std::setw(22) << fmt_entry(zero_acc_sum / num_seeds, zero_traj_sum / num_seeds)
                  << std::setw(22) << fmt_entry(scram_acc_sum / num_seeds, scram_traj_sum / num_seeds)
                  << std::endl;
    }
    std::cout << std::string(96, '=') << "\n" << std::endl;

    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << RolloutMetrics::csv_header() << "\n";
            for (const auto& m : all_metrics) {
                ofs << m.to_csv() << "\n";
            }
            std::cout << "Saved ablation records to: " << csv_path << std::endl;
        }
    }

    return all_metrics;
}

std::vector<RolloutMetrics> ExperimentRunner::run_milestone5d4_controls(
    size_t num_seeds,
    const std::vector<uint32_t>& horizons,
    uint32_t episodes_per_seed,
    const std::string& csv_path
) {
    std::cout << "\n=========================================================================================\n"
              << "  SPARK Milestone 5D.4: Scientific Baselines & Controls Comparison (Open-Loop)\n"
              << "  (" << num_seeds << " independent seeds, " << episodes_per_seed << " episodes/seed, 20 unseen test layouts)\n"
              << "=========================================================================================\n" << std::endl;

    auto splits = GridWorldEnv::generate_split_layouts(100, 20, 20, 3, 7, 7777ULL);
    std::vector<RolloutMetrics> all_metrics;

    // Train or load SPARK recurrent models and Feedforward MLP models
    std::vector<MicroUnitNetwork> spark_models;
    std::vector<MicroUnitNetwork> ff_models;
    spark_models.reserve(num_seeds);
    ff_models.reserve(num_seeds);

    for (size_t s = 0; s < num_seeds; ++s) {
        uint64_t seed = 42ULL + s * 17ULL;
        spark_models.push_back(get_or_train_world_model(seed, 20000, splits, false));
        ff_models.push_back(get_or_train_world_model(seed, 20000, splits, true));
    }

    std::cout << std::left << std::setw(6)  << "H"
              << std::right << std::setw(15) << "Oracle"
              << std::setw(16) << "SPARK TBPTT"
              << std::setw(16) << "Feedforward MLP"
              << std::setw(16) << "Copy-Current"
              << std::setw(16) << "Naive-Move"
              << std::setw(16) << "Random Cell"
              << "\n" << std::string(101, '-') << std::endl;

    for (uint32_t H : horizons) {
        double oracle_acc = 0.0;
        double spark_acc = 0.0, spark_traj = 0.0;
        double ff_acc = 0.0, ff_traj = 0.0;
        double cc_acc = 0.0;
        double nm_acc = 0.0;
        double rand_acc = 0.0;

        for (size_t s = 0; s < num_seeds; ++s) {
            uint64_t seed = 42ULL + s * 17ULL;

            // 1. Oracle
            auto m_oracle = RolloutEngine::evaluate_oracle(splits.test_layouts, H, episodes_per_seed, seed + 10000ULL);
            m_oracle.seed = seed;
            all_metrics.push_back(m_oracle);
            oracle_acc += m_oracle.overall_step_accuracy();

            // 2. SPARK
            auto m_spark = RolloutEngine::evaluate_rollouts(
                spark_models[s], splits.test_layouts, H, episodes_per_seed,
                RolloutMode::OpenLoop, seed + 10000ULL
            );
            m_spark.seed = seed;
            m_spark.model_name = "SPARK_TBPTT";
            all_metrics.push_back(m_spark);
            spark_acc += m_spark.overall_step_accuracy();
            spark_traj += m_spark.complete_trajectory_accuracy();

            // 3. Feedforward MLP
            auto m_ff = RolloutEngine::evaluate_rollouts(
                ff_models[s], splits.test_layouts, H, episodes_per_seed,
                RolloutMode::OpenLoop, seed + 10000ULL
            );
            m_ff.seed = seed;
            m_ff.model_name = "Feedforward_MLP";
            all_metrics.push_back(m_ff);
            ff_acc += m_ff.overall_step_accuracy();
            ff_traj += m_ff.complete_trajectory_accuracy();

            // 4. Copy-Current
            auto m_cc = RolloutEngine::evaluate_baseline(
                splits.test_layouts, H, episodes_per_seed,
                RolloutBaselineType::CopyCurrent, RolloutMode::OpenLoop, seed + 10000ULL
            );
            m_cc.seed = seed;
            all_metrics.push_back(m_cc);
            cc_acc += m_cc.overall_step_accuracy();

            // 5. Naive-Move
            auto m_nm = RolloutEngine::evaluate_baseline(
                splits.test_layouts, H, episodes_per_seed,
                RolloutBaselineType::NaiveMove, RolloutMode::OpenLoop, seed + 10000ULL
            );
            m_nm.seed = seed;
            all_metrics.push_back(m_nm);
            nm_acc += m_nm.overall_step_accuracy();

            // 6. Random Predictor
            auto m_rand = RolloutEngine::evaluate_baseline(
                splits.test_layouts, H, episodes_per_seed,
                RolloutBaselineType::RandomPredictor, RolloutMode::OpenLoop, seed + 10000ULL
            );
            m_rand.seed = seed;
            all_metrics.push_back(m_rand);
            rand_acc += m_rand.overall_step_accuracy();
        }

        auto fmt_acc = [](double acc) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << (acc * 100.0) << "%";
            return ss.str();
        };

        auto fmt_acc_traj = [](double acc, double traj) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1) << (acc * 100.0) << "% (" << (traj * 100.0) << "%)";
            return ss.str();
        };

        std::cout << std::left << "H=" << std::setw(4) << H
                  << std::right << std::setw(15) << fmt_acc(oracle_acc / num_seeds)
                  << std::setw(16) << fmt_acc_traj(spark_acc / num_seeds, spark_traj / num_seeds)
                  << std::setw(16) << fmt_acc_traj(ff_acc / num_seeds, ff_traj / num_seeds)
                  << std::setw(16) << fmt_acc(cc_acc / num_seeds)
                  << std::setw(16) << fmt_acc(nm_acc / num_seeds)
                  << std::setw(16) << fmt_acc(rand_acc / num_seeds)
                  << std::endl;
    }
    std::cout << std::string(101, '=') << "\n" << std::endl;

    if (!csv_path.empty()) {
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << RolloutMetrics::csv_header() << "\n";
            for (const auto& m : all_metrics) {
                ofs << m.to_csv() << "\n";
            }
            std::cout << "Saved control baseline records to: " << csv_path << std::endl;
        }
    }

    return all_metrics;
}

} // namespace spark
