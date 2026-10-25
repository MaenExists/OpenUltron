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

} // namespace spark
