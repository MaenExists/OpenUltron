#pragma once

#include "spark/config.hpp"
#include "spark/graph.hpp"
#include "spark/random.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

namespace spark {

enum class ReadoutMode {
    Sigmoid,
    Softmax,
    Linear
};

class MicroUnitNetwork {
public:
    MicroUnitNetwork(const NetworkConfig& net_config, const SparseGraph& graph);
    MicroUnitNetwork(const NetworkConfig& net_config, const SparseGraphConfig& graph_config);

    // Network lifecycle
    void reset_state();
    void set_state(const std::vector<float>& h);
    const std::vector<float>& state() const { return h_curr_; }
    const std::vector<float>& prev_state() const { return h_prev_; }
    const std::vector<float>& pre_activations() const { return a_curr_; }

    // Forward computation
    // Performs internal microsteps and updates state synchronously.
    void forward_step(
        const std::vector<float>& input,
        std::vector<float>& out_logits,
        std::vector<float>& out_probs,
        ReadoutMode mode = ReadoutMode::Sigmoid
    );

    // Compute single internal recurrent microstep without updating output
    void recurrent_microstep(const std::vector<float>& input);

    // Compute readout from current hidden state
    void compute_readout(
        std::vector<float>& out_logits,
        std::vector<float>& out_probs,
        ReadoutMode mode = ReadoutMode::Sigmoid
    ) const;

    // Mathematical utility functions
    static float stable_sigmoid(float z);
    static void stable_softmax(const std::vector<float>& logits, std::vector<float>& probs);

    // Accessors
    const NetworkConfig& config() const { return config_; }
    const SparseGraph& graph() const { return graph_; }
    SparseGraph& graph() { return graph_; }

    const std::vector<float>& biases() const { return biases_; }
    std::vector<float>& biases() { return biases_; }

    const std::vector<float>& input_weights() const { return U_; }
    std::vector<float>& input_weights() { return U_; }

    const std::vector<float>& output_weights() const { return V_; }
    std::vector<float>& output_weights() { return V_; }

    const std::vector<float>& output_biases() const { return c_; }
    std::vector<float>& output_biases() { return c_; }

    // Validation & Memory profiling
    bool validate(std::string& err_msg) const;
    size_t memory_bytes() const;
    size_t parameter_count() const;

    // Checkpoint serialization & deserialization
    bool save_checkpoint(const std::string& filepath) const;
    static bool load_checkpoint(const std::string& filepath, MicroUnitNetwork& out_net);

private:
    void initialize_parameters(Random& rng);

    NetworkConfig config_;
    SparseGraph graph_;

    // Recurrent state arrays (contiguous float buffers)
    std::vector<float> h_prev_; // h(t-1)
    std::vector<float> h_curr_; // h(t)
    std::vector<float> a_curr_; // a(t) pre-activations
    std::vector<float> biases_; // b_i

    // Projections
    std::vector<float> U_; // Input projection matrix [num_units x input_dim]
    std::vector<float> V_; // Output projection matrix [output_dim x num_units]
    std::vector<float> c_; // Output bias vector [output_dim]
};

} // namespace spark
