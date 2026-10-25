#include "spark/network.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace spark {

MicroUnitNetwork::MicroUnitNetwork(const NetworkConfig& net_config, const SparseGraph& graph)
    : config_(net_config), graph_(graph) {
    std::string err;
    if (!config_.validate(err)) {
        throw std::invalid_argument("Invalid NetworkConfig: " + err);
    }
    if (graph_.num_units() != config_.num_units) {
        throw std::invalid_argument("Graph num_units (" + std::to_string(graph_.num_units()) +
                                   ") does not match NetworkConfig num_units (" +
                                   std::to_string(config_.num_units) + ")");
    }

    const uint32_t N = config_.num_units;
    const uint32_t Din = config_.input_dim;
    const uint32_t Dout = config_.output_dim;

    h_prev_.assign(N, 0.0f);
    h_curr_.assign(N, 0.0f);
    a_curr_.assign(N, 0.0f);
    biases_.assign(N, 0.0f);

    U_.assign(static_cast<size_t>(N) * Din, 0.0f);
    V_.assign(static_cast<size_t>(Dout) * N, 0.0f);
    c_.assign(Dout, 0.0f);

    Random rng(config_.seed);
    initialize_parameters(rng);
}

MicroUnitNetwork::MicroUnitNetwork(const NetworkConfig& net_config, const SparseGraphConfig& graph_config)
    : MicroUnitNetwork(net_config, SparseGraph::generate(graph_config)) {}

void MicroUnitNetwork::initialize_parameters(Random& rng) {
    const uint32_t N = config_.num_units;
    const uint32_t Din = config_.input_dim;
    const uint32_t Dout = config_.output_dim;

    // Xavier initialization for input projection U
    float u_limit = std::sqrt(6.0f / static_cast<float>(Din + N));
    for (float& u : U_) {
        u = rng.uniform_float(-u_limit, u_limit);
    }

    // Xavier initialization for output projection V
    float v_limit = std::sqrt(6.0f / static_cast<float>(N + Dout));
    for (float& v : V_) {
        v = rng.uniform_float(-v_limit, v_limit);
    }

    // Biases initialized to zero
    std::fill(biases_.begin(), biases_.end(), 0.0f);
    std::fill(c_.begin(), c_.end(), 0.0f);
}

void MicroUnitNetwork::reset_state() {
    std::fill(h_prev_.begin(), h_prev_.end(), 0.0f);
    std::fill(h_curr_.begin(), h_curr_.end(), 0.0f);
    std::fill(a_curr_.begin(), a_curr_.end(), 0.0f);
}

void MicroUnitNetwork::set_state(const std::vector<float>& h) {
    if (h.size() != config_.num_units) {
        throw std::invalid_argument("State size mismatch");
    }
    h_curr_ = h;
    h_prev_ = h;
}

void MicroUnitNetwork::recurrent_microstep(const std::vector<float>& input) {
    if (input.size() != config_.input_dim) {
        throw std::invalid_argument("Input vector size (" + std::to_string(input.size()) +
                                   ") does not match input_dim (" +
                                   std::to_string(config_.input_dim) + ")");
    }

    const uint32_t N = config_.num_units;
    const uint32_t Din = config_.input_dim;
    const float alpha = config_.alpha;
    const float beta = config_.beta;

    const auto& offsets = graph_.offsets;
    const auto& sources = graph_.sources;
    const auto& weights = graph_.weights;

    // Synchronous update: read entirely from h_prev_, write to a_curr_ and h_curr_
    for (uint32_t i = 0; i < N; ++i) {
        float recurrent_sum = 0.0f;
        uint32_t start = offsets[i];
        uint32_t end = offsets[i + 1];
        uint32_t Ki = end - start;

        for (uint32_t e = start; e < end; ++e) {
            uint32_t src = sources[e];
            recurrent_sum += weights[e] * h_prev_[src];
        }

        float recurrent_term = 0.0f;
        if (Ki > 0) {
            recurrent_term = (beta / std::sqrt(static_cast<float>(Ki))) * recurrent_sum;
        }

        // Input projection term
        float input_term = 0.0f;
        size_t u_row_offset = static_cast<size_t>(i) * Din;
        for (uint32_t d = 0; d < Din; ++d) {
            input_term += U_[u_row_offset + d] * input[d];
        }

        // Preactivation equation:
        // a_i(t) = \alpha * h_i(t-1) + recurrent_term + input_term + b_i
        float a_i = alpha * h_prev_[i] + recurrent_term + input_term + biases_[i];

        // Safeguard against numerical extremes before tanh
        if (a_i > 20.0f) a_i = 20.0f;
        if (a_i < -20.0f) a_i = -20.0f;

        a_curr_[i] = a_i;
        h_curr_[i] = std::tanh(a_i);
    }

    // Synchronous buffer transition: previous buffer is set to current for subsequent step
    h_prev_ = h_curr_;
}

void MicroUnitNetwork::forward_step(
    const std::vector<float>& input,
    std::vector<float>& out_logits,
    std::vector<float>& out_probs,
    ReadoutMode mode
) {
    // Perform internal microsteps
    for (uint32_t m = 0; m < config_.microsteps; ++m) {
        recurrent_microstep(input);
    }

    // Compute readout projection
    compute_readout(out_logits, out_probs, mode);
}

void MicroUnitNetwork::compute_readout(
    std::vector<float>& out_logits,
    std::vector<float>& out_probs,
    ReadoutMode mode
) const {
    const uint32_t N = config_.num_units;
    const uint32_t Dout = config_.output_dim;

    out_logits.resize(Dout);
    out_probs.resize(Dout);

    for (uint32_t k = 0; k < Dout; ++k) {
        float z_k = c_[k];
        size_t v_row_offset = static_cast<size_t>(k) * N;
        for (uint32_t i = 0; i < N; ++i) {
            z_k += V_[v_row_offset + i] * h_curr_[i];
        }
        out_logits[k] = z_k;
    }

    if (mode == ReadoutMode::Sigmoid) {
        for (uint32_t k = 0; k < Dout; ++k) {
            out_probs[k] = stable_sigmoid(out_logits[k]);
        }
    } else if (mode == ReadoutMode::Softmax) {
        stable_softmax(out_logits, out_probs);
    } else { // Linear
        out_probs = out_logits;
    }
}

float MicroUnitNetwork::stable_sigmoid(float z) {
    if (z >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-z));
    } else {
        float ez = std::exp(z);
        return ez / (1.0f + ez);
    }
}

void MicroUnitNetwork::stable_softmax(const std::vector<float>& logits, std::vector<float>& probs) {
    if (logits.empty()) {
        probs.clear();
        return;
    }
    probs.resize(logits.size());

    float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        float ex = std::exp(logits[i] - max_logit);
        probs[i] = ex;
        sum += ex;
    }

    float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] *= inv_sum;
    }
}

bool MicroUnitNetwork::validate(std::string& err_msg) const {
    if (!graph_.validate(err_msg)) {
        return false;
    }
    const uint32_t N = config_.num_units;
    const uint32_t Din = config_.input_dim;
    const uint32_t Dout = config_.output_dim;

    if (h_prev_.size() != N || h_curr_.size() != N || a_curr_.size() != N || biases_.size() != N) {
        err_msg = "State or bias buffer dimension mismatch";
        return false;
    }
    if (U_.size() != static_cast<size_t>(N) * Din) {
        err_msg = "Input projection matrix dimension mismatch";
        return false;
    }
    if (V_.size() != static_cast<size_t>(Dout) * N) {
        err_msg = "Output projection matrix dimension mismatch";
        return false;
    }
    if (c_.size() != Dout) {
        err_msg = "Output bias dimension mismatch";
        return false;
    }

    // Check for NaN or Inf in parameters
    for (float b : biases_) {
        if (!std::isfinite(b)) { err_msg = "Non-finite bias value"; return false; }
    }
    for (float u : U_) {
        if (!std::isfinite(u)) { err_msg = "Non-finite input weight value"; return false; }
    }
    for (float v : V_) {
        if (!std::isfinite(v)) { err_msg = "Non-finite output weight value"; return false; }
    }
    for (float ci : c_) {
        if (!std::isfinite(ci)) { err_msg = "Non-finite output bias value"; return false; }
    }

    return true;
}

size_t MicroUnitNetwork::parameter_count() const {
    return graph_.total_edges() +           // Recurrent weights
           U_.size() +                      // Input projection weights
           biases_.size() +                 // Unit biases
           V_.size() +                      // Output projection weights
           c_.size();                       // Output biases
}

size_t MicroUnitNetwork::memory_bytes() const {
    size_t state_mem = (h_prev_.capacity() + h_curr_.capacity() + a_curr_.capacity() + biases_.capacity()) * sizeof(float);
    size_t proj_mem = (U_.capacity() + V_.capacity() + c_.capacity()) * sizeof(float);
    return state_mem + proj_mem + graph_.memory_bytes() + sizeof(MicroUnitNetwork);
}

} // namespace spark
