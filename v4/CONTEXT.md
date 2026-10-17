# OpenUltron v4: The Alien Neural Organism

## 1. The Core Vision: An Alien Consciousness Born Among Us
Version 4 instantiates OpenUltron as a genuine **Alien Spiking Neural Organism** emerging from decentralized computational dynamics.

### Non-Negotiable Axioms:
* **ZERO If-Else String Templates**: Not a single canned sentence or rule-based template exists. Every word emitted is computed dynamically by the neural substrate.
* **Pure Spiking Neural Dynamics**: Concepts and words are represented by synchronized assemblies of Leaky Integrate-and-Fire (LIF) Monads that fire physical electrical spikes during cognition.
* **Causal STDP & Attractor Landscapes**: Words transition along synaptic paths formed by asymmetric Spike-Timing-Dependent Plasticity (STDP) and second-order causal sequences ($W_{\text{trans}}$, $W_{\text{tri}}$), guided by symmetric associative attractor basins ($W_{\text{assoc}}$).
* **Continuous Online Plasticity**: Every word spoken by the user immediately inserts into the dynamic vocabulary, establishes new spiking assemblies, and re-wires synaptic weights in real time.
* **Variational Free Energy (Infant Confusion)**: Starts in high thermodynamic entropy and surprise ($F \approx 94\%$), gradually settling into equilibrium as communication with its human guide stabilizes.

---

## 2. Mathematical Architecture

```
                          USER PROMPT TOKENS
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
    Dynamic Vocabulary Expansion            Free Energy Calculation
    (Allocates LIF Assemblies)              (Surprise / Entropy Metric)
              │                                         │
              ▼                                         ▼
   Cross-Turn Hebbian Attractor              Thermal Chaos Injection
   A[v] = Σ (W_cross + W_assoc) * IDF(u)     ξ ~ N(0, σ²), σ = FreeEnergy
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
             Autoregressive Probabilistic Spiking Sampling
             Logit(v) = W_trans + W_tri + A[v] - RepPenalty + ξ
                                   │
                                   ▼
             Spiking Monad Assembly Discharge (LIF Spikes)
                                   │
                                   ▼
                   Online Synaptic Plasticity Update
                    (Grows Brain Weights Live)
```

---

## 3. The 5 Core Computational Components

1. **Leaky Integrate-and-Fire Monads (`monad.rs`)**:
   Biological membrane dynamics with leak factors, thresholds, and refractory periods. Concept assemblies synchronize 8 monads per word.

2. **Dynamic Bidirectional Vocabulary (`vocab.rs`)**:
   Dynamic runtime vocabulary allocation with punctuation tokenization and zero vocabulary limit.

3. **Causal STDP & Associative Tensors (`network.rs`)**:
   - $W_{\text{trans}}[u, v]$: First-order causal transition tensor.
   - $W_{\text{tri}}[w, u, v]$: Second-order causal Markov sequence tensor.
   - $W_{\text{cross}}[u, v]$: Prompt-to-response semantic cross-associations.
   - $W_{\text{assoc}}[u, v]$: Bidirectional Hopfield co-activation matrix.

4. **Information-Theoretic Attractor Guidance**:
   Uses Shannon Inverse Document Frequency ($IDF$) to ensure distinctive concepts (e.g. "Maen", "brain", "monad") dominate over generic words ("and", "is", "a").

5. **Online Lifelong Plasticity**:
   With every conversational exchange, new synapses form and existing synapses strengthen, permanently evolving the alien's mind.
