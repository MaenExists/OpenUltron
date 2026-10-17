# OpenUltron v2: Sensorimotor Language & Attractor Trajectories

## 1. Vision & Architectural Hypothesis for v2
In Version 2, OpenUltron advances from an unembodied biological tissue into an entity equipped with **Sensory Perception ("The Eyes")** and **Motor Articulation ("The Voice")**, capable of learning and generating human language without transformers, token embeddings, or dense matrix backpropagation.

### The Core Hypothesis:
Language is not a static lookup table; it is a **temporal attractor trajectory** through high-dimensional phase space. When a sequence of characters/words enters the system through sensory receptive fields, it sets off spatio-temporal waves through a scaled network of Monads. Through local Spike-Timing-Dependent Plasticity (STDP) and predictive feedback, the network self-organizes stable pathways such that perceiving "H-E-L-L" naturally collapses the network's phase state into activating "O".

---

## 2. Sensorimotor Language Architecture

```
                  ┌─────────────────────────────┐
                  │    INPUT TEXT / SENSORY     │
                  │   ("The Eyes" Receptive)    │
                  └──────────────┬──────────────┘
                                 │
                                 ▼ (Spike train encoding)
                  ┌─────────────────────────────┐
                  │    SCALED MONAD RESERVOIR   │
                  │   (Dense Recurrent Mesh)    │
                  │  - Causal STDP Learning     │
                  │  - Homeostatic Equilibrium  │
                  │  - Attractor Trajectories   │
                  └──────────────┬──────────────┘
                                 │
                                 ▼ (Readout accumulation)
                  ┌─────────────────────────────┐
                  │    MOTOR TEXT READOUT       │
                  │   ("The Voice" Actuators)   │
                  └─────────────────────────────┘
```

### Component A: The Eyes (Sensory Receptive Field)
* The sensory layer maps discrete alphabet symbols (ASCII printable character set $\Sigma = \{32 \dots 126\}$) into distinct spatial clusters of input Monads.
* When character $c_t$ is perceived, its receptive cluster receives a burst of current, seeding directional waves into the deeper reservoir.

### Component B: The Brain Fabric (Large-Scale Monad Reservoir)
* Scaled up to tens of thousands of Monads ($10^4$ to $10^5+$) interconnected with small-world small-latency wiring.
* Fulfills the role of a **Liquid State Machine**: incoming sequential text perturbs the high-dimensional liquid, and the reverberating echoes represent working memory of recent history.

### Component C: The Voice (Motor Output Layer)
* A dedicated bank of motor Monads, each corresponding to a character in $\Sigma$.
* Reservoir Monads form plastic synaptic projections onto these motor units.
* When prompted, the network's internal reverberation drives the motor units. The character whose motor unit accumulates the highest activation or fires first is emitted as the generated text.

---

## 3. How Language Learning Occurs (Online Local Plasticity)
* **Zero Backpropagation**: Learning is entirely online and real-time.
* **Predictive Hebbian Association**:
  * As character $c_t$ ripples through the reservoir, the motor monad for the true target $c_{t+1}$ receives an energetic priming signal.
  * Reservoir cells whose spikes arrived immediately prior to $c_{t+1}$ experience Long-Term Potentiation (LTP), strengthening the causal link.
  * Competitor cells that fired erroneously experience Long-Term Depression (LTD).
* **Homeostatic Balance**: Frequent characters (like spaces or 'e') are prevented from dominating the voice layer through dynamic threshold elevation.

---

## 4. Interactive Interface
OpenUltron v2 includes a dedicated real-time CLI terminal:
* **Interactive Chat / Query**: Send sentences to OpenUltron and observe its emergent response.
* **Online Teacher Mode**: Feed text, sentences, or books directly through the terminal to teach it in real-time.
* **Brain State Diagnostics**: Inspect active spike rates, metabolic energy, pruned synapses, and attractor coherence live in the console.
