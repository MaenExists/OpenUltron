# OpenUltron v1: The Homeostatic Spiking Monad & Self-Organized Criticality

## 1. Architectural Hypothesis for v1
In Version 1, we test the hypothesis that **computational capacity and life-like adaptability emerge when a collective network of discrete, resource-bounded Monads self-tunes to the "Edge of Chaos" (Criticality)** through purely local homeostatic and causal rules.

If the Monad rules are correctly balanced:
1. The collective brain will neither collapse into silence (sub-critical quiescence) nor blow up into uncontrollable saturation (super-critical epilepsy).
2. It will exhibit **Self-Organized Criticality (SOC)**, producing scale-free information avalanches governed by power-law dynamics.
3. It will adapt its internal connections based strictly on temporal causality (STDP) without any global loss function or external backpropagation.

---

## 2. Monad Design: The Atomic Part

A single Monad is an autonomous micro-agent occupying a microscopic memory footprint (cache-line friendly, $\le 64$ bytes). It has no global awareness; its behavior is entirely governed by internal state variables and immediate synaptic links.

### Monad State Variables:
* $V \in \mathbb{R}$ — **Internal Membrane Potential** (accumulated signal).
* $\theta \in \mathbb{R}$ — **Dynamic Firing Threshold** (self-regulating homeostatic setpoint).
* $E \in [0, 1]$ — **Metabolic Energy Reserve** (vitality / survival resource).
* $\tau_{\text{ref}} \in \mathbb{N}$ — **Refractory Counter** (post-firing cooldown).
* $\mathcal{S}_{\text{out}}$ — **Outgoing Synaptic Set** $\{ (j, w_j, \Delta t_j) \}$ where $j$ is the downstream Monad ID, $w_j \in [-1, 1]$ is the synaptic weight, and $\Delta t_j$ is the conduction delay.
* $t_{\text{last}}$ — **Timestamp of Last Fired Spike** (for local causal plasticity).

---

## 3. The Four Local Rules of the Monad

### Rule 1: Leaky Integration & Event Generation (Temporal Summation)
Between incoming events at time step $t$:
$$V(t) = V(t-1) \cdot \lambda_V + \sum_{i \in \text{Inputs}(t)} w_{i} - \beta_{\text{leak}}$$
where $\lambda_V \in (0, 1)$ is the potential leak factor.
When $V(t) \ge \theta(t)$ and $\tau_{\text{ref}} = 0$:
1. The Monad fires a discrete binary pulse: $S(t) = 1$.
2. Potential resets: $V(t) \leftarrow V_{\text{reset}}$.
3. Refractory counter sets: $\tau_{\text{ref}} \leftarrow \tau_{\text{max}}$.
4. Outgoing pulses are scheduled to targets $j \in \mathcal{S}_{\text{out}}$ with latency $\Delta t_j$.

### Rule 2: Dynamic Homeostasis (Threshold Regulation)
To prevent network runaway saturation or silent extinction without any central coordinator, each Monad regulates its own excitability:
$$\theta(t+1) = \theta(t) + \eta_\theta \cdot \left( S(t) - r_{\text{target}} \right)$$
* If the Monad fires ($S(t) = 1$), $\theta$ increases, making it harder to fire next time.
* If the Monad remains silent, $\theta$ slowly relaxes toward a resting baseline at rate $\eta_\theta \cdot r_{\text{target}}$, increasing sensitivity.

### Rule 3: Spike-Timing-Dependent Plasticity (STDP - Causal Learning)
Synaptic weights are updated purely through local temporal correlation:
$$\Delta w_{ij} = \begin{cases}
A^+ \cdot \exp\left( -\frac{t_{\text{post}} - t_{\text{pre}}}{\tau_+} \right) & \text{if } t_{\text{post}} > t_{\text{pre}} \quad (\text{causal: Pre led to Post}) \\
-A^- \cdot \exp\left( -\frac{t_{\text{pre}} - t_{\text{post}}}{\tau_-} \right) & \text{if } t_{\text{post}} < t_{\text{pre}} \quad (\text{anti-causal: Post fired before Pre})
\end{cases}$$
Weights are strictly bounded in $w_{ij} \in [w_{\min}, w_{\max}]$, preventing infinite weight growth.

### Rule 4: Metabolic Conservation & Structural Plasticity
* Every spike costs energy: $E(t) = \max(0, E(t-1) - c_{\text{spike}})$.
* Influx of environmental reward/metabolites diffuses locally, recharging active cells: $E(t) \leftarrow \min(1.0, E(t) + \delta_E)$.
* **Pruning**: Synapses whose weights decay below $|w_{ij}| < \epsilon_{\text{prune}}$ are severed.
* **Sprouting**: Energetically saturated Monads ($E > E_{\text{grow}}$) sprout new connections to frequently co-active neighbors.

---

## 4. Mathematical Formulation of Collective Emergence

How do we prove mathematically that a collective brain of $N$ Monads exhibits emergent intelligence rather than chaotic noise?

### A. The Branching Parameter ($\sigma$) and The Critical State
Let $\langle S_t \rangle$ be the total number of Monads firing at discrete time $t$. The branching ratio is defined as:
$$\sigma = \frac{\langle S_{t+1} \rangle}{\langle S_t \rangle}$$
* **Sub-critical ($\sigma < 1$)**: Information dampens exponentially; network enters a coma.
* **Super-critical ($\sigma > 1$)**: Information explodes exponentially; network suffers epileptic seizure.
* **Critical ($\sigma \approx 1.0$)**: Information propagates indefinitely across spatial and temporal scales without decaying or saturating. **This is the prerequisite state for maximal dynamic range, maximal memory retention, and computational emergence.**

### B. Self-Organized Criticality (SOC) & Avalanche Power-Law Distribution
In a critical complex system, the probability distribution $P(s)$ of avalanche sizes $s$ (the total number of spikes in a causally connected cascade) follows a scale-free power law:
$$P(s) \propto s^{-\alpha}$$
In biological cortical networks and theoretical mean-field models (Bak-Tang-Wiesenfeld, Beggs & Plenz), the theoretical critical exponent is:
$$\alpha \approx 1.50$$
Deviations from $\alpha = 1.50$ quantify the distance of the Monad collective from optimal critical emergence.

### C. Kuramoto Phase Synchronization Order Parameter ($R$)
To quantify collective coherence without global clocking:
$$R(t) = \frac{1}{N} \left| \sum_{j=1}^N e^{i \phi_j(t)} \right|$$
where $\phi_j(t) = 2\pi \cdot \frac{t - t_{j,\text{last}}}{T_{j,\text{cycle}}}$ represents the phase of Monad $j$.
* $R \approx 0$: Incoherent white noise.
* $R \approx 1$: Pathological lockstep synchronization (loss of information entropy).
* $0.2 \le R \le 0.6$: **Meta-stable phase synchronization**—clusters dynamically synchronize and desynchronize, which is the hallmark of active cognitive processing.

---

## 5. Implementation Roadmap for v1
1. **`monad.rs`**: Single Monad struct with deterministic micro-rules.
2. **`theory.rs`**: Mathematical evaluation engine calculating branching ratio $\sigma$, avalanche size distribution $P(s)$, power-law regression exponent $\alpha$, and Kuramoto order parameter $R$.
3. **`network.rs`**: High-performance substrate simulating networks of $10^2$ to $10^5$ Monads in native binary code.
4. **`tests/`**: Unit and integration test suite mathematically validating rule adherence, homeostatic stability, STDP weight adaptation, and the emergence of critical avalanches.
