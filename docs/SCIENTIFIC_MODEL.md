# Scientific Model & Mathematical Formulation

## 1. Network Dynamics

Let $N$ be the number of computational units. The recurrent state at time $t$ is $h(t) \in [-1, 1]^N$.
For each unit $i \in \{0, \dots, N-1\}$:

$$a_i(t) = \alpha h_i(t-1) + \frac{\beta}{\sqrt{K_i}} \sum_{j \in \mathcal{N}(i)} w_{ij} h_j(t-1) + \sum_{d=0}^{D_{in}-1} U_{id} x_d(t) + b_i$$

$$h_i(t) = \tanh(a_i(t))$$

Where:
- $\mathcal{N}(i)$: Set of incoming presynaptic neighbor units to $i$.
- $K_i = |\mathcal{N}(i)|$: In-degree (fan-in) of unit $i$. If $K_i = 0$, the normalization factor is treated as $1.0$ and recurrent sum is $0$.
- $w_{ij}$: Synaptic weight of directed edge $j \to i$.
- $\alpha \in [0, 1]$: Recurrent self-retention coefficient (default: $0.8$).
- $\beta > 0$: Recurrent interaction scale (default: $0.5$).
- $U \in \mathbb{R}^{N \times D_{in}}$: Input projection weight matrix.
- $x(t) \in \mathbb{R}^{D_{in}}$: External input vector at step $t$.
- $b_i \in \mathbb{R}$: Unit bias.

## 2. Readout Interface

The readout maps the distributed hidden representation to the task target:

$$z_k(t) = \sum_{i=0}^{N-1} V_{ki} h_i(t) + c_k, \quad k \in \{0, \dots, D_{out}-1\}$$

For binary prediction / independent outputs:
$$p_k(t) = \sigma(z_k(t)) = \frac{1}{1 + e^{-z_k(t)}}$$

For categorical prediction:
$$p_k(t) = \frac{e^{z_k(t) - \max_j z_j(t)}}{\sum_j e^{z_j(t) - \max_j z_j(t)}}$$

## 3. Loss Functions

Binary Cross Entropy (computed safely from logits):
$$\mathcal{L}_{BCE}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$$

Categorical Cross Entropy:
$$\mathcal{L}_{CE}(z, y) = -\sum_k y_k \log(p_k)$$

## 4. Topologies
- **Random Sparse (Erdős–Rényi variant)**: Each unit randomly chooses $K$ distinct presynaptic sources ($j \neq i$).
- **Clustered Sparse (Small-World-like)**: Units partitioned into $C$ clusters. For unit in cluster $c$, incoming edges are drawn with probability $p_{intra}$ from cluster $c$ and $1 - p_{intra}$ from external clusters.
- **Adaptive Sparse**: Dynamic rewiring based on synaptic utility budgets (introduced in Phase 7).
