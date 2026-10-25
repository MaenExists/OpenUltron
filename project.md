# MASTER ENGINEERING PROMPT — SPARK
## Sparse Predictive Adaptive Recurrent Kernel
### Experimental Architecture for Distributed Emergent Intelligence

**Role:** Principal AI Research Engineer, Computational Neuroscience-Inspired Systems Architect, C++ Performance Engineer, and Scientific Software Developer.

**Development environment:** Google Antigravity with Gemini 3.8 Flash.

**Primary implementation language:** C++17.

**Project type:** Experimental AI architecture, scientific research engine, and CPU-first machine-learning system.

**Project codename:** SPARK.

---

# 1. YOUR MISSION

Build an experimental artificial intelligence architecture called SPARK (Sparse Predictive Adaptive Recurrent Kernel).

SPARK investigates whether sophisticated information-processing capabilities can emerge from interactions among many extremely simple computational units rather than relying exclusively on conventional large, layered neural networks.

This is a genuine research and engineering project. Your job is to implement a working computational architecture, establish scientifically meaningful experiments, test learning mechanisms, measure performance, and iteratively improve the system using experimental evidence.

Do not build a graphical mockup of artificial intelligence. Do not simulate learning with predetermined responses. Do not claim the system demonstrates intelligence unless experiments support that claim.

Our long-term ambition is to discover architectures that may deliver substantial general-purpose intelligence with far fewer computational resources than today's frontier models.

However, frontier-level intelligence on a laptop is an unproven research objective, not an acceptance criterion for the initial implementation.

The immediate goal is to build a working experimental system that can:

1. Contain thousands of extremely simple computational units.
2. Connect those units through efficient sparse graph structures.
3. Process information through distributed, recurrent activity.
4. Learn from observations, prediction errors, and feedback.
5. Develop useful representations without predefined semantic roles.
6. Retain information across multiple time steps.
7. Generalize learned relationships to unfamiliar inputs.
8. Experiment with local learning rules and adaptive connectivity.
9. Support repeated internal computation before producing predictions.
10. Run locally on conventional CPUs with measurable resource requirements.

Do not implement these capabilities simultaneously. Establish a stable foundation and add each mechanism through validated milestones.

# 2. CENTRAL SCIENTIFIC HYPOTHESIS

Our hypothesis is:

> A sufficiently organized network of small computational units may acquire useful, transferable cognitive capabilities through distributed learning, recurrent computation, and adaptive connectivity, without requiring explicit programming of individual concepts or problem-solving rules.

We want to test whether:

- Useful representations arise through learning.
- Recurrent interactions support temporal memory and computation.
- Sparse connectivity can achieve efficient information processing.
- Local learning rules can approximate or replace some aspects of global gradient-based optimization.
- Repeated computation improves performance without proportionally increasing parameter count.
- Adaptive topology improves performance or efficiency.
- A compact system can generalize learned rules to unseen combinations of problems.

Treat every statement above as a testable hypothesis, not an established result.

The project must distinguish between:

**Established components:** Sparse matrices, recurrent neural networks, numerical optimization, backpropagation through time, feedback alignment, supervised prediction, and reinforcement-learning principles.

**Experimental components:** Novel combinations of locally trained units, topology adaptation, compact distributed representations, and emergent problem-solving.

**Unproven ambitions:** Frontier-level general intelligence, major training-efficiency breakthroughs, and replacing conventional foundation-model architectures.

Do not confuse complex-looking activity with intelligent behavior.

# 3. ENGINEERING CONSTRAINTS

Use the following technical requirements.

## Language and build system

- C++17 is mandatory for the core system.
- Use CMake for compilation.
- Target Linux first, but keep the core reasonably portable.
- Use the C++ standard library wherever possible.
- Do not use Rust.
- Do not use Python, PyTorch, TensorFlow, JAX, or a pretrained LLM for core computation.
- Avoid heavy third-party libraries.
- A small, clearly justified dependency can be introduced only when it provides substantial value.
- Do not create a web application or graphical frontend during the initial milestones.
- Implement a command-line interface.
- Prefer simple, testable code over complicated abstraction systems.
- Make CPU execution the default.
- GPU acceleration may be researched later, but must not be required.

## Performance constraints

Design for ordinary consumer laptops, including CPU-only systems.

The initial prototype should work on a machine with approximately:

- 4 CPU cores.
- 8 GB RAM.
- No dedicated GPU.
- A conventional 64-bit operating system.

Begin with 128–1,024 computational units during training.

Later benchmark 10,000–100,000 units for forward inference, depending on available resources.

Do not allocate millions of units until smaller systems have passed functional and scientific tests.

Use optimized contiguous memory structures.

Avoid creating one heap-allocated C++ object for every unit or connection.

Avoid unnecessary dynamic allocations during repeated computation.

Measure memory, time, and accuracy instead of guessing.

# 4. AGENT DEVELOPMENT PROTOCOL

You are operating through Gemini 3.8 Flash in the Antigravity environment.

The scientific architecture is complex. Therefore, development MUST follow a controlled incremental workflow.

Do not generate the entire application in one large code-writing action.

For every milestone:

1. Read `AGENTS.md` and `docs/STATUS.md`.
2. Identify the next incomplete milestone.
3. State the files you intend to create or modify.
4. Define its expected behavior and test criteria.
5. Implement one small, logically complete component.
6. Compile immediately.
7. Run relevant tests.
8. Fix errors before continuing.
9. Record results, limitations, and implementation decisions.
10. Update the project status before ending the milestone.

Do not silently skip failed tests.

Do not invent benchmark results.

Do not claim to have run commands that were not actually executed.

Do not respond with pseudocode when executable C++ is required.

Do not create empty classes or meaningless placeholder methods merely to make the directory structure appear complete.

Do not rewrite working components without a specific technical reason.

Avoid unnecessary abstractions, speculative interfaces, massive classes, and premature optimization.

If a scientific mechanism is uncertain, implement a simple baseline before implementing the experimental alternative.

Keep changes small enough to diagnose failures.

At the end of each development session, report:

- What was implemented.
- Which files were changed.
- The exact commands executed.
- Which tests passed and failed.
- What experimental results were observed.
- What remains incomplete.
- The next recommended milestone.

If context becomes limited, preserve project progress in repository files rather than relying on conversational memory.

# 5. REPOSITORY STRUCTURE

Create the following initial structure, adding files only when their milestone is reached.

```text
spark/
├── CMakeLists.txt
├── AGENTS.md
├── README.md
├── .gitignore
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SCIENTIFIC_MODEL.md
│   ├── RESEARCH_PLAN.md
│   ├── EXPERIMENTS.md
│   ├── DECISIONS.md
│   └── STATUS.md
│
├── include/spark/
│   ├── config.hpp
│   ├── random.hpp
│   ├── graph.hpp
│   ├── network.hpp
│   ├── learning.hpp
│   ├── trainer.hpp
│   ├── environment.hpp
│   ├── metrics.hpp
│   └── checkpoint.hpp
│
├── src/
│   ├── main.cpp
│   ├── graph.cpp
│   ├── network.cpp
│   ├── learning.cpp
│   ├── trainer.cpp
│   ├── environments/
│   ├── metrics.cpp
│   └── checkpoint.cpp
│
├── tests/
│   ├── test_graph.cpp
│   ├── test_forward.cpp
│   ├── test_gradients.cpp
│   ├── test_learning.cpp
│   └── test_checkpoint.cpp
│
├── experiments/
│   ├── configs/
│   └── results/
│
└── scripts/
```

The initial implementation may contain fewer files.

Use `AGENTS.md` for permanent instructions and architectural invariants, not for enormous copied specifications.

Store detailed explanations in the `docs/` files.

Antigravity should be able to resume development using these files without needing the original conversation.

# 6. THE FUNDAMENTAL COMPUTATIONAL UNIT

The basic unit is called a **MicroUnit**.

A MicroUnit is an extremely simple stateful computational element.

It must not be implemented as a complete independent neural network.

Each unit participates in the global computation by receiving signals, updating its state, and influencing connected units.

Initially, it requires only:

- Current activation.
- Previous activation, maintained through double buffering.
- Bias.
- Information about incoming connections.

Long-term learned information is primarily stored in connection weights.

Do not store redundant state inside every unit.

The conceptual objective is eventual low-precision storage, potentially only a few bytes of persistent state per unit.

However, the initial training implementation MUST use 32-bit floating-point values for numerical reliability.

Do not prematurely quantize training states.

## 6.1 Mathematical state transition

Represent the network with a recurrent state vector:

\[
h(t)\in\mathbb{R}^{N}
\]

For each unit \(i\), compute:

\[
a_i(t)=
\alpha h_i(t-1)
+
\frac{\beta}{\sqrt{K_i}}
\sum_{j\in\mathcal{N}(i)}
w_{ij}h_j(t-1)
+
\sum_d U_{id}x_d(t)
+b_i
\]

Then:

\[
h_i(t)=\tanh(a_i(t))
\]

Where:

- \(N\): Number of computational units.
- \(K_i\): Number of incoming recurrent connections to unit \(i\).
- \(h_i(t)\): Current activation.
- \(w_{ij}\): Trainable recurrent connection weight.
- \(x(t)\): Current external input.
- \(U\): Trainable input projection.
- \(b_i\): Trainable bias.
- \(\alpha\): Recurrent state retention coefficient.
- \(\beta\): Recurrent input scaling coefficient.

Use configurable coefficients with stable initial values.

Begin with a modest recurrent contribution to reduce instability.

The precise coefficients should be experimentally tuned, not assumed optimal.

Use synchronous updates:

1. Read all values from the previous-state buffer.
2. Compute all next-state values.
3. Swap the buffers.

Never update states in place when doing so would unintentionally change the mathematical update rule.

Provide a numerically stable implementation of `tanh`.

Implement safeguards against NaN and infinity.

## 6.2 Important restriction

The MicroUnit must have no manually assigned semantic identity.

Do not designate individual units as:

- Mathematics neurons.
- Language neurons.
- Reasoning neurons.
- Memory neurons.
- Consciousness neurons.
- Object-detection neurons.

The architecture may have input and output interfaces, but internal computational units should initially be homogeneous.

Any internal specialization must arise through training, not through manually specified conceptual labels.

# 7. SPARSE CONNECTION ARCHITECTURE

Represent the network as a directed, sparse recurrent graph.

Each computational unit receives a configurable number of incoming connections.

Initial configurations:

- Tiny: 128 units, 8 incoming connections each.
- Small: 512 units, 16 incoming connections each.
- Medium: 1,024 units, 16–32 incoming connections each.
- Larger inference test: 10,000 units, 16–32 incoming connections each.

Do not assume larger configurations will necessarily learn better.

## 7.1 Storage format

Use a compressed-sparse-row-like structure for incoming connections.

Conceptually:

```cpp
struct SparseGraph {
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> sources;
    std::vector<float> weights;
};
```

For unit `i`, its incoming edges occupy:

```cpp
offsets[i] ... offsets[i + 1]
```

Each entry contains:

- Source unit index.
- Connection weight.

The destination is implicitly defined by the row.

Maintain contiguous memory and deterministic traversal order.

Add helper functions for:

- Graph initialization.
- Edge-count validation.
- Graph connectivity statistics.
- Weight statistics.
- Reproducible topology generation.
- Memory measurement.

Validate that connection indices are within bounds.

Prevent duplicate edges unless an explicit experiment requires them.

Avoid self-edges initially because the state equation already contains a self-retention term.

## 7.2 Graph topology variants

Implement these separately:

**Topology A — Random sparse graph**

Each unit receives a random selection of incoming connections.

This is the initial baseline.

**Topology B — Clustered sparse graph**

Organize units into computational clusters.

Most connections are local to a cluster, with a smaller number connecting clusters.

Use configurable local/long-range proportions.

Cluster membership is only an engineering property of the graph, not a semantic assignment.

**Topology C — Adaptive sparse graph**

Add later, after fixed-topology learning is verified.

Weak or consistently unhelpful connections can be removed.

New connections may be introduced using a measurable selection strategy.

Maintain a fixed edge budget during rewiring experiments.

Compare adaptive topology against the original fixed graph.

Never assume that rewiring improves performance.

# 8. EXTERNAL INFORMATION INTERFACE

The network must interact with experimental environments.

Separate the internal recurrent state from external observations.

Define an interface similar to:

```cpp
struct Observation {
    std::vector<float> values;
};
```

Provide a lightweight environment interface supporting:

- Reset.
- Current observation.
- Optional action submission.
- State transition.
- Prediction target.
- Episode termination.
- Evaluation metadata.

Keep environment simulation independent of the neural network.

This separation is essential for scientific evaluation.

## 8.1 Input projection

The input projection maps observable variables to internal units.

For the first prototype, use a small trainable or fixed input projection.

A sparse input projection is preferred for larger experiments.

Use contiguous arrays.

Keep dimensions explicit and validated.

Do not introduce text tokenization, language-model APIs, or multimodal encoders at this stage.

## 8.2 Output projection

Use a lightweight readout:

\[
z_k(t)=\sum_i V_{ki}h_i(t)+c_k
\]

For independent binary prediction targets:

\[
p_k(t)=\sigma(z_k(t))
\]

Where \(\sigma\) is the sigmoid function.

Use stable sigmoid and binary-cross-entropy implementations.

For one-hot classification tasks, optionally implement softmax with categorical cross-entropy as a separate output mode.

Do not use sigmoid and softmax interchangeably.

The output readout is a measurement interface, not a manually programmed reasoning engine.

# 9. PRIMARY LEARNING OBJECTIVE

The first scientifically testable capability will be predictive learning.

The network receives an observation and attempts to predict a target that is not directly available from the current input alone.

The initial objective is:

\[
L_{\text{prediction}}
=
-\frac{1}{TK}
\sum_{t,k}
\left[
y_{tk}\log p_{tk}
+
(1-y_{tk})\log(1-p_{tk})
\right]
\]

Where:

- \(T\) is the number of supervised time steps.
- \(K\) is the output dimension.
- \(y\) is the actual binary target.
- \(p\) is the predicted probability.

Compute the loss stably from logits wherever possible.

For categorical prediction, use categorical cross-entropy.

Every learning experiment must report both training performance and held-out evaluation performance.

We must prevent trivial solutions, data leakage, and memorization from being mistaken for generalization.

# 10. LEARNING ALGORITHM A — EXACT GRADIENT BASELINE

This is a mandatory reference implementation.

Before experimenting with novel local learning rules, establish that the chosen sparse recurrent architecture can learn using a conventional, reliable optimizer.

Implement truncated backpropagation through time (TBPTT) for the model defined above.

Initially support short unroll windows, approximately 4–8 time steps.

Store the forward activations needed for gradient computation.

Calculate gradients for:

- Recurrent connection weights.
- Input projection weights.
- Unit biases.
- Readout weights.
- Readout biases.

## 10.1 Backward recurrence

For binary outputs, the derivative of cross-entropy with respect to the output logits is proportional to:

\[
\frac{\partial L}{\partial z_k(t)}
=
p_k(t)-y_k(t)
\]

Include the correct loss normalization factor.

Propagate the resulting output error into the hidden state.

For each recurrent unit, apply:

\[
\frac{\partial h_i(t)}{\partial a_i(t)}
=
1-h_i(t)^2
\]

Accumulate recurrent parameter gradients across the unroll window.

Propagate hidden-state gradients backward in time through:

- Self-retention.
- Recurrent graph connections.

Do not accidentally discard gradients from future time steps inside the training window.

Treat the hidden state at the beginning of each truncated window as detached.

Document this approximation clearly.

Implement global gradient-norm clipping.

Begin with ordinary stochastic gradient descent.

Introduce Adam only if tests show a need for improved optimization.

Keep optimizer state separate from model parameters.

Training-time memory usage and inference-time memory usage must be measured independently.

## 10.2 Gradient verification

Write finite-difference gradient tests on tiny networks.

For selected parameters:

\[
g_{\text{numerical}}
=
\frac{L(w+\epsilon)-L(w-\epsilon)}
{2\epsilon}
\]

Compare against the analytically implemented gradient.

Use double precision for the numerical checking calculations where practical.

Account for floating-point tolerances.

Test recurrent, input, bias, and readout parameters separately.

Do not proceed with experimental learning algorithms until the gradient baseline passes its correctness tests.

A model that appears to train with incorrect gradients is not a trustworthy scientific baseline.

# 11. LEARNING ALGORITHM B — LOCAL FEEDBACK LEARNING

After establishing the exact-gradient reference, implement a separate experimental learning mode.

The aim is to investigate whether useful recurrent representations can be trained with primarily local information and lightweight feedback.

Do not claim that this is biologically exact or mathematically equivalent to TBPTT.

Use a simplified eligibility-trace and random-feedback mechanism inspired by research on feedback alignment and eligibility-based learning.

## 11.1 Local eligibility

For recurrent connection \(j\rightarrow i\), maintain:

\[
E_{ij}(t)=
\rho E_{ij}(t-1)
+
\left(1-h_i(t)^2\right)h_j(t-1)
\]

Where:

- \(E_{ij}\) is a connection eligibility trace.
- \(\rho\) controls trace decay.
- \(h_j(t-1)\) is the presynaptic activity.
- \(1-h_i(t)^2\) approximates local sensitivity.

This is a deliberately simplified trace. It does not capture all indirect recurrent dependencies and must not be presented as an exact gradient.

## 11.2 Feedback signal

Use a fixed random feedback matrix \(B\) to provide output-error information to internal units:

\[
g_i(t)=
\sum_k B_{ik}\left(p_k(t)-y_k(t)\right)
\]

Initialize \(B\) with a seeded random generator.

Keep \(B\) fixed during the initial experiment.

## 11.3 Local update

Update recurrent weights using:

\[
\Delta w_{ij}
=
-\eta
\frac{\beta}{\sqrt{K_i}}
g_i(t)E_{ij}(t)
\]

Apply appropriate update rules to input weights and biases using the same local feedback idea.

Train the readout using its exact directly available output gradient.

Use the same training and evaluation datasets as the TBPTT reference.

Report:

- Final loss.
- Learning speed.
- Memory usage.
- Training time.
- Generalization performance.
- Numerical stability.

Do not describe the algorithm as superior merely because it uses a more biologically inspired mechanism.

The primary scientific question is whether it can approach the performance of global gradient training while using less memory or more local computation.

If it fails, preserve the experimental results and analyze why.

# 12. EXPERIMENTAL ENVIRONMENTS

We will build a progressively more demanding set of small environments.

Each environment should reveal a particular computational capability.

## Environment 1 — Temporal Memory

Generate random binary input sequences.

At time \(t\), the system receives a bit.

Its task is to output the bit observed \(d\) time steps earlier.

Begin with:

- Delay 1.
- Delay 3.
- Delay 5.

Do not expose the target bit in the current input.

Use separate random seeds for training and testing.

Metrics:

- Bit accuracy.
- Cross-entropy.
- Accuracy versus delay.
- Accuracy versus recurrent network size.

This tests temporal memory rather than general intelligence.

The initial target is to demonstrate reliable learning on a small delay task.

## Environment 2 — Structured Sequence Prediction

Generate sequences using hidden transition rules.

Examples include recurring patterns or small finite-state processes.

The model should predict the next symbol from previous observations.

Compare it with:

- Constant predictor.
- Frequency-based predictor.
- First-order Markov predictor.

Use held-out sequences.

Avoid letting the train/test split share identical generated sequences.

This tests whether recurrent state provides a measurable advantage.

## Environment 3 — Action-Conditioned Grid World

Construct a deterministic 5×5 grid environment.

An agent can take five discrete actions:

- Up.
- Down.
- Left.
- Right.
- Stay.

Optional walls block movement.

Provide an observation containing:

- Agent position.
- Wall positions.
- Goal position.

Encode these using binary or one-hot vectors.

Provide the selected action to the model.

The prediction target is the agent's next position.

The network must learn transition behavior from examples rather than using a hardcoded transition table internally.

The environment simulator itself must implement correct ground-truth transitions.

Use randomized valid grid configurations.

Create separate training, validation, and test instances.

Metrics:

- Next-position prediction accuracy.
- Cross-entropy.
- Performance on unseen configurations.
- Collision-transition accuracy.
- Accuracy on previously unseen action/state combinations.

A learned transition model that predicts the next state correctly is evidence of environment modeling, not necessarily evidence of general causal reasoning.

## Environment 4 — Compositional Generalization

Introduce simple combinations of independent transition rules.

Train the system on some combinations and test it on new combinations.

For example, learn basic rules for movement and obstacle interactions separately, then evaluate combinations that were excluded from training.

Control the split carefully.

Avoid any overlap that could allow test examples to be reconstructed from memorized trajectories.

Compare against the gradient-trained baseline.

Only claim compositional generalization when the held-out test results support it.

## Environment 5 — Planning

Implement only after the world model has reliable transition predictions.

Allow the system to use repeated internal computation or predicted rollouts to evaluate possible actions.

Start with a very short planning horizon and small action space.

Do not claim that transition prediction alone produces planning.

The planning experiment must explicitly test whether the network can select actions that improve goal achievement.

Measure planning success against:

- Random actions.
- Simple heuristic policy.
- An oracle using true environment dynamics.

Clearly distinguish learned world modeling from an external planner operating on top of the learned world model.

# 13. RECURRENT INTERNAL REASONING

A major hypothesis is that a small network may benefit from reusing the same parameters for multiple internal computation steps.

Introduce configurable internal microsteps after the initial one-step recurrent model is validated.

For a single external observation, allow the state to evolve through multiple recurrent updates before generating an output.

Start with:

- 1 microstep.
- 2 microsteps.
- 4 microsteps.
- 8 microsteps.

Keep the same external input fixed during each internal refinement cycle unless a different input policy is explicitly defined.

When training with TBPTT, ensure the actual internal updates are included in the gradient unroll.

Never silently use a training graph that differs from inference.

Evaluate:

- Accuracy versus microsteps.
- Inference time versus microsteps.
- Whether improvements survive held-out testing.
- Whether the same improvements could be obtained by adding parameters instead.

The network must not be called a reasoning model solely because it repeats its calculations.

Repeated internal computation must produce measurable improvements in appropriate tasks.

# 14. ADAPTIVE TOPOLOGY

This is an advanced experimental feature.

Do not implement it in the first prototype.

Once the fixed-topology network learns reliably, allow structural reorganization.

Every configurable number of training steps:

1. Estimate connection usefulness from a defined statistic.
2. Select a small fraction of connections for removal.
3. Propose new connection candidates.
4. Maintain the total edge budget.
5. Rebuild sparse indexing structures safely.
6. Reset optimizer or eligibility state for newly created connections.
7. Evaluate whether the changes improve validation performance.

Test several candidate usefulness measures:

- Absolute weight magnitude.
- Moving average of update magnitude.
- Gradient-derived usefulness where available.

Document tradeoffs and computational overhead.

Do not equate a small connection weight with guaranteed uselessness.

Preserve a fixed-topology baseline under identical training conditions.

Do not introduce dynamic rewiring until the learning algorithms are sufficiently stable to support meaningful comparisons.

# 15. QUANTIZATION AND COMPRESSION

Low memory consumption is an important long-term goal.

However, correctness comes before compression.

Use FP32 model weights and states during initial training.

Once a trained model works, implement an optional INT8 inference representation.

For each output unit or sparse row, store:

- Quantized weight values.
- A scale factor.

An example symmetric quantization mechanism is:

\[
q=\operatorname{clamp}\left(
\operatorname{round}(w/s),-127,127
\right)
\]

with:

\[
s=\frac{\max |w|}{127}
\]

Handle all-zero rows correctly.

During inference, dequantize appropriately or use a validated integer accumulation strategy.

Compare quantized inference against FP32 inference.

Measure:

- Weight storage.
- Total graph storage.
- Runtime memory.
- Prediction differences.
- Task accuracy changes.
- Throughput.

Do not claim the entire system occupies one byte per unit merely because weights are one byte.

Connection IDs, graph indices, activations, readouts, and buffers also consume memory.

Binary and ternary quantization are future experiments, not initial requirements.

# 16. REPRODUCIBILITY

Scientific reproducibility is mandatory.

Every experiment must have:

- A deterministic random seed.
- Explicit configuration.
- Recorded network size.
- Recorded graph topology.
- Recorded training algorithm.
- Recorded learning rate.
- Recorded number of training updates.
- Recorded dataset generator parameters.
- Recorded validation and test seeds.
- Recorded numerical precision.
- Recorded training and inference times.

Implement a reproducible pseudo-random number generator.

Do not rely on unspecified global random state.

Store experiment settings alongside results.

Use CSV as the initial results format.

Example columns:

```text
experiment_id
seed
task
network_units
fan_in
topology
trainer
learning_rate
training_steps
train_loss
validation_loss
test_accuracy
peak_memory_bytes
training_seconds
inference_seconds
```

If certain metrics cannot be measured reliably, record them as unavailable rather than inventing numbers.

Use distinct training, validation, and test seeds.

Do not tune hyperparameters on the test set.

# 17. CHECKPOINT SYSTEM

Implement model checkpointing once training works.

Store:

- File format magic identifier.
- Checkpoint version.
- Network configuration.
- Sparse graph indices.
- Recurrent weights.
- Input projection.
- Output projection.
- Biases.
- Random seed metadata.
- Numerical representation.

Use a documented binary format.

Validate dimensions before allocating memory during loading.

Reject corrupted, truncated, or incompatible checkpoints.

Keep the checkpoint portable across compatible C++ builds where feasible, using explicit field sizes and documented byte order.

Optimizer state may be stored separately for resumable training.

Test:

1. Save a trained model.
2. Load it into a new process.
3. Evaluate the same input.
4. Compare outputs with an appropriate numerical tolerance.

The loaded model must preserve predictive behavior.

# 18. COMMAND-LINE INTERFACE

Build a simple CLI executable named `spark`.

The final interface should support commands resembling:

```bash
spark info

spark train \
  --task delay-copy \
  --units 128 \
  --fan-in 8 \
  --trainer bptt \
  --steps 2000 \
  --seed 42

spark eval \
  --checkpoint checkpoints/model.spark \
  --task delay-copy \
  --split test

spark benchmark \
  --units 10000 \
  --fan-in 16 \
  --ticks 100 \
  --seed 42
```

These are target commands. Implement them incrementally as their underlying functionality becomes available.

Provide clear usage information for invalid commands.

Validate input ranges.

Reject nonsensical configurations before allocating memory.

Provide a quiet mode and a verbose mode.

Do not print huge amounts of data during training.

Training logs should include useful numerical progress.

# 19. SCIENTIFIC BASELINES

The experimental architecture must be compared against simpler alternatives.

Implement baseline models as separate experiment modes where practical.

Initial baselines:

**Baseline A — No learning**

Randomly initialized network, unchanged throughout evaluation.

**Baseline B — Simple statistical prediction**

For sequence tasks, implement a majority/frequency predictor and a small Markov predictor.

**Baseline C — Gradient-trained recurrent network**

SPARK using exact TBPTT.

**Baseline D — Local-learning recurrent network**

SPARK using the experimental feedback-and-eligibility update rule.

Later:

**Baseline E — Dense recurrent network**

A small conventional recurrent network using comparable parameter counts and training data.

A sufficiently small dense configuration may be implemented through the existing computation infrastructure.

Compare model quality under controlled conditions.

Where possible, evaluate both equal parameter budgets and equal training-compute budgets.

Do not interpret a better result under a much larger compute budget as evidence of inherent efficiency.

# 20. METRICS FOR EMERGENT FUNCTIONALITY

We must define emergence operationally.

Simply observing complex activity does not count.

The system may be said to have acquired a capability when:

1. It performs measurably better than an appropriate baseline.
2. Its behavior depends on learned parameters.
3. The capability generalizes to unseen examples.
4. The behavior is not directly hardcoded.
5. Relevant ablations show that the architecture's components contribute to performance.

Possible measurements:

**Learning:** Does test loss improve with training?

**Memory:** How much information can the recurrent state retain across time?

**Prediction:** How well does the model anticipate unseen transitions?

**Generalization:** Can it solve inputs excluded from training?

**Compositionality:** Can it combine previously learned rules?

**Adaptation:** How much additional experience is required to learn a modified environment?

**Parameter efficiency:** What performance is achieved per trainable parameter?

**Compute efficiency:** What performance is achieved per unit of training or inference computation?

**Structural dependence:** What happens when recurrent connections are removed or randomized?

**Iterative computation:** Does increasing internal microsteps improve results?

Avoid presenting aggregate activation entropy or complex visual patterns as direct measures of intelligence.

These may be useful diagnostics, not sufficient evidence.

# 21. ABLATION EXPERIMENTS

Add controlled ablations after the primary model works.

Examples:

- Remove recurrent connections.
- Disable training of recurrent weights.
- Replace clustered topology with random topology.
- Disable connection adaptation.
- Remove eligibility traces.
- Vary recurrent state retention.
- Change graph sparsity.
- Change microstep count.
- Compare fixed and trained input projections.

Run multiple independent seeds for important findings.

Report distributions or means and variation rather than only highlighting the best run.

Document negative results.

Do not manufacture success by repeatedly tuning on held-out test data.

# 22. PERFORMANCE ENGINEERING

Optimize only after correctness is demonstrated.

## CPU implementation

Prefer contiguous vectors and predictable memory access.

Do not use recursive function calls for network computation.

Do not create one operating-system thread per computational unit.

Do not allocate new hidden-state vectors at every update.

Use double buffering.

Avoid repeated copying of large arrays where possible.

Keep sparse edge arrays compact.

Measure the effect of compiler optimization.

Start with portable scalar implementations.

Consider CPU parallelization or SIMD only after profiling.

Do not assume multithreading is automatically faster for small networks.

## Benchmarking

Measure:

- Units processed per second.
- Recurrent edge updates per second.
- Forward-pass latency.
- Total training time.
- Model memory.
- Graph memory.
- Temporary buffer memory.
- Checkpoint size.

Separate forward-only benchmarks from training benchmarks.

Include warmup runs where appropriate.

Use release builds for performance comparisons.

Report hardware and compiler information when available.

# 23. SAFETY AND NUMERICAL RELIABILITY

Implement defensive checks for:

- Invalid graph indices.
- Buffer overflows.
- Incorrect array dimensions.
- Invalid checkpoint lengths.
- NaN values.
- Infinite values.
- Exploding gradients.
- Loss becoming non-finite.
- Empty datasets.
- Invalid learning rates.
- Invalid graph degrees.
- Duplicate edges.

Use sanitizers in development builds when supported.

Avoid undefined behavior.

Favor deterministic single-threaded execution for the initial experimental version.

Introduce parallel execution only after reproducibility is established.

# 24. TESTING REQUIREMENTS

Every component needs automated tests.

## Graph tests

Verify:

- Correct number of units.
- Correct edge counts.
- Valid indices.
- Deterministic initialization.
- Correct CSR offsets.
- No unexpected duplicates.
- Correct graph iteration.

## Forward tests

Verify:

- Correct output dimensions.
- Deterministic results.
- Correct double-buffer behavior.
- Stable outputs for controlled inputs.
- Finite activation values.

## Gradient tests

Verify analytical gradients against numerical finite differences on tiny networks.

## Learning tests

Train on a minimal synthetic problem.

Demonstrate that training can reduce loss relative to its own initialization.

Verify that resetting the seed reproduces results within documented numerical tolerances.

## Checkpoint tests

Verify save/load equivalence.

## Benchmark tests

Verify benchmark execution produces legitimate timing and memory information.

A milestone is not complete merely because compilation succeeds.

# 25. IMPLEMENTATION ROADMAP

The project must be built in the following order.

## PHASE 0 — Research and repository foundation

Create:

- CMake configuration.
- Minimal executable.
- Documentation files.
- Persistent Antigravity rules.
- Reproducible random generator.
- Test infrastructure.
- Basic configuration validation.

Confirm that the project compiles and tests execute.

Do not implement learning yet.

## PHASE 1 — Sparse recurrent computation

Implement:

- Sparse graph.
- MicroUnit state arrays.
- Recurrent state update.
- Input injection.
- Readout.
- Deterministic initialization.
- Forward-pass tests.
- Basic CPU benchmark.

Deliver a functioning, untrained recurrent computational system.

## PHASE 2 — Learning with exact gradients

Implement:

- Binary prediction loss.
- TBPTT.
- Gradient accumulation.
- SGD.
- Gradient clipping.
- Finite-difference tests.
- Temporal memory dataset.
- Training CLI.

Confirm learning on at least one small synthetic task.

## PHASE 3 — Scientific comparison

Implement:

- Validation and test evaluation.
- No-learning baseline.
- Statistical baseline.
- Multiple seeds.
- CSV experiment logs.
- Loss curves represented through exported data.

Establish a reproducible benchmark.

## PHASE 4 — Local learning

Implement:

- Eligibility traces.
- Fixed random feedback.
- Local recurrent-weight updates.
- Local-learning training mode.
- Comparison against TBPTT.

Document both successes and failures.

## PHASE 5 — World-model experiments

Implement:

- Action-conditioned grid environment.
- Observation and action encoding.
- Next-state prediction.
- Held-out world configurations.
- Generalization evaluation.

Validate the system as a learned predictive model.

## PHASE 6 — Iterative internal computation

Implement:

- Configurable microsteps.
- Correct recurrent training through microsteps.
- Compute-versus-accuracy experiments.

Determine whether repeated computation improves generalization.

## PHASE 7 — Adaptive topology

Implement:

- Connection pruning.
- Controlled regrowth.
- Graph reconstruction.
- Fixed-edge-budget experiments.
- Comparisons against static sparsity.

## PHASE 8 — Memory optimization

Implement:

- INT8 inference weights.
- Quantized checkpoint format.
- FP32 comparison.
- Memory profiling.
- Larger-network inference benchmarks.

## PHASE 9 — Higher-level cognitive experiments

Explore:

- Compositional generalization.
- Simple planning.
- Continual adaptation.
- Catastrophic forgetting.
- Multi-environment learning.
- Learned internal representations.

These are research milestones, not guaranteed capabilities.

## PHASE 10 — Research report

Produce a report describing:

- Actual implemented architecture.
- Exact learning algorithms.
- Experimental configurations.
- Measured results.
- Baseline comparisons.
- Resource requirements.
- Failed hypotheses.
- Remaining limitations.
- Recommended future research.

Avoid exaggerated conclusions.

# 26. RESEARCH REFERENCES

Use these as technical background.

Do not copy their architectures wholesale.

**Eligibility propagation and local learning**

Bellec et al. (2020), "A solution to the learning dilemma for recurrent networks of spiking neurons."

https://www.nature.com/articles/s41467-020-17236-y

**Dynamic sparse training**

Evci et al. (2020), "Rigging the Lottery: Making All Tickets Winners."

https://proceedings.mlr.press/v119/evci20a.html

**Low-precision neural networks**

Wang et al., "BitNet: 1-bit Pre-training for Large Language Models."

https://www.jmlr.org/papers/v26/24-2050.html

Use these studies to identify established mechanisms, numerical considerations, and meaningful baselines.

Our system is not required to reproduce those papers.

Document which ideas are borrowed, which are modified, and which constitute genuinely new experiments.

# 27. RESEARCH INTEGRITY RULES

These rules override the desire to produce impressive-looking results.

Do not claim:

- Intelligence has emerged merely because network activity is complex.
- A model understands the world merely because it predicts an observation.
- A smaller model is more intelligent merely because it contains fewer parameters.
- A recurrent model performs reasoning merely because it has multiple computational iterations.
- Local learning is superior without controlled experiments.
- Frontier-level performance has been achieved without comparable evaluation.
- A system is conscious because of its architectural complexity.

Do not disguise conventional neural-network functionality with novel terminology.

Be explicit when SPARK is equivalent or closely related to an existing recurrent-network method.

Novelty must be supported by architectural differences or scientific results, not branding.

Negative experimental results are valuable and must be preserved.

# 28. REQUIRED FIRST ACTIONS

Begin by inspecting the current workspace.

If the repository does not exist, initialize it.

Create a concise implementation plan in the Antigravity planning interface, followed by a concrete task checklist.

Then implement **Phase 0 and Phase 1 only**.

Specifically, deliver:

1. A valid C++17 CMake project.
2. A working command-line executable.
3. Deterministic random-number generation.
4. A sparse recurrent graph.
5. A homogeneous MicroUnit state system using contiguous arrays.
6. A correct recurrent forward update.
7. A small input/output projection.
8. Configuration validation.
9. Automated graph and forward tests.
10. A simple forward-performance benchmark.
11. Scientific architecture documentation.
12. An updated milestone tracking file.

The executable must support a basic self-test or demonstration mode that generates a network, injects numerical input, performs multiple recurrent updates, and reports numerical state statistics.

Demonstrate that changing connection weights or external inputs changes network behavior.

Do not describe this as learned intelligence. Training is introduced in the next phase.

Use the following baseline build commands:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Record the actual outcomes.

Do not fabricate successful test results.

# 29. FINAL INSTRUCTIONS FOR THE CODING AGENT

Think like a scientific software engineer, not an application generator.

The objective is not to maximize the number of files or features.

The objective is to construct a system whose computational behavior can be understood, measured, and improved.

Prioritize, in this order:

1. Mathematical correctness.
2. Reproducibility.
3. Verified learning.
4. Generalization.
5. Scientific interpretability.
6. Resource efficiency.
7. Experimental architectural improvements.
8. Scalability.

Do not introduce complex functionality before validating its prerequisites.

Preserve implementation details in documentation.

Preserve experimental evidence.

Use existing scientific knowledge whenever it helps, but do not abandon the central architectural research question.

**The core question is whether a large number of extremely simple, sparsely connected, recurrent computational units can develop increasingly useful predictive, adaptive, and problem-solving capabilities through learning—and how efficiently they can do so.**

Build the experimental foundation carefully enough that future research can answer this question.

**START NOW: Implement Phases 0 and 1, compile the code, execute the tests, and report the verified results.**
