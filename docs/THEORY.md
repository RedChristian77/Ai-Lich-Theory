# Theoretical Framework: The Weight-Space Hypothesis

This document describes the theoretical foundation behind Lich Theory — a mechanistic hypothesis about how transformer models work internally and why the proposed Lich architecture should be more efficient.

---

## The Weight-Space Hypothesis

**Core claim:** A trained model is essentially one giant attention layer. Each parameter defines the location of a weight in a massive high-dimensional space (1000+ dimensions). The billions of parameters don't make the model "smart" — they define the **geometry of the space** the attention mechanism navigates.

### How Inference Works (Under This Model)

1. **Space construction:** When a model runs, it processes its billions of parameters to construct a high-dimensional weight space. This is where the compute cost actually lives — not in the "thinking," but in building the space.
2. **Navigation:** The attention mechanism then navigates this constructed space to find the region relevant to the input query.
3. **Output:** The relevant region of the space produces the response.

A 70B parameter model isn't smarter than a 7B model. It has a **higher-resolution space** — more defined coordinates, finer-grained regions — which means it can make more precise distinctions. But it still constructs the entire space for every query.

### The Role of the Transformer

The transformer architecture's purpose, under this hypothesis, is **dimensional alignment**. It ensures that all components — the attention mechanism, the weight matrices, the residual connections — are the same shape so they "fit" together. The transformer is plumbing, not intelligence. The intelligence is in the geometry of the trained weight space.

---

## The Efficiency Problem

Current inference is wasteful. Every query constructs the entire weight space, then uses a tiny fraction of it.

### Three Tiers of Efficiency

| Approach | Strategy | Cost |
|----------|----------|------|
| **Monolithic models** | Build the entire space, then search for the relevant corner | Full construction cost every time |
| **Mixture-of-Experts (MoE)** | Spatial indexing — ask "which subspace is relevant?" and activate only those expert regions | Reduced activation, but still one model |
| **Lich architecture** | Identify relevant weight regions **before** construction, then build only from those weights | Construction cost proportional to query, not model size |

The key insight: MoE is a better search strategy within the same space. The Lich architecture is a fundamentally different compute model — **"search first, build only what you need"** instead of **"build everything, search once."**

You don't need the "Japanese history" weights when looking up engine torque. Why construct that part of the space at all?

---

## Headless Models as Pre-Partitioned Weight Spaces

In the proposed Lich architecture, headless models are not just "specialist networks." They are **pre-partitioned weight spaces** — domains that have already been separated out during training.

- Each headless model defines the geometry of a specific domain (code, creative writing, mathematics, etc.)
- The Lich router's job is to identify which domain spaces are relevant to a query
- Only the selected spaces are loaded and constructed
- The router picks which weight spaces to compose, rather than indexing into a monolithic blob of parameters

This is why the architecture should scale efficiently: adding a new domain means adding a new pre-partitioned weight space, not expanding a single monolithic model. The construction cost stays proportional to the query's actual requirements.

---

## Supporting Evidence from This Repository

The experiments in this repo validate the foundational building blocks of this theory:

### Weight-Space Construction is Stable

**Basin formation tests** show that attention weights converge to stable, deterministic patterns — the "space" being constructed is consistent and reproducible, not random. This is a prerequisite for the weight-space hypothesis: if the space were unstable, routing into it would be meaningless.

### Constructed Spaces are Portable

**Persistence tests** demonstrate zero degradation across save/load cycles. The constructed weight space can be serialized, stored, and reconstructed identically. This validates the concept of pre-partitioned weight spaces — they can exist as independent, storable artifacts.

### A Single Routing Layer Can Select the Right Space

**BERT routing at 100% accuracy** proves that a trained attention layer (the Lich) can reliably identify which space is relevant for a given query. This is the "search first" part of "search first, build only what you need."

### Trained Spaces Resist Override

**Preference basin tests** show that once a weight space is trained, its structure dominates — even when given contradictory input. The space's geometry is the authority, not the query. This supports the claim that trained weight spaces are stable structures, not transient activations.

---

## Connections to Existing Research

The weight-space hypothesis is consistent with several established observations:

**LoRA and adapter methods** freeze most of a trained model's parameters and fine-tune a small active portion. This works surprisingly well — which makes sense if the bulk of the model is a correctly constructed weight space that doesn't need to change. You're just adjusting the navigation, not the space.

**Mixture-of-Experts** architectures use learned routing to activate specialist sub-networks. This is spatial indexing within a single model — a less efficient version of what the Lich architecture proposes with separate, pre-partitioned weight spaces.

**The Lottery Ticket Hypothesis** finds that trained networks contain small, sparse substructures that do most of the actual work. Under the weight-space hypothesis, these are the relevant subspaces — the regions of the weight space that matter for a given class of queries.

---

## The Word-Deciphering Experiment

`[DATA PENDING]`

A preliminary experiment trained a single, pure attention layer to decipher words — no transformer stack, no multi-head attention, no feed-forward layers. Just one attention mechanism operating on a weight space.

The experiment demonstrated (crudely) that a single attention layer can navigate a weight space to produce meaningful output. This supports the core claim: the attention mechanism is the navigator, and the weight space is the territory.

**Test data and scripts to be added when available.** The experiment was run on limited hardware and needs to be cleaned up and documented with full reproducibility standards before inclusion.

---

## Predictions and Future Work

If the weight-space hypothesis is correct, the following should be testable:

### Predictions

1. **Layer freezing should be nearly lossless.** Freezing trained layers (beyond what LoRA already demonstrates) and only running the attention routing should produce outputs comparable to full inference — because the frozen layers ARE the space, and the attention layer is the only part that needs to be active.

2. **Domain-partitioned models should compose.** Taking weight spaces trained on different domains and routing between them should produce coherent outputs — because each space is self-contained and the router is the only integration point.

3. **Construction cost should be measurable and separable.** It should be possible to measure the compute cost of "space construction" vs. "space navigation" separately, and the former should dominate.

### Required Experiments

| Experiment | What It Tests | Hardware Needs |
|-----------|---------------|----------------|
| Layer freezing at scale | How much of a trained model can be frozen without output degradation | Mid-range GPU (8GB+) |
| Domain partitioning | Training separate weight spaces and routing between them | Multiple GPUs or sequential training |
| Construction cost profiling | Measuring compute spent on space construction vs. navigation | Profiling tools, any GPU |
| Word-deciphering replication | Reproducing and scaling the pure attention layer experiment | Basic GPU |
| Full Lich prototype | End-to-end: router + headless models + synthesizer | Significant GPU resources |

### Hardware Limitations

Current experiments are constrained by available hardware. The theory's strongest predictions (domain partitioning, full Lich prototype) require resources beyond what's currently available. The experiments in this repo prove the building blocks; scaling them is a resource problem, not a conceptual one.
