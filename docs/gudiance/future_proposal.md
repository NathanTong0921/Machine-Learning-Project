# Future Proposal Notes

## Purpose

This note collects the most plausible next-step directions for the UCI `Predict Students' Dropout and Academic Success` project.

It is not a claim of superiority. It is a proposal draft for future experiments, written to help decide what is worth trying next and what should stay secondary.

## Current signal from experiments

Two facts matter most:

- `TabM-style` / multi-expert models can improve `accuracy`, but
- `Multi-View Gated MLP` is currently more reliable on `macro-F1` and `AUC`.

For this dataset, that trade-off is important because the task is a 3-class student outcome problem:

- `Dropout`
- `Enrolled`
- `Graduate`

The middle class `Enrolled` is usually the hardest one, so `macro-F1` and per-class recall matter more than raw `accuracy`.

## Why this dataset is structurally interesting

The dataset is not just “small tabular classification”. It has several distinct information sources:

- demographic and enrollment background
- family / socioeconomic background
- first-semester academic performance
- second-semester academic performance
- macroeconomic variables

That means the main modeling question is not only “which classifier is stronger”, but also:

- should features be grouped by view?
- should numerical features be encoded differently?
- should the loss reflect the ordinal-like difficulty among the three classes?

## Recommended proposal ranking

### Tier 1: Most promising

#### 1. Ordinal or cost-sensitive classification

This is the cleanest next experiment.

Reason:

- The class geometry is not fully symmetric.
- Misclassifying `Dropout` as `Graduate` is much worse than confusing `Dropout` with `Enrolled`.
- Standard cross-entropy ignores this asymmetry.

What to try:

- `cost-sensitive cross entropy`
- `CORAL` / ordinal regression style heads
- class-weighted loss with an explicit penalty matrix

Why it belongs in the proposal:

- low implementation cost
- directly aligned with the problem structure
- can improve `macro-F1` without changing the whole model family

#### 2. TabPFN as a strong external baseline

TabPFN is worth checking early if the goal is to benchmark a small-to-medium tabular task against a modern foundation-model-style baseline.

Reason:

- the dataset size is well within the regime where TabPFN is commonly discussed
- it already handles mixed tabular structure well in many settings
- it gives a strong reference point for whether the current project is already near the practical ceiling

Proposal role:

- not the main story
- a high-value sanity check baseline

### Tier 2: Good but secondary

#### 3. NAM (Neural Additive Models)

NAM is attractive if the proposal wants transparency.

Why it fits:

- many student outcome variables likely have nonlinear but separable effects
- the project can benefit from feature-wise response curves
- the result is easy to explain to a non-ML audience

What it gives:

- per-feature contribution curves
- a GAM-like narrative with neural flexibility

Why it is secondary:

- the dataset likely contains meaningful feature interactions
- a purely additive model may underfit the cross-feature structure

#### 4. NODE

NODE is reasonable if the proposal wants a tree-neural hybrid baseline.

Why it fits:

- categorical / threshold-like behavior can matter in this dataset
- tree-style splits are often a better inductive bias than a plain MLP when the data is structured but not huge

Why it is secondary:

- the expected gain is uncertain
- the project already has stronger non-tree neural baselines
- it is more useful as a comparison model than as the main thesis

### Tier 3: Interesting but risky

#### 5. FT-Transformer plus richer numerical encoding

This is the most practical “fancy transformer” direction.

Best version for this project:

- stronger numerical embeddings
- explicit view embeddings
- column/type embeddings
- careful regularization

Why it is useful:

- it keeps the transformer story
- it lets the model see the dataset structure explicitly

Why it is risky:

- a naive transformer often underperforms a strong MLP on small tabular data
- more structural tokens do not automatically translate into better generalization

#### 6. Retrieval-augmented tabular prediction

This is the most speculative idea.

Concept:

- for each student, retrieve similar training examples
- feed neighbor statistics or labels as auxiliary evidence

Why it might help:

- `Enrolled` is a middle state
- local neighborhood information may be informative for ambiguous samples

Why it is risky:

- harder to evaluate cleanly
- more engineering overhead
- easy to overfit if retrieval is not carefully controlled

## What should not be the main next step

### KAN

KAN is interesting, but for this project it should stay secondary.

Reason:

- the strongest evidence for KAN is not in small mixed tabular classification
- it adds complexity without giving a clearly better inductive bias than the better tabular-specific baselines above

Use it only if the proposal wants a “function-parameterized” comparison model.

### DCN v2

DCN v2 is useful as a feature-crossing baseline, but it is not the first thing I would bet on.

Reason:

- it is strongest in web-scale recommender / ranking settings
- the current dataset is much smaller and less sparse
- the current project already has a strong view-aware model that captures structure more naturally

### Pure stacking of more MLP layers

This should not be the main story anymore.

Reason:

- the current results already show that better structure matters more than just deeper generic MLPs

## Proposed experiment order

1. Keep `Multi-View Gated MLP` as the main reference model.
2. Add an ordinal / cost-sensitive loss variant.
3. Benchmark `TabPFN` as an external sanity-check baseline.
4. Add `NAM` if interpretability is needed for the report.
5. Add `NODE` only if there is room for another non-tree neural comparison.
6. Only then consider more speculative ideas such as retrieval augmentation.

## Suggested research framing

A good future proposal for this project is not:

- “let us try more advanced models”

A better framing is:

- “can we match the asymmetric, view-structured nature of student-outcome data with a model and loss that reflect the problem geometry?”

That framing naturally supports:

- multi-view feature grouping
- ordinal/cost-sensitive learning
- interpretable nonlinear feature effects
- a strong external baseline such as TabPFN

## Short conclusion

The most defensible next-step proposal is:

- keep `Multi-View Gated MLP` as the core model
- add ordinal / cost-sensitive learning
- use `TabPFN` as a strong external baseline
- use `NAM` as the interpretability-oriented branch

`NODE`, `DCN v2`, `KAN`, and retrieval augmentation are all possible follow-ups, but they are lower priority for this specific dataset and project stage.
