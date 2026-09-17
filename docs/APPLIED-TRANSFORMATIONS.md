# Applied Transformations

## Scope

VAIT evaluates supplied workflow transformation candidates. It does not
synthesize arbitrary replacement implementations or rank candidates by a
single overall score.

The applied transformation path introduced in M3 supports supplied
deterministic Python rule candidates.

## Transformation pipeline

```text
TransformationDescriptor
        |
        v
TransformationRegistry
        |
        v
ApplicabilityContext
        |
        v
Applicability evaluation
        |
        +---- NOT_APPLICABLE
        |
        v
Candidate preparation
        |
        v
ImplementationRunner
        |
        v
Benchmark / verification evidence
        |
        +---- EXACT
        +---- BOUNDED
        +---- REJECT
```

## Applicability is not verification

`NOT_APPLICABLE` means that a transformation does not satisfy the declared
requirements for the supplied workflow context.

It is not a verification rejection because the candidate is not executed.

`REJECT` is a verification decision produced after an applicable candidate
has been evaluated and fails a hard constraint, risk threshold, statistical
threshold, execution requirement, or other supported verification condition.

## Supplied candidates

A `PythonRuleTransformation` contains:

- a versioned transformation descriptor;
- an implementation identity;
- a supplied deterministic Python function.

The transformation layer does not claim that a supplied implementation is
safe. It only prepares an applicable candidate for the existing runner and
verification layers.

## Multi-candidate assessment

`assess_python_rule_transformations()` evaluates supplied candidates
independently.

Assessment results are returned in deterministic canonical-ID order.

VAIT does not automatically select a winner. Each applicable candidate keeps
its own verification evidence and decision.

## AP benchmark example

The Accounts Payable invoice-exception dataset is a synthetic benchmark
domain used to exercise the transformation pipeline.

Run:

```bash
python examples/verify_ap_transformation_v02.py
python examples/assess_ap_transformations_v02.py
```

The multi-candidate example deliberately demonstrates:

- a candidate that is not applicable;
- a candidate that satisfies the bounded verification policy;
- an intentionally unsafe candidate that is rejected.

A `BOUNDED` result is statistical evidence under the declared benchmark,
risk strata, thresholds, and confidence level. It is not a claim of universal
semantic equivalence or production financial safety.

## M3 heterogeneous AI evidence

M3 extends the supplied-candidate model beyond deterministic rules while
preserving the same separation between candidate preparation, applicability,
and verification.

The development evidence is grouped by task family rather than collapsed into
a single leaderboard.

### Structured decision candidates

The AP structured-decision track includes:

- deterministic rule transformations;
- scikit-learn Random Forest;
- XGBoost;
- a PyTorch multilayer perceptron;
- compact hard-label and distilled PyTorch students;
- a TensorFlow/Keras multilayer perceptron.

Candidates are evaluated independently. Similar training metrics do not imply
equivalent verification behaviour.

For example, the PyTorch and Keras AP networks use the same 11-feature input
representation, 32/16 hidden-layer architecture, 963 parameters, and nearly
identical development training accuracy. PyTorch satisfied the AP v0.2 bounded
policy with zero observed disagreements, while Keras was rejected after one
reproducible high-risk boundary disagreement.

This is development evidence, not a general claim that one framework is
superior to another.

### Retrieval and local RAG

The policy-context track includes:

- pinned sentence-transformer embeddings;
- retrieval evaluation;
- local Qwen2.5-0.5B-Instruct generation;
- stricter prompt contracts;
- semantic guarding and fallback routing;
- reviewed faithfulness progression.

The guarded configuration improved surfaced support on the reviewed synthetic
development set, but this does not establish hallucination-free behaviour or
general performance on unseen policy corpora.

### Specialist adaptation

The specialist path includes:

- supervised specialist corpus construction and tokenization checks;
- PEFT/LoRA training;
- reviewed base-versus-LoRA semantic evidence;
- QLoRA/NF4 feasibility on Apple Silicon.

QLoRA evidence establishes that the pinned local model can be loaded in 4-bit
NF4 form, prepared for k-bit training, and complete a real backward pass with
non-zero adapter gradients while frozen base-model parameters remain without
gradients.

The reported model memory footprint is not peak end-to-end training memory.

### Synthetic data

The structured training corpus is generated deterministically with class
balance, integrity checks, policy-label agreement checks, duplicate detection,
and exact-input exclusion against the AP v0.2 evaluation set.

Exact input exclusion does not make the development benchmark statistically
independent. The synthetic labels and benchmark scenarios share the same
policy domain.

### Unified M3 evidence

Run:

    python examples/build_m3_heterogeneous_evidence_v01.py

The resulting development artifact is:

    artifacts/benchmarks/m3-heterogeneous-evidence-v0.1.json

It groups evidence into:

1. structured decision transformations;
2. retrieval and RAG;
3. specialist adaptation;
4. synthetic data.

VAIT deliberately does not compute a cross-task score or automatically select
a winner across these heterogeneous capabilities.

The unified artifact is an evidence index, not a production certification.

## M3 evidence limitations

The current M3 evidence should be interpreted with the following constraints:

- AP v0.2 is a development benchmark, not an unseen blind evaluation;
- structured synthetic labels are policy-derived rather than independent
  ground truth;
- RAG faithfulness judgments are bounded to the reviewed development cases;
- the specialist LoRA corpus and reviewed evaluation set are small;
- QLoRA demonstrates technical feasibility rather than model-quality gains;
- local latency measurements do not establish universal runtime performance;
- `BOUNDED` means that the declared statistical verification policy was
  satisfied for the evaluated benchmark and risk strata, not that two
  implementations are universally semantically equivalent.
