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
