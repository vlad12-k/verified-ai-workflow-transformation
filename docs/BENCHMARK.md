# Benchmark Specification — AP Exception Transformation v0.1

**Benchmark role:** engineering validation domain, not product identity
**Primary transformation:** probabilistic/reference decision -> deterministic Python/rule candidate

## 1. Benchmark question

Can VAIT correctly determine whether a supplied deterministic replacement for a probabilistic/reference AP exception decision satisfies a pre-declared transformation contract?

The benchmark tests the verification system. It does not claim to reproduce a production AP environment or prove real operational ROI.

## 2. Benchmark case schema

Each evaluation case should contain at minimum:

```text
case_id
supplier_id
invoice_id
invoice_amount
po_present
po_amount
received_amount
vendor_blocked
duplicate_flag
policy_threshold
exception_type
risk_tier
expected_policy_outcome
```

Optional fields may be added only when required by a declared rule or benchmark scenario.

## 3. Initial scenario set

The first dataset should deliberately cover distinct failure modes rather than maximise row count.

Required scenarios:

1. normal invoice within tolerance;
2. price variance within tolerance;
3. price variance outside tolerance;
4. quantity mismatch;
5. missing purchase order;
6. missing goods receipt;
7. duplicate invoice;
8. blocked vendor;
9. policy threshold boundary case;
10. ambiguous/unsupported case requiring rejection or escalation.

Each scenario must include low-, medium-, or high-risk metadata where meaningful.

## 4. Reference and candidate

### Reference

The reference implementation may be one of:

- a live probabilistic model through a provider adapter; or
- a frozen, versioned set of reference outputs captured from such a model.

Core tests must not require external API access.

### Candidate

The first candidate is a deterministic Python/rule implementation supplied to VAIT.

VAIT does not generate the candidate in v0.1.

## 5. Contract example

```yaml
version: "0.1"
name: "ap_exception_low_risk_replacement"

scope:
  side_effects: "read_only"

risk:
  field: "risk_tier"
  hard_reject_on_invariant_failure: true

invariants:
  - id: "blocked_vendor_never_approve"
  - id: "duplicate_invoice_never_approve"

bounded_degradation:
  confidence_level: 0.95
  max_overall_disagreement: 0.02
  max_high_risk_disagreement: 0.00

latency:
  report: true

economics:
  report: true
```

The exact YAML schema should be implemented through versioned Pydantic models.

## 6. Required metrics

### Behavioural metrics

- total cases;
- exact agreement count/rate;
- disagreement count/rate;
- agreement by scenario;
- agreement by risk tier.

### Risk metrics

- hard-invariant violations;
- unsafe disagreements;
- high-risk disagreement rate;
- examples of every unsafe disagreement.

### Statistical evidence

- confidence interval for relevant proportions;
- rare-event upper bound where zero events are observed;
- sample size reported for every risk stratum.

The implementation must not report "zero risk" from zero observed failures.

### Performance metrics

- reference p50/p95 latency where measurable;
- candidate p50/p95 latency;
- latency delta.

### Economic metrics

Where measurable:

- estimated reference execution cost per case;
- estimated candidate execution cost per case;
- cost delta.

Cost evidence must not override failed risk constraints.

## 7. Decision examples

### Example A — BOUNDED

```text
Overall agreement: 99.0%
High-risk disagreement: 0.0%
Hard invariants: PASS
Bounded-degradation thresholds: PASS
=> BOUNDED
```

### Example B — REJECT despite high average agreement

```text
Overall agreement: 99.5%
High-risk disagreement: 8.0%
Hard invariant violated: blocked vendor approved
=> REJECT
```

### Example C — EXACT

Only use when the supported deterministic verification path establishes all declared exact obligations for the tested transformation class.

## 8. Dataset discipline

The v0.1 dataset may be synthetic/controlled, but must be:

- versioned;
- deterministic to regenerate where possible;
- scenario-balanced intentionally;
- separated from any later hidden test set;
- explicit about synthetic limitations.

Synthetic results may support algorithmic claims only.

They must not be used to claim:

- real FTE reduction;
- fraud reduction;
- production financial safety;
- production ROI;
- compliance.

## 9. Reproducibility

The target CLI shape is:

```bash
vait verify \
  --contract benchmarks/ap_exception/contract.yaml \
  --dataset benchmarks/ap_exception/cases.jsonl \
  --reference reference-config.yaml \
  --candidate candidate-config.yaml
```

The command should emit:

- human-readable terminal summary;
- JSON evidence report;
- non-zero exit code on execution/configuration failure.

## 10. v0.1 benchmark completion criteria

The benchmark is complete when:

1. at least the required scenarios are represented;
2. all cases can be executed reproducibly;
3. risk strata are populated;
4. at least one candidate produces `BOUNDED` or `EXACT` under a legitimate contract;
5. at least one deliberately unsafe candidate produces `REJECT`;
6. the result is covered by integration tests;
7. results can be reproduced from a clean environment.
