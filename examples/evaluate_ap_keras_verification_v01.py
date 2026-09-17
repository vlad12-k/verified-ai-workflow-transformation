"""Verify the Keras AP candidate on benchmark v0.2."""

from pathlib import Path

from vait.benchmark.loader import load_benchmark
from vait.benchmark.reporting import write_benchmark_report
from vait.benchmark.transformation_assessment import (
    assess_python_transformations,
)
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
)
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.ap_keras import (
    build_keras_transformation,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)

DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-keras-verification-v0.1.json"
)

dataset = load_benchmark(
    DATASET_PATH
)

corpus = build_synthetic_ap_training_corpus(
    excluded_inputs=tuple(
        case.input_data
        for case in dataset.cases
    )
)

candidate = build_keras_transformation(
    corpus
)

context = ApplicabilityContext(
    risk_level=RiskLevel.CRITICAL,
    available_capabilities=frozenset(
        {
            "typed-inputs",
            "tabular-features",
            "trained-model",
            "keras-inference",
        }
    ),
)

policy = BoundedVerificationPolicy(
    max_overall_disagreement_rate=0.15,
    max_high_risk_disagreement_rate=0.25,
    confidence_level=0.95,
    min_total_cases=20,
    min_high_risk_cases=11,
)

report = assess_python_transformations(
    dataset=dataset,
    transformations=(
        candidate,
    ),
    context=context,
    policy=policy,
)

report_path = write_benchmark_report(
    report,
    OUTPUT_PATH,
)

assessment = report.assessments[0]

print("AP Keras bounded verification")
print()
print(
    "Benchmark:",
    f"{report.benchmark_id}@{report.benchmark_version}",
)
print(
    "Candidate:",
    assessment.candidate_implementation_id,
)
print(
    "Applicable:",
    assessment.applicability.is_applicable,
)

verification = assessment.verification

if verification is None:
    raise RuntimeError(
        "Keras candidate was not verified."
    )

print(
    "Decision:",
    verification.decision.value,
)
print(
    "Cases:",
    verification.cases_evaluated,
)

evidence = verification.statistical_evidence

if evidence is None:
    raise RuntimeError(
        "Missing statistical evidence."
    )

print(
    "Disagreements:",
    evidence.disagreements,
)
print(
    "Overall rate:",
    f"{evidence.disagreement_rate:.4f}",
)
print(
    "Overall upper bound:",
    f"{evidence.disagreement_upper_bound:.4f}",
)
print(
    "High-risk cases:",
    evidence.high_risk_cases_evaluated,
)
print(
    "High-risk disagreements:",
    evidence.high_risk_disagreements,
)
print(
    "High-risk rate:",
    f"{evidence.high_risk_disagreement_rate:.4f}",
)

high_risk_upper = (
    evidence.high_risk_disagreement_upper_bound
)

if high_risk_upper is None:
    print(
        "High-risk upper bound: n/a"
    )
else:
    print(
        "High-risk upper bound:",
        f"{high_risk_upper:.4f}",
    )

latency = verification.candidate_latency

if latency is None:
    raise RuntimeError(
        "Missing latency evidence."
    )

print(
    "Mean latency:",
    f"{latency.mean_ms:.4f} ms",
)
print(
    "p50 latency:",
    f"{latency.p50_ms:.4f} ms",
)
print(
    "p95 latency:",
    f"{latency.p95_ms:.4f} ms",
)
print(
    "Max latency:",
    f"{latency.max_ms:.4f} ms",
)
print(
    "Failures:",
    len(verification.failures),
)

print()
print(
    "Evidence report:",
    report_path,
)
