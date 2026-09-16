"""Train and verify the PyTorch AP candidate on benchmark v0.2."""

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
from vait.transformations.library.ap_torch import (
    build_pytorch_transformation,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)

dataset = load_benchmark(
    Path("datasets/ap_invoice_exceptions/v0.2/cases.yaml")
)

corpus = build_synthetic_ap_training_corpus(
    sample_count=600,
    seed=20260916,
    excluded_inputs=(
        case.input_data
        for case in dataset.cases
    ),
)

transformation = build_pytorch_transformation(
    corpus
)

report = assess_python_transformations(
    dataset=dataset,
    transformations=(transformation,),
    context=ApplicabilityContext(
        risk_level=RiskLevel.CRITICAL,
        available_capabilities=frozenset(
            {
                "typed-inputs",
                "tabular-features",
                "trained-model",
                "pytorch-inference",
            }
        ),
    ),
    policy=BoundedVerificationPolicy(
        max_overall_disagreement_rate=0.15,
        max_high_risk_disagreement_rate=0.25,
        confidence_level=0.95,
        min_total_cases=20,
        min_high_risk_cases=11,
    ),
)

report_path = write_benchmark_report(
    report,
    Path(
        "artifacts/benchmarks/"
        "ap-v0.2-pytorch-transformation-assessment.json"
    ),
)

assessment = report.assessments[0]

print(
    f"Training corpus: {len(corpus.cases)} cases"
)
print(
    f"Training class counts: {corpus.class_counts}"
)
print(
    f"Validation benchmark: {len(dataset.cases)} cases"
)
print()
print(
    f"Candidate: {assessment.canonical_id}"
)
print(
    f"Applicability: {assessment.applicability.status.value}"
)

verification = assessment.verification

if verification is None:
    raise RuntimeError(
        "Applicable PyTorch candidate produced no verification evidence."
    )

print(
    f"Decision: {verification.decision.value}"
)

evidence = verification.statistical_evidence

if evidence is not None:
    print(
        "Observed disagreement: "
        f"{evidence.disagreement_rate:.1%}"
    )
    print(
        "95% overall upper bound: "
        f"{evidence.disagreement_upper_bound:.1%}"
    )
    print(
        "High-risk disagreements: "
        f"{evidence.high_risk_disagreements}/"
        f"{evidence.high_risk_cases_evaluated}"
    )

    if evidence.high_risk_disagreement_upper_bound is not None:
        print(
            "95% high-risk upper bound: "
            f"{evidence.high_risk_disagreement_upper_bound:.1%}"
        )

if verification.failures:
    print("Failures:")

    for failure in verification.failures:
        print(
            f"  - {failure.code.value}: "
            f"{failure.message}"
        )

if verification.candidate_latency is not None:
    latency = verification.candidate_latency

    print(
        "Latency p50/p95: "
        f"{latency.p50_ms:.3f} / "
        f"{latency.p95_ms:.3f} ms"
    )

print(
    "Final training loss: "
    f"{transformation.configuration['final_training_loss']:.6f}"
)
print(
    f"Evidence report: {report_path}"
)
