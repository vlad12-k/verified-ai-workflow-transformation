"""Run the applied AP transformation through the VAIT verification pipeline."""

from pathlib import Path

from vait.benchmark.evaluator import evaluate_benchmark
from vait.benchmark.loader import load_benchmark
from vait.benchmark.reporting import write_benchmark_report
from vait.benchmark.verification import verify_benchmark_bounded
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
)
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.synthetic_ap import (
    build_synthetic_ap_transformation,
)
from vait.transformations.python_callable import (
    PythonCallableTransformation,
)
from vait.transformations.registry import TransformationRegistry

dataset = load_benchmark(
    Path("datasets/ap_invoice_exceptions/v0.2/cases.yaml")
)

transformation = build_synthetic_ap_transformation()

registry = TransformationRegistry()
registry.register(transformation)

registered = registry.get(
    transformation.descriptor.transformation_id,
    transformation.descriptor.version,
)

if not isinstance(
    registered,
    PythonCallableTransformation,
):
    raise TypeError(
        "Registered AP transformation does not support "
        "Python candidate preparation."
    )

preparation = registered.prepare_candidate(
    ApplicabilityContext(
        risk_level=RiskLevel.CRITICAL,
        available_capabilities=frozenset({"typed-inputs"}),
    )
)

if preparation.runner is None:
    reasons = ", ".join(
        issue.code.value
        for issue in preparation.applicability.issues
    )
    raise RuntimeError(
        f"Transformation is not applicable: {reasons}"
    )

candidate_configuration = {
    "transformation_id": transformation.descriptor.transformation_id,
    "transformation_version": transformation.descriptor.version,
}

evaluation = evaluate_benchmark(
    dataset=dataset,
    candidate=preparation.runner,
    candidate_configuration=candidate_configuration,
)

verification = verify_benchmark_bounded(
    dataset=dataset,
    candidate=preparation.runner,
    policy=BoundedVerificationPolicy(
        max_overall_disagreement_rate=0.15,
        max_high_risk_disagreement_rate=0.25,
        confidence_level=0.95,
        min_total_cases=20,
        min_high_risk_cases=11,
    ),
    candidate_configuration=candidate_configuration,
)

evaluation_path = write_benchmark_report(
    evaluation,
    Path(
        "artifacts/benchmarks/"
        "ap-v0.2-transformation-evaluation.json"
    ),
)

verification_path = write_benchmark_report(
    verification,
    Path(
        "artifacts/benchmarks/"
        "ap-v0.2-transformation-verification.json"
    ),
)

print(
    "Transformation: "
    f"{transformation.descriptor.canonical_id}"
)
print(
    "Applicability: "
    f"{preparation.applicability.status.value}"
)
print(f"Benchmark: {evaluation.benchmark_id}")
print(f"Version: {evaluation.benchmark_version}")
print(f"Cases evaluated: {evaluation.cases_evaluated}")
print(f"Gold matches: {evaluation.gold_matches}")
print(
    "Gold agreement: "
    f"{evaluation.gold_agreement_rate:.1%}"
)
print(f"Decision: {verification.decision.value}")

if verification.statistical_evidence is not None:
    evidence = verification.statistical_evidence

    print(
        "Observed disagreement: "
        f"{evidence.disagreement_rate:.1%}"
    )
    print(
        "95% overall upper bound: "
        f"{evidence.disagreement_upper_bound:.1%}"
    )

    if evidence.high_risk_disagreement_upper_bound is not None:
        print(
            "95% high-risk upper bound: "
            f"{evidence.high_risk_disagreement_upper_bound:.1%}"
        )

print(f"Evaluation report: {evaluation_path}")
print(f"Verification report: {verification_path}")
