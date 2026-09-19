"""Build controlled M4 inference evidence for the AP rule candidate."""

from pathlib import Path

from vait.benchmark.loader import load_benchmark
from vait.benchmark.verification import verify_benchmark_bounded
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
    VerificationCase,
)
from vait.decision.models import Decision
from vait.inference.benchmark import run_controlled_inference_benchmark
from vait.inference.models import InferenceConfiguration
from vait.inference.reporting import write_inference_report
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.synthetic_ap import (
    build_synthetic_ap_transformation,
)
from vait.transformations.python_callable import (
    PythonCallableTransformation,
)
from vait.transformations.registry import TransformationRegistry

DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4-ap-rules-inference-v0.1.json"
)


def main() -> None:
    """Verify and benchmark the deterministic AP transformation."""
    dataset = load_benchmark(DATASET_PATH)

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
            available_capabilities=frozenset(
                {"typed-inputs"}
            ),
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
        "transformation_id": (
            transformation.descriptor.transformation_id
        ),
        "transformation_version": (
            transformation.descriptor.version
        ),
        "runtime": "local-python-rule",
    }

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

    if verification.decision == Decision.REJECT:
        reasons = ", ".join(
            failure.code.value
            for failure in verification.failures
        )
        raise RuntimeError(
            "Candidate failed verification before inference "
            f"benchmarking: {reasons}"
        )

    if verification.provenance is None:
        raise RuntimeError(
            "Expected verification provenance was not produced."
        )

    cases = [
        VerificationCase(
            id=case.case_id,
            input_data=case.input_data,
            risk_level=case.risk_level,
        )
        for case in dataset.cases
    ]

    report = run_controlled_inference_benchmark(
        runner=preparation.runner,
        cases=cases,
        task_family="structured-decision",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="python-rule",
            device="cpu",
            dtype="native",
            batch_size=1,
            metadata={
                "transformation_id": (
                    transformation.descriptor.transformation_id
                ),
                "transformation_version": (
                    transformation.descriptor.version
                ),
            },
        ),
        warmup_rounds=5,
        measured_rounds=50,
        measure_cold_start=False,
        evidence_metadata={
            "benchmark_id": dataset.benchmark_id,
            "benchmark_version": dataset.version,
            "dataset_fingerprint_sha256": (
                verification.provenance.dataset_fingerprint_sha256
            ),
            "verification_decision": verification.decision.value,
            "verification_cases_evaluated": (
                verification.cases_evaluated
            ),
            "verification_scope": "development-benchmark",
            "cold_start_measurement": (
                "not-applicable-for-in-process-rule-candidate"
            ),
        },
    )

    report_path = write_inference_report(
        report=report,
        path=ARTIFACT_PATH,
    )

    print("M4-A controlled inference benchmark")
    print()
    print(f"Candidate: {report.candidate_implementation_id}")
    print(f"Task family: {report.task_family}")
    print(
        "Verification: "
        f"{verification.decision.value}"
    )
    print(
        "Benchmark: "
        f"{dataset.benchmark_id} v{dataset.version}"
    )
    print()
    print(
        f"Warm-up executions: {report.warmup_iterations}"
    )
    print(
        f"Measured executions: {report.measured_iterations}"
    )
    print(
        "Cold start: not measured for this in-process "
        "deterministic rule candidate."
    )
    print()
    print(
        f"Mean latency: {report.latency.mean_ms:.6f} ms"
    )
    print(
        f"p50 latency: {report.latency.p50_ms:.6f} ms"
    )
    print(
        f"p95 latency: {report.latency.p95_ms:.6f} ms"
    )
    print(
        f"p99 latency: {report.latency.p99_ms:.6f} ms"
    )
    print(
        f"Max latency: {report.latency.max_ms:.6f} ms"
    )

    cases_per_second = report.throughput.cases_per_second

    if cases_per_second is None:
        raise RuntimeError(
            "Expected cases-per-second evidence was not produced."
        )

    print(
        f"Throughput: {cases_per_second:.2f} cases/sec"
    )
    print()
    print(f"Run ID: {report.run_id}")
    print(f"Evidence report: {report_path}")


if __name__ == "__main__":
    main()
