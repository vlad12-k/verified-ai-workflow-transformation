"""Build controlled M4 inference evidence for the AP PyTorch candidate."""

from pathlib import Path
from time import perf_counter_ns

from vait.benchmark.loader import load_benchmark
from vait.benchmark.verification import verify_benchmark_bounded
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
    VerificationCase,
)
from vait.decision.models import Decision
from vait.inference.benchmark import run_controlled_inference_benchmark
from vait.inference.models import (
    InferenceConfiguration,
    InferenceSetupEvidence,
    InferenceWorkload,
)
from vait.inference.reporting import write_inference_report
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.ap_torch import (
    build_pytorch_transformation,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)

DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4-ap-pytorch-inference-v0.1.json"
)

TRAINING_SAMPLE_COUNT = 600
TRAINING_SEED = 20260916


def main() -> None:
    """Train, verify, and benchmark the PyTorch AP candidate."""
    dataset = load_benchmark(DATASET_PATH)

    setup_started = perf_counter_ns()

    corpus = build_synthetic_ap_training_corpus(
        sample_count=TRAINING_SAMPLE_COUNT,
        seed=TRAINING_SEED,
        excluded_inputs=(
            case.input_data
            for case in dataset.cases
        ),
    )

    transformation = build_pytorch_transformation(
        corpus
    )

    preparation = transformation.prepare_candidate(
        ApplicabilityContext(
            risk_level=RiskLevel.CRITICAL,
            available_capabilities=frozenset(
                {
                    "typed-inputs",
                    "tabular-features",
                    "trained-model",
                    "pytorch-inference",
                }
            ),
        )
    )

    setup_duration_ms = (
        perf_counter_ns() - setup_started
    ) / 1_000_000

    if preparation.runner is None:
        reasons = ", ".join(
            issue.code.value
            for issue in preparation.applicability.issues
        )
        raise RuntimeError(
            f"PyTorch transformation is not applicable: {reasons}"
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
        candidate_configuration={
            "transformation_id": (
                transformation.descriptor.transformation_id
            ),
            "transformation_version": (
                transformation.descriptor.version
            ),
            "model_family": "pytorch_mlp",
            "training_seed": TRAINING_SEED,
            "training_cases": TRAINING_SAMPLE_COUNT,
        },
    )

    if verification.decision == Decision.REJECT:
        reasons = ", ".join(
            failure.code.value
            for failure in verification.failures
        )
        raise RuntimeError(
            "PyTorch candidate failed verification before "
            f"optimisation benchmarking: {reasons}"
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

    workload = InferenceWorkload(
        workload_id=dataset.benchmark_id,
        version=dataset.version,
        item_count=len(dataset.cases),
        fingerprint_sha256=(
            verification.provenance.dataset_fingerprint_sha256
        ),
        metadata={
            "scope": "development-benchmark",
            "task": "ap-structured-decision",
        },
    )

    setup = InferenceSetupEvidence(
        duration_ms=setup_duration_ms,
        scope="candidate-preparation",
        included_operations=[
            "training-corpus-build",
            "model-training",
            "candidate-preparation",
        ],
        metadata={
            "included_in_inference_latency": False,
            "training_cases": TRAINING_SAMPLE_COUNT,
            "training_seed": TRAINING_SEED,
            "training_epochs": (
                transformation.configuration["training_epochs"]
            ),
            "training_device": (
                transformation.configuration["training_device"]
            ),
            "final_training_loss": (
                transformation.configuration["final_training_loss"]
            ),
        },
    )

    report = run_controlled_inference_benchmark(
        runner=preparation.runner,
        cases=cases,
        task_family="structured-decision",
        configuration=InferenceConfiguration(
            provider="local",
            runtime="pytorch-eager",
            device="cpu",
            dtype="float32",
            batch_size=1,
            model_id=transformation.implementation_id,
            model_revision=transformation.descriptor.version,
            metadata={
                "model_family": "pytorch_mlp",
                "feature_count": (
                    transformation.configuration["feature_count"]
                ),
                "hidden_dimensions": (
                    transformation.configuration["hidden_dimensions"]
                ),
            },
        ),
        workload=workload,
        setup=setup,
        warmup_rounds=5,
        measured_rounds=50,
        measure_cold_start=False,
        evidence_metadata={
            "verification_decision": verification.decision.value,
            "verification_cases_evaluated": (
                verification.cases_evaluated
            ),
            "verification_scope": "development-benchmark",
            "measurement_scope": "steady-state-model-inference",
            "setup_excluded_from_inference_latency": True,
            "cold_start_measurement": (
                "not-measured-for-prepared-in-process-model"
            ),
        },
    )

    report_path = write_inference_report(
        report=report,
        path=ARTIFACT_PATH,
    )

    print("M4-A PyTorch model-backed inference benchmark")
    print()
    print(f"Candidate: {report.candidate_implementation_id}")
    print(f"Verification: {verification.decision.value}")
    print(
        f"Training corpus: {len(corpus.cases)} cases"
    )
    print(
        f"Setup duration: {setup_duration_ms:.3f} ms"
    )
    print(
        "Setup scope: corpus build + model training + "
        "candidate preparation"
    )
    print()
    print(
        f"Warm-up executions: {report.warmup_iterations}"
    )
    print(
        f"Measured executions: {report.measured_iterations}"
    )
    print(
        "Cold start: not reported for the already-prepared "
        "in-process model."
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
