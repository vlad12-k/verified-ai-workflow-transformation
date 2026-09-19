"""Build controlled M4-H profiling evidence for the verified AP PyTorch candidate."""

from pathlib import Path
from time import perf_counter_ns

from pydantic import BaseModel

from vait.benchmark.loader import load_benchmark
from vait.benchmark.verification import (
    BenchmarkVerificationReport,
    verify_benchmark_bounded,
)
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
    VerificationCase,
)
from vait.decision.models import Decision
from vait.inference.models import (
    InferenceConfiguration,
    InferenceSetupEvidence,
)
from vait.optimisation.admissibility import (
    SearchPointAdmissibility,
    SearchPointVerificationEvidence,
    VerificationEvidenceKind,
    evaluate_search_point_admissibility,
)
from vait.optimisation.compatibility import (
    InferenceCompatibilityContext,
    evaluate_search_point_compatibility,
)
from vait.optimisation.search_space import (
    InferenceSearchPoint,
)
from vait.optimisation.structured_benchmark import (
    StructuredBenchmarkSeries,
    StructuredModelFootprint,
    run_structured_benchmark_series,
)
from vait.transformations.applicability import (
    ApplicabilityContext,
)
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
    "m4h-ap-pytorch-resource-profile-v0.1.json"
)

TRAINING_SAMPLE_COUNT = 600
TRAINING_SEED = 20260916

REPETITIONS = 5
WARMUP_ROUNDS = 5
MEASURED_ROUNDS = 50


class M4HProfilingArtifact(BaseModel):
    """Versioned verification-backed M4-H profiling evidence."""

    schema_version: str = "0.1"

    artifact_id: str
    benchmark_id: str
    benchmark_version: str

    verification: BenchmarkVerificationReport
    admissibility: SearchPointAdmissibility
    profiling: StructuredBenchmarkSeries


def main() -> None:
    """Verify, admit, and profile the PyTorch AP candidate."""
    dataset = load_benchmark(
        DATASET_PATH
    )

    setup_started = perf_counter_ns()

    corpus = build_synthetic_ap_training_corpus(
        sample_count=TRAINING_SAMPLE_COUNT,
        seed=TRAINING_SEED,
        excluded_inputs=(
            case.input_data
            for case in dataset.cases
        ),
    )

    transformation = (
        build_pytorch_transformation(
            corpus
        )
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

    setup_elapsed_ms = (
        perf_counter_ns()
        - setup_started
    ) / 1_000_000

    if preparation.runner is None:
        reasons = ", ".join(
            issue.code.value
            for issue in preparation.applicability.issues
        )

        raise RuntimeError(
            "PyTorch transformation is not applicable: "
            f"{reasons}"
        )

    runner = preparation.runner

    setup = InferenceSetupEvidence(
        duration_ms=setup_elapsed_ms,
        scope="candidate-preparation",
        included_operations=[
            "synthetic-training-corpus-build",
            "pytorch-model-training",
            "candidate-preparation",
        ],
        metadata={
            "included_in_inference_latency": False,
            "training_sample_count": (
                TRAINING_SAMPLE_COUNT
            ),
            "training_seed": TRAINING_SEED,
        },
    )

    policy = BoundedVerificationPolicy(
        max_overall_disagreement_rate=0.15,
        max_high_risk_disagreement_rate=0.25,
        confidence_level=0.95,
        min_total_cases=20,
        min_high_risk_cases=11,
    )

    verification = verify_benchmark_bounded(
        dataset=dataset,
        candidate=runner,
        policy=policy,
        candidate_configuration={
            **transformation.configuration,
            "transformation_id": (
                transformation
                .descriptor
                .transformation_id
            ),
            "transformation_version": (
                transformation
                .descriptor
                .version
            ),
        },
    )

    if verification.decision is Decision.REJECT:
        reasons = ", ".join(
            failure.code.value
            for failure in verification.failures
        )

        raise RuntimeError(
            "PyTorch candidate failed verification "
            f"before profiling: {reasons}"
        )

    if verification.decision is not Decision.BOUNDED:
        raise RuntimeError(
            "Expected bounded statistical verification "
            "for the PyTorch profiling candidate."
        )

    if verification.statistical_evidence is None:
        raise RuntimeError(
            "BOUNDED verification is missing "
            "statistical evidence."
        )

    if verification.provenance is None:
        raise RuntimeError(
            "Expected verification provenance "
            "was not produced."
        )

    configuration = InferenceConfiguration(
        provider="local",
        runtime="pytorch",
        device="cpu",
        dtype="float32",
        batch_size=1,
        model_id=(
            transformation.implementation_id
        ),
        model_revision=(
            transformation.descriptor.version
        ),
        metadata={
            "model_family": "pytorch_mlp",
            "training_seed": TRAINING_SEED,
        },
    )

    search_point = InferenceSearchPoint(
        search_point_id=(
            "synthetic-ap-pytorch-mlp::"
            "cpu-float32-b1"
        ),
        search_space_id=(
            "m4h-ap-pytorch-profiling"
        ),
        task_family="structured-decision",
        candidate_id=(
            "synthetic-ap-pytorch-mlp"
        ),
        implementation_id=(
            transformation.implementation_id
        ),
        configuration_id="cpu-float32-b1",
        configuration=configuration,
        requires_verification=True,
        required_capabilities=(
            transformation
            .descriptor
            .required_capabilities
        ),
        candidate_metadata={
            "transformation_id": (
                transformation
                .descriptor
                .transformation_id
            ),
            "transformation_version": (
                transformation
                .descriptor
                .version
            ),
        },
    )

    compatibility = (
        evaluate_search_point_compatibility(
            search_point,
            InferenceCompatibilityContext(
                available_providers=frozenset(
                    {"local"}
                ),
                available_runtimes=frozenset(
                    {"pytorch"}
                ),
                available_devices=frozenset(
                    {"cpu"}
                ),
                available_dtypes=frozenset(
                    {"float32"}
                ),
                max_batch_size=1,
                available_model_ids=frozenset(
                    {
                        transformation
                        .implementation_id
                    }
                ),
                available_capabilities=(
                    transformation
                    .descriptor
                    .required_capabilities
                ),
            ),
        )
    )

    contract_id = (
        f"benchmark:{dataset.benchmark_id}:"
        f"{dataset.version}:"
        f"{runner.implementation_id}"
    )

    verification_evidence = (
        SearchPointVerificationEvidence(
            search_point_id=(
                search_point.search_point_id
            ),
            implementation_id=(
                search_point.implementation_id
            ),
            contract_id=contract_id,
            decision=verification.decision,
            evidence_kind=(
                VerificationEvidenceKind
                .BOUNDED_STATISTICAL
            ),
            statistical_evidence_present=True,
            failure_codes=tuple(
                failure.code
                for failure in verification.failures
            ),
        )
    )

    admissibility = (
        evaluate_search_point_admissibility(
            compatibility=compatibility,
            verification_evidence=(
                verification_evidence
            ),
        )
    )

    if not admissibility.is_admissible:
        reasons = ", ".join(
            issue.code.value
            for issue in admissibility.issues
        )

        raise RuntimeError(
            "Verified PyTorch search point "
            f"is not admissible: {reasons}"
        )

    parameter_value = (
        transformation.configuration[
            "parameter_count"
        ]
    )

    if (
        isinstance(parameter_value, bool)
        or not isinstance(
            parameter_value,
            int,
        )
    ):
        raise TypeError(
            "PyTorch parameter_count must be an integer."
        )

    parameter_count = parameter_value

    footprint = StructuredModelFootprint(
        parameter_count=parameter_count,
        model_size_bytes=(
            parameter_count * 4
        ),
        metadata={
            "model_size_scope": (
                "float32-parameter-tensor-storage"
            ),
            "bytes_per_parameter": 4,
            "excludes_framework_overhead": True,
            "excludes_optimizer_state": True,
        },
    )

    cases = tuple(
        VerificationCase(
            id=case.case_id,
            input_data=case.input_data,
            risk_level=case.risk_level,
        )
        for case in dataset.cases
    )

    series = run_structured_benchmark_series(
        admissibility=admissibility,
        runner=runner,
        cases=cases,
        repetitions=REPETITIONS,
        warmup_rounds=WARMUP_ROUNDS,
        measured_rounds=MEASURED_ROUNDS,
        setup=setup,
        footprint=footprint,
        evidence_metadata={
            "milestone": "M4-H",
            "benchmark_id": dataset.benchmark_id,
            "benchmark_version": dataset.version,
            "dataset_fingerprint_sha256": (
                verification
                .provenance
                .dataset_fingerprint_sha256
            ),
            "verification_decision": (
                verification.decision.value
            ),
            "profiling_scope": (
                "local-development-evidence"
            ),
            "rss_interpretation": (
                "process-lifetime-high-water-mark"
            ),
        },
    )

    artifact = M4HProfilingArtifact(
        artifact_id=(
            "m4h-ap-pytorch-resource-profile-v0.1"
        ),
        benchmark_id=dataset.benchmark_id,
        benchmark_version=dataset.version,
        verification=verification,
        admissibility=admissibility,
        profiling=series,
    )

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_PATH.write_text(
        artifact.model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    summary = series.repeatability

    print("M4-H PyTorch resource profiling")
    print()
    print(
        "Candidate:",
        series.implementation_id,
    )
    print(
        "Verification:",
        verification.decision.value,
    )
    print(
        "Admissibility:",
        admissibility.status.value,
    )
    print()
    print(
        f"Repetitions: {summary.repetitions}"
    )
    print(
        "Measured executions:",
        summary.measured_executions,
    )
    print(
        "Mean latency:",
        f"{summary.weighted_mean_latency_ms:.6f} ms",
    )
    print(
        "Mean throughput:",
        f"{summary.mean_cases_per_second:.2f} cases/sec",
    )
    print(
        "Resource observations:",
        summary.resource_observations,
    )

    if summary.mean_process_cpu_time_ms is not None:
        print(
            "Mean measured-region CPU time:",
            f"{summary.mean_process_cpu_time_ms:.3f} ms",
        )

    if (
        summary.mean_cpu_time_to_wall_time_ratio
        is not None
    ):
        print(
            "Mean CPU/wall ratio:",
            (
                f"{summary.mean_cpu_time_to_wall_time_ratio:.3f}"
            ),
        )

    if summary.max_process_peak_rss_bytes is not None:
        print(
            "Max process peak RSS:",
            (
                f"{summary.max_process_peak_rss_bytes / 1024**2:.2f} MiB"
            ),
        )

    if (
        summary.mean_process_peak_rss_growth_bytes
        is not None
    ):
        print(
            "Mean peak-RSS growth:",
            (
                f"{summary.mean_process_peak_rss_growth_bytes / 1024**2:.3f} MiB"
            ),
        )

    print(
        "Parameter count:",
        parameter_count,
    )
    print(
        "Float32 parameter storage:",
        f"{footprint.model_size_bytes} bytes",
    )
    print()
    print(
        "RSS note: process-lifetime high-water mark; "
        "not candidate-exclusive resident memory."
    )
    print(
        "Evidence artifact:",
        ARTIFACT_PATH,
    )


if __name__ == "__main__":
    main()
