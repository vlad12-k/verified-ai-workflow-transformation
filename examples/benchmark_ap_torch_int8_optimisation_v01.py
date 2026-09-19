"""Verify and benchmark matched FP32 and TorchAO INT8 AP candidates."""

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import JsonValue

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
from vait.inference.models import InferenceConfiguration
from vait.optimisation.admissibility import (
    SearchPointVerificationEvidence,
    VerificationEvidenceKind,
    evaluate_search_point_admissibility,
)
from vait.optimisation.compatibility import (
    InferenceCompatibilityContext,
    evaluate_search_point_compatibility,
)
from vait.optimisation.search_space import InferenceSearchPoint
from vait.optimisation.structured_batch_benchmark import (
    NativeBatchOperation,
    run_structured_native_batch_series,
)
from vait.optimisation.structured_benchmark import (
    StructuredBenchmarkSeries,
    StructuredModelFootprint,
    run_structured_benchmark_series,
)
from vait.runners.base import ImplementationRunner
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.ap_torch_optimisation import (
    BatchPredictionFunction,
    TorchOptimisationCandidate,
    build_pytorch_precision_candidates,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)
from vait.transformations.python_callable import (
    PythonCallableTransformation,
)

DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4e-ap-torch-int8-optimisation-v0.1.json"
)

TRAINING_SAMPLE_COUNT = 600
TRAINING_SEED = 20260916

BATCH_SIZES = (
    1,
    4,
    8,
    16,
    32,
)

REPETITIONS = 3
WARMUP_ROUNDS = 3
MEASURED_ROUNDS = 20


def _runtime_and_dtype(
    transformation: PythonCallableTransformation,
) -> tuple[str, str]:
    """Return execution identity for one precision candidate."""
    optimisation = (
        transformation.configuration.get(
            "optimisation"
        )
    )

    if optimisation == "torchao-int8-dynamic":
        return (
            "torchao-eager",
            "int8-dynamic",
        )

    return (
        "pytorch-eager",
        "float32",
    )


def _search_point(
    *,
    transformation: PythonCallableTransformation,
    batch_size: int,
) -> InferenceSearchPoint:
    """Build one explicit M4-E search point."""
    runtime, dtype = (
        _runtime_and_dtype(
            transformation
        )
    )

    return InferenceSearchPoint(
        search_point_id=(
            f"{transformation.implementation_id}"
            f"::{runtime}::batch-{batch_size}"
        ),
        search_space_id=(
            "m4e-ap-torch-precision-v01"
        ),
        task_family="structured-decision",
        candidate_id=(
            transformation
            .descriptor
            .transformation_id
        ),
        implementation_id=(
            transformation.implementation_id
        ),
        configuration_id=(
            f"{dtype}-batch-{batch_size}"
        ),
        configuration=InferenceConfiguration(
            provider="local",
            runtime=runtime,
            device="cpu",
            dtype=dtype,
            batch_size=batch_size,
            model_id=(
                transformation
                .implementation_id
            ),
            model_revision=(
                transformation
                .descriptor
                .version
            ),
            metadata={
                "model_family": (
                    transformation
                    .configuration[
                        "model_family"
                    ]
                ),
                "optimisation": (
                    transformation
                    .configuration[
                        "optimisation"
                    ]
                ),
            },
        ),
        requires_verification=True,
        required_capabilities=(
            transformation
            .descriptor
            .required_capabilities
        ),
        candidate_metadata=dict(
            transformation.configuration
        ),
        configuration_metadata={
            "m4_slice": "M4-E",
        },
    )


def _compatibility_context(
    transformation: PythonCallableTransformation,
) -> InferenceCompatibilityContext:
    """Declare the runtime capabilities used by this experiment."""
    return InferenceCompatibilityContext(
        available_providers=frozenset(
            {"local"}
        ),
        available_runtimes=frozenset(
            {
                "pytorch-eager",
                "torchao-eager",
            }
        ),
        available_devices=frozenset(
            {"cpu"}
        ),
        available_dtypes=frozenset(
            {
                "float32",
                "int8-dynamic",
            }
        ),
        max_batch_size=max(
            BATCH_SIZES
        ),
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
    )


def _verification_evidence(
    *,
    search_point: InferenceSearchPoint,
    verification: BenchmarkVerificationReport,
) -> SearchPointVerificationEvidence:
    """Attach bounded benchmark verification to one search point."""
    return SearchPointVerificationEvidence(
        search_point_id=(
            search_point.search_point_id
        ),
        implementation_id=(
            search_point.implementation_id
        ),
        contract_id=(
            "benchmark:"
            f"{verification.benchmark_id}:"
            f"{verification.benchmark_version}:"
            f"{verification.candidate_implementation_id}"
        ),
        decision=verification.decision,
        evidence_kind=(
            VerificationEvidenceKind
            .BOUNDED_STATISTICAL
        ),
        statistical_evidence_present=(
            verification.statistical_evidence
            is not None
        ),
        failure_codes=tuple(
            failure.code
            for failure
            in verification.failures
        ),
    )


def _footprint(
    transformation: PythonCallableTransformation,
) -> StructuredModelFootprint:
    """Build measured model-footprint evidence."""
    parameter_count = (
        transformation.configuration.get(
            "parameter_count"
        )
    )

    model_size_bytes = (
        transformation.configuration.get(
            "serialized_state_bytes"
        )
    )

    if (
        not isinstance(
            parameter_count,
            int,
        )
        or isinstance(
            parameter_count,
            bool,
        )
        or parameter_count <= 0
    ):
        raise RuntimeError(
            "Positive parameter_count evidence is required."
        )

    if (
        not isinstance(
            model_size_bytes,
            int,
        )
        or isinstance(
            model_size_bytes,
            bool,
        )
        or model_size_bytes <= 0
    ):
        raise RuntimeError(
            "Positive serialized model-size evidence is required."
        )

    return StructuredModelFootprint(
        parameter_count=parameter_count,
        model_size_bytes=model_size_bytes,
        metadata={
            "optimisation": (
                transformation
                .configuration[
                    "optimisation"
                ]
            ),
        },
    )


def _adapt_batch_operation(
    operation: BatchPredictionFunction,
) -> NativeBatchOperation:
    """Adapt raw AP inputs to the generic native-batch harness."""

    def execute(
        cases: Sequence[
            VerificationCase
        ],
    ) -> tuple[
        JsonValue,
        ...,
    ]:
        return operation(
            tuple(
                case.input_data
                for case in cases
            )
        )

    return execute


def _verify_batch_equivalence(
    *,
    scalar_runner: ImplementationRunner,
    operation: BatchPredictionFunction,
    cases: Sequence[VerificationCase],
    batch_size: int,
) -> tuple[int, int]:
    """Require native batching to preserve verified scalar behaviour."""
    cases_checked = 0
    disagreements = 0

    for start in range(
        0,
        len(cases),
        batch_size,
    ):
        batch = cases[
            start : start + batch_size
        ]

        batch_outputs = operation(
            tuple(
                case.input_data
                for case in batch
            )
        )

        if len(batch_outputs) != len(
            batch
        ):
            raise RuntimeError(
                "Native batch operation returned "
                "an invalid number of outputs."
            )

        for case, batch_output in zip(
            batch,
            batch_outputs,
            strict=True,
        ):
            scalar_observation = (
                scalar_runner.execute(
                    case
                )
            )

            if (
                scalar_observation.error
                is not None
            ):
                raise RuntimeError(
                    "Scalar candidate failed during "
                    "batch-equivalence checking: "
                    f"{scalar_observation.error}"
                )

            cases_checked += 1

            if (
                scalar_observation.output
                != batch_output
            ):
                disagreements += 1

    return (
        cases_checked,
        disagreements,
    )


def _benchmark_candidate(
    *,
    candidate: TorchOptimisationCandidate,
    cases: tuple[
        VerificationCase,
        ...,
    ],
    verification: BenchmarkVerificationReport,
) -> dict[str, object]:
    """Benchmark one candidate only after verification-backed admissibility."""
    transformation = (
        candidate.transformation
    )

    preparation = (
        transformation.prepare_candidate(
            ApplicabilityContext(
                risk_level=(
                    RiskLevel.CRITICAL
                ),
                available_capabilities=(
                    transformation
                    .descriptor
                    .required_capabilities
                ),
            )
        )
    )

    if preparation.runner is None:
        raise RuntimeError(
            "Candidate preparation unexpectedly failed: "
            f"{transformation.implementation_id}"
        )

    runner = preparation.runner

    scalar_point = _search_point(
        transformation=transformation,
        batch_size=1,
    )

    scalar_compatibility = (
        evaluate_search_point_compatibility(
            scalar_point,
            _compatibility_context(
                transformation
            ),
        )
    )

    scalar_admissibility = (
        evaluate_search_point_admissibility(
            compatibility=(
                scalar_compatibility
            ),
            verification_evidence=(
                _verification_evidence(
                    search_point=(
                        scalar_point
                    ),
                    verification=(
                        verification
                    ),
                )
            ),
        )
    )

    footprint = _footprint(
        transformation
    )

    scalar_series: (
        StructuredBenchmarkSeries
        | None
    ) = None

    if (
        scalar_admissibility
        .is_admissible
    ):
        scalar_series = (
            run_structured_benchmark_series(
                admissibility=(
                    scalar_admissibility
                ),
                runner=runner,
                cases=cases,
                repetitions=REPETITIONS,
                warmup_rounds=(
                    WARMUP_ROUNDS
                ),
                measured_rounds=(
                    MEASURED_ROUNDS
                ),
                footprint=footprint,
                evidence_metadata={
                    "m4_slice": "M4-E",
                    "verification_decision": (
                        verification
                        .decision
                        .value
                    ),
                    "optimisation": (
                        transformation
                        .configuration[
                            "optimisation"
                        ]
                    ),
                },
            )
        )

    batch_results: list[
        dict[str, object]
    ] = []

    for batch_size in BATCH_SIZES:
        batch_point = _search_point(
            transformation=(
                transformation
            ),
            batch_size=batch_size,
        )

        compatibility = (
            evaluate_search_point_compatibility(
                batch_point,
                _compatibility_context(
                    transformation
                ),
            )
        )

        admissibility = (
            evaluate_search_point_admissibility(
                compatibility=(
                    compatibility
                ),
                verification_evidence=(
                    _verification_evidence(
                        search_point=(
                            batch_point
                        ),
                        verification=(
                            verification
                        ),
                    )
                ),
            )
        )

        if not admissibility.is_admissible:
            batch_results.append(
                {
                    "batch_size": (
                        batch_size
                    ),
                    "admissibility": (
                        admissibility.model_dump(
                            mode="json"
                        )
                    ),
                    "batch_equivalence": None,
                    "benchmark": None,
                }
            )

            continue

        (
            checked,
            disagreements,
        ) = _verify_batch_equivalence(
            scalar_runner=runner,
            operation=(
                candidate.batch_operation
            ),
            cases=cases,
            batch_size=batch_size,
        )

        equivalence_passed = (
            disagreements == 0
        )

        batch_series = None

        if equivalence_passed:
            batch_series = (
                run_structured_native_batch_series(
                    admissibility=(
                        admissibility
                    ),
                    implementation_id=(
                        transformation
                        .implementation_id
                    ),
                    operation=(
                        _adapt_batch_operation(
                            candidate
                            .batch_operation
                        )
                    ),
                    cases=cases,
                    repetitions=(
                        REPETITIONS
                    ),
                    warmup_rounds=(
                        WARMUP_ROUNDS
                    ),
                    measured_rounds=(
                        MEASURED_ROUNDS
                    ),
                    footprint=footprint,
                    evidence_metadata={
                        "m4_slice": "M4-E",
                        "verification_decision": (
                            verification
                            .decision
                            .value
                        ),
                        "batch_equivalence_cases": (
                            checked
                        ),
                        "batch_equivalence_disagreements": (
                            disagreements
                        ),
                    },
                )
            )

        batch_results.append(
            {
                "batch_size": (
                    batch_size
                ),
                "admissibility": (
                    admissibility.model_dump(
                        mode="json"
                    )
                ),
                "batch_equivalence": {
                    "cases": checked,
                    "disagreements": (
                        disagreements
                    ),
                    "passed": (
                        equivalence_passed
                    ),
                },
                "benchmark": (
                    batch_series.model_dump(
                        mode="json"
                    )
                    if batch_series
                    is not None
                    else None
                ),
            }
        )

    return {
        "implementation_id": (
            transformation
            .implementation_id
        ),
        "configuration": dict(
            transformation.configuration
        ),
        "verification": (
            verification.model_dump(
                mode="json"
            )
        ),
        "scalar_admissibility": (
            scalar_admissibility.model_dump(
                mode="json"
            )
        ),
        "scalar_benchmark": (
            scalar_series.model_dump(
                mode="json"
            )
            if scalar_series
            is not None
            else None
        ),
        "batch_results": (
            batch_results
        ),
    }


def _print_performance(
    *,
    candidate: TorchOptimisationCandidate,
    result: dict[str, object],
) -> None:
    """Print a concise human-readable experiment summary."""
    transformation = (
        candidate.transformation
    )

    print()
    print(
        transformation
        .implementation_id
    )

    scalar_raw = result[
        "scalar_benchmark"
    ]

    if not isinstance(
        scalar_raw,
        dict,
    ):
        print(
            "  performance: EXCLUDED"
        )
        return

    repeatability = scalar_raw.get(
        "repeatability"
    )

    if not isinstance(
        repeatability,
        dict,
    ):
        raise RuntimeError(
            "Scalar repeatability evidence is missing."
        )

    print(
        "  scalar:",
        f"{repeatability['mean_cases_per_second']:.2f}",
        "cases/s,",
        f"{repeatability['weighted_mean_latency_ms']:.6f}",
        "ms",
    )

    raw_batches = result[
        "batch_results"
    ]

    if not isinstance(
        raw_batches,
        list,
    ):
        raise RuntimeError(
            "Batch evidence is missing."
        )

    for raw_batch in raw_batches:
        if not isinstance(
            raw_batch,
            dict,
        ):
            raise RuntimeError(
                "Invalid batch evidence."
            )

        benchmark = raw_batch.get(
            "benchmark"
        )

        if not isinstance(
            benchmark,
            dict,
        ):
            print(
                "  batch",
                raw_batch.get(
                    "batch_size"
                ),
                "EXCLUDED",
            )
            continue

        batch_repeatability = (
            benchmark.get(
                "repeatability"
            )
        )

        if not isinstance(
            batch_repeatability,
            dict,
        ):
            raise RuntimeError(
                "Batch repeatability evidence is missing."
            )

        print(
            "  batch",
            raw_batch[
                "batch_size"
            ],
            f"{batch_repeatability['mean_cases_per_second']:.2f}",
            "cases/s,",
            f"{batch_repeatability['weighted_mean_latency_ms']:.6f}",
            "ms",
        )


def main() -> None:
    """Run verify-first FP32 versus TorchAO INT8 optimisation evidence."""
    dataset = load_benchmark(
        DATASET_PATH
    )

    corpus = (
        build_synthetic_ap_training_corpus(
            sample_count=(
                TRAINING_SAMPLE_COUNT
            ),
            seed=TRAINING_SEED,
            excluded_inputs=(
                case.input_data
                for case in dataset.cases
            ),
        )
    )

    (
        fp32_candidate,
        int8_candidate,
    ) = build_pytorch_precision_candidates(
        corpus
    )

    candidates = (
        fp32_candidate,
        int8_candidate,
    )

    cases = tuple(
        VerificationCase(
            id=case.case_id,
            input_data=case.input_data,
            risk_level=case.risk_level,
        )
        for case in dataset.cases
    )

    policy = (
        BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.15,
            max_high_risk_disagreement_rate=0.25,
            confidence_level=0.95,
            min_total_cases=20,
            min_high_risk_cases=11,
        )
    )

    results: list[
        dict[str, object]
    ] = []

    for candidate in candidates:
        transformation = (
            candidate.transformation
        )

        preparation = (
            transformation.prepare_candidate(
                ApplicabilityContext(
                    risk_level=(
                        RiskLevel.CRITICAL
                    ),
                    available_capabilities=(
                        transformation
                        .descriptor
                        .required_capabilities
                    ),
                )
            )
        )

        if preparation.runner is None:
            raise RuntimeError(
                "Candidate preparation failed: "
                f"{transformation.implementation_id}"
            )

        verification = (
            verify_benchmark_bounded(
                dataset=dataset,
                candidate=(
                    preparation.runner
                ),
                policy=policy,
                candidate_configuration={
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
                    **transformation.configuration,
                },
            )
        )

        print()
        print(
            transformation
            .implementation_id,
            "verification:",
            verification.decision.value,
        )

        result = _benchmark_candidate(
            candidate=candidate,
            cases=cases,
            verification=(
                verification
            ),
        )

        results.append(
            result
        )

        _print_performance(
            candidate=candidate,
            result=result,
        )

    fp32_bytes = (
        fp32_candidate
        .transformation
        .configuration[
            "serialized_state_bytes"
        ]
    )

    int8_bytes = (
        int8_candidate
        .transformation
        .configuration[
            "serialized_state_bytes"
        ]
    )

    if (
        not isinstance(
            fp32_bytes,
            int,
        )
        or isinstance(
            fp32_bytes,
            bool,
        )
        or not isinstance(
            int8_bytes,
            int,
        )
        or isinstance(
            int8_bytes,
            bool,
        )
    ):
        raise RuntimeError(
            "Serialized-size evidence is invalid."
        )

    size_reduction_percent = (
        (
            1.0
            - (
                int8_bytes
                / fp32_bytes
            )
        )
        * 100.0
    )

    payload = {
        "schema_version": "0.1",
        "milestone": "M4-E",
        "experiment": (
            "ap-pytorch-fp32-vs-torchao-int8-dynamic"
        ),
        "verify_first_optimise_second": True,
        "training_sample_count": (
            TRAINING_SAMPLE_COUNT
        ),
        "training_seed": (
            TRAINING_SEED
        ),
        "batch_sizes": list(
            BATCH_SIZES
        ),
        "repetitions": (
            REPETITIONS
        ),
        "warmup_rounds": (
            WARMUP_ROUNDS
        ),
        "measured_rounds": (
            MEASURED_ROUNDS
        ),
        "compression": {
            "fp32_serialized_state_bytes": (
                fp32_bytes
            ),
            "int8_serialized_state_bytes": (
                int8_bytes
            ),
            "int8_to_fp32_size_ratio": (
                int8_bytes
                / fp32_bytes
            ),
            "size_reduction_percent": (
                size_reduction_percent
            ),
        },
        "results": results,
        "limitations": [
            (
                "Verification uses the AP v0.2 "
                "development benchmark."
            ),
            (
                "Performance evidence is local to the "
                "recorded execution environment."
            ),
            (
                "Serialized state size includes format "
                "and quantization metadata overhead."
            ),
            (
                "A REJECT candidate is excluded from "
                "performance optimisation regardless "
                "of potential speed or size benefits."
            ),
        ],
    }

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("Compression")
    print(
        "  FP32 bytes:",
        fp32_bytes,
    )
    print(
        "  INT8 bytes:",
        int8_bytes,
    )
    print(
        "  reduction:",
        f"{size_reduction_percent:.2f}%",
    )
    print()
    print(
        f"Wrote {ARTIFACT_PATH}"
    )


if __name__ == "__main__":
    main()
