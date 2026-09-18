"""Verify and benchmark FP16 and BF16 PyTorch AP candidates."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    StructuredModelFootprint,
    run_structured_benchmark_series,
)
from vait.runners.base import ImplementationRunner
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.ap_torch_optimisation import (
    BatchPredictionFunction,
    TorchOptimisationCandidate,
    build_pytorch_lower_precision_candidates,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)

DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4e-ap-torch-lower-precision-v0.1.json"
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


def _dtype(
    candidate: TorchOptimisationCandidate,
) -> str:
    """Return declared lower precision."""
    value = (
        candidate
        .transformation
        .configuration
        .get("precision")
    )

    if not isinstance(value, str):
        raise RuntimeError(
            "Lower-precision candidate lacks precision metadata."
        )

    return value


def _search_point(
    *,
    candidate: TorchOptimisationCandidate,
    batch_size: int,
) -> InferenceSearchPoint:
    """Build one declared lower-precision search point."""
    transformation = candidate.transformation
    dtype = _dtype(candidate)

    return InferenceSearchPoint(
        search_point_id=(
            f"{transformation.implementation_id}"
            f"::pytorch-eager::{dtype}::batch-{batch_size}"
        ),
        search_space_id=(
            "m4e-ap-torch-lower-precision-v01"
        ),
        task_family="structured-decision",
        candidate_id=(
            transformation.descriptor.transformation_id
        ),
        implementation_id=(
            transformation.implementation_id
        ),
        configuration_id=(
            f"{dtype}-batch-{batch_size}"
        ),
        configuration=InferenceConfiguration(
            provider="local",
            runtime="pytorch-eager",
            device="cpu",
            dtype=dtype,
            batch_size=batch_size,
            model_id=(
                transformation.implementation_id
            ),
            model_revision=(
                transformation.descriptor.version
            ),
            metadata={
                "model_family": (
                    transformation.configuration[
                        "model_family"
                    ]
                ),
                "optimisation": (
                    transformation.configuration[
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
    candidate: TorchOptimisationCandidate,
) -> InferenceCompatibilityContext:
    """Declare measured lower-precision CPU capabilities."""
    transformation = candidate.transformation

    return InferenceCompatibilityContext(
        available_providers=frozenset(
            {"local"}
        ),
        available_runtimes=frozenset(
            {"pytorch-eager"}
        ),
        available_devices=frozenset(
            {"cpu"}
        ),
        available_dtypes=frozenset(
            {
                "float16",
                "bfloat16",
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
    """Attach bounded verification evidence to one search point."""
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
            for failure in verification.failures
        ),
    )


def _positive_int(
    value: JsonValue | None,
    *,
    name: str,
) -> int:
    """Require positive integer footprint evidence."""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise RuntimeError(
            f"{name} must be a positive integer."
        )

    return value


def _footprint(
    candidate: TorchOptimisationCandidate,
) -> StructuredModelFootprint:
    """Build measured footprint evidence."""
    configuration = (
        candidate
        .transformation
        .configuration
    )

    return StructuredModelFootprint(
        parameter_count=_positive_int(
            configuration.get(
                "parameter_count"
            ),
            name="parameter_count",
        ),
        model_size_bytes=_positive_int(
            configuration.get(
                "serialized_state_bytes"
            ),
            name="serialized_state_bytes",
        ),
        metadata={
            "optimisation": (
                configuration[
                    "optimisation"
                ]
            ),
            "precision": (
                configuration[
                    "precision"
                ]
            ),
        },
    )


def _adapt_batch_operation(
    operation: BatchPredictionFunction,
) -> NativeBatchOperation:
    """Adapt AP input dictionaries to the generic batch harness."""

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
    runner: ImplementationRunner,
    operation: BatchPredictionFunction,
    cases: Sequence[VerificationCase],
    batch_size: int,
) -> tuple[int, int]:
    """Require native batch outputs to equal verified scalar outputs."""
    checked = 0
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
                "Lower-precision batch operation returned "
                "an invalid number of outputs."
            )

        for (
            case,
            batch_output,
        ) in zip(
            batch,
            batch_outputs,
            strict=True,
        ):
            observation = runner.execute(
                case
            )

            if observation.error is not None:
                raise RuntimeError(
                    "Scalar lower-precision execution failed: "
                    f"{observation.error}"
                )

            checked += 1

            if (
                observation.output
                != batch_output
            ):
                disagreements += 1

    return (
        checked,
        disagreements,
    )


def _run_candidate(
    *,
    candidate: TorchOptimisationCandidate,
    cases: tuple[
        VerificationCase,
        ...,
    ],
    verification: BenchmarkVerificationReport,
) -> dict[str, Any]:
    """Run admissibility and performance evidence for one precision."""
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
            "Lower-precision candidate preparation failed: "
            f"{transformation.implementation_id}"
        )

    runner = preparation.runner
    context = _compatibility_context(
        candidate
    )
    footprint = _footprint(
        candidate
    )

    scalar_point = _search_point(
        candidate=candidate,
        batch_size=1,
    )

    scalar_compatibility = (
        evaluate_search_point_compatibility(
            scalar_point,
            context,
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

    scalar_series = None

    if scalar_admissibility.is_admissible:
        scalar_series = (
            run_structured_benchmark_series(
                admissibility=(
                    scalar_admissibility
                ),
                runner=runner,
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
                    "precision": (
                        _dtype(candidate)
                    ),
                    "verification_decision": (
                        verification
                        .decision
                        .value
                    ),
                },
            )
        )

    batch_results: list[
        dict[str, Any]
    ] = []

    for batch_size in BATCH_SIZES:
        point = _search_point(
            candidate=candidate,
            batch_size=batch_size,
        )

        compatibility = (
            evaluate_search_point_compatibility(
                point,
                context,
            )
        )

        admissibility = (
            evaluate_search_point_admissibility(
                compatibility=(
                    compatibility
                ),
                verification_evidence=(
                    _verification_evidence(
                        search_point=point,
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
            runner=runner,
            operation=(
                candidate.batch_operation
            ),
            cases=cases,
            batch_size=batch_size,
        )

        equivalence_passed = (
            disagreements == 0
        )

        series = None

        if equivalence_passed:
            series = (
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
                        "precision": (
                            _dtype(
                                candidate
                            )
                        ),
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
                    series.model_dump(
                        mode="json"
                    )
                    if series is not None
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
            if scalar_series is not None
            else None
        ),
        "batch_results": (
            batch_results
        ),
    }


def main() -> None:
    """Build verify-first FP16 and BF16 M4-E evidence."""
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

    candidates = (
        build_pytorch_lower_precision_candidates(
            corpus
        )
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
        dict[str, Any]
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

        result = _run_candidate(
            candidate=candidate,
            cases=cases,
            verification=(
                verification
            ),
        )

        results.append(
            result
        )

        scalar = result[
            "scalar_benchmark"
        ]

        if isinstance(
            scalar,
            dict,
        ):
            repeatability = scalar[
                "repeatability"
            ]

            print(
                "  scalar:",
                f"{repeatability['mean_cases_per_second']:.2f}",
                "cases/s,",
                f"{repeatability['weighted_mean_latency_ms']:.6f}",
                "ms",
            )
        else:
            print(
                "  scalar: EXCLUDED"
            )

        for batch_result in result[
            "batch_results"
        ]:
            benchmark = (
                batch_result[
                    "benchmark"
                ]
            )

            if not isinstance(
                benchmark,
                dict,
            ):
                print(
                    "  batch",
                    batch_result[
                        "batch_size"
                    ],
                    "EXCLUDED",
                )
                continue

            repeatability = (
                benchmark[
                    "repeatability"
                ]
            )

            print(
                "  batch",
                batch_result[
                    "batch_size"
                ],
                f"{repeatability['mean_cases_per_second']:.2f}",
                "cases/s,",
                f"{repeatability['weighted_mean_latency_ms']:.6f}",
                "ms",
            )

    payload: dict[
        str,
        Any,
    ] = {
        "schema_version": "0.1",
        "milestone": "M4-E",
        "experiment": (
            "ap-pytorch-lower-precision-fp16-bf16"
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
        "results": (
            results
        ),
        "limitations": [
            (
                "Lower-precision support and performance "
                "are environment-specific."
            ),
            (
                "Probe-level class agreement was not used "
                "as verification evidence."
            ),
            (
                "Each lower-precision candidate was "
                "re-verified against the full AP benchmark."
            ),
            (
                "A REJECT candidate is excluded from "
                "performance benchmarking."
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
    print(
        f"Wrote {ARTIFACT_PATH}"
    )


if __name__ == "__main__":
    main()
