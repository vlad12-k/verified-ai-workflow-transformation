"""Verify and benchmark the static-shape TorchInductor AP candidate."""

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
from vait.inference.models import (
    InferenceConfiguration,
    InferenceSetupEvidence,
)
from vait.optimisation.admissibility import (
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
from vait.optimisation.structured_batch_benchmark import (
    NativeBatchOperation,
    run_structured_native_batch_series,
)
from vait.optimisation.structured_benchmark import (
    StructuredModelFootprint,
    run_structured_benchmark_series,
)
from vait.runners.base import ImplementationRunner
from vait.transformations.applicability import (
    ApplicabilityContext,
)
from vait.transformations.library.ap_torch_optimisation import (
    BatchPredictionFunction,
    TorchCompilationCandidate,
    TorchCompileSetup,
    build_pytorch_compilation_candidate,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)

DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4e-ap-torch-compile-optimisation-v0.1.json"
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


def _search_point(
    *,
    implementation_id: str,
    transformation_id: str,
    version: str,
    required_capabilities: frozenset[str],
    batch_size: int,
) -> InferenceSearchPoint:
    """Build one declared static-shape Inductor search point."""
    return InferenceSearchPoint(
        search_point_id=(
            f"{implementation_id}"
            f"::pytorch-inductor::batch-{batch_size}"
        ),
        search_space_id=(
            "m4e-ap-torch-compile-v01"
        ),
        task_family=(
            "structured-decision"
        ),
        candidate_id=(
            transformation_id
        ),
        implementation_id=(
            implementation_id
        ),
        configuration_id=(
            f"inductor-fp32-batch-{batch_size}"
        ),
        configuration=(
            InferenceConfiguration(
                provider="local",
                runtime="pytorch-inductor",
                device="cpu",
                dtype="float32",
                batch_size=batch_size,
                model_id=implementation_id,
                model_revision=version,
                metadata={
                    "compile_backend": (
                        "inductor"
                    ),
                    "compile_dynamic": False,
                    "compile_fullgraph": True,
                    "static_batch_size": (
                        batch_size
                    ),
                },
            )
        ),
        requires_verification=True,
        required_capabilities=(
            required_capabilities
        ),
        candidate_metadata={
            "optimisation": (
                "torch-compile-inductor"
            ),
        },
        configuration_metadata={
            "m4_slice": "M4-E",
        },
    )


def _compatibility_context(
    *,
    implementation_id: str,
    required_capabilities: frozenset[str],
) -> InferenceCompatibilityContext:
    """Declare the runtime available to this experiment."""
    return InferenceCompatibilityContext(
        available_providers=frozenset(
            {"local"}
        ),
        available_runtimes=frozenset(
            {"pytorch-inductor"}
        ),
        available_devices=frozenset(
            {"cpu"}
        ),
        available_dtypes=frozenset(
            {"float32"}
        ),
        max_batch_size=max(
            BATCH_SIZES
        ),
        available_model_ids=frozenset(
            {implementation_id}
        ),
        available_capabilities=(
            required_capabilities
        ),
    )


def _verification_evidence(
    *,
    search_point: InferenceSearchPoint,
    verification: BenchmarkVerificationReport,
) -> SearchPointVerificationEvidence:
    """Attach bounded verification to one compiled search point."""
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


def _setup_evidence(
    setup: TorchCompileSetup,
) -> InferenceSetupEvidence:
    """Represent compile cost separately from steady-state inference."""
    return InferenceSetupEvidence(
        duration_ms=(
            setup.total_setup_ms
        ),
        scope=(
            "static-shape-torchinductor-compilation"
        ),
        included_operations=[
            "torch.compile-wrapper-creation",
            "first-compiled-call",
        ],
        metadata={
            "batch_size": (
                setup.batch_size
            ),
            "wrapper_creation_ms": (
                setup.wrapper_creation_ms
            ),
            "first_call_compile_ms": (
                setup.first_call_compile_ms
            ),
            "included_in_inference_latency": False,
        },
    )


def _adapt_batch_operation(
    operation: BatchPredictionFunction,
) -> NativeBatchOperation:
    """Adapt raw AP inputs to the generic batch benchmark harness."""

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
    """Compare fixed-shape batch outputs with verified scalar outputs."""
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
                "Compiled batch operation returned "
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
                    "Scalar compiled inference failed "
                    "during equivalence checking: "
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


def _positive_int(
    value: JsonValue | None,
    *,
    name: str,
) -> int:
    """Require positive integer model evidence."""
    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
        or value <= 0
    ):
        raise RuntimeError(
            f"{name} must be a positive integer."
        )

    return value


def _run_candidate(
    *,
    candidate: TorchCompilationCandidate,
    cases: tuple[
        VerificationCase,
        ...,
    ],
    verification: BenchmarkVerificationReport,
) -> dict[str, Any]:
    """Run admissibility and benchmark evidence for the compiled candidate."""
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
            "Compiled candidate preparation failed."
        )

    runner = preparation.runner

    footprint = StructuredModelFootprint(
        parameter_count=_positive_int(
            transformation.configuration.get(
                "parameter_count"
            ),
            name="parameter_count",
        ),
        model_size_bytes=_positive_int(
            transformation.configuration.get(
                "serialized_state_bytes"
            ),
            name="serialized_state_bytes",
        ),
        metadata={
            "optimisation": (
                "torch-compile-inductor"
            ),
        },
    )

    context = _compatibility_context(
        implementation_id=(
            transformation.implementation_id
        ),
        required_capabilities=(
            transformation
            .descriptor
            .required_capabilities
        ),
    )

    scalar_point = _search_point(
        implementation_id=(
            transformation.implementation_id
        ),
        transformation_id=(
            transformation
            .descriptor
            .transformation_id
        ),
        version=(
            transformation
            .descriptor
            .version
        ),
        required_capabilities=(
            transformation
            .descriptor
            .required_capabilities
        ),
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
                setup=_setup_evidence(
                    candidate
                    .setup_by_batch_size[
                        1
                    ]
                ),
                footprint=footprint,
                evidence_metadata={
                    "m4_slice": "M4-E",
                    "optimisation": (
                        "torch-compile-inductor"
                    ),
                    "verification_decision": (
                        verification
                        .decision
                        .value
                    ),
                    "compile_cost_excluded_from_steady_state": True,
                },
            )
        )

    batch_results: list[
        dict[str, Any]
    ] = []

    for batch_size in BATCH_SIZES:
        point = _search_point(
            implementation_id=(
                transformation
                .implementation_id
            ),
            transformation_id=(
                transformation
                .descriptor
                .transformation_id
            ),
            version=(
                transformation
                .descriptor
                .version
            ),
            required_capabilities=(
                transformation
                .descriptor
                .required_capabilities
            ),
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

        setup = (
            candidate
            .setup_by_batch_size[
                batch_size
            ]
        )

        if not admissibility.is_admissible:
            batch_results.append(
                {
                    "batch_size": (
                        batch_size
                    ),
                    "setup": (
                        _setup_evidence(
                            setup
                        ).model_dump(
                            mode="json"
                        )
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

        operation = (
            candidate
            .batch_operations[
                batch_size
            ]
        )

        (
            checked,
            disagreements,
        ) = _verify_batch_equivalence(
            runner=runner,
            operation=operation,
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
                            operation
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
                    setup=(
                        _setup_evidence(
                            setup
                        )
                    ),
                    footprint=footprint,
                    evidence_metadata={
                        "m4_slice": "M4-E",
                        "optimisation": (
                            "torch-compile-inductor"
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
                        "compile_cost_excluded_from_steady_state": True,
                    },
                )
            )

        batch_results.append(
            {
                "batch_size": (
                    batch_size
                ),
                "setup": (
                    _setup_evidence(
                        setup
                    ).model_dump(
                        mode="json"
                    )
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
    """Build verify-first static TorchInductor M4-E evidence."""
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

    print(
        "Building static TorchInductor "
        "candidate configurations..."
    )

    candidate = (
        build_pytorch_compilation_candidate(
            corpus,
            batch_sizes=(
                BATCH_SIZES
            ),
        )
    )

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
            "Compiled candidate preparation failed."
        )

    verification = (
        verify_benchmark_bounded(
            dataset=dataset,
            candidate=(
                preparation.runner
            ),
            policy=(
                BoundedVerificationPolicy(
                    max_overall_disagreement_rate=0.15,
                    max_high_risk_disagreement_rate=0.25,
                    confidence_level=0.95,
                    min_total_cases=20,
                    min_high_risk_cases=11,
                )
            ),
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
        transformation.implementation_id,
        "verification:",
        verification.decision.value,
    )

    cases = tuple(
        VerificationCase(
            id=case.case_id,
            input_data=(
                case.input_data
            ),
            risk_level=(
                case.risk_level
            ),
        )
        for case in dataset.cases
    )

    result = _run_candidate(
        candidate=candidate,
        cases=cases,
        verification=verification,
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

        print()
        print(
            "scalar:",
            f"{repeatability['mean_cases_per_second']:.2f}",
            "cases/s,",
            f"{repeatability['weighted_mean_latency_ms']:.6f}",
            "ms",
        )
    else:
        print()
        print(
            "scalar: EXCLUDED"
        )

    print()
    print(
        "Static batch evidence"
    )

    for batch_result in result[
        "batch_results"
    ]:
        batch_size = (
            batch_result[
                "batch_size"
            ]
        )

        setup = (
            batch_result[
                "setup"
            ]
        )

        benchmark = (
            batch_result[
                "benchmark"
            ]
        )

        print(
            f"batch {batch_size}"
        )
        print(
            "  compile setup:",
            f"{setup['duration_ms']:.3f}",
            "ms",
        )

        if not isinstance(
            benchmark,
            dict,
        ):
            print(
                "  benchmark: EXCLUDED"
            )
            continue

        repeatability = (
            benchmark[
                "repeatability"
            ]
        )

        print(
            "  throughput:",
            f"{repeatability['mean_cases_per_second']:.2f}",
            "cases/s",
        )
        print(
            "  mean latency:",
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
            "ap-pytorch-torchinductor-static-shape"
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
        "result": result,
        "limitations": [
            (
                "Compilation setup cost is measured "
                "separately from steady-state inference."
            ),
            (
                "TorchInductor cache state is not reset "
                "between development runs."
            ),
            (
                "Each declared batch size uses its own "
                "static-shape compiled execution path."
            ),
            (
                "Tail batches are padded to the declared "
                "static batch size and padding overhead "
                "remains inside benchmark timing."
            ),
            (
                "Performance evidence is local to the "
                "recorded execution environment."
            ),
            (
                "A REJECT candidate is never admitted "
                "to performance optimisation."
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
