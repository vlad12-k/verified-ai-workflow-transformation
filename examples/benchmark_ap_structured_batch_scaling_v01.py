"""Build M4-D native batch-scaling evidence for PyTorch AP candidates."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

from pydantic import JsonValue

from vait.benchmark.loader import load_benchmark
from vait.benchmark.verification import verify_benchmark_bounded
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
    VerificationCase,
)
from vait.decision.models import Decision
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
)
from vait.runners.base import ImplementationRunner
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.ap_distillation_candidates import (
    build_ap_distillation_native_batch_candidates,
)
from vait.transformations.library.ap_ml import (
    build_random_forest_native_batch_candidate,
    build_xgboost_native_batch_candidate,
)
from vait.transformations.library.ap_torch import (
    build_pytorch_native_batch_candidate,
)
from vait.transformations.library.ap_training import (
    APTrainingCorpus,
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
    "m4-structured-batch-scaling-v0.1.json"
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

CandidateBatchOperation = Callable[
    [Sequence[dict[str, JsonValue]]],
    tuple[JsonValue, ...],
]


class BatchCandidate(NamedTuple):
    """One trained candidate plus its genuine vectorised inference path."""

    transformation: PythonCallableTransformation
    operation: CandidateBatchOperation


def _build_candidates(
    corpus: APTrainingCorpus,
) -> tuple[
    BatchCandidate,
    ...,
]:
    """Build teacher and both compact student candidates once."""
    teacher_transformation, teacher_operation = (
        build_pytorch_native_batch_candidate(
            corpus
        )
    )

    (
        hard_pair,
        distilled_pair,
    ) = (
        build_ap_distillation_native_batch_candidates(
            corpus
        )
    )

    hard_transformation, hard_operation = (
        hard_pair
    )

    (
        distilled_transformation,
        distilled_operation,
    ) = distilled_pair

    (
        random_forest_transformation,
        random_forest_operation,
    ) = build_random_forest_native_batch_candidate(
        corpus
    )

    (
        xgboost_transformation,
        xgboost_operation,
    ) = build_xgboost_native_batch_candidate(
        corpus
    )

    return (
        BatchCandidate(
            transformation=random_forest_transformation,
            operation=random_forest_operation,
        ),
        BatchCandidate(
            transformation=xgboost_transformation,
            operation=xgboost_operation,
        ),
        BatchCandidate(
            transformation=teacher_transformation,
            operation=teacher_operation,
        ),
        BatchCandidate(
            transformation=hard_transformation,
            operation=hard_operation,
        ),
        BatchCandidate(
            transformation=distilled_transformation,
            operation=distilled_operation,
        ),
    )


def _runtime_for(
    transformation: PythonCallableTransformation,
) -> str:
    """Return the declared runtime for one structured model family."""
    model_family = transformation.configuration.get(
        "model_family"
    )

    if model_family == "random_forest":
        return "sklearn"

    if model_family == "xgboost":
        return "xgboost"

    return "pytorch-eager"


def _dtype_for(
    transformation: PythonCallableTransformation,
) -> str:
    """Return the inference dtype used by one structured model family."""
    model_family = transformation.configuration.get(
        "model_family"
    )

    if model_family in {
        "random_forest",
        "xgboost",
    }:
        return "native"

    return "float32"


def _adapt_operation(
    operation: CandidateBatchOperation,
) -> NativeBatchOperation:
    """Adapt raw AP dictionaries to the generic batch harness."""

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
    operation: CandidateBatchOperation,
    cases: Sequence[VerificationCase],
    batch_size: int,
) -> tuple[
    int,
    int,
]:
    """Compare native-batch outputs with the verified scalar execution path."""
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
                "Native batch path returned "
                "an invalid output count."
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
                    "Verified scalar path failed "
                    "during batch-equivalence checking: "
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


def _footprint(
    transformation: PythonCallableTransformation,
) -> StructuredModelFootprint | None:
    """Return parameter-count evidence only when the candidate exposes it."""
    parameter_count = (
        transformation.configuration.get(
            "parameter_count"
        )
    )

    if (
        isinstance(
            parameter_count,
            int,
        )
        and not isinstance(
            parameter_count,
            bool,
        )
        and parameter_count > 0
    ):
        return StructuredModelFootprint(
            parameter_count=parameter_count
        )

    return None


def main() -> None:
    """Verify, re-check vectorised behaviour, then measure batch scaling."""
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

    candidates = _build_candidates(
        corpus
    )

    cases = tuple(
        VerificationCase(
            id=case.case_id,
            input_data=case.input_data,
            risk_level=case.risk_level,
        )
        for case in dataset.cases
    )

    verification_policy = (
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

        scalar_runner = (
            preparation.runner
        )

        verification = (
            verify_benchmark_bounded(
                dataset=dataset,
                candidate=scalar_runner,
                policy=verification_policy,
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
            transformation.implementation_id
        )
        print(
            "  scalar verification:",
            verification.decision.value,
        )

        if (
            verification.decision
            is Decision.REJECT
        ):
            results.append(
                {
                    "implementation_id": (
                        transformation
                        .implementation_id
                    ),
                    "verification": (
                        verification.model_dump(
                            mode="json"
                        )
                    ),
                    "batch_results": [],
                    "excluded": (
                        "scalar-verification-reject"
                    ),
                }
            )
            continue

        candidate_batch_results: list[
            dict[str, object]
        ] = []

        for batch_size in BATCH_SIZES:
            (
                equivalence_cases,
                equivalence_disagreements,
            ) = _verify_batch_equivalence(
                scalar_runner=scalar_runner,
                operation=(
                    candidate.operation
                ),
                cases=cases,
                batch_size=batch_size,
            )

            equivalence_passed = (
                equivalence_disagreements
                == 0
            )

            search_point = (
                InferenceSearchPoint(
                    search_point_id=(
                        transformation
                        .descriptor
                        .transformation_id
                        + "::batch-"
                        + str(batch_size)
                    ),
                    search_space_id=(
                        "m4d-pytorch-batch-v01"
                    ),
                    task_family=(
                        "structured-decision"
                    ),
                    candidate_id=(
                        transformation
                        .descriptor
                        .transformation_id
                    ),
                    implementation_id=(
                        transformation
                        .implementation_id
                    ),
                    configuration_id=(
                        f"batch-{batch_size}"
                    ),
                    configuration=(
                        InferenceConfiguration(
                            provider="local",
                            runtime=(
                                _runtime_for(
                                    transformation
                                )
                            ),
                            device="cpu",
                            dtype=(
                                _dtype_for(
                                    transformation
                                )
                            ),
                            batch_size=(
                                batch_size
                            ),
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
                            },
                        )
                    ),
                    requires_verification=True,
                    required_capabilities=(
                        transformation
                        .descriptor
                        .required_capabilities
                    ),
                    candidate_metadata=dict(
                        transformation
                        .configuration
                    ),
                )
            )

            compatibility = (
                evaluate_search_point_compatibility(
                    search_point,
                    InferenceCompatibilityContext(
                        available_providers=(
                            frozenset(
                                {"local"}
                            )
                        ),
                        available_runtimes=(
                            frozenset(
                                {
                                    "sklearn",
                                    "xgboost",
                                    "pytorch-eager",
                                }
                            )
                        ),
                        available_devices=(
                            frozenset(
                                {"cpu"}
                            )
                        ),
                        available_dtypes=(
                            frozenset(
                                {
                                    "native",
                                    "float32",
                                }
                            )
                        ),
                        max_batch_size=32,
                        available_model_ids=(
                            frozenset(
                                {
                                    transformation
                                    .implementation_id
                                }
                            )
                        ),
                        available_capabilities=(
                            transformation
                            .descriptor
                            .required_capabilities
                        ),
                    ),
                )
            )

            verification_evidence = (
                SearchPointVerificationEvidence(
                    search_point_id=(
                        search_point
                        .search_point_id
                    ),
                    implementation_id=(
                        search_point
                        .implementation_id
                    ),
                    contract_id=(
                        "benchmark:"
                        f"{dataset.benchmark_id}:"
                        f"{dataset.version}:"
                        f"{transformation.implementation_id}"
                    ),
                    decision=(
                        verification.decision
                    ),
                    evidence_kind=(
                        VerificationEvidenceKind
                        .BOUNDED_STATISTICAL
                    ),
                    statistical_evidence_present=(
                        verification
                        .statistical_evidence
                        is not None
                    ),
                    failure_codes=tuple(
                        failure.code
                        for failure
                        in verification.failures
                    ),
                )
            )

            admissibility = (
                evaluate_search_point_admissibility(
                    compatibility=(
                        compatibility
                    ),
                    verification_evidence=(
                        verification_evidence
                    ),
                )
            )

            series = None

            if (
                admissibility.is_admissible
                and equivalence_passed
            ):
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
                            _adapt_operation(
                                candidate.operation
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
                        footprint=(
                            _footprint(
                                transformation
                            )
                        ),
                        evidence_metadata={
                            "m4_slice": "M4-D",
                            "verification_decision": (
                                verification
                                .decision
                                .value
                            ),
                            "batch_equivalence_scope": (
                                "ap-v0.2-development-benchmark"
                            ),
                            "batch_equivalence_cases": (
                                equivalence_cases
                            ),
                            "batch_equivalence_disagreements": (
                                equivalence_disagreements
                            ),
                        },
                    )
                )

            candidate_batch_results.append(
                {
                    "batch_size": (
                        batch_size
                    ),
                    "batch_equivalence": {
                        "cases": (
                            equivalence_cases
                        ),
                        "disagreements": (
                            equivalence_disagreements
                        ),
                        "passed": (
                            equivalence_passed
                        ),
                    },
                    "admissibility": (
                        admissibility.model_dump(
                            mode="json"
                        )
                    ),
                    "benchmark": (
                        series.model_dump(
                            mode="json"
                        )
                        if series is not None
                        else None
                    ),
                }
            )

            if series is None:
                print(
                    f"  batch {batch_size}: "
                    "EXCLUDED"
                )
                continue

            repeatability = (
                series.repeatability
            )

            print(
                f"  batch {batch_size}: "
                f"{repeatability.mean_cases_per_second:.2f} cases/s, "
                f"{repeatability.weighted_mean_latency_ms:.6f} ms "
                "mean batch latency"
            )

        results.append(
            {
                "implementation_id": (
                    transformation
                    .implementation_id
                ),
                "parameter_count": (
                    transformation
                    .configuration
                    .get(
                        "parameter_count"
                    )
                ),
                "verification": (
                    verification.model_dump(
                        mode="json"
                    )
                ),
                "batch_results": (
                    candidate_batch_results
                ),
            }
        )

    payload = {
        "schema_version": "0.1",
        "milestone": "M4-D",
        "experiment": (
            "pytorch-native-batch-scaling"
        ),
        "batch_sizes": list(
            BATCH_SIZES
        ),
        "training_sample_count": (
            TRAINING_SAMPLE_COUNT
        ),
        "training_seed": (
            TRAINING_SEED
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
        "results": results,
        "limitations": [
            (
                "Batch behavioural equivalence is checked "
                "against the verified scalar path on the "
                "AP v0.2 development benchmark only."
            ),
            (
                "These measurements describe the current "
                "local execution environment and are not "
                "universal performance claims."
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
