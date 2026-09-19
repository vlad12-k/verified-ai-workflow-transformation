"""Build unified M4-D structured inference optimisation evidence."""

import json
from pathlib import Path

from vait.benchmark.loader import load_benchmark
from vait.benchmark.models import BenchmarkDataset
from vait.benchmark.verification import verify_benchmark_bounded
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
from vait.optimisation.structured_benchmark import (
    StructuredModelFootprint,
    run_structured_benchmark_series,
)
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.ap_distillation_candidates import (
    build_ap_distillation_transformations,
)
from vait.transformations.library.ap_keras import (
    build_keras_transformation,
)
from vait.transformations.library.ap_ml import (
    build_random_forest_transformation,
    build_xgboost_transformation,
)
from vait.transformations.library.ap_torch import (
    build_pytorch_transformation,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)
from vait.transformations.library.synthetic_ap import (
    build_synthetic_ap_transformation,
)
from vait.transformations.python_callable import (
    PythonCallableTransformation,
)

DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4-structured-optimisation-evidence-v0.1.json"
)

TRAINING_SAMPLE_COUNT = 600
TRAINING_SEED = 20260916

REPETITIONS = 3
WARMUP_ROUNDS = 3
MEASURED_ROUNDS = 20


def _runtime_for(
    transformation: PythonCallableTransformation,
) -> str:
    """Return the declared local runtime for one candidate."""
    model_family = transformation.configuration.get(
        "model_family"
    )

    if model_family == "random_forest":
        return "sklearn"

    if model_family == "xgboost":
        return "xgboost"

    if model_family in {
        "pytorch_mlp",
        "pytorch_mlp_student",
    }:
        return "pytorch-eager"

    if model_family == "keras_mlp":
        return "keras-eager"

    return "python"


def _dtype_for(
    transformation: PythonCallableTransformation,
) -> str:
    """Return the declared scalar inference dtype."""
    model_family = transformation.configuration.get(
        "model_family"
    )

    if model_family in {
        "pytorch_mlp",
        "pytorch_mlp_student",
        "keras_mlp",
    }:
        return "float32"

    return "native"


def _footprint_for(
    transformation: PythonCallableTransformation,
) -> StructuredModelFootprint | None:
    """Build available model-footprint evidence without inventing values."""
    value = transformation.configuration.get(
        "parameter_count"
    )

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    ):
        return StructuredModelFootprint(
            parameter_count=value
        )

    return None


def _build_transformations() -> tuple[
    BenchmarkDataset,
    tuple[
        PythonCallableTransformation,
        ...,
    ],
]:
    """Build the complete declared M4-D structured candidate family."""
    dataset = load_benchmark(
        DATASET_PATH
    )

    corpus = build_synthetic_ap_training_corpus(
        sample_count=TRAINING_SAMPLE_COUNT,
        seed=TRAINING_SEED,
        excluded_inputs=(
            case.input_data
            for case in dataset.cases
        ),
    )

    hard_student, distilled_student = (
        build_ap_distillation_transformations(
            corpus
        )
    )

    transformations: tuple[
        PythonCallableTransformation,
        ...,
    ] = (
        build_synthetic_ap_transformation(),
        build_random_forest_transformation(
            corpus
        ),
        build_xgboost_transformation(
            corpus
        ),
        build_pytorch_transformation(
            corpus
        ),
        hard_student,
        distilled_student,
        build_keras_transformation(
            corpus
        ),
    )

    return dataset, transformations


def main() -> None:
    """Verify first, then benchmark only admissible structured candidates."""
    dataset, transformations = (
        _build_transformations()
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

    compatibility_context = (
        InferenceCompatibilityContext(
            available_providers=frozenset(
                {"local"}
            ),
            available_runtimes=frozenset(
                {
                    "python",
                    "sklearn",
                    "xgboost",
                    "pytorch-eager",
                    "keras-eager",
                }
            ),
            available_devices=frozenset(
                {"cpu"}
            ),
            available_dtypes=frozenset(
                {
                    "native",
                    "float32",
                }
            ),
            max_batch_size=32,
            available_model_ids=frozenset(
                transformation.implementation_id
                for transformation in transformations
            ),
            available_capabilities=frozenset(
                capability
                for transformation in transformations
                for capability
                in transformation.descriptor.required_capabilities
            ),
        )
    )

    results: list[dict[str, object]] = []

    for transformation in transformations:
        applicability_context = (
            ApplicabilityContext(
                risk_level=RiskLevel.CRITICAL,
                available_capabilities=(
                    transformation.descriptor.required_capabilities
                ),
            )
        )

        preparation = (
            transformation.prepare_candidate(
                applicability_context
            )
        )

        if preparation.runner is None:
            raise RuntimeError(
                "Structured candidate could not be prepared: "
                f"{transformation.implementation_id}"
            )

        configuration = InferenceConfiguration(
            provider="local",
            runtime=_runtime_for(
                transformation
            ),
            device="cpu",
            dtype=_dtype_for(
                transformation
            ),
            batch_size=1,
            model_id=(
                transformation.implementation_id
            ),
            model_revision=(
                transformation.descriptor.version
            ),
            metadata=dict(
                transformation.configuration
            ),
        )

        candidate_id = (
            transformation.descriptor.transformation_id
        )

        search_point = InferenceSearchPoint(
            search_point_id=(
                f"{candidate_id}::scalar-b1"
            ),
            search_space_id=(
                "m4d-ap-structured-v01"
            ),
            task_family=(
                "structured-decision"
            ),
            candidate_id=candidate_id,
            implementation_id=(
                transformation.implementation_id
            ),
            configuration_id=(
                "scalar-b1"
            ),
            configuration=configuration,
            requires_verification=True,
            required_capabilities=(
                transformation.descriptor.required_capabilities
            ),
            candidate_metadata=dict(
                transformation.configuration
            ),
        )

        verification = (
            verify_benchmark_bounded(
                dataset=dataset,
                candidate=preparation.runner,
                policy=verification_policy,
                candidate_configuration={
                    "transformation_id": (
                        transformation.descriptor.transformation_id
                    ),
                    "transformation_version": (
                        transformation.descriptor.version
                    ),
                    **transformation.configuration,
                },
            )
        )

        compatibility = (
            evaluate_search_point_compatibility(
                search_point,
                compatibility_context,
            )
        )

        verification_evidence = (
            SearchPointVerificationEvidence(
                search_point_id=(
                    search_point.search_point_id
                ),
                implementation_id=(
                    search_point.implementation_id
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
                    verification.statistical_evidence
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
                compatibility=compatibility,
                verification_evidence=(
                    verification_evidence
                ),
            )
        )

        benchmark = None

        if admissibility.is_admissible:
            benchmark = (
                run_structured_benchmark_series(
                    admissibility=admissibility,
                    runner=preparation.runner,
                    cases=cases,
                    repetitions=REPETITIONS,
                    warmup_rounds=WARMUP_ROUNDS,
                    measured_rounds=(
                        MEASURED_ROUNDS
                    ),
                    footprint=_footprint_for(
                        transformation
                    ),
                    evidence_metadata={
                        "m4_slice": "M4-D",
                        "candidate_family": (
                            transformation.configuration.get(
                                "model_family",
                                "deterministic-rule",
                            )
                        ),
                        "verification_decision": (
                            verification.decision.value
                        ),
                    },
                )
            )

        results.append(
            {
                "candidate_id": candidate_id,
                "implementation_id": (
                    transformation.implementation_id
                ),
                "configuration": (
                    configuration.model_dump(
                        mode="json"
                    )
                ),
                "verification": (
                    verification.model_dump(
                        mode="json"
                    )
                ),
                "admissibility": (
                    admissibility.model_dump(
                        mode="json"
                    )
                ),
                "benchmark": (
                    benchmark.model_dump(
                        mode="json"
                    )
                    if benchmark is not None
                    else None
                ),
            }
        )

        benchmark_status = (
            "benchmarked"
            if benchmark is not None
            else "excluded"
        )

        print(
            transformation.implementation_id,
            verification.decision.value,
            benchmark_status,
        )

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": "0.1",
        "milestone": "M4-D",
        "benchmark_id": (
            dataset.benchmark_id
        ),
        "benchmark_version": (
            dataset.version
        ),
        "training_sample_count": (
            TRAINING_SAMPLE_COUNT
        ),
        "training_seed": TRAINING_SEED,
        "repetitions": REPETITIONS,
        "warmup_rounds": WARMUP_ROUNDS,
        "measured_rounds": (
            MEASURED_ROUNDS
        ),
        "results": results,
        "limitations": [
            (
                "Development benchmark evidence only; "
                "AP v0.2 is not an independent blind benchmark."
            ),
            (
                "Only verification-admissible candidates "
                "enter performance benchmarking."
            ),
            (
                "Scalar batch_size=1 evidence does not yet "
                "establish native batch-scaling performance."
            ),
        ],
    }

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
