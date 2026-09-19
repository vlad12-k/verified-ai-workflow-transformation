"""Build the final comparative evidence artifact for M4-D."""

import json
from pathlib import Path
from typing import Any

SCALAR_PATH = Path(
    "artifacts/benchmarks/"
    "m4-structured-optimisation-evidence-v0.1.json"
)

BATCH_PATH = Path(
    "artifacts/benchmarks/"
    "m4-structured-batch-scaling-v0.1.json"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4d-structured-comparison-v0.1.json"
)

TEACHER_ID = "synthetic-ap-pytorch-mlp-v1"
HARD_STUDENT_ID = (
    "synthetic-ap-pytorch-student-hard-label-v1"
)
DISTILLED_STUDENT_ID = (
    "synthetic-ap-pytorch-student-distilled-v1"
)

EXPECTED_BATCH_SIZES = (
    1,
    4,
    8,
    16,
    32,
)


def _load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON evidence artifact."""
    if not path.exists():
        raise RuntimeError(
            f"Required evidence artifact is missing: {path}"
        )

    payload = json.loads(
        path.read_text()
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"Evidence artifact must contain an object: {path}"
        )

    return payload


def _index_results(
    payload: dict[str, Any],
) -> dict[
    str,
    dict[str, Any],
]:
    """Index artifact results by implementation identity."""
    raw_results = payload.get(
        "results"
    )

    if not isinstance(
        raw_results,
        list,
    ):
        raise RuntimeError(
            "Evidence artifact does not contain a results list."
        )

    indexed: dict[
        str,
        dict[str, Any],
    ] = {}

    for raw_result in raw_results:
        if not isinstance(
            raw_result,
            dict,
        ):
            raise RuntimeError(
                "Each evidence result must be an object."
            )

        implementation_id = (
            raw_result.get(
                "implementation_id"
            )
        )

        if not isinstance(
            implementation_id,
            str,
        ):
            raise RuntimeError(
                "Evidence result is missing implementation_id."
            )

        if implementation_id in indexed:
            raise RuntimeError(
                "Duplicate implementation_id in evidence: "
                f"{implementation_id}"
            )

        indexed[
            implementation_id
        ] = raw_result

    return indexed


def _verification_decision(
    result: dict[str, Any],
) -> str:
    """Return one recorded verification decision."""
    verification = result.get(
        "verification"
    )

    if not isinstance(
        verification,
        dict,
    ):
        raise RuntimeError(
            "Result is missing verification evidence."
        )

    decision = verification.get(
        "decision"
    )

    if not isinstance(
        decision,
        str,
    ):
        raise RuntimeError(
            "Verification evidence is missing decision."
        )

    return decision


def _scalar_summary(
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract comparable scalar inference evidence."""
    benchmark = result.get(
        "benchmark"
    )

    if benchmark is None:
        return None

    if not isinstance(
        benchmark,
        dict,
    ):
        raise RuntimeError(
            "Scalar benchmark evidence must be an object."
        )

    repeatability = benchmark.get(
        "repeatability"
    )

    if not isinstance(
        repeatability,
        dict,
    ):
        raise RuntimeError(
            "Scalar benchmark is missing repeatability evidence."
        )

    configuration = result.get(
        "configuration"
    )

    if not isinstance(
        configuration,
        dict,
    ):
        raise RuntimeError(
            "Scalar result is missing configuration evidence."
        )

    return {
        "runtime": configuration.get(
            "runtime"
        ),
        "dtype": configuration.get(
            "dtype"
        ),
        "weighted_mean_latency_ms": (
            repeatability.get(
                "weighted_mean_latency_ms"
            )
        ),
        "mean_cases_per_second": (
            repeatability.get(
                "mean_cases_per_second"
            )
        ),
        "run_mean_latency_stdev_ms": (
            repeatability.get(
                "run_mean_latency_stdev_ms"
            )
        ),
        "run_mean_latency_cv": (
            repeatability.get(
                "run_mean_latency_cv"
            )
        ),
        "throughput_stdev": (
            repeatability.get(
                "throughput_stdev"
            )
        ),
        "throughput_cv": (
            repeatability.get(
                "throughput_cv"
            )
        ),
        "worst_p95_latency_ms": (
            repeatability.get(
                "worst_p95_latency_ms"
            )
        ),
        "worst_p99_latency_ms": (
            repeatability.get(
                "worst_p99_latency_ms"
            )
        ),
        "worst_max_latency_ms": (
            repeatability.get(
                "worst_max_latency_ms"
            )
        ),
    }


def _batch_summary(
    result: dict[str, Any],
) -> list[
    dict[str, Any]
]:
    """Extract native batch-scaling evidence."""
    raw_batch_results = result.get(
        "batch_results"
    )

    if not isinstance(
        raw_batch_results,
        list,
    ):
        raise RuntimeError(
            "Batch result is missing batch_results."
        )

    summaries: list[
        dict[str, Any]
    ] = []

    for item in raw_batch_results:
        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                "Batch evidence item must be an object."
            )

        batch_size = item.get(
            "batch_size"
        )

        equivalence = item.get(
            "batch_equivalence"
        )

        benchmark = item.get(
            "benchmark"
        )

        if not isinstance(
            batch_size,
            int,
        ):
            raise RuntimeError(
                "Batch evidence is missing batch_size."
            )

        if not isinstance(
            equivalence,
            dict,
        ):
            raise RuntimeError(
                "Batch evidence is missing equivalence evidence."
            )

        if not isinstance(
            benchmark,
            dict,
        ):
            raise RuntimeError(
                "An admissible M4-D batch point "
                "is missing benchmark evidence."
            )

        repeatability = benchmark.get(
            "repeatability"
        )

        if not isinstance(
            repeatability,
            dict,
        ):
            raise RuntimeError(
                "Batch benchmark is missing repeatability evidence."
            )

        summaries.append(
            {
                "batch_size": batch_size,
                "equivalence_cases": (
                    equivalence.get(
                        "cases"
                    )
                ),
                "equivalence_disagreements": (
                    equivalence.get(
                        "disagreements"
                    )
                ),
                "equivalence_passed": (
                    equivalence.get(
                        "passed"
                    )
                ),
                "weighted_mean_batch_latency_ms": (
                    repeatability.get(
                        "weighted_mean_latency_ms"
                    )
                ),
                "mean_cases_per_second": (
                    repeatability.get(
                        "mean_cases_per_second"
                    )
                ),
                "run_mean_latency_stdev_ms": (
                    repeatability.get(
                        "run_mean_latency_stdev_ms"
                    )
                ),
                "run_mean_latency_cv": (
                    repeatability.get(
                        "run_mean_latency_cv"
                    )
                ),
                "throughput_stdev": (
                    repeatability.get(
                        "throughput_stdev"
                    )
                ),
                "throughput_cv": (
                    repeatability.get(
                        "throughput_cv"
                    )
                ),
                "worst_p95_batch_latency_ms": (
                    repeatability.get(
                        "worst_p95_latency_ms"
                    )
                ),
                "worst_p99_batch_latency_ms": (
                    repeatability.get(
                        "worst_p99_latency_ms"
                    )
                ),
                "worst_max_batch_latency_ms": (
                    repeatability.get(
                        "worst_max_latency_ms"
                    )
                ),
            }
        )

    observed_sizes = tuple(
        item["batch_size"]
        for item in summaries
    )

    if observed_sizes != EXPECTED_BATCH_SIZES:
        raise RuntimeError(
            "Unexpected batch-scaling matrix: "
            f"{observed_sizes}"
        )

    if any(
        item[
            "equivalence_passed"
        ]
        is not True
        or item[
            "equivalence_disagreements"
        ]
        != 0
        for item in summaries
    ):
        raise RuntimeError(
            "Native batch behaviour diverged from "
            "the verified scalar path."
        )

    return summaries


def _parameter_count(
    scalar_result: dict[str, Any],
    batch_result: dict[str, Any] | None,
) -> int | None:
    """Resolve parameter evidence without inventing unavailable values."""
    if batch_result is not None:
        value = batch_result.get(
            "parameter_count"
        )

        if (
            isinstance(
                value,
                int,
            )
            and not isinstance(
                value,
                bool,
            )
            and value > 0
        ):
            return value

    configuration = scalar_result.get(
        "configuration"
    )

    if isinstance(
        configuration,
        dict,
    ):
        metadata = configuration.get(
            "metadata"
        )

        if isinstance(
            metadata,
            dict,
        ):
            value = metadata.get(
                "parameter_count"
            )

            if (
                isinstance(
                    value,
                    int,
                )
                and not isinstance(
                    value,
                    bool,
                )
                and value > 0
            ):
                return value

    return None


def _parameter_reduction_percent(
    *,
    teacher_parameters: int,
    student_parameters: int,
) -> float:
    """Calculate student parameter reduction relative to the teacher."""
    if (
        teacher_parameters <= 0
        or student_parameters <= 0
    ):
        raise ValueError(
            "Parameter counts must be positive."
        )

    return (
        1.0
        - (
            student_parameters
            / teacher_parameters
        )
    ) * 100.0


def _batch_point(
    points: list[
        dict[str, Any]
    ],
    batch_size: int,
) -> dict[str, Any]:
    """Return one batch-size comparison point."""
    for point in points:
        if point.get(
            "batch_size"
        ) == batch_size:
            return point

    raise RuntimeError(
        f"Missing batch size {batch_size}."
    )


def _student_comparison(
    *,
    student_id: str,
    teacher: dict[str, Any],
    student: dict[str, Any],
) -> dict[str, Any]:
    """Compare one compact student against the PyTorch teacher."""
    teacher_parameters = teacher.get(
        "parameter_count"
    )
    student_parameters = student.get(
        "parameter_count"
    )

    if not isinstance(
        teacher_parameters,
        int,
    ):
        raise RuntimeError(
            "Teacher parameter_count evidence is missing."
        )

    if not isinstance(
        student_parameters,
        int,
    ):
        raise RuntimeError(
            "Student parameter_count evidence is missing."
        )

    teacher_batches = teacher.get(
        "native_batch"
    )
    student_batches = student.get(
        "native_batch"
    )

    if not isinstance(
        teacher_batches,
        list,
    ) or not isinstance(
        student_batches,
        list,
    ):
        raise RuntimeError(
            "Teacher/student batch evidence is missing."
        )

    batch_comparisons: list[
        dict[str, Any]
    ] = []

    for batch_size in EXPECTED_BATCH_SIZES:
        teacher_point = _batch_point(
            teacher_batches,
            batch_size,
        )
        student_point = _batch_point(
            student_batches,
            batch_size,
        )

        teacher_throughput = teacher_point.get(
            "mean_cases_per_second"
        )
        student_throughput = student_point.get(
            "mean_cases_per_second"
        )

        teacher_latency = teacher_point.get(
            "weighted_mean_batch_latency_ms"
        )
        student_latency = student_point.get(
            "weighted_mean_batch_latency_ms"
        )

        if not isinstance(
            teacher_throughput,
            (int, float),
        ) or not isinstance(
            student_throughput,
            (int, float),
        ):
            raise RuntimeError(
                "Teacher/student throughput evidence is incomplete."
            )

        if not isinstance(
            teacher_latency,
            (int, float),
        ) or not isinstance(
            student_latency,
            (int, float),
        ):
            raise RuntimeError(
                "Teacher/student latency evidence is incomplete."
            )

        batch_comparisons.append(
            {
                "batch_size": batch_size,
                "teacher_cases_per_second": (
                    teacher_throughput
                ),
                "student_cases_per_second": (
                    student_throughput
                ),
                "student_to_teacher_throughput_ratio": (
                    student_throughput
                    / teacher_throughput
                ),
                "teacher_mean_batch_latency_ms": (
                    teacher_latency
                ),
                "student_mean_batch_latency_ms": (
                    student_latency
                ),
                "student_to_teacher_latency_ratio": (
                    student_latency
                    / teacher_latency
                ),
            }
        )

    return {
        "student_implementation_id": student_id,
        "teacher_verification": teacher[
            "verification_decision"
        ],
        "student_verification": student[
            "verification_decision"
        ],
        "teacher_parameter_count": (
            teacher_parameters
        ),
        "student_parameter_count": (
            student_parameters
        ),
        "parameter_reduction_percent": (
            _parameter_reduction_percent(
                teacher_parameters=(
                    teacher_parameters
                ),
                student_parameters=(
                    student_parameters
                ),
            )
        ),
        "batch_comparisons": (
            batch_comparisons
        ),
    }


def main() -> None:
    """Build final M4-D structured comparative evidence."""
    scalar_payload = _load_json(
        SCALAR_PATH
    )
    batch_payload = _load_json(
        BATCH_PATH
    )

    scalar_results = _index_results(
        scalar_payload
    )
    batch_results = _index_results(
        batch_payload
    )

    candidate_ids = sorted(
        scalar_results
    )

    candidates: list[
        dict[str, Any]
    ] = []

    candidate_index: dict[
        str,
        dict[str, Any],
    ] = {}

    for implementation_id in candidate_ids:
        scalar_result = scalar_results[
            implementation_id
        ]

        batch_result = batch_results.get(
            implementation_id
        )

        verification_decision = (
            _verification_decision(
                scalar_result
            )
        )

        scalar = _scalar_summary(
            scalar_result
        )

        native_batch = (
            _batch_summary(
                batch_result
            )
            if batch_result is not None
            else None
        )

        parameter_count = (
            _parameter_count(
                scalar_result,
                batch_result,
            )
        )

        execution_path_overhead = None

        if (
            scalar is not None
            and native_batch is not None
        ):
            scalar_latency = scalar.get(
                "weighted_mean_latency_ms"
            )

            batch_one = _batch_point(
                native_batch,
                1,
            )

            native_latency = batch_one.get(
                "weighted_mean_batch_latency_ms"
            )

            if isinstance(
                scalar_latency,
                (int, float),
            ) and isinstance(
                native_latency,
                (int, float),
            ):
                execution_path_overhead = {
                    "scalar_runner_latency_ms": (
                        scalar_latency
                    ),
                    "native_batch1_latency_ms": (
                        native_latency
                    ),
                    "scalar_minus_native_ms": (
                        scalar_latency
                        - native_latency
                    ),
                    "scope": (
                        "Observed execution-path difference only; "
                        "not an isolated pure framework-overhead claim."
                    ),
                }

        candidate_summary = {
            "implementation_id": (
                implementation_id
            ),
            "verification_decision": (
                verification_decision
            ),
            "parameter_count": (
                parameter_count
            ),
            "scalar": scalar,
            "native_batch": (
                native_batch
            ),
            "execution_path_overhead_evidence": (
                execution_path_overhead
            ),
        }

        candidates.append(
            candidate_summary
        )

        candidate_index[
            implementation_id
        ] = candidate_summary

    for required_id in (
        TEACHER_ID,
        HARD_STUDENT_ID,
        DISTILLED_STUDENT_ID,
    ):
        if required_id not in candidate_index:
            raise RuntimeError(
                "Missing required teacher/student evidence: "
                f"{required_id}"
            )

    teacher = candidate_index[
        TEACHER_ID
    ]

    hard_student = candidate_index[
        HARD_STUDENT_ID
    ]

    distilled_student = candidate_index[
        DISTILLED_STUDENT_ID
    ]

    teacher_student_comparison: dict[str, Any] = {
        "teacher_implementation_id": (
            TEACHER_ID
        ),
        "hard_label_student": (
            _student_comparison(
                student_id=(
                    HARD_STUDENT_ID
                ),
                teacher=teacher,
                student=hard_student,
            )
        ),
        "distilled_student": (
            _student_comparison(
                student_id=(
                    DISTILLED_STUDENT_ID
                ),
                teacher=teacher,
                student=distilled_student,
            )
        ),
    }

    keras = candidate_index.get(
        "synthetic-ap-keras-mlp-v1"
    )

    if (
        keras is None
        or keras[
            "verification_decision"
        ]
        != "REJECT"
        or keras[
            "scalar"
        ]
        is not None
        or keras[
            "native_batch"
        ]
        is not None
    ):
        raise RuntimeError(
            "Keras REJECT evidence must remain excluded "
            "from performance optimisation."
        )

    payload = {
        "schema_version": "0.1",
        "milestone": "M4-D",
        "title": (
            "Structured inference optimisation comparison"
        ),
        "verify_first_optimise_second": True,
        "source_artifacts": [
            str(
                SCALAR_PATH
            ),
            str(
                BATCH_PATH
            ),
        ],
        "batch_sizes": list(
            EXPECTED_BATCH_SIZES
        ),
        "candidates": candidates,
        "teacher_student_comparison": (
            teacher_student_comparison
        ),
        "limitations": [
            (
                "AP v0.2 is development benchmark evidence "
                "and is not an independent blind evaluation."
            ),
            (
                "Performance measurements describe this local "
                "execution environment and are not universal claims."
            ),
            (
                "Scalar-runner versus native-batch-1 differences "
                "include execution-path effects and must not be "
                "interpreted as isolated framework overhead."
            ),
            (
                "Classical model parameter counts are left unavailable "
                "where the existing candidate evidence does not expose "
                "a directly comparable parameter-count definition."
            ),
            (
                "Keras remains verification REJECT and therefore "
                "does not enter performance optimisation."
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

    print("M4-D structured comparison")
    print()

    for candidate in candidates:
        print(
            candidate[
                "implementation_id"
            ],
            candidate[
                "verification_decision"
            ],
        )

    print()
    print("Teacher/student compression")

    for key in (
        "hard_label_student",
        "distilled_student",
    ):
        comparison = (
            teacher_student_comparison[
                key
            ]
        )

        print(
            comparison[
                "student_implementation_id"
            ]
        )
        print(
            "  parameters:",
            comparison[
                "student_parameter_count"
            ],
            "/",
            comparison[
                "teacher_parameter_count"
            ],
        )
        print(
            "  parameter reduction:",
            f"{comparison['parameter_reduction_percent']:.2f}%",
        )

    print()
    print(
        f"Wrote {ARTIFACT_PATH}"
    )


if __name__ == "__main__":
    main()
