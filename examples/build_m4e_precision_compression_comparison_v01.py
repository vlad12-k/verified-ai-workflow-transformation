"""Build consolidated M4-E precision, compression, and compilation evidence."""

import json
from pathlib import Path
from typing import Any

INT8_ARTIFACT = Path(
    "artifacts/benchmarks/"
    "m4e-ap-torch-int8-optimisation-v0.1.json"
)

COMPILE_ARTIFACT = Path(
    "artifacts/benchmarks/"
    "m4e-ap-torch-compile-optimisation-v0.1.json"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "m4e-precision-compression-comparison-v0.1.json"
)

FP32_ID = (
    "synthetic-ap-pytorch-mlp-fp32-v1"
)

INT8_ID = (
    "synthetic-ap-pytorch-mlp-int8-dynamic-v1"
)

INDUCTOR_ID = (
    "synthetic-ap-pytorch-mlp-inductor-v1"
)

BATCH_SIZES = (
    1,
    4,
    8,
    16,
    32,
)


def _load(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON evidence artifact."""
    if not path.exists():
        raise RuntimeError(
            f"Required artifact does not exist: {path}"
        )

    raw = json.loads(
        path.read_text()
    )

    if not isinstance(
        raw,
        dict,
    ):
        raise RuntimeError(
            f"Artifact must contain a JSON object: {path}"
        )

    return raw


def _repeatability_summary(
    benchmark: Any,
) -> dict[str, float] | None:
    """Extract comparable steady-state benchmark evidence."""
    if benchmark is None:
        return None

    if not isinstance(
        benchmark,
        dict,
    ):
        raise RuntimeError(
            "Benchmark evidence must be an object."
        )

    repeatability = benchmark.get(
        "repeatability"
    )

    if not isinstance(
        repeatability,
        dict,
    ):
        raise RuntimeError(
            "Benchmark repeatability evidence is missing."
        )

    return {
        "mean_latency_ms": float(
            repeatability[
                "weighted_mean_latency_ms"
            ]
        ),
        "mean_cases_per_second": float(
            repeatability[
                "mean_cases_per_second"
            ]
        ),
        "worst_p95_latency_ms": float(
            repeatability[
                "worst_p95_latency_ms"
            ]
        ),
        "worst_p99_latency_ms": float(
            repeatability[
                "worst_p99_latency_ms"
            ]
        ),
        "worst_max_latency_ms": float(
            repeatability[
                "worst_max_latency_ms"
            ]
        ),
    }


def _candidate_summary(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Normalise one optimisation candidate into comparison evidence."""
    implementation_id = result.get(
        "implementation_id"
    )

    if not isinstance(
        implementation_id,
        str,
    ):
        raise RuntimeError(
            "Candidate implementation_id is missing."
        )

    verification = result.get(
        "verification"
    )

    if not isinstance(
        verification,
        dict,
    ):
        raise RuntimeError(
            f"Verification evidence missing for {implementation_id}."
        )

    decision = verification.get(
        "decision"
    )

    if not isinstance(
        decision,
        str,
    ):
        raise RuntimeError(
            f"Verification decision missing for {implementation_id}."
        )

    batch_results = result.get(
        "batch_results"
    )

    if not isinstance(
        batch_results,
        list,
    ):
        raise RuntimeError(
            f"Batch results missing for {implementation_id}."
        )

    batches: list[
        dict[str, Any]
    ] = []

    observed_batch_sizes: list[
        int
    ] = []

    for item in batch_results:
        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                "Batch result must be an object."
            )

        batch_size = item.get(
            "batch_size"
        )

        if (
            not isinstance(
                batch_size,
                int,
            )
            or isinstance(
                batch_size,
                bool,
            )
        ):
            raise RuntimeError(
                "Invalid batch_size evidence."
            )

        observed_batch_sizes.append(
            batch_size
        )

        equivalence = item.get(
            "batch_equivalence"
        )

        if not isinstance(
            equivalence,
            dict,
        ):
            raise RuntimeError(
                f"Batch equivalence missing for "
                f"{implementation_id} batch={batch_size}."
            )

        if (
            equivalence.get(
                "passed"
            )
            is not True
            or equivalence.get(
                "disagreements"
            )
            != 0
        ):
            raise RuntimeError(
                f"Batch equivalence failed for "
                f"{implementation_id} batch={batch_size}."
            )

        setup_ms = None

        setup = item.get(
            "setup"
        )

        if isinstance(
            setup,
            dict,
        ):
            raw_duration = setup.get(
                "duration_ms"
            )

            if isinstance(
                raw_duration,
                (int, float),
            ) and not isinstance(
                raw_duration,
                bool,
            ):
                setup_ms = float(
                    raw_duration
                )

        batches.append(
            {
                "batch_size": (
                    batch_size
                ),
                "batch_equivalence_passed": True,
                "batch_equivalence_disagreements": 0,
                "compile_setup_ms": (
                    setup_ms
                ),
                "steady_state": (
                    _repeatability_summary(
                        item.get(
                            "benchmark"
                        )
                    )
                ),
            }
        )

    if tuple(
        observed_batch_sizes
    ) != BATCH_SIZES:
        raise RuntimeError(
            f"Unexpected batch-size evidence for "
            f"{implementation_id}: "
            f"{observed_batch_sizes}"
        )

    configuration = result.get(
        "configuration"
    )

    if not isinstance(
        configuration,
        dict,
    ):
        raise RuntimeError(
            f"Configuration evidence missing for {implementation_id}."
        )

    return {
        "implementation_id": (
            implementation_id
        ),
        "verification_decision": (
            decision
        ),
        "parameter_count": (
            configuration.get(
                "parameter_count"
            )
        ),
        "serialized_state_bytes": (
            configuration.get(
                "serialized_state_bytes"
            )
        ),
        "optimisation": (
            configuration.get(
                "optimisation"
            )
        ),
        "scalar": (
            _repeatability_summary(
                result.get(
                    "scalar_benchmark"
                )
            )
        ),
        "native_batch": batches,
    }


def _batch_map(
    candidate: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Index candidate batch evidence by declared batch size."""
    raw = candidate[
        "native_batch"
    ]

    if not isinstance(
        raw,
        list,
    ):
        raise RuntimeError(
            "Candidate native_batch evidence is invalid."
        )

    return {
        int(
            item[
                "batch_size"
            ]
        ): item
        for item in raw
    }


def _performance_comparison(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare candidate steady-state evidence with FP32 eager."""
    baseline_scalar = baseline.get(
        "scalar"
    )

    candidate_scalar = candidate.get(
        "scalar"
    )

    if not isinstance(
        baseline_scalar,
        dict,
    ) or not isinstance(
        candidate_scalar,
        dict,
    ):
        raise RuntimeError(
            "Scalar evidence required for comparison."
        )

    baseline_batches = _batch_map(
        baseline
    )

    candidate_batches = _batch_map(
        candidate
    )

    batch_comparisons: list[
        dict[str, float | int]
    ] = []

    for batch_size in BATCH_SIZES:
        baseline_point = baseline_batches[
            batch_size
        ][
            "steady_state"
        ]

        candidate_point = candidate_batches[
            batch_size
        ][
            "steady_state"
        ]

        if not isinstance(
            baseline_point,
            dict,
        ) or not isinstance(
            candidate_point,
            dict,
        ):
            raise RuntimeError(
                "Steady-state batch evidence required."
            )

        baseline_throughput = float(
            baseline_point[
                "mean_cases_per_second"
            ]
        )

        candidate_throughput = float(
            candidate_point[
                "mean_cases_per_second"
            ]
        )

        baseline_latency = float(
            baseline_point[
                "mean_latency_ms"
            ]
        )

        candidate_latency = float(
            candidate_point[
                "mean_latency_ms"
            ]
        )

        batch_comparisons.append(
            {
                "batch_size": batch_size,
                "throughput_ratio_vs_fp32": (
                    candidate_throughput
                    / baseline_throughput
                ),
                "latency_ratio_vs_fp32": (
                    candidate_latency
                    / baseline_latency
                ),
            }
        )

    baseline_scalar_throughput = float(
        baseline_scalar[
            "mean_cases_per_second"
        ]
    )

    candidate_scalar_throughput = float(
        candidate_scalar[
            "mean_cases_per_second"
        ]
    )

    baseline_scalar_latency = float(
        baseline_scalar[
            "mean_latency_ms"
        ]
    )

    candidate_scalar_latency = float(
        candidate_scalar[
            "mean_latency_ms"
        ]
    )

    return {
        "candidate_implementation_id": (
            candidate[
                "implementation_id"
            ]
        ),
        "scalar_throughput_ratio_vs_fp32": (
            candidate_scalar_throughput
            / baseline_scalar_throughput
        ),
        "scalar_latency_ratio_vs_fp32": (
            candidate_scalar_latency
            / baseline_scalar_latency
        ),
        "batch_comparisons": (
            batch_comparisons
        ),
    }


def main() -> None:
    """Build consolidated M4-E comparison evidence."""
    int8_artifact = _load(
        INT8_ARTIFACT
    )

    compile_artifact = _load(
        COMPILE_ARTIFACT
    )

    raw_results = int8_artifact.get(
        "results"
    )

    if not isinstance(
        raw_results,
        list,
    ):
        raise RuntimeError(
            "INT8 artifact results are missing."
        )

    indexed: dict[
        str,
        dict[str, Any],
    ] = {}

    for result in raw_results:
        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                "INT8 experiment result must be an object."
            )

        implementation_id = result.get(
            "implementation_id"
        )

        if not isinstance(
            implementation_id,
            str,
        ):
            raise RuntimeError(
                "INT8 experiment implementation_id missing."
            )

        indexed[
            implementation_id
        ] = result

    if set(
        indexed
    ) != {
        FP32_ID,
        INT8_ID,
    }:
        raise RuntimeError(
            "Unexpected FP32/INT8 candidate set."
        )

    compile_result = (
        compile_artifact.get(
            "result"
        )
    )

    if not isinstance(
        compile_result,
        dict,
    ):
        raise RuntimeError(
            "Compile experiment result missing."
        )

    if (
        compile_result.get(
            "implementation_id"
        )
        != INDUCTOR_ID
    ):
        raise RuntimeError(
            "Unexpected Inductor implementation identity."
        )

    fp32 = _candidate_summary(
        indexed[
            FP32_ID
        ]
    )

    int8 = _candidate_summary(
        indexed[
            INT8_ID
        ]
    )

    inductor = _candidate_summary(
        compile_result
    )

    for candidate in (
        fp32,
        int8,
        inductor,
    ):
        if (
            candidate[
                "verification_decision"
            ]
            != "BOUNDED"
        ):
            raise RuntimeError(
                "Expected current M4-E candidates "
                "to remain BOUNDED."
            )

    compression = (
        int8_artifact.get(
            "compression"
        )
    )

    if not isinstance(
        compression,
        dict,
    ):
        raise RuntimeError(
            "Compression evidence is missing."
        )

    comparisons = {
        INT8_ID: _performance_comparison(
            baseline=fp32,
            candidate=int8,
        ),
        INDUCTOR_ID: _performance_comparison(
            baseline=fp32,
            candidate=inductor,
        ),
    }

    payload: dict[
        str,
        Any,
    ] = {
        "schema_version": "0.1",
        "milestone": "M4-E",
        "comparison": (
            "structured-pytorch-precision-"
            "compression-compilation"
        ),
        "verify_first_optimise_second": True,
        "baseline_implementation_id": (
            FP32_ID
        ),
        "candidates": [
            fp32,
            int8,
            inductor,
        ],
        "compression": (
            compression
        ),
        "performance_comparisons": (
            comparisons
        ),
        "observations": [
            (
                "All three measured candidates remained "
                "BOUNDED under the same AP verification policy."
            ),
            (
                "TorchAO INT8 preserved admissibility but "
                "did not improve steady-state performance "
                "for this small CPU model."
            ),
            (
                "TorchInductor preserved admissibility but "
                "did not outperform eager FP32 for the "
                "measured scalar or declared batch sizes."
            ),
            (
                "Compilation setup cost is separate from "
                "steady-state inference evidence."
            ),
            (
                "Performance results are environment-specific "
                "and do not imply the same outcome for larger "
                "models or different hardware."
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "M4-E precision/compression comparison"
    )
    print()

    for candidate in (
        fp32,
        int8,
        inductor,
    ):
        print(
            candidate[
                "implementation_id"
            ],
            candidate[
                "verification_decision"
            ],
        )

    print()
    print(
        "INT8 size reduction:",
        f"{float(compression['size_reduction_percent']):.2f}%",
    )

    print()
    print(
        "Steady-state ratios versus eager FP32"
    )

    for implementation_id in (
        INT8_ID,
        INDUCTOR_ID,
    ):
        comparison = comparisons[
            implementation_id
        ]

        print()
        print(
            implementation_id
        )

        print(
            "  scalar throughput ratio:",
            f"{comparison['scalar_throughput_ratio_vs_fp32']:.4f}",
        )

        print(
            "  scalar latency ratio:",
            f"{comparison['scalar_latency_ratio_vs_fp32']:.4f}",
        )

        for point in comparison[
            "batch_comparisons"
        ]:
            print(
                "  batch",
                point[
                    "batch_size"
                ],
                "throughput ratio",
                f"{point['throughput_ratio_vs_fp32']:.4f}",
                "latency ratio",
                f"{point['latency_ratio_vs_fp32']:.4f}",
            )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
