"""Build final consolidated M4-E optimisation evidence."""

import json
from pathlib import Path
from typing import Any

BASE_COMPARISON_PATH = Path(
    "artifacts/benchmarks/"
    "m4e-precision-compression-comparison-v0.1.json"
)

LOWER_PRECISION_PATH = Path(
    "artifacts/benchmarks/"
    "m4e-ap-torch-lower-precision-v0.1.json"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "m4e-final-precision-compression-compilation-v0.1.json"
)

FP32_ID = "synthetic-ap-pytorch-mlp-fp32-v1"
FP16_ID = "synthetic-ap-pytorch-mlp-fp16-v1"
BF16_ID = "synthetic-ap-pytorch-mlp-bf16-v1"
INT8_ID = "synthetic-ap-pytorch-mlp-int8-dynamic-v1"
INDUCTOR_ID = "synthetic-ap-pytorch-mlp-inductor-v1"

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
    """Load one required JSON artifact."""
    if not path.exists():
        raise RuntimeError(
            f"Required artifact does not exist: {path}"
        )

    payload = json.loads(
        path.read_text()
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"Artifact must contain an object: {path}"
        )

    return payload


def _mean_performance(
    benchmark: Any,
) -> dict[str, float]:
    """Extract comparable steady-state performance."""
    if not isinstance(
        benchmark,
        dict,
    ):
        raise RuntimeError(
            "Benchmark evidence is missing."
        )

    repeatability = benchmark.get(
        "repeatability"
    )

    if not isinstance(
        repeatability,
        dict,
    ):
        raise RuntimeError(
            "Repeatability evidence is missing."
        )

    return {
        "mean_cases_per_second": float(
            repeatability[
                "mean_cases_per_second"
            ]
        ),
        "mean_latency_ms": float(
            repeatability[
                "weighted_mean_latency_ms"
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
    }


def _normalise_lower_precision(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Normalise one FP16/BF16 result."""
    implementation_id = result.get(
        "implementation_id"
    )

    if not isinstance(
        implementation_id,
        str,
    ):
        raise RuntimeError(
            "Lower-precision implementation ID missing."
        )

    verification = result.get(
        "verification"
    )

    if not isinstance(
        verification,
        dict,
    ):
        raise RuntimeError(
            f"Verification missing for {implementation_id}."
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

    configuration = result.get(
        "configuration"
    )

    if not isinstance(
        configuration,
        dict,
    ):
        raise RuntimeError(
            f"Configuration missing for {implementation_id}."
        )

    baseline_bytes = configuration.get(
        "baseline_serialized_state_bytes"
    )

    candidate_bytes = configuration.get(
        "serialized_state_bytes"
    )

    if (
        not isinstance(
            baseline_bytes,
            int,
        )
        or isinstance(
            baseline_bytes,
            bool,
        )
        or baseline_bytes <= 0
    ):
        raise RuntimeError(
            f"Invalid baseline size for {implementation_id}."
        )

    if (
        not isinstance(
            candidate_bytes,
            int,
        )
        or isinstance(
            candidate_bytes,
            bool,
        )
        or candidate_bytes <= 0
    ):
        raise RuntimeError(
            f"Invalid candidate size for {implementation_id}."
        )

    raw_batches = result.get(
        "batch_results"
    )

    if not isinstance(
        raw_batches,
        list,
    ):
        raise RuntimeError(
            f"Batch evidence missing for {implementation_id}."
        )

    batches: list[
        dict[str, Any]
    ] = []

    observed_sizes: list[
        int
    ] = []

    for item in raw_batches:
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
                "Invalid batch size."
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

        observed_sizes.append(
            batch_size
        )

        batches.append(
            {
                "batch_size": batch_size,
                "batch_equivalence_passed": True,
                "batch_equivalence_disagreements": 0,
                "steady_state": (
                    _mean_performance(
                        item.get(
                            "benchmark"
                        )
                    )
                ),
            }
        )

    if tuple(
        observed_sizes
    ) != BATCH_SIZES:
        raise RuntimeError(
            f"Unexpected batch sizes for "
            f"{implementation_id}: {observed_sizes}"
        )

    return {
        "implementation_id": (
            implementation_id
        ),
        "verification_decision": (
            decision
        ),
        "precision": (
            configuration.get(
                "precision"
            )
        ),
        "optimisation": (
            configuration.get(
                "optimisation"
            )
        ),
        "parameter_count": (
            configuration.get(
                "parameter_count"
            )
        ),
        "serialized_state_bytes": (
            candidate_bytes
        ),
        "baseline_serialized_state_bytes": (
            baseline_bytes
        ),
        "serialized_state_size_ratio": (
            candidate_bytes
            / baseline_bytes
        ),
        "size_reduction_percent": (
            (
                1.0
                - (
                    candidate_bytes
                    / baseline_bytes
                )
            )
            * 100.0
        ),
        "scalar": (
            _mean_performance(
                result.get(
                    "scalar_benchmark"
                )
            )
        ),
        "native_batch": batches,
    }


def _batch_index(
    candidate: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Index normalised batch evidence."""
    batches = candidate.get(
        "native_batch"
    )

    if not isinstance(
        batches,
        list,
    ):
        raise RuntimeError(
            "Native batch evidence missing."
        )

    return {
        int(
            item[
                "batch_size"
            ]
        ): item
        for item in batches
    }


def _comparison(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare one candidate with eager FP32."""
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
            "Scalar comparison evidence missing."
        )

    baseline_batches = _batch_index(
        baseline
    )

    candidate_batches = _batch_index(
        candidate
    )

    batch_comparisons: list[
        dict[str, Any]
    ] = []

    for batch_size in BATCH_SIZES:
        baseline_steady = (
            baseline_batches[
                batch_size
            ][
                "steady_state"
            ]
        )

        candidate_steady = (
            candidate_batches[
                batch_size
            ][
                "steady_state"
            ]
        )

        if not isinstance(
            baseline_steady,
            dict,
        ) or not isinstance(
            candidate_steady,
            dict,
        ):
            raise RuntimeError(
                "Batch performance evidence missing."
            )

        baseline_throughput = float(
            baseline_steady[
                "mean_cases_per_second"
            ]
        )

        candidate_throughput = float(
            candidate_steady[
                "mean_cases_per_second"
            ]
        )

        baseline_latency = float(
            baseline_steady[
                "mean_latency_ms"
            ]
        )

        candidate_latency = float(
            candidate_steady[
                "mean_latency_ms"
            ]
        )

        throughput_ratio = (
            candidate_throughput
            / baseline_throughput
        )

        latency_ratio = (
            candidate_latency
            / baseline_latency
        )

        batch_comparisons.append(
            {
                "batch_size": (
                    batch_size
                ),
                "throughput_ratio_vs_fp32": (
                    throughput_ratio
                ),
                "throughput_change_percent": (
                    (
                        throughput_ratio
                        - 1.0
                    )
                    * 100.0
                ),
                "latency_ratio_vs_fp32": (
                    latency_ratio
                ),
                "latency_change_percent": (
                    (
                        latency_ratio
                        - 1.0
                    )
                    * 100.0
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

    scalar_throughput_ratio = (
        candidate_scalar_throughput
        / baseline_scalar_throughput
    )

    scalar_latency_ratio = (
        candidate_scalar_latency
        / baseline_scalar_latency
    )

    return {
        "candidate_implementation_id": (
            candidate[
                "implementation_id"
            ]
        ),
        "scalar_throughput_ratio_vs_fp32": (
            scalar_throughput_ratio
        ),
        "scalar_throughput_change_percent": (
            (
                scalar_throughput_ratio
                - 1.0
            )
            * 100.0
        ),
        "scalar_latency_ratio_vs_fp32": (
            scalar_latency_ratio
        ),
        "scalar_latency_change_percent": (
            (
                scalar_latency_ratio
                - 1.0
            )
            * 100.0
        ),
        "batch_comparisons": (
            batch_comparisons
        ),
    }


def main() -> None:
    """Build final M4-E evidence across all measured candidates."""
    base = _load(
        BASE_COMPARISON_PATH
    )

    lower = _load(
        LOWER_PRECISION_PATH
    )

    raw_base_candidates = base.get(
        "candidates"
    )

    if not isinstance(
        raw_base_candidates,
        list,
    ):
        raise RuntimeError(
            "Base comparison candidates missing."
        )

    base_candidates: dict[
        str,
        dict[str, Any],
    ] = {}

    for candidate in raw_base_candidates:
        if not isinstance(
            candidate,
            dict,
        ):
            raise RuntimeError(
                "Base candidate must be an object."
            )

        implementation_id = candidate.get(
            "implementation_id"
        )

        if not isinstance(
            implementation_id,
            str,
        ):
            raise RuntimeError(
                "Base candidate implementation ID missing."
            )

        base_candidates[
            implementation_id
        ] = candidate

    expected_base = {
        FP32_ID,
        INT8_ID,
        INDUCTOR_ID,
    }

    if set(
        base_candidates
    ) != expected_base:
        raise RuntimeError(
            "Unexpected base candidate set."
        )

    raw_lower_results = lower.get(
        "results"
    )

    if not isinstance(
        raw_lower_results,
        list,
    ):
        raise RuntimeError(
            "Lower-precision results missing."
        )

    lower_candidates: dict[
        str,
        dict[str, Any],
    ] = {}

    for result in raw_lower_results:
        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                "Lower-precision result must be an object."
            )

        candidate = (
            _normalise_lower_precision(
                result
            )
        )

        lower_candidates[
            candidate[
                "implementation_id"
            ]
        ] = candidate

    if set(
        lower_candidates
    ) != {
        FP16_ID,
        BF16_ID,
    }:
        raise RuntimeError(
            "Unexpected lower-precision candidate set."
        )

    candidates = {
        **base_candidates,
        **lower_candidates,
    }

    expected_all = {
        FP32_ID,
        FP16_ID,
        BF16_ID,
        INT8_ID,
        INDUCTOR_ID,
    }

    if set(
        candidates
    ) != expected_all:
        raise RuntimeError(
            "Final M4-E candidate set is incomplete."
        )

    for candidate in candidates.values():
        if (
            candidate.get(
                "verification_decision"
            )
            != "BOUNDED"
        ):
            raise RuntimeError(
                "Final comparison contains a "
                "non-BOUNDED candidate."
            )

    baseline = candidates[
        FP32_ID
    ]

    comparisons = {
        implementation_id: (
            _comparison(
                baseline=baseline,
                candidate=candidates[
                    implementation_id
                ],
            )
        )
        for implementation_id in (
            FP16_ID,
            BF16_ID,
            INT8_ID,
            INDUCTOR_ID,
        )
    }

    payload: dict[
        str,
        Any,
    ] = {
        "schema_version": "0.1",
        "milestone": "M4-E",
        "scope": (
            "precision-compression-compilation"
        ),
        "verify_first_optimise_second": True,
        "baseline_implementation_id": (
            FP32_ID
        ),
        "candidate_order": [
            FP32_ID,
            FP16_ID,
            BF16_ID,
            INT8_ID,
            INDUCTOR_ID,
        ],
        "candidates": [
            candidates[
                implementation_id
            ]
            for implementation_id in (
                FP32_ID,
                FP16_ID,
                BF16_ID,
                INT8_ID,
                INDUCTOR_ID,
            )
        ],
        "performance_comparisons": (
            comparisons
        ),
        "evidence_observations": [
            (
                "All measured M4-E candidates remained "
                "BOUNDED under benchmark verification."
            ),
            (
                "FP16 and BF16 reduced serialized model "
                "state size relative to eager FP32."
            ),
            (
                "FP16 and BF16 traded weaker scalar and "
                "small-batch performance for stronger "
                "throughput at larger measured batches."
            ),
            (
                "TorchAO dynamic INT8 preserved bounded "
                "behaviour but did not improve measured "
                "steady-state performance for this model."
            ),
            (
                "TorchInductor preserved bounded behaviour "
                "but did not outperform eager FP32 across "
                "the declared measured batch range."
            ),
            (
                "Compilation setup cost is tracked separately "
                "from steady-state inference performance."
            ),
            (
                "Performance conclusions are scoped to the "
                "measured model, workload, runtime, and "
                "execution environment."
            ),
        ],
        "source_artifacts": [
            str(
                BASE_COMPARISON_PATH
            ),
            str(
                LOWER_PRECISION_PATH
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
        "M4-E final comparison"
    )
    print()

    for implementation_id in (
        FP32_ID,
        FP16_ID,
        BF16_ID,
        INT8_ID,
        INDUCTOR_ID,
    ):
        candidate = candidates[
            implementation_id
        ]

        print(
            implementation_id,
            candidate[
                "verification_decision"
            ],
        )

    print()
    print(
        "Performance versus eager FP32"
    )

    for implementation_id in (
        FP16_ID,
        BF16_ID,
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
            "  scalar throughput change:",
            f"{comparison['scalar_throughput_change_percent']:+.2f}%",
        )

        for point in comparison[
            "batch_comparisons"
        ]:
            print(
                "  batch",
                point[
                    "batch_size"
                ],
                "throughput change",
                f"{point['throughput_change_percent']:+.2f}%",
            )

    print()
    print(
        "Lower-precision footprint"
    )

    for implementation_id in (
        FP16_ID,
        BF16_ID,
    ):
        candidate = candidates[
            implementation_id
        ]

        print(
            " ",
            implementation_id,
            "size reduction",
            f"{candidate['size_reduction_percent']:.2f}%",
        )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
