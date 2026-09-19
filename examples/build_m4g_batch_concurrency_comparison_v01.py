"""Build M4-G batching versus concurrency scaling comparison evidence."""

import json
from pathlib import Path
from typing import Any

CONCURRENCY_PATH = Path(
    "artifacts/benchmarks/"
    "m4g-smollm2-concurrency-v0.1.json"
)

BATCHING_PATH = Path(
    "artifacts/benchmarks/"
    "m4g-smollm2-native-batching-v0.1.json"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "m4g-batching-concurrency-comparison-v0.1.json"
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one benchmark artifact."""
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"Expected object payload: {path}"
        )

    return payload


def require_number(
    value: object,
    *,
    label: str,
) -> float:
    """Return one required numeric value."""
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int | float,
        )
    ):
        raise RuntimeError(
            f"Expected numeric value: {label}"
        )

    return float(
        value
    )


def concurrency_rows(
    payload: dict[str, Any],
) -> dict[int, dict[str, float]]:
    """Extract comparable concurrency evidence."""
    rows: dict[
        int,
        dict[str, float],
    ] = {}

    series = payload.get(
        "series"
    )

    if not isinstance(
        series,
        list,
    ):
        raise RuntimeError(
            "Concurrency artifact has no series."
        )

    for item in series:
        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                "Invalid concurrency series item."
            )

        concurrency = item.get(
            "concurrency"
        )

        report = item.get(
            "report"
        )

        if not isinstance(
            concurrency,
            int,
        ):
            raise RuntimeError(
                "Invalid concurrency value."
            )

        if not isinstance(
            report,
            dict,
        ):
            raise RuntimeError(
                "Invalid concurrency report."
            )

        throughput = report.get(
            "throughput"
        )

        latency = report.get(
            "latency"
        )

        if not isinstance(
            throughput,
            dict,
        ):
            raise RuntimeError(
                "Missing concurrency throughput."
            )

        if not isinstance(
            latency,
            dict,
        ):
            raise RuntimeError(
                "Missing concurrency latency."
            )

        rows[
            concurrency
        ] = {
            "requests_per_second": (
                require_number(
                    throughput.get(
                        "requests_per_second"
                    ),
                    label=(
                        "concurrency requests/s"
                    ),
                )
            ),
            "tokens_per_second": (
                require_number(
                    throughput.get(
                        "tokens_per_second"
                    ),
                    label=(
                        "concurrency tokens/s"
                    ),
                )
            ),
            "mean_completion_latency_ms": (
                require_number(
                    latency.get(
                        "mean_ms"
                    ),
                    label=(
                        "concurrency mean latency"
                    ),
                )
            ),
            "p95_completion_latency_ms": (
                require_number(
                    latency.get(
                        "p95_ms"
                    ),
                    label=(
                        "concurrency p95 latency"
                    ),
                )
            ),
        }

    return rows


def batching_rows(
    payload: dict[str, Any],
) -> dict[int, dict[str, float]]:
    """Extract comparable native-batch evidence."""
    rows: dict[
        int,
        dict[str, float],
    ] = {}

    series = payload.get(
        "series"
    )

    if not isinstance(
        series,
        list,
    ):
        raise RuntimeError(
            "Batching artifact has no series."
        )

    for item in series:
        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                "Invalid batching series item."
            )

        batch_size = item.get(
            "batch_size"
        )

        report = item.get(
            "report"
        )

        if not isinstance(
            batch_size,
            int,
        ):
            raise RuntimeError(
                "Invalid batch-size value."
            )

        if not isinstance(
            report,
            dict,
        ):
            raise RuntimeError(
                "Invalid batching report."
            )

        throughput = report.get(
            "throughput"
        )

        latency = report.get(
            "latency"
        )

        if not isinstance(
            throughput,
            dict,
        ):
            raise RuntimeError(
                "Missing batching throughput."
            )

        if not isinstance(
            latency,
            dict,
        ):
            raise RuntimeError(
                "Missing batching latency."
            )

        rows[
            batch_size
        ] = {
            "requests_per_second": (
                require_number(
                    throughput.get(
                        "requests_per_second"
                    ),
                    label=(
                        "batching requests/s"
                    ),
                )
            ),
            "tokens_per_second": (
                require_number(
                    throughput.get(
                        "tokens_per_second"
                    ),
                    label=(
                        "batching tokens/s"
                    ),
                )
            ),
            "mean_completion_latency_ms": (
                require_number(
                    latency.get(
                        "mean_ms"
                    ),
                    label=(
                        "batching mean latency"
                    ),
                )
            ),
            "p95_completion_latency_ms": (
                require_number(
                    latency.get(
                        "p95_ms"
                    ),
                    label=(
                        "batching p95 latency"
                    ),
                )
            ),
        }

    return rows


def main() -> None:
    """Build one comparable M4-G execution-scaling artifact."""
    concurrency_payload = load_json(
        CONCURRENCY_PATH
    )

    batching_payload = load_json(
        BATCHING_PATH
    )

    concurrency = concurrency_rows(
        concurrency_payload
    )

    batching = batching_rows(
        batching_payload
    )

    shared_widths = sorted(
        set(
            concurrency
        )
        & set(
            batching
        )
    )

    if not shared_widths:
        raise RuntimeError(
            "No shared execution widths "
            "between experiments."
        )

    comparisons: list[
        dict[str, Any]
    ] = []

    print(
        "=== M4-G BATCHING / CONCURRENCY COMPARISON ==="
    )

    for width in shared_widths:
        concurrent = concurrency[
            width
        ]

        batched = batching[
            width
        ]

        throughput_ratio = (
            concurrent[
                "requests_per_second"
            ]
            / batched[
                "requests_per_second"
            ]
        )

        latency_ratio = (
            concurrent[
                "mean_completion_latency_ms"
            ]
            / batched[
                "mean_completion_latency_ms"
            ]
        )

        comparisons.append(
            {
                "execution_width": width,
                "concurrency": concurrent,
                "native_batching": batched,
                "concurrency_to_batch_throughput_ratio": (
                    throughput_ratio
                ),
                "concurrency_to_batch_mean_latency_ratio": (
                    latency_ratio
                ),
            }
        )

        print()
        print(
            "width:",
            width,
        )

        print(
            "  concurrency req/s:",
            f"{concurrent['requests_per_second']:.3f}",
        )

        print(
            "  native batch req/s:",
            f"{batched['requests_per_second']:.3f}",
        )

        print(
            "  concurrency / batch throughput:",
            f"{throughput_ratio:.3f}x",
        )

        print(
            "  concurrency mean latency ms:",
            f"{concurrent['mean_completion_latency_ms']:.3f}",
        )

        print(
            "  native batch mean latency ms:",
            f"{batched['mean_completion_latency_ms']:.3f}",
        )

        print(
            "  concurrency / batch latency:",
            f"{latency_ratio:.3f}x",
        )

    concurrency_c1 = concurrency[
        1
    ]

    batching_b1 = batching[
        1
    ]

    concurrency_relative = {
        str(
            width
        ): {
            "throughput_speedup_vs_c1": (
                concurrency[
                    width
                ][
                    "requests_per_second"
                ]
                / concurrency_c1[
                    "requests_per_second"
                ]
            ),
            "mean_latency_ratio_vs_c1": (
                concurrency[
                    width
                ][
                    "mean_completion_latency_ms"
                ]
                / concurrency_c1[
                    "mean_completion_latency_ms"
                ]
            ),
        }
        for width in sorted(
            concurrency
        )
    }

    batching_relative = {
        str(
            width
        ): {
            "throughput_speedup_vs_b1": (
                batching[
                    width
                ][
                    "requests_per_second"
                ]
                / batching_b1[
                    "requests_per_second"
                ]
            ),
            "mean_latency_ratio_vs_b1": (
                batching[
                    width
                ][
                    "mean_completion_latency_ms"
                ]
                / batching_b1[
                    "mean_completion_latency_ms"
                ]
            ),
        }
        for width in sorted(
            batching
        )
    }

    payload: dict[
        str,
        Any,
    ] = {
        "schema_version": "0.1",
        "milestone": "M4-G",
        "experiment": (
            "batching-concurrency-comparison"
        ),
        "verify_first_optimise_second": True,
        "model": (
            batching_payload.get(
                "model"
            )
        ),
        "workload_case": "bank-01",
        "semantic_verification": {
            "concurrency_precheck_passed": (
                concurrency_payload
                .get(
                    "verification",
                    {},
                )
                .get(
                    "semantic_pass"
                )
                is True
            ),
            "native_batch_scalar_passed": (
                batching_payload
                .get(
                    "verification",
                    {},
                )
                .get(
                    "scalar_semantic_pass"
                )
                is True
            ),
            "native_batch_equivalence_passed": all(
                check.get(
                    "semantic_pass"
                )
                is True
                and check.get(
                    "outputs_identical"
                )
                is True
                and check.get(
                    "matches_scalar"
                )
                is True
                for check
                in batching_payload
                .get(
                    "verification",
                    {},
                )
                .get(
                    "batch_checks",
                    [],
                )
                if isinstance(
                    check,
                    dict,
                )
            ),
        },
        "within_axis_scaling": {
            "concurrency": (
                concurrency_relative
            ),
            "native_batching": (
                batching_relative
            ),
        },
        "same_width_comparison": (
            comparisons
        ),
        "observations": [
            (
                "Concurrency produced substantial "
                "aggregate throughput scaling at "
                "widths 2 and 4."
            ),
            (
                "Native batching preserved deterministic "
                "semantic behaviour through batch_size=8."
            ),
            (
                "Native batching produced only modest "
                "throughput improvement while batch "
                "completion latency increased strongly "
                "with batch size."
            ),
            (
                "This artifact records bounded runtime "
                "evidence and does not make the final "
                "multi-objective optimisation decision."
            ),
        ],
        "limitations": [
            (
                "Measurements are from one pinned "
                "SmolLM2 CPU runtime and one previously "
                "verified AP-policy workload case."
            ),
            (
                "Separate runs have slightly different "
                "width-1 baselines, so within-axis "
                "relative scaling is the primary "
                "comparison for speedup."
            ),
            (
                "Caching is not varied in either "
                "experiment."
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
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
