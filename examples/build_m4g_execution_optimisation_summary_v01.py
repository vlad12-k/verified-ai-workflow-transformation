"""Build unified M4-G batching, concurrency, and caching evidence."""

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

COMPARISON_PATH = Path(
    "artifacts/benchmarks/"
    "m4g-batching-concurrency-comparison-v0.1.json"
)

CACHE_PATH = Path(
    "artifacts/benchmarks/"
    "m4g-smollm2-response-cache-v0.1.json"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "m4g-execution-optimisation-summary-v0.1.json"
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required benchmark artifact."""
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
            f"Expected JSON object: {path}"
        )

    return payload


def require_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    """Require one JSON object."""
    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"Expected object: {label}"
        )

    return value


def require_list(
    value: object,
    *,
    label: str,
) -> list[Any]:
    """Require one JSON array."""
    if not isinstance(
        value,
        list,
    ):
        raise RuntimeError(
            f"Expected list: {label}"
        )

    return value


def require_number(
    value: object,
    *,
    label: str,
) -> float:
    """Require one non-boolean numeric evidence value."""
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


def main() -> None:
    """Build bounded M4-G execution optimisation evidence."""
    concurrency = load_json(
        CONCURRENCY_PATH
    )

    batching = load_json(
        BATCHING_PATH
    )

    comparison = load_json(
        COMPARISON_PATH
    )

    cache = load_json(
        CACHE_PATH
    )

    concurrency_verification = (
        require_mapping(
            concurrency.get(
                "verification"
            ),
            label=(
                "concurrency verification"
            ),
        )
    )

    if (
        concurrency_verification.get(
            "semantic_pass"
        )
        is not True
        or concurrency_verification.get(
            "outputs_identical"
        )
        is not True
    ):
        raise RuntimeError(
            "Concurrency verify-first gate "
            "was not satisfied."
        )

    batching_verification = (
        require_mapping(
            batching.get(
                "verification"
            ),
            label=(
                "batching verification"
            ),
        )
    )

    if (
        batching_verification.get(
            "scalar_semantic_pass"
        )
        is not True
    ):
        raise RuntimeError(
            "Native-batch scalar semantic "
            "verification was not satisfied."
        )

    batch_checks = require_list(
        batching_verification.get(
            "batch_checks"
        ),
        label="native batch checks",
    )

    if not batch_checks:
        raise RuntimeError(
            "Native-batch verification "
            "contains no batch checks."
        )

    for check in batch_checks:
        checked = require_mapping(
            check,
            label="native batch check",
        )

        if (
            checked.get(
                "semantic_pass"
            )
            is not True
            or checked.get(
                "outputs_identical"
            )
            is not True
            or checked.get(
                "matches_scalar"
            )
            is not True
        ):
            raise RuntimeError(
                "Native-batch equivalence gate "
                "was not satisfied."
            )

    cold_miss = require_mapping(
        cache.get(
            "cold_miss"
        ),
        label="cache cold miss",
    )

    warm_hits = require_mapping(
        cache.get(
            "warm_hits"
        ),
        label="cache warm hits",
    )

    invalidation = require_mapping(
        cache.get(
            "invalidation"
        ),
        label="cache invalidation",
    )

    statistics = require_mapping(
        cache.get(
            "statistics"
        ),
        label="cache statistics",
    )

    if (
        cold_miss.get(
            "semantic_pass"
        )
        is not True
        or warm_hits.get(
            "semantic_pass"
        )
        is not True
        or warm_hits.get(
            "outputs_match_cold"
        )
        is not True
        or invalidation.get(
            "semantic_pass"
        )
        is not True
        or invalidation.get(
            "post_clear_output_matches_cold"
        )
        is not True
    ):
        raise RuntimeError(
            "Response-cache semantic or "
            "invalidation gate failed."
        )

    cold_latency_ms = require_number(
        cold_miss.get(
            "latency_ms"
        ),
        label="cold cache latency",
    )

    warm_mean_latency_ms = (
        require_number(
            warm_hits.get(
                "mean_latency_ms"
            ),
            label=(
                "warm cache mean latency"
            ),
        )
    )

    post_clear_latency_ms = (
        require_number(
            invalidation.get(
                "post_clear_latency_ms"
            ),
            label=(
                "post-clear miss latency"
            ),
        )
    )

    provider_generations_avoided = (
        require_number(
            statistics.get(
                "provider_generations_avoided"
            ),
            label=(
                "provider generations avoided"
            ),
        )
    )

    cold_to_warm_speedup = (
        cold_latency_ms
        / warm_mean_latency_ms
    )

    post_clear_to_warm_speedup = (
        post_clear_latency_ms
        / warm_mean_latency_ms
    )

    same_width = require_list(
        comparison.get(
            "same_width_comparison"
        ),
        label="same-width comparison",
    )

    print(
        "=== M4-G EXECUTION OPTIMISATION SUMMARY ==="
    )

    print()
    print(
        "verify-first gates: PASS"
    )

    print()
    print(
        "concurrency:"
    )

    for item in require_list(
        concurrency.get(
            "relative_scaling"
        ),
        label="concurrency scaling",
    ):
        row = require_mapping(
            item,
            label="concurrency scaling row",
        )

        throughput_speedup = require_number(
            row.get(
                "throughput_speedup_vs_c1"
            ),
            label=(
                "concurrency throughput speedup"
            ),
        )

        latency_ratio = require_number(
            row.get(
                "mean_latency_ratio_vs_c1"
            ),
            label=(
                "concurrency latency ratio"
            ),
        )

        print(
            "  c=",
            row.get(
                "concurrency"
            ),
            " throughput_speedup=",
            f"{throughput_speedup:.3f}x",
            " latency_ratio=",
            f"{latency_ratio:.3f}x",
            sep="",
        )

    print()
    print(
        "native batching:"
    )

    for item in require_list(
        batching.get(
            "relative_scaling"
        ),
        label="batch scaling",
    ):
        row = require_mapping(
            item,
            label="batch scaling row",
        )

        throughput_speedup = require_number(
            row.get(
                "throughput_speedup_vs_b1"
            ),
            label=(
                "batch throughput speedup"
            ),
        )

        batch_latency_ratio = require_number(
            row.get(
                "batch_latency_ratio_vs_b1"
            ),
            label=(
                "batch latency ratio"
            ),
        )

        print(
            "  b=",
            row.get(
                "batch_size"
            ),
            " throughput_speedup=",
            f"{throughput_speedup:.3f}x",
            " batch_latency_ratio=",
            f"{batch_latency_ratio:.3f}x",
            sep="",
        )

    print()
    print(
        "same-width concurrency vs batching:"
    )

    for item in same_width:
        row = require_mapping(
            item,
            label="same-width row",
        )

        throughput_ratio = require_number(
            row.get(
                "concurrency_to_batch_throughput_ratio"
            ),
            label=(
                "same-width throughput ratio"
            ),
        )

        latency_ratio = require_number(
            row.get(
                "concurrency_to_batch_mean_latency_ratio"
            ),
            label=(
                "same-width latency ratio"
            ),
        )

        print(
            "  width=",
            row.get(
                "execution_width"
            ),
            " throughput_ratio=",
            f"{throughput_ratio:.3f}x",
            " latency_ratio=",
            f"{latency_ratio:.3f}x",
            sep="",
        )

    print()
    print(
        "deterministic response cache:"
    )

    print(
        "  cold_latency_ms:",
        f"{cold_latency_ms:.3f}",
    )

    print(
        "  warm_mean_latency_ms:",
        f"{warm_mean_latency_ms:.6f}",
    )

    print(
        "  post_clear_latency_ms:",
        f"{post_clear_latency_ms:.3f}",
    )

    print(
        "  cold_to_warm_speedup:",
        f"{cold_to_warm_speedup:.3f}x",
    )

    print(
        "  post_clear_to_warm_speedup:",
        f"{post_clear_to_warm_speedup:.3f}x",
    )

    print(
        "  provider_generations_avoided:",
        int(
            provider_generations_avoided
        ),
    )

    payload: dict[
        str,
        Any,
    ] = {
        "schema_version": "0.1",
        "milestone": "M4-G",
        "title": (
            "Batching, concurrency, and "
            "deterministic caching evidence"
        ),
        "verify_first_optimise_second": (
            True
        ),
        "bounded_scope": {
            "model_id": (
                "HuggingFaceTB/"
                "SmolLM2-135M-Instruct"
            ),
            "model_revision": (
                "12fd25f77366fa6b3b4b768ec3050bf629380bac"
            ),
            "device": "cpu",
            "workload_case": (
                "bank-01"
            ),
            "raw_model_global_rag_admissibility": (
                False
            ),
            "interpretation": (
                "Execution-scaling evidence "
                "for one previously semantically "
                "verified deterministic workload case."
            ),
        },
        "verification": {
            "concurrency_semantic_precheck": (
                True
            ),
            "concurrency_deterministic_precheck": (
                True
            ),
            "native_batch_scalar_semantic_pass": (
                True
            ),
            "native_batch_semantic_equivalence": (
                True
            ),
            "cache_cold_semantic_pass": (
                True
            ),
            "cache_warm_exact_output_preservation": (
                True
            ),
            "cache_invalidation_restores_miss": (
                True
            ),
        },
        "concurrency": {
            "relative_scaling": (
                concurrency.get(
                    "relative_scaling"
                )
            ),
        },
        "native_batching": {
            "relative_scaling": (
                batching.get(
                    "relative_scaling"
                )
            ),
        },
        "batching_vs_concurrency": {
            "same_width_comparison": (
                same_width
            ),
        },
        "deterministic_response_cache": {
            "cold_latency_ms": (
                cold_latency_ms
            ),
            "warm_mean_latency_ms": (
                warm_mean_latency_ms
            ),
            "post_clear_latency_ms": (
                post_clear_latency_ms
            ),
            "cold_to_warm_speedup": (
                cold_to_warm_speedup
            ),
            "post_clear_to_warm_speedup": (
                post_clear_to_warm_speedup
            ),
            "provider_generations_avoided": (
                int(
                    provider_generations_avoided
                )
            ),
            "marginal_input_tokens_on_hit": (
                0
            ),
            "marginal_output_tokens_on_hit": (
                0
            ),
        },
        "observations": [
            (
                "Concurrency produced substantial "
                "aggregate throughput scaling at "
                "execution widths 2 and 4."
            ),
            (
                "Native batching preserved exact "
                "deterministic semantics through "
                "batch_size=8 but produced only "
                "modest throughput scaling."
            ),
            (
                "At shared execution widths 2 and 4, "
                "concurrency produced higher throughput "
                "and lower completion latency than "
                "native batching on this runtime."
            ),
            (
                "Exact deterministic response-cache "
                "hits avoided model generation and "
                "provider token work."
            ),
            (
                "Explicit cache invalidation restored "
                "model-generation latency and a cache "
                "miss, confirming the measured warm "
                "path was cache reuse."
            ),
        ],
        "not_yet_decided": [
            (
                "No final multi-objective optimisation "
                "configuration is selected in M4-G."
            ),
            (
                "Final trade-off selection remains "
                "the responsibility of M4-I."
            ),
        ],
        "kv_cache_scope": {
            "transformer_kv_cache_varied": (
                False
            ),
            "reason": (
                "Transformer generation cache behaviour "
                "is distinct from application response "
                "caching and is not required to support "
                "the M4-G application-cache evidence."
            ),
            "next_evidence_phase": (
                "M4-H profiling/resource evidence"
            ),
        },
        "limitations": [
            (
                "Execution results are bounded to "
                "one pinned local CPU runtime and one "
                "semantically verified AP-policy case."
            ),
            (
                "The raw SmolLM2 RAG candidate remains "
                "globally inadmissible under M4-F; "
                "these execution experiments do not "
                "override that result."
            ),
            (
                "The large response-cache speedup "
                "applies only to exact reusable "
                "deterministic requests."
            ),
            (
                "Cross-run absolute latency baselines "
                "can vary; within-run relative scaling "
                "is the primary performance evidence."
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
