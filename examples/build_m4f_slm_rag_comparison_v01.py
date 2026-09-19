"""Build the final M4-F SLM/RAG quality-performance comparison."""

import json
from pathlib import Path
from typing import Any

SMOL_RAW_PATH = Path(
    "artifacts/benchmarks/"
    "m4f-ap-policy-smollm2-provider-rag-v0.1.json"
)

SMOL_GUARDED_PATH = Path(
    "artifacts/benchmarks/"
    "m4f-ap-policy-smollm2-"
    "guarded-provider-rag-v0.1.json"
)

QWEN_RAW_PATH = Path(
    "artifacts/benchmarks/"
    "m4f-ap-policy-qwen2.5-0.5b-provider-rag-v0.1.json"
)

QWEN_GUARDED_PATH = Path(
    "artifacts/benchmarks/"
    "m4f-ap-policy-qwen2.5-0.5b-"
    "guarded-provider-rag-v0.1.json"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "m4f-slm-rag-comparison-v0.1.json"
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one required benchmark artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark artifact: {path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"Artifact must contain an object: {path}"
        )

    return payload


def require_float(
    value: Any,
    *,
    label: str,
) -> float:
    """Return one required numeric value."""
    if not isinstance(
        value,
        int | float,
    ):
        raise TypeError(
            f"{label} must be numeric."
        )

    return float(
        value
    )


def require_int(
    value: Any,
    *,
    label: str,
) -> int:
    """Return one required integer value."""
    if not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{label} must be an integer."
        )

    return value


def quality_metrics(
    payload: dict[str, Any],
) -> dict[str, float]:
    """Extract structural RAG quality metrics."""
    quality = payload.get(
        "quality"
    )

    if not isinstance(
        quality,
        dict,
    ):
        raise TypeError(
            "Artifact is missing quality evidence."
        )

    names = (
        "evidence_coverage",
        "citation_accuracy",
        "abstention_accuracy",
        "answerable_success_rate",
        "overall_success_rate",
    )

    return {
        name: require_float(
            quality.get(
                name
            ),
            label=(
                f"quality.{name}"
            ),
        )
        for name in names
    }


def raw_summary(
    *,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one unguarded SLM/RAG experiment."""
    inference = payload.get(
        "inference"
    )

    if not isinstance(
        inference,
        dict,
    ):
        raise TypeError(
            f"{name} is missing inference evidence."
        )

    semantic_pass_rate = require_float(
        inference.get(
            "semantic_guard_pass_rate"
        ),
        label=(
            f"{name}.semantic_guard_pass_rate"
        ),
    )

    provider_failures = require_int(
        inference.get(
            "provider_failures"
        ),
        label=(
            f"{name}.provider_failures"
        ),
    )

    quality = quality_metrics(
        payload
    )

    structurally_valid = all(
        value == 1.0
        for value
        in quality.values()
    )

    semantic_admissible = (
        structurally_valid
        and provider_failures == 0
        and semantic_pass_rate == 1.0
    )

    return {
        "name": name,
        "path_type": "raw",
        "quality": quality,
        "provider_failures": (
            provider_failures
        ),
        "raw_semantic_pass_rate": (
            semantic_pass_rate
        ),
        "surfaced_semantic_pass_rate": None,
        "primary_surfaced": None,
        "fallback_surfaced": None,
        "generation_tokens_per_second": (
            require_float(
                inference.get(
                    "generation_tokens_per_second"
                ),
                label=(
                    f"{name}.generation_tokens_per_second"
                ),
            )
        ),
        "mean_end_to_end_latency_ms": (
            require_float(
                inference.get(
                    "mean_end_to_end_latency_ms"
                ),
                label=(
                    f"{name}.mean_end_to_end_latency_ms"
                ),
            )
        ),
        "structurally_valid": (
            structurally_valid
        ),
        "semantic_admissible": (
            semantic_admissible
        ),
    }


def guarded_summary(
    *,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one semantic-guarded SLM/RAG experiment."""
    inference = payload.get(
        "inference"
    )
    guard = payload.get(
        "guard"
    )

    if not isinstance(
        inference,
        dict,
    ):
        raise TypeError(
            f"{name} is missing inference evidence."
        )

    if not isinstance(
        guard,
        dict,
    ):
        raise TypeError(
            f"{name} is missing guard evidence."
        )

    raw_pass_rate = require_float(
        guard.get(
            "raw_primary_pass_rate"
        ),
        label=(
            f"{name}.raw_primary_pass_rate"
        ),
    )

    surfaced_pass_rate = require_float(
        guard.get(
            "surfaced_pass_rate"
        ),
        label=(
            f"{name}.surfaced_pass_rate"
        ),
    )

    primary_surfaced = require_int(
        guard.get(
            "primary_surfaced"
        ),
        label=(
            f"{name}.primary_surfaced"
        ),
    )

    fallback_surfaced = require_int(
        guard.get(
            "fallback_surfaced"
        ),
        label=(
            f"{name}.fallback_surfaced"
        ),
    )

    provider_failures = require_int(
        inference.get(
            "provider_failures"
        ),
        label=(
            f"{name}.provider_failures"
        ),
    )

    quality = quality_metrics(
        payload
    )

    structurally_valid = all(
        value == 1.0
        for value
        in quality.values()
    )

    semantic_admissible = (
        structurally_valid
        and provider_failures == 0
        and surfaced_pass_rate == 1.0
    )

    return {
        "name": name,
        "path_type": "guarded",
        "quality": quality,
        "provider_failures": (
            provider_failures
        ),
        "raw_semantic_pass_rate": (
            raw_pass_rate
        ),
        "surfaced_semantic_pass_rate": (
            surfaced_pass_rate
        ),
        "primary_surfaced": (
            primary_surfaced
        ),
        "fallback_surfaced": (
            fallback_surfaced
        ),
        "generation_tokens_per_second": (
            require_float(
                inference.get(
                    "generation_tokens_per_second"
                ),
                label=(
                    f"{name}.generation_tokens_per_second"
                ),
            )
        ),
        "mean_end_to_end_latency_ms": (
            require_float(
                inference.get(
                    "mean_end_to_end_latency_ms"
                ),
                label=(
                    f"{name}.mean_end_to_end_latency_ms"
                ),
            )
        ),
        "structurally_valid": (
            structurally_valid
        ),
        "semantic_admissible": (
            semantic_admissible
        ),
    }


def relative_change(
    candidate: float,
    baseline: float,
) -> float:
    """Return fractional change relative to baseline."""
    if baseline == 0.0:
        raise ZeroDivisionError(
            "Comparison baseline must be non-zero."
        )

    return (
        candidate
        / baseline
    ) - 1.0


def main() -> None:
    """Build and print final M4-F comparison evidence."""
    smol_raw = raw_summary(
        name="smollm2_raw",
        payload=load_json(
            SMOL_RAW_PATH
        ),
    )

    smol_guarded = guarded_summary(
        name="smollm2_guarded",
        payload=load_json(
            SMOL_GUARDED_PATH
        ),
    )

    qwen_raw = raw_summary(
        name="qwen2.5_0.5b_raw",
        payload=load_json(
            QWEN_RAW_PATH
        ),
    )

    qwen_guarded = guarded_summary(
        name="qwen2.5_0.5b_guarded",
        payload=load_json(
            QWEN_GUARDED_PATH
        ),
    )

    candidates = [
        smol_raw,
        smol_guarded,
        qwen_raw,
        qwen_guarded,
    ]

    smol_guarded_tps = require_float(
        smol_guarded[
            "generation_tokens_per_second"
        ],
        label="smollm2_guarded throughput",
    )

    qwen_guarded_tps = require_float(
        qwen_guarded[
            "generation_tokens_per_second"
        ],
        label="qwen_guarded throughput",
    )

    smol_guarded_latency = require_float(
        smol_guarded[
            "mean_end_to_end_latency_ms"
        ],
        label="smollm2_guarded latency",
    )

    qwen_guarded_latency = require_float(
        qwen_guarded[
            "mean_end_to_end_latency_ms"
        ],
        label="qwen_guarded latency",
    )

    comparison = {
        "qwen_vs_smollm2_guarded": {
            "throughput_ratio": (
                qwen_guarded_tps
                / smol_guarded_tps
            ),
            "throughput_change": (
                relative_change(
                    qwen_guarded_tps,
                    smol_guarded_tps,
                )
            ),
            "latency_ratio": (
                qwen_guarded_latency
                / smol_guarded_latency
            ),
            "latency_change": (
                relative_change(
                    qwen_guarded_latency,
                    smol_guarded_latency,
                )
            ),
            "raw_semantic_pass_delta": (
                require_float(
                    qwen_guarded[
                        "raw_semantic_pass_rate"
                    ],
                    label="qwen raw semantic rate",
                )
                - require_float(
                    smol_guarded[
                        "raw_semantic_pass_rate"
                    ],
                    label="smollm2 raw semantic rate",
                )
            ),
            "additional_primary_surfaced": (
                require_int(
                    qwen_guarded[
                        "primary_surfaced"
                    ],
                    label="qwen primary surfaced",
                )
                - require_int(
                    smol_guarded[
                        "primary_surfaced"
                    ],
                    label="smollm2 primary surfaced",
                )
            ),
        }
    }

    if smol_raw[
        "semantic_admissible"
    ]:
        raise RuntimeError(
            "SmolLM2 raw path unexpectedly passed "
            "the semantic gate."
        )

    if qwen_raw[
        "semantic_admissible"
    ]:
        raise RuntimeError(
            "Qwen raw path unexpectedly passed "
            "the semantic gate."
        )

    if not smol_guarded[
        "semantic_admissible"
    ]:
        raise RuntimeError(
            "SmolLM2 guarded path failed "
            "the surfaced semantic gate."
        )

    if not qwen_guarded[
        "semantic_admissible"
    ]:
        raise RuntimeError(
            "Qwen guarded path failed "
            "the surfaced semantic gate."
        )

    payload = {
        "schema_version": "0.1",
        "milestone": "M4-F",
        "experiment": (
            "slm-rag-quality-performance-comparison"
        ),
        "verify_first_optimise_second": True,
        "candidates": candidates,
        "comparison": comparison,
        "observations": [
            (
                "Neither raw SLM candidate passes the "
                "bounded deterministic semantic gate."
            ),
            (
                "Both guarded paths achieve complete "
                "surfaced semantic-guard passage on "
                "this benchmark."
            ),
            (
                "Guarded-path performance must be "
                "interpreted together with fallback "
                "dependency, not throughput alone."
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

    print(
        "M4-F SLM/RAG comparison"
    )
    print()

    for candidate in candidates:
        print(
            candidate[
                "name"
            ]
        )
        print(
            "  raw semantic pass:",
            f"{require_float(candidate['raw_semantic_pass_rate'], label='raw semantic'):.3f}",
        )

        surfaced = candidate[
            "surfaced_semantic_pass_rate"
        ]

        if surfaced is not None:
            print(
                "  surfaced semantic pass:",
                f"{require_float(surfaced, label='surfaced semantic'):.3f}",
            )
            print(
                "  primary surfaced:",
                candidate[
                    "primary_surfaced"
                ],
            )
            print(
                "  fallback surfaced:",
                candidate[
                    "fallback_surfaced"
                ],
            )

        print(
            "  tokens/s:",
            f"{require_float(candidate['generation_tokens_per_second'], label='throughput'):.3f}",
        )
        print(
            "  mean E2E ms:",
            f"{require_float(candidate['mean_end_to_end_latency_ms'], label='latency'):.3f}",
        )
        print(
            "  semantic admissible:",
            candidate[
                "semantic_admissible"
            ],
        )
        print()

    qwen_vs_smol = comparison[
        "qwen_vs_smollm2_guarded"
    ]

    print(
        "Qwen guarded vs SmolLM2 guarded"
    )
    print(
        "  throughput ratio:",
        f"{qwen_vs_smol['throughput_ratio']:.3f}x",
    )
    print(
        "  throughput change:",
        f"{qwen_vs_smol['throughput_change'] * 100:+.2f}%",
    )
    print(
        "  latency ratio:",
        f"{qwen_vs_smol['latency_ratio']:.3f}x",
    )
    print(
        "  latency change:",
        f"{qwen_vs_smol['latency_change'] * 100:+.2f}%",
    )
    print(
        "  raw semantic pass delta:",
        f"{qwen_vs_smol['raw_semantic_pass_delta']:+.3f}",
    )
    print(
        "  additional primary surfaced:",
        qwen_vs_smol[
            "additional_primary_surfaced"
        ],
    )

    print()
    print(
        f"Wrote {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
