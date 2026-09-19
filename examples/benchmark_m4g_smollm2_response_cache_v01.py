"""Benchmark deterministic application-level response caching for SmolLM2."""

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
from typing import Any

from vait.inference.providers.cache import (
    DeterministicCachingProvider,
)
from vait.inference.providers.huggingface import (
    LocalHuggingFaceProvider,
)
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
)
from vait.rag.models import (
    RAGContextDocument,
    RAGGenerationRequest,
)
from vait.rag.prompt_contracts import (
    RAGPromptContract,
    build_rag_messages,
)
from vait.rag.semantic_guard import (
    assess_semantic_drift,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

MODEL_REVISION = (
    "12fd25f77366fa6b3b4b768ec3050bf629380bac"
)

QUERY = (
    "What must happen before paying a supplier "
    "after its bank account changes?"
)

CONTEXT = (
    "A supplier bank account change requires independent "
    "verification before any payment is released."
)

WARM_HITS = 5

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4g-smollm2-response-cache-v0.1.json"
)


def build_request(
    *,
    request_id: str,
) -> ProviderInferenceRequest:
    """Build the exact deterministic bank-policy request."""
    rag_request = RAGGenerationRequest(
        query=QUERY,
        context_documents=(
            RAGContextDocument(
                document_id="bank-change-policy",
                text=CONTEXT,
                score=1.0,
                rank=1,
            ),
        ),
    )

    rag_messages = build_rag_messages(
        rag_request,
        context_documents=(
            rag_request.context_documents
        ),
        contract=(
            RAGPromptContract.STRICT_V2
        ),
    )

    messages = tuple(
        InferenceMessage(
            role=InferenceRole(
                message["role"]
            ),
            content=message["content"],
        )
        for message
        in rag_messages
    )

    return ProviderInferenceRequest(
        request_id=request_id,
        model_id=MODEL_ID,
        messages=messages,
        max_output_tokens=80,
        temperature=0.0,
        metadata={
            "milestone": "M4-G",
            "experiment": (
                "deterministic-response-cache"
            ),
            "source_case": "bank-01",
            "prompt_contract": (
                RAGPromptContract.STRICT_V2.value
            ),
            "model_revision": MODEL_REVISION,
        },
    )


def timed_generate(
    *,
    provider: DeterministicCachingProvider,
    request_id: str,
) -> tuple[
    ProviderInferenceResponse,
    float,
]:
    """Execute one cached-provider request and capture outer latency."""
    request = build_request(
        request_id=request_id
    )

    started = perf_counter_ns()

    response = provider.generate(
        request
    )

    latency_ms = (
        perf_counter_ns()
        - started
    ) / 1_000_000

    if not response.succeeded:
        raise RuntimeError(
            "Cache experiment provider failure: "
            f"{response.error}"
        )

    if response.output_text is None:
        raise RuntimeError(
            "Successful cache response has no output."
        )

    return (
        response,
        latency_ms,
    )


def require_cache_status(
    response: ProviderInferenceResponse,
    *,
    expected: str,
) -> None:
    """Require one exact cache provenance state."""
    actual = response.metadata.get(
        "cache_status"
    )

    if actual != expected:
        raise RuntimeError(
            "Unexpected cache status: "
            f"expected={expected}, actual={actual}"
        )


def require_semantic_pass(
    response: ProviderInferenceResponse,
) -> None:
    """Require the surfaced answer to pass the bounded semantic guard."""
    if response.output_text is None:
        raise RuntimeError(
            "Semantic verification requires output text."
        )

    assessment = assess_semantic_drift(
        context_text=CONTEXT,
        answer_text=(
            response.output_text.strip()
        ),
    )

    if not assessment.passed:
        risks = [
            issue.risk.value
            for issue
            in assessment.issues
        ]

        raise RuntimeError(
            "Semantic cache verification failed: "
            f"{risks}"
        )


def usage_payload(
    response: ProviderInferenceResponse,
) -> dict[str, int] | None:
    """Return serialisable token evidence."""
    if response.usage is None:
        return None

    return {
        "input_tokens": (
            response.usage.input_tokens
        ),
        "output_tokens": (
            response.usage.output_tokens
        ),
        "total_tokens": (
            response.usage.total_tokens
        ),
    }


def main() -> None:
    """Run cold, warm, and invalidated cache measurements."""
    print(
        "=== M4-G DETERMINISTIC RESPONSE CACHE ==="
    )
    print(
        "model:",
        MODEL_ID,
    )
    print(
        "revision:",
        MODEL_REVISION,
    )
    print(
        "case:",
        "bank-01",
    )

    print()
    print(
        "=== PROVIDER SETUP ==="
    )

    setup_started = perf_counter_ns()

    underlying = LocalHuggingFaceProvider(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        local_files_only=True,
    )

    provider = DeterministicCachingProvider(
        provider=underlying
    )

    setup_ms = (
        perf_counter_ns()
        - setup_started
    ) / 1_000_000

    print(
        "provider_setup_ms:",
        f"{setup_ms:.3f}",
    )

    print()
    print(
        "=== COLD MISS ==="
    )

    cold_response, cold_latency_ms = (
        timed_generate(
            provider=provider,
            request_id=(
                "m4g-cache-cold"
            ),
        )
    )

    require_cache_status(
        cold_response,
        expected="miss",
    )

    require_semantic_pass(
        cold_response
    )

    cold_answer = (
        cold_response.output_text
        or ""
    ).strip()

    print(
        "cache_status:",
        cold_response.metadata[
            "cache_status"
        ],
    )

    print(
        "cold_latency_ms:",
        f"{cold_latency_ms:.3f}",
    )

    print(
        "cold_usage:",
        usage_payload(
            cold_response
        ),
    )

    print(
        "answer:",
        cold_answer,
    )

    print()
    print(
        "=== WARM HITS ==="
    )

    warm_latencies_ms: list[
        float
    ] = []

    warm_responses: list[
        ProviderInferenceResponse
    ] = []

    for index in range(
        WARM_HITS
    ):
        response, latency_ms = (
            timed_generate(
                provider=provider,
                request_id=(
                    "m4g-cache-warm-"
                    f"{index}"
                ),
            )
        )

        require_cache_status(
            response,
            expected="hit",
        )

        require_semantic_pass(
            response
        )

        answer = (
            response.output_text
            or ""
        ).strip()

        if answer != cold_answer:
            raise RuntimeError(
                "Cache hit output differs "
                "from cold provider output."
            )

        if response.usage is None:
            raise RuntimeError(
                "Cache hit must expose zero-work "
                "token evidence."
            )

        if (
            response.usage.input_tokens
            != 0
            or response.usage.output_tokens
            != 0
        ):
            raise RuntimeError(
                "Cache hit incorrectly reports "
                "provider token work."
            )

        warm_responses.append(
            response
        )

        warm_latencies_ms.append(
            latency_ms
        )

        print(
            "hit=",
            index + 1,
            " latency_ms=",
            f"{latency_ms:.6f}",
            " usage=",
            usage_payload(
                response
            ),
            sep="",
        )

    warm_mean_ms = fmean(
        warm_latencies_ms
    )

    warm_min_ms = min(
        warm_latencies_ms
    )

    warm_max_ms = max(
        warm_latencies_ms
    )

    speedup = (
        cold_latency_ms
        / warm_mean_ms
    )

    print()
    print(
        "warm_mean_latency_ms:",
        f"{warm_mean_ms:.6f}",
    )

    print(
        "warm_min_latency_ms:",
        f"{warm_min_ms:.6f}",
    )

    print(
        "warm_max_latency_ms:",
        f"{warm_max_ms:.6f}",
    )

    print(
        "cold_to_warm_speedup:",
        f"{speedup:.3f}x",
    )

    print()
    print(
        "=== INVALIDATION ==="
    )

    cache_size_before_clear = (
        provider.cache_size
    )

    provider.clear()

    cache_size_after_clear = (
        provider.cache_size
    )

    if (
        cache_size_before_clear
        != 1
    ):
        raise RuntimeError(
            "Expected exactly one cached "
            "semantic request before clear()."
        )

    if (
        cache_size_after_clear
        != 0
    ):
        raise RuntimeError(
            "Cache clear() did not invalidate "
            "the stored response."
        )

    (
        invalidated_response,
        invalidated_latency_ms,
    ) = timed_generate(
        provider=provider,
        request_id=(
            "m4g-cache-after-clear"
        ),
    )

    require_cache_status(
        invalidated_response,
        expected="miss",
    )

    require_semantic_pass(
        invalidated_response
    )

    invalidated_answer = (
        invalidated_response.output_text
        or ""
    ).strip()

    if (
        invalidated_answer
        != cold_answer
    ):
        raise RuntimeError(
            "Post-invalidation model output "
            "differs from cold verified output."
        )

    print(
        "cache_size_before_clear:",
        cache_size_before_clear,
    )

    print(
        "cache_size_after_clear:",
        cache_size_after_clear,
    )

    print(
        "post_clear_status:",
        invalidated_response.metadata[
            "cache_status"
        ],
    )

    print(
        "post_clear_latency_ms:",
        f"{invalidated_latency_ms:.3f}",
    )

    print()
    print(
        "=== CACHE EVIDENCE ==="
    )

    print(
        "hits:",
        provider.cache_hits,
    )

    print(
        "misses:",
        provider.cache_misses,
    )

    print(
        "bypasses:",
        provider.cache_bypasses,
    )

    print(
        "provider_generations_avoided:",
        provider.cache_hits,
    )

    payload: dict[
        str,
        Any,
    ] = {
        "schema_version": "0.1",
        "milestone": "M4-G",
        "experiment": (
            "verified-smollm2-"
            "deterministic-response-cache"
        ),
        "verify_first_optimise_second": (
            True
        ),
        "scope": {
            "cache_layer": (
                "application-level-"
                "deterministic-response-cache"
            ),
            "transformer_kv_cache_varied": (
                False
            ),
            "native_batching_varied": (
                False
            ),
            "concurrency_varied": (
                False
            ),
            "workload_case": "bank-01",
            "temperature": 0.0,
        },
        "model": {
            "model_id": MODEL_ID,
            "model_revision": (
                MODEL_REVISION
            ),
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
        },
        "setup": {
            "provider_setup_ms": (
                setup_ms
            ),
        },
        "cold_miss": {
            "latency_ms": (
                cold_latency_ms
            ),
            "usage": usage_payload(
                cold_response
            ),
            "cache_status": (
                cold_response.metadata[
                    "cache_status"
                ]
            ),
            "semantic_pass": True,
            "answer": cold_answer,
        },
        "warm_hits": {
            "count": WARM_HITS,
            "latencies_ms": (
                warm_latencies_ms
            ),
            "mean_latency_ms": (
                warm_mean_ms
            ),
            "min_latency_ms": (
                warm_min_ms
            ),
            "max_latency_ms": (
                warm_max_ms
            ),
            "cold_to_warm_speedup": (
                speedup
            ),
            "outputs_match_cold": True,
            "semantic_pass": True,
            "marginal_input_tokens": 0,
            "marginal_output_tokens": 0,
        },
        "invalidation": {
            "cache_size_before_clear": (
                cache_size_before_clear
            ),
            "cache_size_after_clear": (
                cache_size_after_clear
            ),
            "post_clear_status": (
                invalidated_response.metadata[
                    "cache_status"
                ]
            ),
            "post_clear_latency_ms": (
                invalidated_latency_ms
            ),
            "post_clear_usage": (
                usage_payload(
                    invalidated_response
                )
            ),
            "post_clear_output_matches_cold": (
                True
            ),
            "semantic_pass": True,
        },
        "statistics": {
            "cache_hits": (
                provider.cache_hits
            ),
            "cache_misses": (
                provider.cache_misses
            ),
            "cache_bypasses": (
                provider.cache_bypasses
            ),
            "provider_generations_avoided": (
                provider.cache_hits
            ),
        },
        "limitations": [
            (
                "This cache reuses only exact "
                "deterministic semantic requests."
            ),
            (
                "Positive-temperature requests "
                "bypass the cache."
            ),
            (
                "Failures are never cached."
            ),
            (
                "This experiment does not vary "
                "Transformer KV-cache behaviour."
            ),
            (
                "Results use one pinned SmolLM2 "
                "CPU runtime and one previously "
                "verified AP-policy workload case."
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
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Wrote {ARTIFACT_PATH}"
    )


if __name__ == "__main__":
    main()
