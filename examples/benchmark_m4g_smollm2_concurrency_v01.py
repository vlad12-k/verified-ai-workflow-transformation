"""Benchmark verified SmolLM2 provider concurrency scaling."""

import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from vait.inference.models import (
    InferenceConfiguration,
)
from vait.inference.providers.benchmark import (
    run_provider_concurrent_generative_benchmark,
)
from vait.inference.providers.huggingface import (
    LocalHuggingFaceProvider,
)
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderInferenceRequest,
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

MODEL_ID = (
    "HuggingFaceTB/"
    "SmolLM2-135M-Instruct"
)

MODEL_REVISION = (
    "12fd25f77366fa6b3b4b768ec3050bf629380bac"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4g-smollm2-concurrency-v0.1.json"
)

QUERY = (
    "What must happen before paying a supplier "
    "after its bank account changes?"
)

CONTEXT = (
    "A supplier bank account change requires independent "
    "verification before any payment is released."
)

CONCURRENCY_LEVELS = (
    1,
    2,
    4,
)

WARMUP_ROUNDS = 1
MEASURED_ROUNDS = 3
MAX_OUTPUT_TOKENS = 80


def build_request(
    *,
    request_id: str,
) -> ProviderInferenceRequest:
    """Build the exact verified bank-policy provider request."""
    rag_request = RAGGenerationRequest(
        query=QUERY,
        context_documents=(
            RAGContextDocument(
                document_id=(
                    "bank-change-policy"
                ),
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
            content=message[
                "content"
            ],
        )
        for message
        in rag_messages
    )

    return ProviderInferenceRequest(
        request_id=request_id,
        model_id=MODEL_ID,
        messages=messages,
        max_output_tokens=(
            MAX_OUTPUT_TOKENS
        ),
        temperature=0.0,
        metadata={
            "milestone": "M4-G",
            "experiment": (
                "concurrency-scaling"
            ),
            "source_case": "bank-01",
            "prompt_contract": (
                RAGPromptContract
                .STRICT_V2
                .value
            ),
            "model_revision": (
                MODEL_REVISION
            ),
        },
    )


def verify_request(
    *,
    provider: LocalHuggingFaceProvider,
) -> dict[str, Any]:
    """Verify deterministic semantic behaviour before optimisation."""
    outputs: list[
        str
    ] = []

    latencies_ms: list[
        float
    ] = []

    for index in range(
        2
    ):
        request = build_request(
            request_id=(
                f"m4g-bank-precheck-{index}"
            )
        )

        started = perf_counter_ns()

        response = provider.generate(
            request
        )

        outer_latency_ms = (
            perf_counter_ns()
            - started
        ) / 1_000_000

        if not response.succeeded:
            error = response.error

            raise RuntimeError(
                "Pre-verification provider "
                "request failed: "
                f"{error}"
            )

        if response.output_text is None:
            raise RuntimeError(
                "Pre-verification response "
                "has no output text."
            )

        answer = (
            response.output_text.strip()
        )

        assessment = (
            assess_semantic_drift(
                context_text=CONTEXT,
                answer_text=answer,
            )
        )

        if not assessment.passed:
            issues = [
                issue.risk.value
                for issue
                in assessment.issues
            ]

            raise RuntimeError(
                "Verify-first semantic gate "
                f"failed: {issues}"
            )

        outputs.append(
            answer
        )

        latencies_ms.append(
            outer_latency_ms
        )

    if len(
        set(
            outputs
        )
    ) != 1:
        raise RuntimeError(
            "Verify-first determinism gate "
            "failed: repeated outputs differ."
        )

    return {
        "case_id": "bank-01",
        "query": QUERY,
        "context": CONTEXT,
        "repetitions": 2,
        "outputs_identical": True,
        "semantic_pass": True,
        "answer": outputs[0],
        "outer_latencies_ms": (
            latencies_ms
        ),
    }


def require_rate(
    value: float | None,
    *,
    label: str,
) -> float:
    """Return one required benchmark throughput value."""
    if value is None:
        raise RuntimeError(
            f"Missing throughput value: {label}"
        )

    return value


def main() -> None:
    """Run verified concurrency scaling evidence."""
    print(
        "=== M4-G VERIFIED CONCURRENCY SCALING ==="
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

    setup_started = (
        perf_counter_ns()
    )

    provider = (
        LocalHuggingFaceProvider(
            model_id=MODEL_ID,
            revision=(
                MODEL_REVISION
            ),
            device="cpu",
            local_files_only=True,
        )
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
        "=== VERIFY FIRST ==="
    )

    verification = (
        verify_request(
            provider=provider
        )
    )

    print(
        "outputs_identical:",
        verification[
            "outputs_identical"
        ],
    )

    print(
        "semantic_pass:",
        verification[
            "semantic_pass"
        ],
    )

    print(
        "answer:",
        verification[
            "answer"
        ],
    )

    configuration = (
        InferenceConfiguration(
            provider=(
                provider.provider_id
            ),
            runtime=(
                provider.runtime_id
            ),
            device="cpu",
            dtype="float32",
            batch_size=1,
            model_id=MODEL_ID,
            model_revision=(
                MODEL_REVISION
            ),
            metadata={
                "execution_axis": (
                    "concurrency"
                ),
                "batching_enabled": (
                    False
                ),
                "prompt_contract": (
                    RAGPromptContract
                    .STRICT_V2
                    .value
                ),
                "source_case": (
                    "bank-01"
                ),
            },
        )
    )

    base_request = (
        build_request(
            request_id=(
                "m4g-bank-concurrency"
            )
        )
    )

    reports: list[
        dict[str, Any]
    ] = []

    print()
    print(
        "=== CONCURRENCY SERIES ==="
    )

    for concurrency in (
        CONCURRENCY_LEVELS
    ):
        report = (
            run_provider_concurrent_generative_benchmark(
                provider=provider,
                request=(
                    base_request
                ),
                configuration=(
                    configuration
                ),
                concurrency=(
                    concurrency
                ),
                warmup_rounds=(
                    WARMUP_ROUNDS
                ),
                measured_rounds=(
                    MEASURED_ROUNDS
                ),
                evidence_metadata={
                    "milestone": (
                        "M4-G"
                    ),
                    "experiment": (
                        "verified-provider-"
                        "concurrency-scaling"
                    ),
                    "source_case": (
                        "bank-01"
                    ),
                    "semantic_precheck": (
                        True
                    ),
                },
            )
        )

        requests_per_second = (
            require_rate(
                report
                .throughput
                .requests_per_second,
                label=(
                    "requests_per_second"
                ),
            )
        )

        tokens_per_second = (
            require_rate(
                report
                .throughput
                .tokens_per_second,
                label=(
                    "tokens_per_second"
                ),
            )
        )

        print()
        print(
            "concurrency:",
            concurrency,
        )

        print(
            "measured_requests:",
            report.measured_iterations,
        )

        print(
            "mean_request_latency_ms:",
            f"{report.latency.mean_ms:.3f}",
        )

        print(
            "p95_request_latency_ms:",
            f"{report.latency.p95_ms:.3f}",
        )

        print(
            "requests_per_second:",
            f"{requests_per_second:.3f}",
        )

        print(
            "tokens_per_second:",
            f"{tokens_per_second:.3f}",
        )

        reports.append(
            {
                "concurrency": (
                    concurrency
                ),
                "report": (
                    report.model_dump(
                        mode="json"
                    )
                ),
            }
        )

    baseline_report = (
        reports[0][
            "report"
        ]
    )

    baseline_throughput = (
        baseline_report[
            "throughput"
        ][
            "requests_per_second"
        ]
    )

    baseline_latency = (
        baseline_report[
            "latency"
        ][
            "mean_ms"
        ]
    )

    if not isinstance(
        baseline_throughput,
        int | float,
    ):
        raise RuntimeError(
            "Missing baseline request throughput."
        )

    if not isinstance(
        baseline_latency,
        int | float,
    ):
        raise RuntimeError(
            "Missing baseline latency."
        )

    comparison: list[
        dict[str, Any]
    ] = []

    print()
    print(
        "=== RELATIVE SCALING ==="
    )

    for item in reports:
        concurrency = int(
            item[
                "concurrency"
            ]
        )

        report_payload = (
            item[
                "report"
            ]
        )

        throughput = (
            report_payload[
                "throughput"
            ][
                "requests_per_second"
            ]
        )

        latency = (
            report_payload[
                "latency"
            ][
                "mean_ms"
            ]
        )

        if not isinstance(
            throughput,
            int | float,
        ):
            raise RuntimeError(
                "Missing request throughput."
            )

        if not isinstance(
            latency,
            int | float,
        ):
            raise RuntimeError(
                "Missing request latency."
            )

        throughput_speedup = (
            float(
                throughput
            )
            / float(
                baseline_throughput
            )
        )

        scaling_efficiency = (
            throughput_speedup
            / concurrency
        )

        latency_ratio = (
            float(
                latency
            )
            / float(
                baseline_latency
            )
        )

        comparison.append(
            {
                "concurrency": (
                    concurrency
                ),
                "throughput_speedup_vs_c1": (
                    throughput_speedup
                ),
                "scaling_efficiency": (
                    scaling_efficiency
                ),
                "mean_latency_ratio_vs_c1": (
                    latency_ratio
                ),
            }
        )

        print(
            "concurrency=",
            concurrency,
            " throughput_speedup=",
            f"{throughput_speedup:.3f}x",
            " efficiency=",
            f"{scaling_efficiency:.3f}",
            " latency_ratio=",
            f"{latency_ratio:.3f}x",
            sep="",
        )

    payload: dict[
        str,
        Any,
    ] = {
        "schema_version": "0.1",
        "milestone": "M4-G",
        "experiment": (
            "verified-smollm2-"
            "provider-concurrency-scaling"
        ),
        "verify_first_optimise_second": (
            True
        ),
        "scope": {
            "optimisation_axis": (
                "concurrency"
            ),
            "batch_size": 1,
            "batching_varied": False,
            "cache_policy_varied": False,
            "workload_case": (
                "bank-01"
            ),
            "raw_model_global_admissibility": (
                False
            ),
            "evidence_interpretation": (
                "runtime-scaling evidence "
                "for one semantically verified "
                "deterministic workload case"
            ),
        },
        "model": {
            "model_id": (
                MODEL_ID
            ),
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
        "verification": (
            verification
        ),
        "series": (
            reports
        ),
        "relative_scaling": (
            comparison
        ),
        "limitations": [
            (
                "This experiment isolates provider "
                "concurrency and does not vary native "
                "batch size."
            ),
            (
                "This experiment does not establish "
                "global raw-model RAG admissibility."
            ),
            (
                "The workload is one previously "
                "semantically verified AP policy case."
            ),
            (
                "Application-level caching is not "
                "enabled or evaluated in this run."
            ),
            (
                "Transformer KV-cache policy is not "
                "varied in this run."
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
