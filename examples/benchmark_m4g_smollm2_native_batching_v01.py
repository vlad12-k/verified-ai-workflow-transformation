"""Benchmark verified SmolLM2 native generative batch scaling."""

import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from vait.inference.models import InferenceConfiguration
from vait.inference.providers.benchmark import (
    run_provider_native_batch_generative_benchmark,
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
from vait.rag.semantic_guard import assess_semantic_drift

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

MODEL_REVISION = (
    "12fd25f77366fa6b3b4b768ec3050bf629380bac"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "m4g-smollm2-native-batching-v0.1.json"
)

QUERY = (
    "What must happen before paying a supplier "
    "after its bank account changes?"
)

CONTEXT = (
    "A supplier bank account change requires independent "
    "verification before any payment is released."
)

BATCH_SIZES = (
    1,
    2,
    4,
    8,
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
                document_id="bank-change-policy",
                text=CONTEXT,
                score=1.0,
                rank=1,
            ),
        ),
    )

    rag_messages = build_rag_messages(
        rag_request,
        context_documents=rag_request.context_documents,
        contract=RAGPromptContract.STRICT_V2,
    )

    messages = tuple(
        InferenceMessage(
            role=InferenceRole(
                message["role"]
            ),
            content=message["content"],
        )
        for message in rag_messages
    )

    return ProviderInferenceRequest(
        request_id=request_id,
        model_id=MODEL_ID,
        messages=messages,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0,
        metadata={
            "milestone": "M4-G",
            "experiment": (
                "native-batch-scaling"
            ),
            "source_case": "bank-01",
            "prompt_contract": (
                RAGPromptContract.STRICT_V2.value
            ),
            "model_revision": MODEL_REVISION,
        },
    )


def require_rate(
    value: float | None,
    *,
    label: str,
) -> float:
    """Return one required throughput rate."""
    if value is None:
        raise RuntimeError(
            f"Missing throughput value: {label}"
        )

    return value


def require_number(
    value: object,
    *,
    label: str,
) -> float:
    """Narrow one machine-readable numeric evidence value."""
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
            f"Expected numeric evidence: {label}"
        )

    return float(
        value
    )


def verify_native_batch_equivalence(
    *,
    provider: LocalHuggingFaceProvider,
) -> dict[str, Any]:
    """Verify batching preserves scalar deterministic semantics."""
    scalar_request = build_request(
        request_id="m4g-batch-scalar-reference"
    )

    scalar_response = provider.generate(
        scalar_request
    )

    if not scalar_response.succeeded:
        raise RuntimeError(
            "Scalar verification request failed: "
            f"{scalar_response.error}"
        )

    if scalar_response.output_text is None:
        raise RuntimeError(
            "Scalar verification response "
            "has no output text."
        )

    scalar_answer = (
        scalar_response.output_text.strip()
    )

    scalar_assessment = (
        assess_semantic_drift(
            context_text=CONTEXT,
            answer_text=scalar_answer,
        )
    )

    if not scalar_assessment.passed:
        raise RuntimeError(
            "Scalar semantic verification failed."
        )

    batch_checks: list[
        dict[str, Any]
    ] = []

    for batch_size in BATCH_SIZES:
        requests = tuple(
            build_request(
                request_id=(
                    "m4g-batch-precheck-"
                    f"{batch_size}-"
                    f"{item_index}"
                )
            )
            for item_index in range(
                batch_size
            )
        )

        responses = provider.generate_batch(
            requests
        )

        if len(
            responses
        ) != batch_size:
            raise RuntimeError(
                "Native batch verification returned "
                "an unexpected response count."
            )

        outputs: list[
            str
        ] = []

        for response in responses:
            if not response.succeeded:
                raise RuntimeError(
                    "Native batch verification "
                    f"failed: {response.error}"
                )

            if response.output_text is None:
                raise RuntimeError(
                    "Native batch verification "
                    "response has no output text."
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
                risks = [
                    issue.risk.value
                    for issue
                    in assessment.issues
                ]

                raise RuntimeError(
                    "Native batch semantic gate "
                    f"failed at batch_size="
                    f"{batch_size}: {risks}"
                )

            outputs.append(
                answer
            )

        outputs_identical = (
            len(
                set(
                    outputs
                )
            )
            == 1
        )

        matches_scalar = all(
            output
            == scalar_answer
            for output in outputs
        )

        if not outputs_identical:
            raise RuntimeError(
                "Native batch determinism gate "
                f"failed at batch_size={batch_size}."
            )

        if not matches_scalar:
            raise RuntimeError(
                "Native batching changed scalar "
                f"generation at batch_size={batch_size}."
            )

        batch_checks.append(
            {
                "batch_size": batch_size,
                "responses": len(
                    responses
                ),
                "outputs_identical": (
                    outputs_identical
                ),
                "matches_scalar": (
                    matches_scalar
                ),
                "semantic_pass": True,
            }
        )

    return {
        "case_id": "bank-01",
        "scalar_semantic_pass": True,
        "scalar_answer": scalar_answer,
        "batch_checks": batch_checks,
    }


def main() -> None:
    """Run verified native generative batch scaling."""
    print(
        "=== M4-G VERIFIED NATIVE BATCH SCALING ==="
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

    provider = LocalHuggingFaceProvider(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        local_files_only=True,
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
        verify_native_batch_equivalence(
            provider=provider
        )
    )

    print(
        "scalar_semantic_pass:",
        verification[
            "scalar_semantic_pass"
        ],
    )

    print(
        "scalar_answer:",
        verification[
            "scalar_answer"
        ],
    )

    for check in verification[
        "batch_checks"
    ]:
        print(
            "batch_size=",
            check[
                "batch_size"
            ],
            " semantic_pass=",
            check[
                "semantic_pass"
            ],
            " outputs_identical=",
            check[
                "outputs_identical"
            ],
            " matches_scalar=",
            check[
                "matches_scalar"
            ],
            sep="",
        )

    base_request = build_request(
        request_id="m4g-bank-native-batch"
    )

    reports: list[
        dict[str, Any]
    ] = []

    print()
    print(
        "=== NATIVE BATCH SERIES ==="
    )

    for batch_size in BATCH_SIZES:
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
                batch_size=batch_size,
                model_id=MODEL_ID,
                model_revision=(
                    MODEL_REVISION
                ),
                metadata={
                    "execution_axis": (
                        "native-batching"
                    ),
                    "concurrency_varied": (
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

        report = (
            run_provider_native_batch_generative_benchmark(
                provider=provider,
                request=base_request,
                configuration=(
                    configuration
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
                        "native-batch-scaling"
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

        request_equivalent_latency = (
            require_number(
                report.evidence_metadata[
                    "request_equivalent_mean_latency_ms"
                ],
                label=(
                    "request_equivalent_mean_latency_ms"
                ),
            )
        )

        print()
        print(
            "batch_size:",
            batch_size,
        )

        print(
            "measured_batch_invocations:",
            report.measured_iterations,
        )

        print(
            "measured_requests:",
            report.evidence_metadata[
                "measured_requests"
            ],
        )

        print(
            "mean_batch_latency_ms:",
            f"{report.latency.mean_ms:.3f}",
        )

        print(
            "p95_batch_latency_ms:",
            f"{report.latency.p95_ms:.3f}",
        )

        print(
            "request_equivalent_mean_latency_ms:",
            f"{request_equivalent_latency:.3f}",
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
                "batch_size": batch_size,
                "requests_per_second": (
                    requests_per_second
                ),
                "tokens_per_second": (
                    tokens_per_second
                ),
                "mean_batch_latency_ms": (
                    report.latency.mean_ms
                ),
                "request_equivalent_mean_latency_ms": (
                    request_equivalent_latency
                ),
                "report": report.model_dump(
                    mode="json"
                ),
            }
        )

    baseline = reports[
        0
    ]

    baseline_throughput = float(
        baseline[
            "requests_per_second"
        ]
    )

    baseline_batch_latency = float(
        baseline[
            "mean_batch_latency_ms"
        ]
    )

    baseline_request_equivalent_latency = float(
        baseline[
            "request_equivalent_mean_latency_ms"
        ]
    )

    relative_scaling: list[
        dict[str, Any]
    ] = []

    print()
    print(
        "=== RELATIVE SCALING ==="
    )

    for result in reports:
        batch_size = int(
            result[
                "batch_size"
            ]
        )

        throughput = float(
            result[
                "requests_per_second"
            ]
        )

        batch_latency = float(
            result[
                "mean_batch_latency_ms"
            ]
        )

        request_equivalent_latency = float(
            result[
                "request_equivalent_mean_latency_ms"
            ]
        )

        throughput_speedup = (
            throughput
            / baseline_throughput
        )

        scaling_efficiency = (
            throughput_speedup
            / batch_size
        )

        batch_latency_ratio = (
            batch_latency
            / baseline_batch_latency
        )

        request_equivalent_latency_ratio = (
            request_equivalent_latency
            / baseline_request_equivalent_latency
        )

        relative_scaling.append(
            {
                "batch_size": batch_size,
                "throughput_speedup_vs_b1": (
                    throughput_speedup
                ),
                "scaling_efficiency": (
                    scaling_efficiency
                ),
                "batch_latency_ratio_vs_b1": (
                    batch_latency_ratio
                ),
                "request_equivalent_latency_ratio_vs_b1": (
                    request_equivalent_latency_ratio
                ),
            }
        )

        print(
            "batch_size=",
            batch_size,
            " throughput_speedup=",
            f"{throughput_speedup:.3f}x",
            " efficiency=",
            f"{scaling_efficiency:.3f}",
            " batch_latency_ratio=",
            f"{batch_latency_ratio:.3f}x",
            " request_equivalent_latency_ratio=",
            f"{request_equivalent_latency_ratio:.3f}x",
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
            "native-batch-scaling"
        ),
        "verify_first_optimise_second": (
            True
        ),
        "scope": {
            "optimisation_axis": (
                "native-batching"
            ),
            "batch_sizes": list(
                BATCH_SIZES
            ),
            "concurrency_varied": False,
            "cache_policy_varied": False,
            "workload_case": "bank-01",
            "raw_model_global_admissibility": (
                False
            ),
            "evidence_interpretation": (
                "runtime native-batch scaling "
                "for one semantically verified "
                "deterministic workload case"
            ),
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
        "verification": verification,
        "series": reports,
        "relative_scaling": (
            relative_scaling
        ),
        "limitations": [
            (
                "This experiment isolates genuine "
                "native batching and does not vary "
                "request concurrency."
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
