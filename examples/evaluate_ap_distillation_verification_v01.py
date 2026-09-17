"""Verify AP teacher and compact students on benchmark v0.2."""

from pathlib import Path

from vait.benchmark.loader import load_benchmark
from vait.benchmark.reporting import write_benchmark_report
from vait.benchmark.transformation_assessment import (
    assess_python_transformations,
)
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
)
from vait.transformations.applicability import (
    ApplicabilityContext,
)
from vait.transformations.library.ap_distillation_candidates import (
    build_ap_distillation_transformations,
)
from vait.transformations.library.ap_torch import (
    build_pytorch_transformation,
)
from vait.transformations.library.ap_training import (
    build_synthetic_ap_training_corpus,
)

DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

OUTPUT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-distillation-verification-v0.1.json"
)

dataset = load_benchmark(
    DATASET_PATH
)

corpus = build_synthetic_ap_training_corpus(
    excluded_inputs=tuple(
        case.input_data
        for case in dataset.cases
    )
)

teacher = build_pytorch_transformation(
    corpus
)

hard_student, distilled_student = (
    build_ap_distillation_transformations(
        corpus
    )
)

context = ApplicabilityContext(
    risk_level=RiskLevel.CRITICAL,
    available_capabilities=frozenset(
        {
            "typed-inputs",
            "tabular-features",
            "trained-model",
            "pytorch-inference",
        }
    ),
)

policy = BoundedVerificationPolicy(
    max_overall_disagreement_rate=0.15,
    max_high_risk_disagreement_rate=0.25,
    confidence_level=0.95,
    min_total_cases=20,
    min_high_risk_cases=11,
)

report = assess_python_transformations(
    dataset=dataset,
    transformations=(
        teacher,
        hard_student,
        distilled_student,
    ),
    context=context,
    policy=policy,
)

report_path = write_benchmark_report(
    report,
    OUTPUT_PATH,
)

print("AP distillation bounded verification")
print()
print(
    "Benchmark:",
    f"{report.benchmark_id}@{report.benchmark_version}",
)
print(
    "Candidates:",
    len(report.assessments),
)
print()

for assessment in report.assessments:
    print(
        assessment.candidate_implementation_id
    )

    print(
        "  applicable:",
        assessment.applicability.is_applicable,
    )

    verification = assessment.verification

    if verification is None:
        print(
            "  decision: NOT_APPLICABLE"
        )
        print()
        continue

    print(
        "  decision:",
        verification.decision.value,
    )

    print(
        "  cases:",
        verification.cases_evaluated,
    )

    evidence = verification.statistical_evidence

    if evidence is not None:
        print(
            "  disagreements:",
            evidence.disagreements,
        )

        print(
            "  overall rate:",
            f"{evidence.disagreement_rate:.4f}",
        )

        print(
            "  overall upper bound:",
            f"{evidence.disagreement_upper_bound:.4f}",
        )

        print(
            "  high-risk cases:",
            evidence.high_risk_cases_evaluated,
        )

        print(
            "  high-risk disagreements:",
            evidence.high_risk_disagreements,
        )

        print(
            "  high-risk rate:",
            f"{evidence.high_risk_disagreement_rate:.4f}",
        )

        if (
            evidence.high_risk_disagreement_upper_bound
            is None
        ):
            print(
                "  high-risk upper bound: n/a"
            )
        else:
            print(
                "  high-risk upper bound:",
                f"{evidence.high_risk_disagreement_upper_bound:.4f}",
            )

    latency = verification.candidate_latency

    if latency is not None:
        print(
            "  mean latency:",
            f"{latency.mean_ms:.4f} ms",
        )

        print(
            "  p50 latency:",
            f"{latency.p50_ms:.4f} ms",
        )

        print(
            "  p95 latency:",
            f"{latency.p95_ms:.4f} ms",
        )

        print(
            "  max latency:",
            f"{latency.max_ms:.4f} ms",
        )

    print(
        "  failures:",
        len(verification.failures),
    )

    print()

print(
    "Evidence report:",
    report_path,
)
