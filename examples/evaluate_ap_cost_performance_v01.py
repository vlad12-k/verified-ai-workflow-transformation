"""Build controlled AP cost and performance evidence for M2."""

from pathlib import Path

from vait.benchmark.loader import load_benchmark
from vait.benchmark.reporting import write_benchmark_report
from vait.benchmark.verification import verify_benchmark_bounded
from vait.contracts.models import (
    BoundedVerificationPolicy,
    RiskLevel,
)
from vait.economics.metrics import (
    CostEstimate,
    CostEvidenceKind,
)
from vait.transformations.applicability import ApplicabilityContext
from vait.transformations.library.synthetic_ap import (
    build_synthetic_ap_transformation,
)
from vait.transformations.registry import TransformationRegistry

DATASET_PATH = Path(
    "datasets/ap_invoice_exceptions/v0.2/cases.yaml"
)

ARTIFACT_PATH = Path(
    "artifacts/benchmarks/"
    "ap-v0.2-cost-performance-evidence-v0.1.json"
)


def build_declared_costs() -> tuple[CostEstimate, CostEstimate]:
    """Create controlled development cost assumptions.

    These values are scenario inputs for validating VAIT's economics
    evidence pipeline. They are not provider prices or production ROI
    claims.
    """
    common_metadata = {
        "scope": "controlled-development-only",
        "production_price_claim": False,
        "provider_price_claim": False,
    }

    reference = CostEstimate(
        amount_per_case=0.10,
        currency="USD",
        evidence_kind=CostEvidenceKind.DECLARED,
        source="controlled-m2-cost-scenario",
        pricing_version="scenario-v0.1",
        metadata=common_metadata,
    )

    candidate = CostEstimate(
        amount_per_case=0.02,
        currency="USD",
        evidence_kind=CostEvidenceKind.DECLARED,
        source="controlled-m2-cost-scenario",
        pricing_version="scenario-v0.1",
        metadata=common_metadata,
    )

    return reference, candidate


def main() -> None:
    """Run bounded verification with controlled economic evidence."""
    dataset = load_benchmark(DATASET_PATH)

    transformation = build_synthetic_ap_transformation()

    registry = TransformationRegistry()
    registry.register(transformation)

    registered = registry.get(
        transformation.descriptor.transformation_id,
        transformation.descriptor.version,
    )

    preparation = registered.prepare_candidate(
        ApplicabilityContext(
            risk_level=RiskLevel.CRITICAL,
            available_capabilities=frozenset(
                {"typed-inputs"}
            ),
        )
    )

    if preparation.runner is None:
        reasons = ", ".join(
            issue.code.value
            for issue in preparation.applicability.issues
        )
        raise RuntimeError(
            f"Transformation is not applicable: {reasons}"
        )

    reference_cost, candidate_cost = build_declared_costs()

    candidate_configuration = {
        "transformation_id": (
            transformation.descriptor.transformation_id
        ),
        "transformation_version": (
            transformation.descriptor.version
        ),
        "evidence_scope": "controlled-development-only",
    }

    verification = verify_benchmark_bounded(
        dataset=dataset,
        candidate=preparation.runner,
        policy=BoundedVerificationPolicy(
            max_overall_disagreement_rate=0.15,
            max_high_risk_disagreement_rate=0.25,
            confidence_level=0.95,
            min_total_cases=20,
            min_high_risk_cases=11,
        ),
        candidate_configuration=candidate_configuration,
        reference_cost=reference_cost,
        candidate_cost=candidate_cost,
    )

    economic = verification.economic_evidence

    if economic is None:
        raise RuntimeError(
            "Expected economic evidence was not produced."
        )

    if economic.cost is None:
        raise RuntimeError(
            "Expected cost comparison was not produced."
        )

    report_path = write_benchmark_report(
        verification,
        ARTIFACT_PATH,
    )

    print("M2 cost/performance evidence")
    print()
    print(f"Benchmark: {verification.benchmark_id}")
    print(f"Version: {verification.benchmark_version}")
    print(f"Cases: {verification.cases_evaluated}")
    print(f"Decision: {verification.decision.value}")
    print()
    print(
        "Declared reference cost/case: "
        f"{economic.cost.reference.amount_per_case:.4f} "
        f"{economic.cost.reference.currency}"
    )
    print(
        "Declared candidate cost/case: "
        f"{economic.cost.candidate.amount_per_case:.4f} "
        f"{economic.cost.candidate.currency}"
    )
    print(
        "Cost delta/case: "
        f"{economic.cost.delta_per_case:.4f} "
        f"{economic.cost.reference.currency}"
    )

    if economic.cost.relative_delta is not None:
        print(
            "Relative cost delta: "
            f"{economic.cost.relative_delta:.1%}"
        )

    print()

    candidate_latency = verification.candidate_latency

    if candidate_latency is None:
        raise RuntimeError(
            "Expected candidate latency evidence was not produced."
        )

    print(
        "Candidate p50 latency: "
        f"{candidate_latency.p50_ms:.6f} ms"
    )
    print(
        "Candidate p95 latency: "
        f"{candidate_latency.p95_ms:.6f} ms"
    )
    print(
        "Reference latency: not measured; benchmark gold labels "
        "are not a timed reference implementation."
    )
    print(
        "Comparative latency delta: not reported."
    )
    print()
    print(
        "Scope: controlled development assumptions only; "
        "no production pricing or ROI claim."
    )
    print(f"Evidence report: {report_path}")


if __name__ == "__main__":
    main()
