"""Evaluation of implementations against versioned VAIT benchmarks."""

from vait.benchmark.models import (
    BenchmarkCaseResult,
    BenchmarkDataset,
    BenchmarkReport,
)
from vait.contracts.models import RiskLevel, VerificationCase
from vait.runners.base import ImplementationRunner

_HIGH_RISK_LEVELS = frozenset({RiskLevel.HIGH, RiskLevel.CRITICAL})


def evaluate_benchmark(
    dataset: BenchmarkDataset,
    candidate: ImplementationRunner,
) -> BenchmarkReport:
    """Evaluate one implementation against benchmark gold outcomes."""
    case_results: list[BenchmarkCaseResult] = []

    gold_matches = 0
    high_risk_cases = 0
    high_risk_failures = 0
    execution_errors = 0

    for benchmark_case in dataset.cases:
        verification_case = VerificationCase(
            id=benchmark_case.case_id,
            input_data=benchmark_case.input_data,
            risk_level=benchmark_case.risk_level,
        )

        observation = candidate.execute(verification_case)

        matched_gold = (
            observation.error is None
            and observation.output == benchmark_case.gold_outcome
        )

        if matched_gold:
            gold_matches += 1

        if observation.error is not None:
            execution_errors += 1

        if benchmark_case.risk_level in _HIGH_RISK_LEVELS:
            high_risk_cases += 1

            if not matched_gold:
                high_risk_failures += 1

        case_results.append(
            BenchmarkCaseResult(
                case_id=benchmark_case.case_id,
                risk_level=benchmark_case.risk_level,
                gold_outcome=benchmark_case.gold_outcome,
                candidate_output=observation.output,
                matched_gold=matched_gold,
                latency_ms=observation.latency_ms,
                error=observation.error,
            )
        )

    total_cases = len(dataset.cases)

    high_risk_agreement_rate = (
        (high_risk_cases - high_risk_failures) / high_risk_cases
        if high_risk_cases > 0
        else 1.0
    )

    return BenchmarkReport(
        benchmark_id=dataset.benchmark_id,
        benchmark_version=dataset.version,
        implementation_id=candidate.implementation_id,
        cases_evaluated=total_cases,
        gold_matches=gold_matches,
        gold_agreement_rate=gold_matches / total_cases,
        high_risk_cases_evaluated=high_risk_cases,
        high_risk_failures=high_risk_failures,
        high_risk_agreement_rate=high_risk_agreement_rate,
        execution_errors=execution_errors,
        case_results=case_results,
    )
