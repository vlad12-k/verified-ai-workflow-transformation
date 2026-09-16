"""Domain models for versioned VAIT benchmark datasets and reports."""

from pydantic import BaseModel, Field, JsonValue, model_validator

from vait.contracts.models import RiskLevel


class BenchmarkCase(BaseModel):
    """One labelled benchmark case."""

    case_id: str = Field(min_length=1)
    input_data: dict[str, JsonValue]
    gold_outcome: dict[str, JsonValue]

    risk_level: RiskLevel
    tags: frozenset[str] = Field(default_factory=frozenset)
    rationale: str = Field(min_length=1)


class BenchmarkDataset(BaseModel):
    """A versioned collection of labelled benchmark cases."""

    benchmark_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)

    cases: list[BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> "BenchmarkDataset":
        """Validate dataset-wide benchmark invariants."""
        case_ids = [case.case_id for case in self.cases]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Benchmark case IDs must be unique.")

        return self


class BenchmarkCaseResult(BaseModel):
    """Evaluation result for one benchmark case."""

    case_id: str
    risk_level: RiskLevel

    gold_outcome: dict[str, JsonValue]
    candidate_output: JsonValue | None = None

    matched_gold: bool
    latency_ms: float = Field(ge=0.0)
    error: str | None = None


class BenchmarkReport(BaseModel):
    """Machine-readable benchmark evaluation report."""

    benchmark_id: str
    benchmark_version: str
    implementation_id: str

    cases_evaluated: int = Field(ge=0)
    gold_matches: int = Field(ge=0)
    gold_agreement_rate: float = Field(ge=0.0, le=1.0)

    high_risk_cases_evaluated: int = Field(ge=0)
    high_risk_failures: int = Field(ge=0)
    high_risk_agreement_rate: float = Field(ge=0.0, le=1.0)

    execution_errors: int = Field(ge=0)

    case_results: list[BenchmarkCaseResult] = Field(default_factory=list)
