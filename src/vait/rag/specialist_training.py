"""Versioned instruction-tuning data for AP policy specialists."""

import hashlib
import json

from pydantic import BaseModel, Field


class SpecialistTrainingExample(BaseModel):
    """One instruction-tuning example for grounded AP policy reasoning."""

    example_id: str = Field(min_length=1)
    split: str = Field(pattern=r"^(train|dev)$")
    policy_context: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class SpecialistTrainingCorpus(BaseModel):
    """Versioned specialist instruction-tuning corpus."""

    corpus_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    examples: tuple[SpecialistTrainingExample, ...] = Field(
        min_length=2
    )

    @property
    def train_examples(
        self,
    ) -> tuple[SpecialistTrainingExample, ...]:
        """Return training examples."""
        return tuple(
            example
            for example in self.examples
            if example.split == "train"
        )

    @property
    def dev_examples(
        self,
    ) -> tuple[SpecialistTrainingExample, ...]:
        """Return development examples."""
        return tuple(
            example
            for example in self.examples
            if example.split == "dev"
        )

    @property
    def sha256(self) -> str:
        """Return deterministic corpus fingerprint."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()


def build_ap_policy_specialist_corpus(
) -> SpecialistTrainingCorpus:
    """Build the bounded AP policy specialist corpus."""
    examples = (
        SpecialistTrainingExample(
            example_id="train-duplicate-hold",
            split="train",
            policy_context=(
                "Potential duplicate supplier invoices must be "
                "held for investigation before payment is approved."
            ),
            question=(
                "How should a suspected duplicate supplier invoice "
                "be treated before payment?"
            ),
            answer=(
                "It must be held for investigation before payment "
                "is approved."
            ),
        ),
        SpecialistTrainingExample(
            example_id="train-bank-verification",
            split="train",
            policy_context=(
                "A supplier bank account change requires "
                "independent verification before payment."
            ),
            question=(
                "What control applies when supplier banking details "
                "have recently changed?"
            ),
            answer=(
                "Independent verification is required before payment."
            ),
        ),
        SpecialistTrainingExample(
            example_id="train-missing-po-exception",
            split="train",
            policy_context=(
                "Invoices without a purchase order reference require "
                "manual review unless an approved exception applies."
            ),
            question=(
                "What is required for an invoice that lacks a PO "
                "reference when no approved exception exists?"
            ),
            answer=(
                "The invoice requires manual review."
            ),
        ),
        SpecialistTrainingExample(
            example_id="train-mismatch-review",
            split="train",
            policy_context=(
                "Any non-zero difference between an invoice amount "
                "and purchase order amount requires review. Large "
                "differences may require escalation."
            ),
            question=(
                "What does the policy require for a small non-zero "
                "invoice and PO amount difference?"
            ),
            answer=(
                "The difference requires review."
            ),
        ),
        SpecialistTrainingExample(
            example_id="train-mismatch-escalation",
            split="train",
            policy_context=(
                "Any non-zero difference between an invoice amount "
                "and purchase order amount requires review. Large "
                "differences may require escalation."
            ),
            question=(
                "How does the policy treat a large invoice and PO "
                "amount difference?"
            ),
            answer=(
                "It requires review and may require escalation."
            ),
        ),
        SpecialistTrainingExample(
            example_id="train-unknown-supplier",
            split="train",
            policy_context=(
                "Invoices from vendors not in the approved supplier "
                "master must be reviewed before processing."
            ),
            question=(
                "What control applies to an invoice from an "
                "unapproved vendor?"
            ),
            answer=(
                "It must be reviewed before processing."
            ),
        ),
        SpecialistTrainingExample(
            example_id="train-tax-id",
            split="train",
            policy_context=(
                "An invalid or unverifiable supplier tax identifier "
                "requires invoice review before approval."
            ),
            question=(
                "What action is required if a supplier tax ID is "
                "invalid?"
            ),
            answer=(
                "The invoice requires review before approval."
            ),
        ),
        SpecialistTrainingExample(
            example_id="train-non-positive",
            split="train",
            policy_context=(
                "Invoices with zero or negative payable amounts "
                "require review because they may represent credits, "
                "corrections, or invalid transactions."
            ),
            question=(
                "What control applies to an invoice with a negative "
                "payable amount?"
            ),
            answer=(
                "The invoice requires review."
            ),
        ),
        SpecialistTrainingExample(
            example_id="train-approval-threshold",
            split="train",
            policy_context=(
                "Invoices above the delegated approval threshold "
                "require authorization from an appropriately senior "
                "approver."
            ),
            question=(
                "What authorization is required when an invoice is "
                "above the delegated approval threshold?"
            ),
            answer=(
                "Authorization from an appropriately senior approver "
                "is required."
            ),
        ),
        SpecialistTrainingExample(
            example_id="train-abstain-hr",
            split="train",
            policy_context=(
                "Duplicate invoices must be reviewed before payment."
            ),
            question=(
                "What is the annual leave carry-over allowance?"
            ),
            answer="INSUFFICIENT_EVIDENCE",
        ),
        SpecialistTrainingExample(
            example_id="train-abstain-travel",
            split="train",
            policy_context=(
                "Supplier bank account changes require independent "
                "verification before payment."
            ),
            question=(
                "Which hotel chains are approved for business travel?"
            ),
            answer="INSUFFICIENT_EVIDENCE",
        ),
        SpecialistTrainingExample(
            example_id="dev-conditional-escalation",
            split="dev",
            policy_context=(
                "Any amount difference requires review. Material "
                "differences may also require escalation."
            ),
            question=(
                "Does every amount difference require escalation?"
            ),
            answer=(
                "No. Every amount difference requires review, while "
                "material differences may also require escalation."
            ),
        ),
        SpecialistTrainingExample(
            example_id="dev-exception-preservation",
            split="dev",
            policy_context=(
                "Invoices without a purchase order require manual "
                "review unless an approved exception applies."
            ),
            question=(
                "Is manual review always required when a PO is "
                "missing?"
            ),
            answer=(
                "No. Manual review is required unless an approved "
                "exception applies."
            ),
        ),
        SpecialistTrainingExample(
            example_id="dev-abstain-equipment",
            split="dev",
            policy_context=(
                "Invoices from unknown suppliers require review."
            ),
            question=(
                "How often should employee laptops be replaced?"
            ),
            answer="INSUFFICIENT_EVIDENCE",
        ),
    )

    return SpecialistTrainingCorpus(
        corpus_id="ap-policy-specialist",
        version="0.1",
        examples=examples,
    )
