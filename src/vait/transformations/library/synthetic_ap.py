"""Synthetic AP benchmark transformation used for VAIT applied evaluation."""

from pydantic import JsonValue

from vait.contracts.models import Effect, RiskLevel
from vait.transformations.models import (
    TransformationCategory,
    TransformationDescriptor,
)
from vait.transformations.python_rule import PythonRuleTransformation


def synthetic_ap_policy(data: dict[str, JsonValue]) -> JsonValue:
    """Apply the deterministic synthetic AP decision policy."""
    invoice_amount = data["invoice_amount"]
    purchase_order_amount = data["purchase_order_amount"]

    purchase_order_present = data["purchase_order_present"]
    supplier_known = data["supplier_known"]
    duplicate_invoice = data["duplicate_invoice"]
    bank_details_changed = data["bank_details_changed"]
    tax_id_valid = data["tax_id_valid"]

    if not isinstance(invoice_amount, (int, float)) or isinstance(
        invoice_amount,
        bool,
    ):
        raise ValueError("invoice_amount must be numeric")

    if bank_details_changed is True or duplicate_invoice is True:
        return {"decision": "HOLD"}

    if invoice_amount <= 0:
        return {"decision": "REVIEW"}

    if tax_id_valid is not True:
        return {"decision": "REVIEW"}

    if purchase_order_present is not True:
        return {"decision": "REVIEW"}

    if supplier_known is not True:
        return {"decision": "REVIEW"}

    if not isinstance(
        purchase_order_amount,
        (int, float),
    ) or isinstance(purchase_order_amount, bool):
        return {"decision": "REVIEW"}

    difference = abs(invoice_amount - purchase_order_amount)

    if difference >= 1000:
        return {"decision": "HOLD"}

    if difference > 0:
        return {"decision": "REVIEW"}

    return {"decision": "RECOMMEND_APPROVE"}


def build_synthetic_ap_transformation() -> PythonRuleTransformation:
    """Build the supplied deterministic AP benchmark transformation."""
    descriptor = TransformationDescriptor(
        transformation_id="synthetic-ap-rule-replacement",
        version="1.0.0",
        name="Synthetic AP deterministic rule replacement",
        description=(
            "Supplied deterministic Python rule candidate for the "
            "synthetic AP invoice-exception benchmark."
        ),
        category=TransformationCategory.OPTIMIZATION,
        declared_effects=frozenset({Effect.NONE}),
        supported_risk_levels=frozenset(RiskLevel),
        required_capabilities=frozenset({"typed-inputs"}),
    )

    return PythonRuleTransformation(
        descriptor=descriptor,
        implementation_id="synthetic-ap-policy-v0.2",
        function=synthetic_ap_policy,
    )
