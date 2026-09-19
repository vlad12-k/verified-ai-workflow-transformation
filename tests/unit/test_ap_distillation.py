"""Tests for AP knowledge-distillation primitives."""

import pytest
import torch

from vait.transformations.library.ap_distillation import (
    APDistillationStudent,
    distillation_loss,
    trainable_parameter_count,
)
from vait.transformations.library.ap_tabular import (
    AP_TABULAR_FEATURE_NAMES,
)
from vait.transformations.library.ap_torch import (
    APTabularNetwork,
)


def test_student_produces_three_decision_logits() -> None:
    """Student should preserve the AP three-class contract."""
    model = APDistillationStudent()

    features = torch.zeros(
        (
            4,
            len(AP_TABULAR_FEATURE_NAMES),
        ),
        dtype=torch.float32,
    )

    logits = model(features)

    assert logits.shape == (4, 3)


def test_student_is_smaller_than_teacher() -> None:
    """Distillation student should use fewer parameters than teacher."""
    teacher = APTabularNetwork()
    student = APDistillationStudent()

    teacher_parameters = trainable_parameter_count(
        teacher
    )
    student_parameters = trainable_parameter_count(
        student
    )

    assert student_parameters < teacher_parameters
    assert student_parameters > 0


def test_distillation_loss_is_finite_and_backpropagates() -> None:
    """Combined hard and soft supervision should train the student."""
    student = APDistillationStudent()

    features = torch.randn(
        (
            5,
            len(AP_TABULAR_FEATURE_NAMES),
        ),
        dtype=torch.float32,
    )

    teacher_logits = torch.randn(
        (5, 3),
        dtype=torch.float32,
        requires_grad=True,
    )

    hard_labels = torch.tensor(
        [0, 1, 2, 0, 1],
        dtype=torch.long,
    )

    student_logits = student(
        features
    )

    loss = distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        hard_labels=hard_labels,
        temperature=2.0,
        hard_label_weight=0.5,
    )

    assert torch.isfinite(loss)

    torch.autograd.backward(loss)

    student_gradients = [
        parameter.grad
        for parameter in student.parameters()
        if parameter.requires_grad
    ]

    assert all(
        gradient is not None
        for gradient in student_gradients
    )

    assert any(
        gradient is not None
        and torch.count_nonzero(
            gradient
        ).item()
        > 0
        for gradient in student_gradients
    )

    assert teacher_logits.grad is None


@pytest.mark.parametrize(
    "temperature",
    [
        0.0,
        -1.0,
    ],
)
def test_distillation_loss_rejects_invalid_temperature(
    temperature: float,
) -> None:
    """Temperature must remain strictly positive."""
    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        distillation_loss(
            student_logits=torch.zeros(
                (2, 3)
            ),
            teacher_logits=torch.zeros(
                (2, 3)
            ),
            hard_labels=torch.tensor(
                [0, 1]
            ),
            temperature=temperature,
        )


@pytest.mark.parametrize(
    "hard_label_weight",
    [
        -0.1,
        1.1,
    ],
)
def test_distillation_loss_rejects_invalid_hard_weight(
    hard_label_weight: float,
) -> None:
    """Hard-label mixing weight must remain a probability."""
    with pytest.raises(
        ValueError,
        match="hard_label_weight",
    ):
        distillation_loss(
            student_logits=torch.zeros(
                (2, 3)
            ),
            teacher_logits=torch.zeros(
                (2, 3)
            ),
            hard_labels=torch.tensor(
                [0, 1]
            ),
            hard_label_weight=hard_label_weight,
        )


def test_distillation_loss_rejects_mismatched_logits() -> None:
    """Teacher and student class tensors must align."""
    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        distillation_loss(
            student_logits=torch.zeros(
                (2, 3)
            ),
            teacher_logits=torch.zeros(
                (2, 2)
            ),
            hard_labels=torch.tensor(
                [0, 1]
            ),
        )


def test_hard_label_student_training_is_reproducible() -> None:
    """Hard-label training should reproduce the same fitted state."""
    from vait.transformations.library.ap_distillation import (
        train_hard_label_student,
    )

    features = torch.randn(
        (12, len(AP_TABULAR_FEATURE_NAMES)),
        generator=torch.Generator().manual_seed(123),
    )

    labels = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        dtype=torch.long,
    )

    first = train_hard_label_student(
        features=features,
        hard_labels=labels,
        seed=99,
        epochs=20,
    )

    second = train_hard_label_student(
        features=features,
        hard_labels=labels,
        seed=99,
        epochs=20,
    )

    assert first.final_loss == pytest.approx(
        second.final_loss
    )

    for first_parameter, second_parameter in zip(
        first.model.parameters(),
        second.model.parameters(),
        strict=True,
    ):
        assert torch.equal(
            first_parameter,
            second_parameter,
        )


def test_distilled_student_training_is_reproducible() -> None:
    """Distilled training should reproduce the same fitted state."""
    from vait.transformations.library.ap_distillation import (
        train_distilled_student,
    )

    generator = torch.Generator().manual_seed(321)

    features = torch.randn(
        (12, len(AP_TABULAR_FEATURE_NAMES)),
        generator=generator,
    )

    teacher_logits = torch.randn(
        (12, 3),
        generator=generator,
    )

    labels = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        dtype=torch.long,
    )

    first = train_distilled_student(
        features=features,
        hard_labels=labels,
        teacher_logits=teacher_logits,
        seed=99,
        epochs=20,
    )

    second = train_distilled_student(
        features=features,
        hard_labels=labels,
        teacher_logits=teacher_logits,
        seed=99,
        epochs=20,
    )

    assert first.final_loss == pytest.approx(
        second.final_loss
    )

    for first_parameter, second_parameter in zip(
        first.model.parameters(),
        second.model.parameters(),
        strict=True,
    ):
        assert torch.equal(
            first_parameter,
            second_parameter,
        )


def test_student_training_rejects_invalid_optimisation_settings() -> None:
    """Training should reject unusable optimisation configuration."""
    from vait.transformations.library.ap_distillation import (
        train_distilled_student,
        train_hard_label_student,
    )

    features = torch.zeros(
        (
            2,
            len(AP_TABULAR_FEATURE_NAMES),
        )
    )

    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    teacher_logits = torch.zeros(
        (2, 3)
    )

    with pytest.raises(
        ValueError,
        match="epochs",
    ):
        train_hard_label_student(
            features=features,
            hard_labels=labels,
            seed=1,
            epochs=0,
        )

    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        train_distilled_student(
            features=features,
            hard_labels=labels,
            teacher_logits=teacher_logits,
            seed=1,
            learning_rate=0.0,
        )
