"""Knowledge-distillation primitives for synthetic AP classifiers."""

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vait.transformations.library.ap_tabular import (
    AP_TABULAR_FEATURE_NAMES,
)

_DECISION_CLASS_COUNT = 3

_STUDENT_HIDDEN_DIMENSION = 8


class APDistillationStudent(nn.Module):
    """Compact student network for AP decision distillation."""

    def __init__(self) -> None:
        """Initialize the compact student classifier."""
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                len(AP_TABULAR_FEATURE_NAMES),
                _STUDENT_HIDDEN_DIMENSION,
            ),
            nn.ReLU(),
            nn.Linear(
                _STUDENT_HIDDEN_DIMENSION,
                _DECISION_CLASS_COUNT,
            ),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """Return student decision logits."""
        return cast(
            Tensor,
            self.network(features),
        )


def distillation_loss(
    *,
    student_logits: Tensor,
    teacher_logits: Tensor,
    hard_labels: Tensor,
    temperature: float = 2.0,
    hard_label_weight: float = 0.5,
) -> Tensor:
    """Combine hard-label supervision with teacher soft targets."""
    if temperature <= 0.0:
        raise ValueError(
            "temperature must be greater than zero."
        )

    if not 0.0 <= hard_label_weight <= 1.0:
        raise ValueError(
            "hard_label_weight must be between 0.0 and 1.0."
        )

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher logits must have identical shapes."
        )

    if student_logits.ndim != 2:
        raise ValueError(
            "Logits must have shape [batch, classes]."
        )

    if hard_labels.ndim != 1:
        raise ValueError(
            "hard_labels must have shape [batch]."
        )

    if hard_labels.shape[0] != student_logits.shape[0]:
        raise ValueError(
            "hard_labels batch size must match logits."
        )

    hard_loss = F.cross_entropy(
        student_logits,
        hard_labels,
    )

    scaled_student = F.log_softmax(
        student_logits / temperature,
        dim=1,
    )

    scaled_teacher = F.softmax(
        teacher_logits.detach() / temperature,
        dim=1,
    )

    soft_loss = F.kl_div(
        scaled_student,
        scaled_teacher,
        reduction="batchmean",
    ) * (temperature**2)

    return (
        hard_label_weight * hard_loss
        + (1.0 - hard_label_weight) * soft_loss
    )


def trainable_parameter_count(
    model: nn.Module,
) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


class APDistillationTrainingResult:
    """Evidence from one bounded AP student-training experiment."""

    def __init__(
        self,
        *,
        model: APDistillationStudent,
        final_loss: float,
        epochs: int,
        learning_rate: float,
        temperature: float | None,
        hard_label_weight: float,
    ) -> None:
        """Store trained student and reproducible training metadata."""
        self.model = model
        self.final_loss = final_loss
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.hard_label_weight = hard_label_weight


def train_hard_label_student(
    *,
    features: Tensor,
    hard_labels: Tensor,
    seed: int,
    epochs: int = 500,
    learning_rate: float = 0.01,
) -> APDistillationTrainingResult:
    """Train the compact student from hard labels only."""
    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    torch.manual_seed(seed)

    model = APDistillationStudent()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    model.train()

    final_loss = 0.0

    for _ in range(epochs):
        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            features
        )

        loss = F.cross_entropy(
            logits,
            hard_labels,
        )

        torch.autograd.backward(loss)
        optimizer.step()

        final_loss = float(
            loss.detach().item()
        )

    model.eval()

    return APDistillationTrainingResult(
        model=model,
        final_loss=final_loss,
        epochs=epochs,
        learning_rate=learning_rate,
        temperature=None,
        hard_label_weight=1.0,
    )


def train_distilled_student(
    *,
    features: Tensor,
    hard_labels: Tensor,
    teacher_logits: Tensor,
    seed: int,
    epochs: int = 500,
    learning_rate: float = 0.01,
    temperature: float = 2.0,
    hard_label_weight: float = 0.5,
) -> APDistillationTrainingResult:
    """Train the compact student using teacher soft targets."""
    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    torch.manual_seed(seed)

    model = APDistillationStudent()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    model.train()

    final_loss = 0.0

    for _ in range(epochs):
        optimizer.zero_grad(
            set_to_none=True
        )

        student_logits = model(
            features
        )

        loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            hard_labels=hard_labels,
            temperature=temperature,
            hard_label_weight=hard_label_weight,
        )

        torch.autograd.backward(loss)
        optimizer.step()

        final_loss = float(
            loss.detach().item()
        )

    model.eval()

    return APDistillationTrainingResult(
        model=model,
        final_loss=final_loss,
        epochs=epochs,
        learning_rate=learning_rate,
        temperature=temperature,
        hard_label_weight=hard_label_weight,
    )
