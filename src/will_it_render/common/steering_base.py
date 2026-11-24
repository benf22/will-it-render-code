"""The ``SteeringModel`` abstract base class.

Hoisted out of ``steering.steering_model`` so that baseline implementations
(``baselines/*/loss.py``) can subclass it without importing the ``steering`` package —
which imported ``baselines`` back, forming a cycle.
``steering.steering_model`` re-exports this name for backwards compatibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class SteeringModel(ABC):
    """Maps one step's attention captures → a steering loss."""

    @abstractmethod
    def should_steer(self, step_idx: int) -> bool:
        """True if this denoising step should be optimized (a probe applies here)."""

    @abstractmethod
    def loss(
        self,
        captures: list,
        step_idx: int,
        *,
        target_words_indices: dict[str, list[int]],
    ) -> tuple[torch.Tensor, dict]:
        """Return ``(scalar_loss, metrics)`` for one step. Only called when
        ``should_steer(step_idx)`` is True. ``captures`` are ``BlockCapture``s with
        live (non-detached) attention maps; ``target_words_indices`` maps each target
        word to its (subword) token indices into the text sequence — one probe score is
        computed per word and the per-word losses are averaged."""
