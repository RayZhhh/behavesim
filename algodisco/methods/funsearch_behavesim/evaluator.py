# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Any

from algodisco.base.evaluator import EvalResult, Evaluator


class BehaviorEvalResult(EvalResult):
    """Common specialization for evaluators that must return behavior data."""

    behavior: Any


class FunSearchBehaveSimEvaluator(Evaluator[BehaviorEvalResult], ABC):

    @abstractmethod
    def evaluate_program(self, program_str: str):
        """Evaluate a given program.

        Args:
            program_str: The raw program text.

        Returns:
            Returns the evaluation result.
        """
        raise NotImplementedError(
            "Must provide an evaluator for a python program. "
            "Override this method in a subclass."
        )
