# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Any, Generic, NotRequired, Optional, TypedDict, TypeVar


class EvalResult(TypedDict):
    """The result of an evaluation.

    This only describes the minimum shared contract across evaluators.
    Method-specific evaluators can define a narrower TypedDict subclass when they
    need extra required keys such as ``behavior``.
    """

    score: float
    execution_time: NotRequired[Optional[float]]
    error_msg: NotRequired[Optional[str]]


TResult_co = TypeVar("TResult_co", bound=EvalResult, covariant=True)


class Evaluator(ABC, Generic[TResult_co]):
    """Base class for program evaluators.

    The generic parameter captures the concrete evaluation result shape. Most
    evaluators can return ``EvalResult`` directly. Evaluators with extra
    required keys should return a TypedDict subclass of ``EvalResult``.
    """

    @abstractmethod
    def evaluate_program(self, program_str: str) -> TResult_co:
        """Evaluate a given program.

        Args:
            program_str: The raw program text.

        Returns:
            Returns the evaluation result. This should be a dictionary
            containing at least a ``score`` key.
        """
        raise NotImplementedError(
            "Must provide an evaluator for a python program. "
            "Override this method in a subclass."
        )
