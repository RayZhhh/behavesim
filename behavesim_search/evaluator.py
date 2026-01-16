from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Any, TypedDict

import adtools.evaluator

class EvalResult(TypedDict):
    score: float
    behavior: Any


class BehaveSimSearchEvaluator(adtools.evaluator.PyEvaluator, ABC):
    def __init__(
        self,
        exec_code: bool = True,
        debug_mode: bool = False,
    ):
        super().__init__(exec_code=exec_code, debug_mode=debug_mode)

    @abstractmethod
    def evaluate_program(
        self,
        program_str: str,
        callable_functions_dict: Dict[str, Callable] | None,
        callable_functions_list: List[Callable] | None,
        callable_classes_dict: Dict[str, Callable] | None,
        callable_classes_list: List[Callable] | None,
        **kwargs,
    ) -> EvalResult:
        """Evaluate a given program.

        Args:
            program_str: The raw program text.
            callable_functions_dict: A dict maps function name to callable function.
            callable_functions_list: A list of callable functions.
            callable_classes_dict: A dict maps class name to callable class.
            callable_classes_list: A list of callable classes.

        Returns:
            Returns the evaluation result.
        """
        raise NotImplementedError(
            "Must provide an evaluator for a python program. "
            "Override this method in a subclass."
        )
