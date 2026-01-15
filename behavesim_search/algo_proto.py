import uuid
from typing import Any, List

from adtools import PyProgram


class AlgoProto:

    def __init__(
        self,
        program: PyProgram,
        behavior: Any,
        score: float,
    ):
        """Represents a prototype of an algorithm.

        This class encapsulates the core components of an algorithm, including its
        source code, conceptual idea, and performance metrics. Each instance is
        assigned a unique identifier upon creation.

        Args:
            idea: A natural language description of the core logic or thought process behind the algorithm.
            program: The source code of the algorithm, encapsulated in a PyProgram object.
            behavior: The execution trajectory or behavior of the algorithm on a set of test cases.
            score: The performance score of the algorithm.
        """
        self.id: str = str(uuid.uuid4())
        self.program = program
        self.behavior = behavior
        self.score = score
        self._temp_var = {}

    def __getitem__(self, key: str) -> Any:
        return self._temp_var.get(key, {})

    def __setitem__(self, key: str, value: Any):
        self._temp_var[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Converts the AlgoProto to a dictionary."""
        return {
            "id": self.id,
            "program_source": str(self.program),  # Using str(PyProgram)
            "behavior": self.behavior,  # Assumes behavior is JSON-serializable
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlgoProto":
        """Creates an AlgoProto from a dictionary."""
        from adtools import PyProgram
        import numpy as np

        instance = cls(
            program=PyProgram.from_text(
                data["program_source"]
            ),  # Using PyProgram.from_text(str)
            behavior=data["behavior"],
            score=data["score"],
        )
        instance.id = data["id"]
        # Parents and children object lists are reconstructed at the AlgoDatabase level.
        return instance
