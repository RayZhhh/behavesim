# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import uuid
import copy
from typing import Any, Dict, List


class AlgoProto:

    def __init__(
        self,
        algo_id: str = "",
        program: str = "",
        language: str = "",
        score: Any = None,
    ):
        self.algo_id = algo_id or str(uuid.uuid4())
        self.program = program
        self.language = language
        self.score = score
        self.metadata = {}

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        assert key not in ["program", "language", "algo_id", "score", "metadata"]
        self.metadata[key] = value

    def __getitem__(self, key):
        return self.metadata[key]

    def get(self, key, default=None):
        assert isinstance(key, str)
        if key in ["program", "language", "algo_id", "score", "metadata"]:
            return getattr(self, key)
        return self.metadata.get(key, default)

    def pop(self, key, *args):
        assert isinstance(key, str)
        assert key not in ["program", "language", "algo_id", "score", "metadata"]
        return self.metadata.pop(key, *args)

    def get_markdown_code_block(self) -> str:
        return f"```{self.language}\n{str(self.program).strip()}\n```"

    def update(self, data: Dict) -> None:
        """Update metadata from dict."""
        data_copy = data.copy()

        if "program" in data_copy:
            self.program = data_copy.pop("program")
        if "language" in data_copy:
            self.language = data_copy.pop("language")
        if "algo_id" in data_copy:
            self.algo_id = data_copy.pop("algo_id")
        if "score" in data_copy:
            self.score = data_copy.pop("score")
        if "metadata" in data_copy:
            attrs = data_copy.pop("metadata")
            assert isinstance(attrs, dict)
            self.metadata.update(attrs)
        if "attributes" in data_copy:
            attrs = data_copy.pop("attributes")
            assert isinstance(attrs, dict)
            self.metadata.update(attrs)

        for k, v in data_copy.items():
            self[k] = v

    @classmethod
    def from_program_str(cls, program: str) -> "AlgoProto":
        """Initialize AlgoProto from program string."""
        return cls.from_dict({"program": program})

    @classmethod
    def from_dict(cls, data: Dict) -> "AlgoProto":
        """Initialize AlgoProto from dict."""
        # Assert each key in data dict is string
        for k in data:
            assert isinstance(k, str), f"Key '{k}' must be a string."

        data_copy = data.copy()

        program = data_copy.pop("program", "")
        language = data_copy.pop("language", "")
        algo_id = data_copy.pop("algo_id", "")
        score = data_copy.pop("score", None)

        algo_proto = cls(
            program=program, language=language, algo_id=algo_id, score=score
        )

        if "metadata" in data_copy:
            attrs = data_copy.pop("metadata")
            assert isinstance(attrs, dict)
            algo_proto.metadata.update(attrs)
        if "attributes" in data_copy:
            attrs = data_copy.pop("attributes")
            assert isinstance(attrs, dict)
            algo_proto.metadata.update(attrs)

        for k, v in data_copy.items():
            algo_proto[k] = v

        return algo_proto

    def to_dict(self) -> Dict:
        data = {
            "algo_id": self.algo_id,
            "program": self.program,
            "language": self.language,
            "score": self.score,
        }
        data.update(self.metadata)
        return data

    def __deepcopy__(self, memo: Dict):
        """
        Creates a deep copy of the object with a newly generated algo_id.

        This method ensures that when copy.deepcopy() is called, the original
        algo_id is not copied, triggering the constructor to generate a fresh UUID.
        """
        # Create a new instance without passing the original algo_id to trigger a new UUID generation
        new_instance = self.__class__(
            program=copy.deepcopy(self.program, memo),
            language=copy.deepcopy(self.language, memo),
            score=copy.deepcopy(self.score, memo),
        )

        # Track the new instance in the memo to handle potential cyclic references
        memo[id(self)] = new_instance

        # Deeply copy the metadata dictionary
        new_instance.metadata = copy.deepcopy(self.metadata, memo)
        return new_instance

    def keep_metadata_keys(self, keys: str | List[str]):
        """Keep only the specified keys in metadata and remove others."""
        if isinstance(keys, str):
            keys = [keys]
        keys_to_keep = set(keys)
        self.metadata = {k: v for k, v in self.metadata.items() if k in keys_to_keep}
