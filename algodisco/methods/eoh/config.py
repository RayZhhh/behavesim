# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from algodisco.base.search_method import SearchConfigBase


class EoHOperators(Enum): ...


@dataclass
class EoHConfig(SearchConfigBase):
    """Configuration for an EoH (Evolution of Heuristics) Search run.

    Attributes:
        template_program: Template program source code used to initialize search.
        task_description: Natural-language task description injected into prompts.
        max_samples: Maximum number of candidates to sample before stopping.
        pop_size: Target population size maintained by the database.
        selection_num: Number of parents selected for multi-parent operators.
        use_e2_operator: Whether to enable the E2 crossover-style operator.
        use_m1_operator: Whether to enable the M1 mutation operator.
        use_m2_operator: Whether to enable the M2 mutation operator.
        llm_max_tokens: Optional max token limit for each LLM response.
        llm_timeout_seconds: Timeout in seconds for each LLM request.
        db_save_frequency: Frequency for persisting population snapshots to the logger.
        init_samples_ratio: Multiplier controlling samples used during initialization.
        keep_metadata_keys: Candidate metadata keys preserved when saving results.
    """

    template_program: str
    task_description: str = ""
    max_samples: Optional[int] = field(default=1000, kw_only=True)

    # Population parameters
    pop_size: int = 10
    selection_num: int = 2

    # Operator flags
    use_e2_operator: bool = True
    use_m1_operator: bool = True
    use_m2_operator: bool = True

    # Search parameters
    llm_max_tokens: Optional[int] = None
    llm_timeout_seconds: int = 120
    db_save_frequency: Optional[int] = 100

    # Initialization phase
    init_samples_ratio: float = 2.0  # Sample 2 * pop_size to initialize

    # Metadata keys to keep when saving
    keep_metadata_keys: List[str] = field(
        default_factory=lambda: [
            "idea",
            "sample_time",
            "eval_time",
            "execution_time",
            "error_msg",
            "prompt",
            "response_text",
        ]
    )
