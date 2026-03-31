# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

from dataclasses import dataclass, field
from typing import Optional, List
from algodisco.base.search_method import SearchConfigBase


@dataclass
class FunSearchConfig(SearchConfigBase):
    """Configuration for a FunSearch run.

    Attributes:
        template_program: Template program source code used to seed the search.
        task_description: Natural-language task description injected into prompts.
        idea_prompt: Whether to use idea-oriented prompting variants.
        max_samples: Maximum number of candidates to sample before stopping.
        examples_per_prompt: Number of archived examples included in each prompt.
        samples_per_prompt: Number of candidates requested from the LLM per prompt.
        llm_max_tokens: Optional max token limit for each LLM response.
        llm_timeout_seconds: Timeout in seconds for each LLM request.
        db_num_islands: Number of islands maintained in the program database.
        db_max_island_capacity: Optional per-island archive capacity limit.
        db_reset_period: Period in seconds for island reset/rebalancing.
        db_cluster_sampling_temperature_init: Initial temperature for cluster sampling.
        db_cluster_sampling_temperature_period: Annealing period for cluster sampling temperature.
        db_save_frequency: Frequency for persisting database snapshots to the logger.
        keep_metadata_keys: Candidate metadata keys preserved when saving results.
    """

    template_program: str
    task_description: str = ""
    idea_prompt: bool = False
    max_samples: Optional[int] = field(default=1000, kw_only=True)
    examples_per_prompt: int = 2
    samples_per_prompt: int = 4
    llm_max_tokens: Optional[int] = None
    llm_timeout_seconds: int = 120

    # Database specific configs
    db_num_islands: int = 10
    db_max_island_capacity: Optional[int] = None
    db_reset_period: int = 4 * 60 * 60
    db_cluster_sampling_temperature_init: float = 0.1
    db_cluster_sampling_temperature_period: int = 30_000
    db_save_frequency: Optional[int] = 100

    # Metadata keys to keep when saving
    keep_metadata_keys: List[str] = field(
        default_factory=lambda: [
            "sample_time",
            "eval_time",
            "execution_time",
            "error_msg",
            "prompt",
            "response_text",
        ]
    )
