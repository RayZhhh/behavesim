# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

from dataclasses import dataclass, field
from typing import Optional, List
from algodisco.base.search_method import SearchConfigBase
from algodisco.methods.funsearch_behavesim.similarity_calculator import (
    BehaveSimCalculator,
)


@dataclass
class BehaveSimSearchConfig(SearchConfigBase):
    """Configuration for a BehaveSim Search run.

    Attributes:
        template_program: Template program source code used to seed the search.
        task_description: Natural-language task description injected into prompts.
        max_samples: Maximum number of candidates to sample before stopping.
        examples_per_prompt: Number of archived examples included in each prompt.
        samples_per_prompt: Number of candidates requested from the LLM per prompt.
        inter_island_selection_p: Probability of selecting parents across islands.
        llm_max_tokens: Optional max token limit for each LLM response.
        llm_timeout_seconds: Timeout in seconds for each LLM request.
        db_save_frequency: Frequency for persisting database snapshots to the logger.
        enable_database_reclustering: Whether to periodically rebuild behavior clusters.
        recluster_threshold: Number of inserts between reclustering passes.
        db_algo_sim_calculator: Similarity calculator used for behavior comparison.
        db_num_islands: Number of islands maintained in the algorithm database.
        db_max_island_capacity: Optional per-island archive capacity limit.
        db_cluster_sampling_temperature_init: Initial temperature for cluster sampling.
        db_cluster_sampling_temperature_period: Annealing period for cluster sampling temperature.
        db_num_sim_caculator_workers: Number of worker processes for similarity computation.
        db_async_register: Whether to register algorithms asynchronously.
        keep_metadata_keys: Candidate metadata keys preserved when saving results.
    """

    template_program: str
    task_description: str = ""
    max_samples: Optional[int] = field(default=1000, kw_only=True)
    examples_per_prompt: int = 2
    samples_per_prompt: int = 4
    inter_island_selection_p: float = 0.5
    llm_max_tokens: Optional[int] = None
    llm_timeout_seconds: int = 120
    db_save_frequency: Optional[int] = 100
    enable_database_reclustering: bool = True
    recluster_threshold: int = 100
    db_algo_sim_calculator: BehaveSimCalculator = field(
        default_factory=BehaveSimCalculator
    )
    db_num_islands: int = 10
    db_max_island_capacity: Optional[int] = None
    db_cluster_sampling_temperature_init: float = 0.1
    db_cluster_sampling_temperature_period: int = 30_000
    db_num_sim_caculator_workers: int = 1
    db_async_register: bool = False

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
