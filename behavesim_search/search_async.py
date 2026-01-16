"""
Main search logic for BehaveSim Search.
This implementation is inspired by Google's FunSearch but adapted for the BehaveSim Search framework.
"""

import asyncio
import logging
import pickle
import random
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from adtools import PyProgram
from adtools.lm import LanguageModel

from behavesim_search.algo_database import AlgoDatabase, AlgoDatabaseConfig
from behavesim_search.algo_proto import AlgoProto
from behavesim_search.evaluator import BehaveSimSearchEvaluator
from behavesim_search.logger import PickleLoggerWithSwanLab
from behavesim_search.prompt import SearchPromptConstructor

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class BehaveSimSearchConfig:
    """Configuration for an BehaveSim Search run."""

    task_description: str
    template_program: str | PyProgram
    database_config: AlgoDatabaseConfig = field(default_factory=AlgoDatabaseConfig)
    evaluate_timeout_seconds: Optional[float] = 30.0
    num_samplers: int = 4
    num_evaluators: int = 4
    examples_per_prompt: int = 2
    samples_per_prompt: int = 4
    max_samples: Optional[int] = 1000
    inter_island_selection_p: float = 0.5
    llm_max_tokens: Optional[int] = None
    llm_timeout_seconds: int = 120
    redirect_to_devnull: bool = True
    db_save_frequency: Optional[int] = 100
    enable_database_reclustering: bool = True
    recluster_threshold: int = 100


class BehaveSimSearchAsync:
    """Core class for the BehaveSim search process, using asyncio."""

    def __init__(
        self,
        config: BehaveSimSearchConfig,
        llm: LanguageModel,
        evaluator: BehaveSimSearchEvaluator,
        logger: Optional[PickleLoggerWithSwanLab] = None,
        prompt_constructor: SearchPromptConstructor = SearchPromptConstructor(),
    ):
        """
        Initializes the BehaveSim Search.

        Args:
            config: Configuration object for the search.
            llm: Language model for code generation.
            evaluator: The evaluator to score programs.
            logger: Logger for experiment tracking.
            prompt_constructor: The component for building prompts.
        """
        self._config = config
        self._template_program_str = str(self._config.template_program)
        # Parse the template program
        self._template_program = PyProgram.from_text(self._template_program_str)
        if not self._template_program:
            raise ValueError("The provided template program is not valid Python code.")

        self._llm = llm
        self._evaluator = evaluator
        self._database = AlgoDatabase(config.database_config)
        self._logger = logger
        self._prompt_constructor = prompt_constructor

        self._lock = asyncio.Lock()
        self._samples_count = 0
        self._evaluator_semaphore = asyncio.Semaphore(self._config.num_evaluators)
        self._stop_event = asyncio.Event()
        self._initial_recluster_done = False

    def _save_database(self, sample_num: int):
        """Saves the current state of the database to a numbered pickle file."""
        if not self._logger:
            return

        database_dict = self._database.to_dict()
        log_dir = self._logger._logdir
        db_dir = Path(log_dir) / "db"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_save_path = db_dir / f"database_{sample_num}.pkl"
        with open(db_save_path, "wb") as f:
            pickle.dump(database_dict, f)
        logging.info(f"Saved database snapshot to {db_save_path}")

    async def run(self):
        """Starts the search process."""
        try:
            # First, evaluate and register the template program
            logging.info("Evaluating template program...")
            eval_results = await asyncio.to_thread(
                self._evaluator.secure_evaluate,
                self._template_program_str,
                timeout_seconds=self._config.evaluate_timeout_seconds,
                redirect_to_devnull=self._config.redirect_to_devnull,
            )
            eval_time = eval_results["evaluate_time"]
            result = eval_results["result"]

            if result is None or result["score"] is None:
                raise RuntimeError(
                    f"The template program failed evaluation. Error: {eval_results['error_msg']}"
                )

            logging.info(
                f"Template program evaluated successfully. Score: {result['score']:.4f}, Time: {eval_time:.2f}s"
            )

            template_proto = AlgoProto(
                program=self._template_program,
                behavior=result["behavior"],
                score=result["score"],
            )
            self._database.register_algo(template_proto)
            async with self._lock:
                self._samples_count += 1  # Increment for template
                current_sample_num = self._samples_count
                score_str = "None"
                if template_proto.score is not None:
                    score_str = f"{template_proto.score:.4f}"
                logging.info(
                    f"Registered Algo #{current_sample_num} (Template): "
                    f"Score={score_str}, "
                    f"SampleTime=N/A, EvalTime={eval_time:.2f}s"
                )
                if self._logger:
                    log_entry = {
                        "sample_num": current_sample_num,
                        "score": template_proto.score,
                        "program": str(template_proto.program),
                        "island_id": -1,  # Template doesn't belong to any island initially
                        "sample_time": 0.0,  # N/A for template, setting to 0.0
                        "eval_time": eval_time,
                    }
                    self._logger.log_to_cache(log_entry)

            # Start sampler tasks
            logging.info(f"Starting {self._config.num_samplers} sampler tasks...")
            sampler_tasks = [
                asyncio.create_task(self._sample_evaluate_register_loop())
                for _ in range(self._config.num_samplers)
            ]

            await asyncio.gather(*sampler_tasks)

        except (KeyboardInterrupt, SystemExit):
            logging.info("Search interrupted by user.")
            self._stop_event.set()
        except Exception:
            logging.error("An unexpected error occurred during the search process.")
            traceback.print_exc()
            self._stop_event.set()
        finally:
            self._save_database(self._samples_count)
            if self._logger:
                logging.info("Finalizing logger...")
                self._logger.finish()
            logging.info("Search finished.")

    async def _sample_evaluate_register_loop(self):
        """The main loop for a single sampler task."""
        while not self._stop_event.is_set():
            async with self._lock:
                if (
                    self._config.max_samples is not None
                    and self._samples_count >= self._config.max_samples
                ):
                    self._stop_event.set()
                    break

            try:
                examples, island_ids = self._select_examples()
                sorted_examples = sorted(examples, key=lambda p: p.score, reverse=True)
                example_programs = [p.program for p in sorted_examples]
                prompt = self._prompt_constructor.construct_prompt(
                    self._config.task_description, example_programs
                )

                for _ in range(self._config.samples_per_prompt):
                    if self._stop_event.is_set():
                        break
                    await self._prompt_and_evaluate(prompt, island_ids)

            except (KeyboardInterrupt, SystemExit):
                self._stop_event.set()
                break
            except Exception:
                logging.warning(f"Exception in sampler task: {traceback.format_exc()}")
                await asyncio.sleep(5)  # Cooldown after an error

    def _select_examples(self) -> tuple[List[AlgoProto], List[int]]:
        """Selects a batch of example algorithms from the database."""
        if random.random() < self._config.inter_island_selection_p:
            # Inter-island selection: k islands, 1 sample from each.
            parent_selection = self._database.select_algos(
                num_islands=self._config.examples_per_prompt, samples_per_island=1
            )
        else:
            # Intra-island selection: 1 island, k samples from it.
            parent_selection = self._database.select_algos(
                num_islands=1, samples_per_island=self._config.examples_per_prompt
            )
        examples = [item[0] for item in parent_selection]
        island_ids = [item[1] for item in parent_selection]
        return examples, island_ids

    async def _prompt_and_evaluate(self, prompt: str, island_ids: List[int]):
        """Generates a new sample, evaluates it, generates idea summary, and registers it."""
        sample_start_time = time.time()
        response_text = await asyncio.to_thread(
            self._llm.chat_completion,
            prompt,
            max_tokens=self._config.llm_max_tokens,
            timeout_seconds=self._config.llm_timeout_seconds,
        )
        sample_time = time.time() - sample_start_time

        new_code_str = self._prompt_constructor.extract_code(response_text)

        if not new_code_str:
            return

        async with self._evaluator_semaphore:
            eval_results = await asyncio.to_thread(
                self._evaluator.secure_evaluate,
                new_code_str,
                timeout_seconds=self._config.evaluate_timeout_seconds,
                redirect_to_devnull=self._config.redirect_to_devnull,
            )

            result = eval_results["result"]

            await self._register_new_proto(
                new_code_str,
                result,
                island_ids,
                sample_time,
                eval_results["evaluate_time"],
            )

    async def _register_new_proto(
        self,
        new_code_str: str,
        result: Optional[dict],
        island_ids: List[int],
        sample_time: float,
        eval_time: float,
    ):
        """Creates a new AlgoProto and registers it in the database and logger."""
        new_program = PyProgram.from_text(new_code_str)
        if not new_program:
            return

        score = result.get("score") if result else None

        async with self._lock:
            if (
                self._config.max_samples is not None
                and self._samples_count >= self._config.max_samples
            ):
                return
            self._samples_count += 1
            current_sample_num = self._samples_count

            if score is not None:
                new_proto = AlgoProto(
                    program=new_program,
                    behavior=result["behavior"],
                    score=score,
                )
                self._database.register_algo(new_proto)

            score_str = "None"
            if score is not None:
                score_str = f"{score:.4f}"
            logging.info(
                f"Algo #{current_sample_num}: "
                f"Score={score_str}, "
                f"SampleTime={sample_time:.2f}s, EvalTime={eval_time:.2f}s"
            )

            if self._logger:
                log_entry = {
                    "sample_num": current_sample_num,
                    "score": score,
                    "program": str(new_program),
                    "island_id": island_ids[0] if island_ids else -1,
                    "sample_time": sample_time,
                    "eval_time": eval_time,
                }
                self._logger.log_to_cache(log_entry)

            if (
                self._config.db_save_frequency is not None
                and current_sample_num % self._config.db_save_frequency == 0
            ):
                self._save_database(current_sample_num)

            # Check if we should trigger the initial database clustering
            if (
                self._config.enable_database_reclustering
                and not self._initial_recluster_done
                and self._database.get_total_algo_count()
                >= self._config.recluster_threshold
            ):
                logging.info(
                    f"Reached {self._database.get_total_algo_count()} algorithms, "
                    f"triggering database reclustering as per threshold ({self._config.recluster_threshold})."
                )
                self._database.cluster_and_reassign_islands()
                self._initial_recluster_done = True
                # Save the database state immediately after this significant event
                # self._save_database(current_sample_num)
