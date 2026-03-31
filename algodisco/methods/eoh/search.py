# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import copy
import logging
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from algodisco.base.llm import LanguageModel
from algodisco.base.algo import AlgoProto
from algodisco.base.evaluator import EvalResult, Evaluator
from algodisco.base.search_method import IterativeSearchBase
from algodisco.base.logger import AlgoSearchLoggerBase
from algodisco.common.timer import Timer
from algodisco.common.logging_utils import format_time_info, format_error_box

from algodisco.methods.eoh.config import EoHConfig
from algodisco.methods.eoh.database import EoHDatabase
from algodisco.methods.eoh.prompt import EoHPromptAdapter

# Configure basic logging.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EoHSearch(IterativeSearchBase):
    """Core class for the EoH search process, using threading."""

    def __init__(
        self,
        config: EoHConfig,
        evaluator: Evaluator[EvalResult],
        llm: LanguageModel = None,
        logger: Optional[AlgoSearchLoggerBase] = None,
        prompt_constructor: EoHPromptAdapter = EoHPromptAdapter(),
        *,
        tool_mode=False,
    ):
        # --- Tool mode assertion ---
        assert llm or tool_mode
        # ---------------------------

        self._config = config
        self._template_program_str = str(self._config.template_program)
        if not self._template_program_str:
            raise ValueError("The provided template program is empty.")

        self._llm = llm
        self._evaluator: Evaluator[EvalResult] = evaluator
        self._database = EoHDatabase(config.pop_size)
        self._logger = logger
        self._prompt_constructor = prompt_constructor

        self._lock = threading.Lock()
        self._samples_count = 0
        self._evaluator_semaphore = threading.Semaphore(self._config.num_evaluators)
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=self._config.num_samplers)

        # Debug mode: print all errors during search (can be set after instantiation)
        # When True, errors are logged at ERROR level instead of WARNING
        self.debug_mode = False
        # When True and debug_mode is True, exit immediately on error
        self.debug_mode_crash = False

    def _save_database(self, sample_num: int):
        """Saves the current state of the database using the logger."""
        if not self._logger:
            return

        database_dict = self._database.to_dict()
        database_dict["sample_num"] = sample_num
        self._logger.log_dict(database_dict, "database")
        logging.info(f"Saved database snapshot for sample #{sample_num} to logger.")

    @override
    def initialize(self):
        """Initializes the search process by evaluating the template program."""
        # Set log flush frequencies
        db_frequency = getattr(self._config, "db_save_frequency", 1)
        algo_frequency = getattr(self._config, "algo", 2000)
        if self._logger:
            self._logger.set_log_item_flush_frequency(
                {
                    "database": db_frequency,
                    "algo": algo_frequency,
                }
            )

        logging.info("Evaluating template program...")

        template_proto = AlgoProto(
            program=self._template_program_str,
            language=self._config.language,
        )

        with Timer(template_proto, "eval_time"):
            results = self._evaluator.evaluate_program(template_proto.program)

        if results is None or results.get("score") is None:
            # EoH template might fail, but we usually want a baseline.
            # We'll register it if it works, otherwise we rely on i1.
            logging.warning("Template program failed evaluation.")
        else:
            template_proto.score = results["score"]
            template_proto["idea"] = "Initial template baseline."
            self._database.register_algo(template_proto)

        with self._lock:
            self._samples_count += 1
            template_proto["island_id"] = -1
            self._log(template_proto, is_template=True)

    def run(self):
        """Starts the search process."""
        try:
            self.initialize()

            logging.info(f"Starting {self._config.num_samplers} sampler threads...")

            # Start sampler threads
            threads = []
            for _ in range(self._config.num_samplers):
                t = threading.Thread(target=self._generate_evaluate_register_loop)
                t.start()
                threads.append(t)

            # Wait for all threads to complete
            for t in threads:
                t.join()

        except (KeyboardInterrupt, SystemExit):
            logging.info("Search interrupted by user.")
            self._stop_event.set()
        except Exception as e:
            error_msg = traceback.format_exc()
            logging.error("An unexpected error occurred during the search process.")
            if self.debug_mode:
                logging.error(format_error_box(error_msg))
                if self.debug_mode_crash:
                    logging.error("Debug mode crash: exiting immediately.")
                    self._stop_event.set()
                    import sys

                    sys.exit(1)
            self._stop_event.set()
        finally:
            self._executor.shutdown(wait=True)
            with self._lock:
                self._save_database(self._samples_count)
            if self._logger:
                logging.info("Finalizing logger...")
                self._logger.finish()
            logging.info("Search finished.")

    @override
    def is_stopped(self) -> bool:
        return self._stop_event.is_set() or (
            self._config.max_samples is not None
            and self._samples_count >= self._config.max_samples
        )

    @override
    def current_num_samples(self) -> int:
        with self._lock:
            return self._samples_count

    @override
    def get_config(self) -> EoHConfig:
        return self._config

    def _generate_evaluate_register_loop(self):
        """The main loop for a single sampler thread."""
        while not self.is_stopped():
            with self._lock:
                if self.is_stopped():
                    self._stop_event.set()
                    break

            try:
                candidate = self.select_and_create_prompt()
                if not candidate:
                    time.sleep(1)
                    continue

                # EoH usually generates one sample per prompt per operator call
                candidate = self.generate(candidate)
                candidate = self.extract_algo_from_response(candidate)
                candidate = self.evaluate(candidate)
                self.register(candidate)

            except (KeyboardInterrupt, SystemExit):
                self._stop_event.set()
                break
            except Exception as e:
                logging.warning(
                    f"Exception in sampler thread: {traceback.format_exc()}"
                )
                if self.debug_mode:
                    logging.error(f"Debug mode: error in sampler thread: {e}")
                    logging.error(format_error_box(traceback.format_exc()))
                    if self.debug_mode_crash:
                        logging.error("Debug mode crash: exiting immediately.")
                        self._stop_event.set()
                        import sys

                        sys.exit(1)
                time.sleep(1)

    @override
    def select_and_create_prompt(self) -> Optional[AlgoProto]:
        """Selects parents and operator, then creates a prompt.

        AlgoProto keys set:
            - operator: str, the evolutionary operator (i1, e1, e2, m1, m2)
            - parents: list of selected parent AlgoProto objects (for e1, e2, m1, m2)
            - prompt: str, the constructed prompt for generation
        """
        current_pop_size = len(self._database)

        # Determine operator
        if (
            self._samples_count
            < self._config.pop_size * self._config.init_samples_ratio
            or current_pop_size < 2
        ):
            op = "i1"
        else:
            # Weighted choice based on config
            ops = ["e1"]
            if self._config.use_e2_operator:
                ops.append("e2")
            if self._config.use_m1_operator:
                ops.append("m1")
            if self._config.use_m2_operator:
                ops.append("m2")
            op = random.choice(ops)

        candidate = AlgoProto(language=self._config.language)
        candidate["operator"] = op

        if op == "i1":
            prompt = self._prompt_constructor.construct_prompt_i1(
                self._config.task_description,
                self._template_program_str,
                self._config.language,
            )
        elif op in ["e1", "e2"]:
            parents = self._database.select_algos(self._config.selection_num)
            if not parents:
                return None
            candidate["parents"] = parents
            if op == "e1":
                prompt = self._prompt_constructor.construct_prompt_e1(
                    self._config.task_description,
                    parents,
                    self._template_program_str,
                    self._config.language,
                )
            else:
                prompt = self._prompt_constructor.construct_prompt_e2(
                    self._config.task_description,
                    parents,
                    self._template_program_str,
                    self._config.language,
                )
        elif op in ["m1", "m2"]:
            parent = self._database.select_algos(1)
            if not parent:
                return None
            candidate["parents"] = parent
            if op == "m1":
                prompt = self._prompt_constructor.construct_prompt_m1(
                    self._config.task_description,
                    parent[0],
                    self._template_program_str,
                    self._config.language,
                )
            else:
                prompt = self._prompt_constructor.construct_prompt_m2(
                    self._config.task_description,
                    parent[0],
                    self._template_program_str,
                    self._config.language,
                )
        else:
            return None

        candidate["prompt"] = prompt
        return candidate

    @override
    @override
    def generate(self, candidate: AlgoProto) -> AlgoProto:
        """Generates a response from the LLM.

        AlgoProto keys set:
            - response_text: str, the raw LLM response
            - sample_time: float, time taken for LLM call (via Timer)
        """
        assert (
            self._llm is not None
        ), "LLM is required for generate(). Use tool_mode=False or provide an LLM."
        with Timer(candidate, "sample_time"):
            response_text = self._llm.chat_completion(
                candidate["prompt"],
                self._config.llm_max_tokens,
                self._config.llm_timeout_seconds,
            )
        candidate["response_text"] = response_text
        return candidate

    @override
    def extract_algo_from_response(self, candidate: AlgoProto) -> AlgoProto:
        response_text = candidate.get("response_text", "")
        idea = self._prompt_constructor.extract_idea(response_text)
        code = self._prompt_constructor.extract_code(
            response_text, language=candidate.language
        )

        if idea:
            candidate["idea"] = idea
        if code:
            candidate.program = code
        return candidate

    @override
    def evaluate(self, candidate: AlgoProto) -> AlgoProto:
        if not candidate or not candidate.program:
            return candidate

        with Timer(candidate, "eval_time"):
            with self._evaluator_semaphore:
                results = self._evaluator.evaluate_program(candidate.program)

        if results:
            # Always record execution_time and error_msg if available
            if "execution_time" in results:
                candidate["execution_time"] = results["execution_time"]
            if "error_msg" in results:
                candidate["error_msg"] = results["error_msg"]
            # Record score if available
            if results.get("score") is not None:
                candidate.score = results["score"]

        return candidate

    @override
    def register(self, algo_proto: AlgoProto):
        if not algo_proto or not algo_proto.program:
            return

        with self._lock:
            if self.is_stopped():
                return

            self._samples_count += 1

            if algo_proto.score is not None:
                self._database.register_algo(algo_proto)

            self._log(algo_proto)

    def _log(self, algo_proto: AlgoProto, is_template: bool = False):
        current_sample_num = self._samples_count
        op = algo_proto.get("operator", "template")

        if (
            self._config.db_save_frequency is not None
            and current_sample_num % self._config.db_save_frequency == 0
        ):
            self._save_database(current_sample_num)

        tag = " (Template)" if is_template else f" ({op})"
        algo_id_str = f"#{current_sample_num}{tag}"

        score_val = algo_proto.score
        score_str = f"{score_val:10.4f}" if score_val is not None else f"{'None':>10}"

        sample_time_val = algo_proto.get("sample_time", 0.0)
        sample_time_str = (
            f"{sample_time_val:6.2f}s" if not is_template else f"{'N/A':>7}"
        )

        eval_time_val = algo_proto.get("eval_time", 0.0)
        execution_time_val = algo_proto.get("execution_time", None)
        time_info = format_time_info(eval_time_val, execution_time_val)

        logging.info(
            f"Algo {algo_id_str:<18} | "
            f"Score: {score_str} | "
            f"Sample: {sample_time_str} | "
            f"{time_info}"
        )

        if self._logger:
            # Keep only specified metadata keys for logging
            algo_proto.keep_metadata_keys(self._config.keep_metadata_keys)
            log_entry = algo_proto.to_dict()
            log_entry.update(
                {
                    "sample_num": current_sample_num,
                    "operator": op,
                    "sample_time": 0.0 if is_template else sample_time_val,
                }
            )

            # Add population size as stat
            log_entry["pop_size"] = len(self._database)
            log_entry["best_score"] = self._database.get_best_score()

            self._logger.log_dict(log_entry, "algo")
