# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import multiprocessing
import sys
import time
import traceback
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import psutil

from algodisco.toolkit.sandbox.utils import _redirect_to_devnull

__all__ = ["ExecutionResults", "SandboxExecutorSimple"]

# Set multiprocessing start method to 'fork' on macOS/Linux for better compatibility.
if sys.platform.startswith("darwin") or sys.platform.startswith("linux"):
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # Already set


class ExecutionResults(TypedDict):
    result: Any
    execution_time: float
    error_msg: str


class SandboxExecutorSimple:
    """Simplified sandbox executor using Queue-based result passing.

    This implementation uses multiprocessing.Queue to pass results between
    parent and child processes, avoiding complex shared memory management.
    Each execution creates a new process for simplicity.
    """

    def __init__(
        self,
        evaluate_worker: Any,
        recur_kill_eval_proc: bool = False,
        debug_mode: bool = False,
        *,
        join_timeout_seconds: int = 10,
    ):
        """Simplified evaluator using multiprocessing.Queue for result passing.

        Args:
            evaluate_worker: The worker object to be executed.
            recur_kill_eval_proc: If True, kill child processes when they are terminated.
            debug_mode: Debug mode.
            join_timeout_seconds: Timeout in seconds to wait for the process to finish.
        """
        self.evaluate_worker = evaluate_worker
        self.debug_mode = debug_mode
        self.recur_kill_eval_proc = recur_kill_eval_proc
        self.join_timeout_seconds = join_timeout_seconds

    def _kill_process_and_its_children(self, process: multiprocessing.Process):
        """Kill a process and all its child processes."""
        children_processes = []
        if self.recur_kill_eval_proc:
            try:
                parent = psutil.Process(process.pid)
                children_processes = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                children_processes = []

        process.terminate()
        process.join(timeout=self.join_timeout_seconds)
        if process.is_alive():
            process.kill()
            process.join()

        for child in children_processes:
            try:
                if self.debug_mode:
                    print(
                        f"Killing process {process.pid}'s children process {child.pid}"
                    )
                if child.is_running():
                    child.terminate()
            except Exception:
                if self.debug_mode:
                    traceback.print_exc()

    def _execute_in_subprocess(
        self,
        worker_execute_method_name: str,
        method_args: Optional[List | Tuple],
        method_kwargs: Optional[Dict],
        result_queue: multiprocessing.Queue,
        redirect_to_devnull: bool,
    ):
        """Execute the worker method in a subprocess and put result in queue."""
        if redirect_to_devnull:
            _redirect_to_devnull()

        if hasattr(self.evaluate_worker, worker_execute_method_name):
            method_to_call = getattr(self.evaluate_worker, worker_execute_method_name)
        else:
            raise RuntimeError(
                f"Method named '{worker_execute_method_name}' not found."
            )

        try:
            args = method_args or []
            kwargs = method_kwargs or {}
            res = method_to_call(*args, **kwargs)
            result_queue.put(("success", res))
        except Exception:
            if self.debug_mode:
                traceback.print_exc()
            result_queue.put(("error", str(traceback.format_exc())))

    def secure_execute(
        self,
        worker_execute_method_name: str,
        method_args: Optional[List | Tuple] = None,
        method_kwargs: Optional[Dict] = None,
        timeout_seconds: int | float = None,
        redirect_to_devnull: bool = False,
    ) -> ExecutionResults:
        """Execute the worker method in a new process.

        Args:
            worker_execute_method_name: Name of the worker execute method.
            method_args: Arguments of the worker execute method.
            method_kwargs: Keyword arguments of the worker execute method.
            timeout_seconds: Return None if execution time exceeds this value.
            redirect_to_devnull: Redirect any output to '/dev/null'.

        Returns:
            ExecutionResults containing result, execution_time, and error_msg.
        """
        evaluate_start_time = time.time()
        result_queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=self._execute_in_subprocess,
            args=(
                worker_execute_method_name,
                method_args,
                method_kwargs,
                result_queue,
                redirect_to_devnull,
            ),
        )
        process.start()

        try:
            status, payload = result_queue.get(timeout=timeout_seconds)
            eval_time = time.time() - evaluate_start_time
        except Empty:
            if self.debug_mode:
                print(f"DEBUG: evaluation time exceeds {timeout_seconds}s.")
            self._kill_process_and_its_children(process)
            return ExecutionResults(
                result=None,
                execution_time=time.time() - evaluate_start_time,
                error_msg="Evaluation timeout.",
            )
        except Exception:
            if self.debug_mode:
                traceback.print_exc()
            self._kill_process_and_its_children(process)
            return ExecutionResults(
                result=None,
                execution_time=time.time() - evaluate_start_time,
                error_msg=str(traceback.format_exc()),
            )
        finally:
            self._kill_process_and_its_children(process)

        if status == "success":
            return ExecutionResults(
                result=payload, execution_time=eval_time, error_msg=""
            )
        else:
            return ExecutionResults(
                result=None, execution_time=eval_time, error_msg=payload
            )
