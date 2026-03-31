# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import multiprocessing
import pickle
import time
import uuid
from multiprocessing import shared_memory
from queue import Empty
from typing import Any, Dict, List, TypedDict, Optional, Tuple
import multiprocessing.managers
import traceback

import psutil

from algodisco.toolkit.sandbox.utils import _redirect_to_devnull

__all__ = ["ExecutionResults", "SandboxExecutor"]


class ExecutionResults(TypedDict):
    result: Any
    execution_time: float
    error_msg: str


class SandboxExecutor:

    def __init__(
        self,
        evaluate_worker: Any,
        recur_kill_eval_proc: bool = False,
        debug_mode: bool = False,
        *,
        join_timeout_seconds: int = 10,
        shm_max_size_bytes: int = 20 * 1024 * 1024,
    ):
        """Evaluator using multiprocessing for secure, isolated execution.

        Args:
            evaluate_worker: The worker object to be executed.
            recur_kill_eval_proc: If using 'self.secure_evaluate', kill children processes
                when they are terminated. Note that it is suggested to set to 'False' if the evaluation process
                does not start new processes.
            debug_mode: Debug mode.
            join_timeout_seconds: Timeout in seconds to wait for the process to finish. Kill the process if timeout.
            shm_max_size_bytes: Max size of shared memory in bytes (default 20MB).
        """
        self.evaluate_worker = evaluate_worker
        self.debug_mode = debug_mode
        self.recur_kill_eval_proc = recur_kill_eval_proc
        self.join_timeout_seconds = join_timeout_seconds
        self.shm_max_size_bytes = shm_max_size_bytes

    def _kill_process_and_its_children(self, process: multiprocessing.Process):
        # Find all children processes
        children_processes = []
        if self.recur_kill_eval_proc:
            try:
                parent = psutil.Process(process.pid)
                children_processes = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                children_processes = []

        # Terminate parent process
        process.terminate()
        process.join(timeout=self.join_timeout_seconds)
        if process.is_alive():
            process.kill()
            process.join()

        # Kill all children processes
        for child in children_processes:
            try:
                if self.debug_mode:
                    print(
                        f"Killing process {process.pid}'s children process {child.pid}"
                    )
                if child.is_running():
                    child.terminate()
            except:
                if self.debug_mode:
                    traceback.print_exc()

    def _execute_and_put_res_in_shared_memory(
        self,
        worker_execute_method_name: str,
        method_args: Optional[List | Tuple],
        method_kwargs: Optional[Dict],
        meta_queue: multiprocessing.Queue,
        redirect_to_devnull: bool,
        shm_name_id: str,
    ):
        """Evaluate and store result in shared memory (for large results)."""
        # Redirect STDOUT and STDERR to '/dev/null'
        if redirect_to_devnull:
            _redirect_to_devnull()

        if hasattr(self.evaluate_worker, worker_execute_method_name):
            method_to_call = getattr(self.evaluate_worker, worker_execute_method_name)
        else:
            raise RuntimeError(
                f"Method named '{worker_execute_method_name}' not found."
            )

        # Execute and get results
        # noinspection PyBroadException
        try:
            # Execute the target method and get result
            args = method_args or []
            kwargs = method_kwargs or {}
            res = method_to_call(*args, **kwargs)

            # Dump the results to data
            data = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)

            # Check if data size exceeds shared memory capacity
            if len(data) > self.shm_max_size_bytes:
                raise RuntimeError(
                    f"Result size ({len(data)} bytes) exceeds shared memory capacity ({self.shm_max_size_bytes} bytes)."
                )

            # Attach to the shared memory created by the parent
            shm = shared_memory.SharedMemory(name=shm_name_id)

            # Write data
            shm.buf[: len(data)] = data
            # We only need to send back the size, as the parent already knows the name.
            # Sending (True, size) to indicate success.
            meta_queue.put((True, len(data)))
            # Child closes its handle
            shm.close()
        except:
            if self.debug_mode:
                traceback.print_exc()
            # Put the exception message to the queue
            # Sending (False, error_message) to indicate failure.
            meta_queue.put((False, str(traceback.format_exc())))

    def secure_execute(
        self,
        worker_execute_method_name: str,
        method_args: Optional[List | Tuple] = None,
        method_kwargs: Optional[Dict] = None,
        timeout_seconds: int | float = None,
        redirect_to_devnull: bool = False,
    ) -> ExecutionResults:
        """Evaluate program in a new process.
        This enables timeout restriction and output redirection.

        Args:
            worker_execute_method_name: Name of the worker execute method.
            method_args: Arguments of the worker execute method.
            method_kwargs: Keyword arguments of the worker execute method.
            timeout_seconds: return 'None' if the execution time exceeds 'timeout_seconds'.
            redirect_to_devnull: redirect any output to '/dev/null'.

        Returns:
            Returns the execution results. If the 'get_execution_time' is True,
            the return value will be (Results, Time).
        """
        # Evaluate and get results
        # noinspection PyBroadException
        process = None
        parent_shm = None
        evaluate_start_time = time.time()
        try:
            # Create a meta queue to get meta information from the evaluation process
            meta_queue = multiprocessing.Queue()
            # Generate a unique name for the shared memory block in the PARENT process.
            unique_shm_name = f"psm_{uuid.uuid4().hex[:8]}"

            # Parent creates the shared memory
            parent_shm = shared_memory.SharedMemory(
                create=True, name=unique_shm_name, size=self.shm_max_size_bytes
            )

            process = multiprocessing.Process(
                target=self._execute_and_put_res_in_shared_memory,
                args=(
                    worker_execute_method_name,
                    method_args,
                    method_kwargs,
                    meta_queue,
                    redirect_to_devnull,
                    unique_shm_name,
                ),
            )
            process.start()

            try:
                # Try to get the metadata before timeout
                meta = meta_queue.get(timeout=timeout_seconds)
                # Calculate evaluation time
                eval_time = time.time() - evaluate_start_time
            except Empty:
                if self.debug_mode:
                    print(f"DEBUG: evaluation time exceeds {timeout_seconds}s.")

                # Evaluation timeout happens, we return 'None' as well as the actual evaluate time
                return ExecutionResults(
                    result=None,
                    execution_time=time.time() - evaluate_start_time,
                    error_msg="Evaluation timeout.",
                )

            # The 'meta' is now (Success_Flag, Data_Size_or_Error_Msg)
            success, payload = meta

            if not success:
                # Payload is the error message
                error_msg = payload
                result = None
            else:
                error_msg = ""
                # Payload is the size of the data
                size = payload
                # Use the existing shared memory in parent
                buf = bytes(parent_shm.buf[:size])
                # Load results from buffer
                result = pickle.loads(buf)

            return ExecutionResults(
                result=result, execution_time=eval_time, error_msg=error_msg
            )
        except:
            if self.debug_mode:
                print(f"DEBUG: exception in shared evaluate:\n{traceback.format_exc()}")

            return ExecutionResults(
                result=None,
                execution_time=time.time() - evaluate_start_time,
                error_msg=str(traceback.format_exc()),
            )
        finally:
            if process is not None and process.ident is not None:
                self._kill_process_and_its_children(process)
            # Critical Cleanup: Ensure the shared memory is unlinked from the OS
            # This runs whether the process finished, timed out, or crashed
            if parent_shm is not None:
                try:
                    parent_shm.close()
                    parent_shm.unlink()
                except Exception:
                    pass
