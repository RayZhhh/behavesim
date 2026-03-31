# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

from typing import Optional, List, Literal, Dict, Any
import os
import subprocess
import sys
from pathlib import Path
import psutil
import time

import openai.types.chat
import requests

from algodisco.base.llm import LanguageModel


def _print_cmd_list(cmd_list, gpus, host, port):
    print("\n" + "=" * 80)
    print(f"[SGLang] Launching SGLang on GPU:{gpus}; URL: http://{host}:{port}")
    print("=" * 80)
    cmd = cmd_list[0] + " \\\n"
    for c in cmd_list[1:]:
        cmd += "    " + c + " \\\n"
    print(cmd.strip())
    print("=" * 80 + "\n", flush=True)


class SGLangServer(LanguageModel):
    def __init__(
        self,
        model_path: str,
        port: int,
        gpus: int | list[int],
        tokenizer_path: Optional[str] = None,
        context_length: int = 16384,
        max_lora_rank: Optional[int] = None,
        host: str = "0.0.0.0",
        mem_fraction_static: float = 0.85,
        deploy_timeout_seconds: int = 600,
        *,
        launch_sglang_in_init=True,
        sglang_log_level: Literal["debug", "info", "warning", "error"] = "info",
        silent_mode: bool = False,
        env_variable_dict: Optional[Dict[str, str]] = None,
        sglang_serve_args: Optional[List[str]] = None,
        sglang_serve_kwargs: Optional[Dict[str, str]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Deploy an SGLang server on specified GPUs.

        Args:
            model_path: Path to the model to deploy.
            port: Port to deploy.
            gpus: List of GPUs to deploy.
            tokenizer_path: Path to the tokenizer. Defaults to model_path.
            context_length: The context length (mapped to --context-length).
            max_lora_rank: Max rank of LoRA adapter. Defaults to `None`.
                Set this to enable LoRA support (mapped to --max-lora-rank).
            host: Host address for SGLang server.
            mem_fraction_static: The memory fraction for static allocation (mapped to --mem-fraction-static).
            deploy_timeout_seconds: Timeout to deploy (in seconds).
            launch_sglang_in_init: Launch SGLang during initialization of this class.
            sglang_log_level: Log level.
            silent_mode: Silent mode.
            env_variable_dict: Environment variables to use.
            sglang_serve_args: Additional arguments to pass to sglang server, e.g., ['--enable-flashinfer'],
                or ['--attention-backend', 'triton']
            sglang_serve_kwargs: Keyword arguments to pass to sglang server.
            chat_template_kwargs: Keyword arguments for chat template (passed during request).

        Example:
            # deploy a model on GPU 0 and 1 with LoRA support
            llm = SGLangServer(
                model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                port=30000,
                gpus=[0, 1],
                max_lora_rank=16,  # Enable LoRA
                sglang_serve_args=['--attention-backend', 'triton']
            )

            # Load an adapter
            llm.load_lora_adapter("my_adapter", "/path/to/adapter")

            # Use the adapter
            llm.chat_completion("Hello", lora_name="my_adapter")
        """
        self._model_path = model_path
        self._port = port
        self._gpus = gpus
        self._tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else model_path
        )
        self._context_length = context_length
        self._max_lora_rank = max_lora_rank
        self._host = host
        self._mem_fraction_static = mem_fraction_static
        self._deploy_timeout_seconds = deploy_timeout_seconds
        self._sglang_log_level = sglang_log_level
        self._silent_mode = silent_mode
        self._env_variable_dict = env_variable_dict
        self._sglang_serve_args = sglang_serve_args
        self._sglang_serve_kwargs = sglang_serve_kwargs
        self._chat_template_kwargs = chat_template_kwargs
        self._server_process = None

        # Deploy SGLang
        if launch_sglang_in_init:
            self.launch_sglang_server()

    def launch_sglang_server(self, detach: bool = False, skip_if_running: bool = False):
        try:
            import sglang
        except ImportError:
            raise

        if skip_if_running and self._is_server_running():
            print(
                f"[SGLang] Server already running on http://{self._host}:{self._port}. "
                f"Skipping launch."
            )
            return

        self._detached = detach
        self._server_process = self._launch_sglang(detach=detach)
        self._wait_for_server()

    def _launch_sglang(self, detach: bool = False):
        """Launch an SGLang server and return the subprocess."""
        if isinstance(self._gpus, int):
            gpus = str(self._gpus)
            tp_size = 1
        else:
            gpus = ",".join([str(g) for g in self._gpus])
            tp_size = len(self._gpus)

        executable_path = sys.executable

        # SGLang launch command structure
        cmd = [
            executable_path,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self._model_path,
            "--tokenizer-path",
            self._tokenizer_path,
            "--port",
            str(self._port),
            "--host",
            self._host,
            "--context-length",
            str(self._context_length),
            "--mem-fraction-static",
            str(self._mem_fraction_static),
            "--tp",
            str(tp_size),
            "--trust-remote-code",
        ]

        # Enable LoRA support if rank is specified
        if self._max_lora_rank is not None:
            cmd.extend(["--max-lora-rank", str(self._max_lora_rank)])
            # SGLang sometimes requires disabling radix cache for LoRA in older versions,
            # but newer versions support it. If you face issues, consider adding "--disable-radix-cache".

        # Other args for sglang serve
        if self._sglang_serve_args is not None:
            for arg in self._sglang_serve_args:
                cmd.append(arg)

        # Other kwargs for sglang serve
        if self._sglang_serve_kwargs is not None:
            for kwarg, value in self._sglang_serve_kwargs.items():
                cmd.extend([kwarg, value])

        # Environmental variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["LOG_LEVEL"] = self._sglang_log_level.upper()

        # Handle NCCL issues if using multiple GPUs
        if tp_size > 1:
            env["NCCL_P2P_DISABLE"] = "1"

        if self._env_variable_dict is not None:
            for k, v in self._env_variable_dict.items():
                env[k] = v

        _print_cmd_list(cmd, gpus=self._gpus, host=self._host, port=self._port)

        # Launch using subprocess
        stdout = Path(os.devnull).open("w") if self._silent_mode else None
        preexec_fn = os.setsid if detach and sys.platform != "win32" else None
        proc = subprocess.Popen(
            cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT, preexec_fn=preexec_fn
        )
        return proc

    def _kill_process(self):
        if getattr(self, "_detached", False):
            print(
                f"[SGLang] Server on port {self._port} is detached. Not killing process."
            )
            return

        try:
            # Get child processes before terminating parent
            try:
                parent = psutil.Process(self._server_process.pid)
                children = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                children = []

            # Terminate parent process
            self._server_process.terminate()
            self._server_process.wait(timeout=5)
            print(f"[SGLang] terminated process: {self._server_process.pid}")

            # Kill any remaining children
            for child in children:
                try:
                    child.terminate()
                    child.wait(timeout=2)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
        except subprocess.TimeoutExpired:
            self._server_process.kill()
            print(f"[SGLang] killed process: {self._server_process.pid}")

    def _is_server_running(self):
        """Check if an SGLang server is already running on the given host and port."""
        health = f"http://{self._host}:{self._port}/health"
        try:
            if requests.get(health, timeout=1).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        return False

    def _wait_for_server(self):
        """Check server state and /health endpoint."""
        for _ in range(self._deploy_timeout_seconds):
            if self._server_process.poll() is not None:
                sys.exit(f"[SGLang] crashed (exit {self._server_process.returncode})")

            if self._is_server_running():
                return
            time.sleep(1)

        print("[SGLang] failed to start within timeout")
        self._kill_process()
        sys.exit("[SGLang] failed to start within timeout")

    def unload_lora_adapter(self, lora_name: str):
        """Unload lora adapter given the lora name via native SGLang endpoint.

        Args:
            lora_name: Lora adapter name.
        """
        # Note: SGLang native endpoints are often at root, not /v1
        url = f"http://{self._host}:{self._port}/unload_lora_adapter"
        headers = {"Content-Type": "application/json"}
        try:
            payload = {"lora_name": lora_name}
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                print(f"[SGLang] Unloaded LoRA adapter: {lora_name}")
            else:
                print(f"[SGLang] Failed to unload LoRA: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[SGLang] Error unloading LoRA: {e}")

    def load_lora_adapter(
        self, lora_name: str, new_adapter_path: str, num_trails: int = 5
    ):
        """Dynamically load a LoRA adapter via native SGLang endpoint.

        Args:
            lora_name: LoRA adapter name.
            new_adapter_path: Path to the new LoRA adapter weights.
        """
        if self._max_lora_rank is None:
            raise ValueError(
                'LoRA is not enabled. Please set "max_lora_rank" in __init__.'
            )

        # Unload first to ensure clean state (optional but safer for updates)
        self.unload_lora_adapter(lora_name)

        url = f"http://{self._host}:{self._port}/load_lora_adapter"
        headers = {"Content-Type": "application/json"}
        payload = {"lora_name": lora_name, "lora_path": new_adapter_path}

        for i in range(num_trails):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                if response.status_code == 200:
                    print(
                        f"[SGLang] Successfully loaded LoRA adapter: {lora_name} from {new_adapter_path}"
                    )
                    return True
                else:
                    print(
                        f"[SGLang] Failed to load LoRA adapter. "
                        f"Status code: {response.status_code}, Response: {response.text}"
                    )
                    # Don't retry immediately if it's a client error (4xx)
                    if 400 <= response.status_code < 500:
                        return False
            except requests.exceptions.RequestException:
                time.sleep(1)
                continue

        print(f"[SGLang] Error loading LoRA adapter after {num_trails} trails.")
        return False

    def close(self):
        """Shut down SGLang server and kill all processes."""
        if self._server_process is not None:
            self._kill_process()

    def reload(self):
        """Reload the SGLang server."""
        self.launch_sglang_server()

    def chat_completion(
        self,
        message: str | List[openai.types.chat.ChatCompletionMessageParam],
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        lora_name: Optional[str] = None,
        temperature: float = 0.9,
        top_p: float = 0.9,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat completion query to the SGLang server.

        Args:
            message: The message in str or openai format.
            max_tokens: The maximum number of tokens to generate.
            timeout_seconds: The timeout seconds.
            lora_name: Name of the LoRA adapter to use for this request.
            temperature: The temperature parameter.
            top_p: The top p parameter.
            chat_template_kwargs: Chat template kwargs.
        """
        data = {
            "messages": [
                (
                    {"role": "user", "content": message.strip()}
                    if isinstance(message, str)
                    else message
                )
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        data: Dict[str, Any]

        # In SGLang OpenAI API, the 'model' parameter routes to the adapter
        if lora_name is not None:
            data["model"] = lora_name
        else:
            data["model"] = self._model_path

        if self._chat_template_kwargs is not None:
            data["chat_template_kwargs"] = self._chat_template_kwargs
        elif chat_template_kwargs is not None:
            data["chat_template_kwargs"] = chat_template_kwargs

        url = f"http://{self._host}:{self._port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            url, headers=headers, json=data, timeout=timeout_seconds
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def embedding(
        self,
        text: str | List[str],
        dimensions: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        lora_name: Optional[str] = None,
        **kwargs,
    ) -> List[float] | List[List[float]]:
        """Generate embeddings for the given text(s)."""
        is_str_input = isinstance(text, str)
        if is_str_input:
            text = [text]

        data = {"input": text, "model": lora_name if lora_name else self._model_path}
        data: Dict[str, Any]

        if dimensions is not None:
            data["dimensions"] = dimensions

        data.update(kwargs)

        url = f"http://{self._host}:{self._port}/v1/embeddings"
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            url, headers=headers, json=data, timeout=timeout_seconds
        )
        response.raise_for_status()

        response_json = response.json()
        embeddings = [item["embedding"] for item in response_json["data"]]

        if is_str_input:
            return embeddings[0]
        return embeddings
