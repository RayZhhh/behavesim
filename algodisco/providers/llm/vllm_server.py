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
    print(f"[vLLM] Launching vLLM on GPU:{gpus}; URL: https://{host}:{port}")
    print("=" * 80)
    cmd = cmd_list[0] + " \\\n"
    for c in cmd_list[1:]:
        cmd += "    " + c + " \\\n"
    print(cmd.strip())
    print("=" * 80 + "\n", flush=True)


class VLLMServer(LanguageModel):
    def __init__(
        self,
        model_path: str,
        port: int,
        gpus: int | list[int],
        tokenizer_path: Optional[str] = None,
        max_model_len: int = 16384,
        max_lora_rank: Optional[int] = None,
        host: str = "0.0.0.0",
        mem_util: float = 0.85,
        deploy_timeout_seconds: int = 600,
        *,
        launch_vllm_in_init=True,
        enforce_eager: bool = False,
        vllm_log_level: Literal[
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        ] = "INFO",
        silent_mode: bool = False,
        env_variable_dict: Optional[Dict[str, str]] = None,
        vllm_serve_args: Optional[List[str]] = None,
        vllm_serve_kwargs: Optional[Dict[str, str]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Deploy an LLM on specified GPUs.

        Args:
            model_path: Path to the model to deploy.
            tokenizer_path: Path to the tokenizer to use.
            port: Port to deploy.
            gpus: List of GPUs to deploy.
            max_lora_rank: Max rank of LoRA adapter. Defaults to `None` which disables LoRA adapter.
            host: Host address for vLLM server.
            mem_util: Memory utility for each vLLM deployment.
            deploy_timeout_seconds: Timeout to deploy (in seconds).
            launch_vllm_in_init: Launch vLLM during initialization of this class.
            enforce_eager: Enforce eager mode.
            vllm_log_level: Log level of vLLM server.
            silent_mode: Silent mode.
            env_variable_dict: Environment variables to use for vLLM server, e.g., {'KEY': 'VALUE'}.
            vllm_serve_args: Arguments to pass to vLLM server, e.g., ['--enable-reasoning'].
            vllm_serve_kwargs: Keyword arguments to pass to vLLM server, e.g., {'--reasoning-parser': 'deepseek-r1'}.
            chat_template_kwargs: Keyword arguments to pass to chat template, e.g., {'enable_thinking': False}.

        Example:
            # deploy a model on GPU 0 and 1
            llm = VLLMServer(
                model_path='path/to/model',
                tokenizer_path='path/to/tokenizer',
                gpus=[0, 1],  # set gpus=0 or gpus=[0] if you only use one GPU
                port=12001,
                mem_util=0.8
            )
            # draw sample using base model
            llm.draw_sample('hello')

            # load adapter and draw sample
            llm.load_lora_adapter('adapter_1', '/path/to/adapter')
            llm.draw_sample('hello', lora_name='adapter_1')

            # unload adapter
            llm.unload_lora_adapter('adapter_1')

            # release resources
            llm.close()
        """
        self._model_path = model_path
        self._port = port
        self._gpus = gpus
        self._tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else model_path
        )
        self._max_model_len = max_model_len
        self._max_lora_rank = max_lora_rank
        self._host = host
        self._mem_util = mem_util
        self._deploy_timeout_seconds = deploy_timeout_seconds
        self._enforce_eager = enforce_eager
        self._vllm_log_level = vllm_log_level
        self._silent_mode = silent_mode
        self._env_variable_dict = env_variable_dict
        self._vllm_serve_args = vllm_serve_args
        self._vllm_serve_kwargs = vllm_serve_kwargs
        self._chat_template_kwargs = chat_template_kwargs
        self._vllm_server_process = None

        # Deploy vLLMs
        if launch_vllm_in_init:
            self.launch_vllm_server()

    def launch_vllm_server(self, detach: bool = False, skip_if_running: bool = False):
        try:
            import vllm
        except ImportError:
            raise

        if skip_if_running and self._is_server_running():
            print(
                f"[vLLM] Server already running on http://{self._host}:{self._port}. "
                f"Skipping launch."
            )
            return

        self._detached = detach
        self._vllm_server_process = self._launch_vllm(detach=detach)
        self._wait_for_vllm()

    def _launch_vllm(self, detach: bool = False):
        """Launch a vLLM server and return the subprocess."""
        if isinstance(self._gpus, int):
            gpus = str(self._gpus)
        else:
            gpus = ",".join([str(g) for g in self._gpus])

        executable_path = sys.executable
        cmd = [
            executable_path,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self._model_path,
            "--tokenizer",
            self._tokenizer_path,
            "--max_model_len",
            str(self._max_model_len),
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--gpu-memory-utilization",
            str(self._mem_util),
            "--tensor-parallel-size",
            str(len(self._gpus)) if isinstance(self._gpus, list) else "1",
            "--trust-remote-code",
            "--chat-template-content-format",
            "string",
        ]

        if self._enforce_eager:
            cmd.append("--enforce_eager")

        # Other args for vllm serve
        if self._vllm_serve_args is not None:
            for arg in self._vllm_serve_args:
                cmd.append(arg)

        # Other kwargs for vllm serve
        if self._vllm_serve_kwargs is not None:
            for kwarg, value in self._vllm_serve_kwargs.items():
                cmd.extend([kwarg, value])

        # Environmental variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["VLLM_LOGGING_LEVEL"] = self._vllm_log_level

        # FIXME: These code are required for my machine :(
        # FIXME: This may due to the bad NCCL configuration :(
        if isinstance(self._gpus, list) and len(self._gpus) > 1:
            # set NCCL environment variable
            env["NCCL_P2P_DISABLE"] = "1"
            # disable custom all reduce
            cmd.append("--disable-custom-all-reduce")

        # Enable LoRA dynamic loading
        if self._max_lora_rank is not None:
            cmd.extend(
                [
                    "--enable-lora",
                    "--max-lora-rank",
                    str(self._max_lora_rank),
                ]
            )
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

        # Other env variables
        if self._env_variable_dict is not None:
            for k, v in self._env_variable_dict.items():
                env[k] = v

        _print_cmd_list(cmd, gpus=self._gpus, host=self._host, port=self._port)

        # Launch vllm using subprocess
        stdout = Path(os.devnull).open("w") if self._silent_mode else None
        preexec_fn = os.setsid if detach and sys.platform != "win32" else None
        proc = subprocess.Popen(
            cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT, preexec_fn=preexec_fn
        )
        return proc

    def _kill_vllm_process(self):
        if getattr(self, "_detached", False):
            print(
                f"[vLLM] Server on port {self._port} is detached. Not killing process."
            )
            return

        try:
            # Get child processes before terminating parent
            try:
                parent = psutil.Process(self._vllm_server_process.pid)
                children = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                children = []

            # Terminate parent process
            self._vllm_server_process.terminate()
            self._vllm_server_process.wait(timeout=5)
            print(f"[vLLM] terminated process: {self._vllm_server_process.pid}")

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
            self._vllm_server_process.kill()
            print(f"[vLLM] killed process: {self._vllm_server_process.pid}")

    def _is_server_running(self):
        """Check if a vLLM server is already running on the given host and port."""
        health = f"http://{self._host}:{self._port}/health"
        try:
            if requests.get(health, timeout=1).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        return False

    def _wait_for_vllm(self):
        """Check each vLLM server's state and check /health. Kill all vLLM server processes if timeout."""
        for _ in range(self._deploy_timeout_seconds):
            # check process status
            if self._vllm_server_process.poll() is not None:
                sys.exit(
                    f"[vLLM] crashed (exit {self._vllm_server_process.returncode})"
                )

            # check server status
            if self._is_server_running():
                return
            time.sleep(1)

        # Servers fail to initialize
        print("[vLLM] failed to start within timeout")
        self._kill_vllm_process()
        sys.exit("[vLLM] failed to start within timeout")

    def unload_lora_adapter(self, lora_name: str):
        """Unload lora adapter given the lora name.
        Args:
            lora_name: Lora adapter name.
        """
        lora_api_url = f"http://{self._host}:{self._port}/v1/unload_lora_adapter"
        headers = {"Content-Type": "application/json"}
        try:
            payload = {"lora_name": lora_name}
            requests.post(lora_api_url, json=payload, headers=headers, timeout=10)
        except requests.exceptions.RequestException:
            pass

    def load_lora_adapter(
        self, lora_name: str, new_adapter_path: str, num_trails: int = 5
    ):
        """Dynamically load a LoRA adapter.

        Args:
            lora_name: LoRA adapter name.
            new_adapter_path: Path to the new LoRA adapter weights.
        """
        # First unload lora adapter
        self.unload_lora_adapter(lora_name)

        if self._max_lora_rank is None:
            raise ValueError(
                'LoRA is not enabled for this VLLMServer instance, since "max_lora_rank" is not set.'
            )

        # Prepare the payload for LoRA update
        payload = {"lora_name": lora_name, "lora_path": new_adapter_path}
        headers = {"Content-Type": "application/json"}
        lora_api_url = f"http://{self._host}:{self._port}/v1/load_lora_adapter"

        # Repeatedly trying to load lora adapters
        for i in range(num_trails):
            try:
                response = requests.post(
                    lora_api_url, json=payload, headers=headers, timeout=60
                )
                if response.status_code == 200:
                    print(
                        f"[vLLM] Successfully load LoRA adapter: {lora_name} from {new_adapter_path}"
                    )
                else:
                    print(
                        f"[vLLM] Failed to load LoRA adapter. "
                        f"Status code: {response.status_code}, Response: {response.text}"
                    )
                return True
            except requests.exceptions.RequestException:
                continue

        print(f"[vLLM] Error loading LoRA adapter.")
        return False

    def close(self):
        """Shut down vLLM server and kill all vLLM processes."""
        if self._vllm_server_process is not None:
            self._kill_vllm_process()

    def reload(self):
        """Reload the vllm server."""
        self.launch_vllm_server()

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
        """Send a chat completion query with OpenAI format to the vLLM server.
        Return the response content.

        Args:
            message: The message in str or openai format.
            max_tokens: The maximum number of tokens to generate.
            timeout_seconds: The timeout seconds.
            lora_name: Lora adapter name. Defaults to None which uses base model.
            temperature: The temperature parameter.
            top_p: The top p parameter.
            chat_template_kwargs: The chat template kwargs, e.g., {'enable_thinking': False}.
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

        # Use the specified lora adapter
        if lora_name is not None:
            data["model"] = lora_name
        # Chat template keyword args
        if self._chat_template_kwargs is not None:
            data["chat_template_kwargs"] = self._chat_template_kwargs
        elif chat_template_kwargs is not None:
            data["chat_template_kwargs"] = chat_template_kwargs
        # Request
        url = f"http://{self._host}:{self._port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            url, headers=headers, json=data, timeout=timeout_seconds
        )
        return response.json()["choices"][0]["message"]["content"]

    def embedding(
        self,
        text: str | List[str],
        dimensions: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        lora_name: Optional[str] = None,
        **kwargs,
    ) -> List[float] | List[List[float]]:
        """Generate embeddings for the given text(s).

        Args:
            text: The text or a list of texts to embed.
            dimensions: The number of dimensions for the output embeddings.
            timeout_seconds: The timeout seconds.
            lora_name: Lora adapter name. Defaults to None which uses base model.

        Returns:
            The embedding for the text, or a list of embeddings for the list of texts.
        """
        is_str_input = isinstance(text, str)
        if is_str_input:
            text = [text]

        # Prepare arguments for the API call
        data = {"input": text}
        data: Dict[str, Any]

        if dimensions is not None:
            data["dimensions"] = dimensions

        if lora_name is not None:
            data["model"] = lora_name

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
