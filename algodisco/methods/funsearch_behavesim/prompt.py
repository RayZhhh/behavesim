# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import sys
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from algodisco.base.algo import AlgoProto
from algodisco.toolkit.program_parser.utils import extract_code_from_response

_TASK_DESCRIPTION_TEMPLATE = dedent("""
    Here is the task description:
    <task>
    {task_description}
    </task>

    Below are some existing {language_capitalized} algorithms that attempt to solve this task.
    They are provided for context on different approaches.
    """).strip()

_NO_TASK_DESCRIPTION_TEMPLATE = (
    "Please help me design a novel {language_capitalized} algorithm function. "
    "Here is an example algorithm function implementation:"
)

_VERSION_TEMPLATE = dedent("""
    [Version {version}]
    {code_block}
    """).strip()

_OUTPERFORMS_TEMPLATE = (
    "We find that the below version outperforms [Version {version}]."
)

_IMPROVEMENT_REQUEST_TEMPLATE = dedent("""
    Please generate an improved version of the algorithm.
    Think outside the box. Do not modify the function signature (i.e., function name, args, ...)
    Please generate your algorithm in ```{language} ...``` block.
    Only output the code and do not give additional outputs.
    Your code should be concise if possible.
    """).strip()


class PromptAdapter:
    """Constructs prompts for the LLM to generate new algorithm variations."""

    def _wrap_markdown_code_block(self, code: str, language: str = "python") -> str:
        return f"```{language}\n{code.strip()}\n```"

    def _render_prompt(self, template: str, **kwargs: str) -> str:
        return template.format(**kwargs).strip()

    def construct_prompt(
        self, task_description: str, sorted_algos: List[AlgoProto]
    ) -> str:
        """Constructs a few-shot prompt to guide the LLM."""
        language = sorted_algos[0].language if sorted_algos else "python"
        language_capitalized = language.capitalize()

        if task_description:
            prompt = self._render_prompt(
                _TASK_DESCRIPTION_TEMPLATE,
                task_description=task_description,
                language_capitalized=language_capitalized,
            )
        else:
            prompt = self._render_prompt(
                _NO_TASK_DESCRIPTION_TEMPLATE,
                language_capitalized=language_capitalized,
            )

        if len(sorted_algos) > 1:
            prompt += "\n\n"
        else:
            prompt += "\n"

        prompt += (
            self._render_prompt(
                _VERSION_TEMPLATE,
                version="1",
                code_block=self._wrap_markdown_code_block(
                    str(sorted_algos[0].program), language
                ),
            )
            if len(sorted_algos) > 1
            else self._wrap_markdown_code_block(str(sorted_algos[0].program), language)
        )

        for i in range(1, len(sorted_algos)):
            prompt += "\n\n"
            prompt += self._render_prompt(
                _OUTPERFORMS_TEMPLATE,
                version=str(i),
            )
            prompt += "\n"
            prompt += self._render_prompt(
                _VERSION_TEMPLATE,
                version=str(i + 1),
                code_block=self._wrap_markdown_code_block(
                    str(sorted_algos[i].program), language
                ),
            )

        prompt += "\n\n"
        prompt += self._render_prompt(
            _IMPROVEMENT_REQUEST_TEMPLATE,
            language=language,
        )
        return prompt

    def extract_code(self, response: str, language: str = "python") -> Optional[str]:
        """Extracts the Python code block from the LLM's response."""
        return extract_code_from_response(response, language)


if __name__ == "__main__":
    adapter = PromptAdapter()
    algos = [
        AlgoProto(program="def choose_move(state):\n    return 0", language="python"),
        AlgoProto(
            program="def choose_move(state):\n    return len(state) % 3",
            language="python",
        ),
        AlgoProto(
            program="def choose_move(state):\n    return sum(state) % 3",
            language="python",
        ),
    ]

    print("=== prompt ===")
    print(adapter.construct_prompt("Design a better behavior policy.", algos))
    print("\n=== extracted code ===")
    print(adapter.extract_code("```python\ndef choose_move(state):\n    return 1\n```"))
