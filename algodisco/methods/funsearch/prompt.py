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

_TASK_DESCRIPTION_TEMPLATE = "Task Description: {task_description}"

_INTRO_TEMPLATE = dedent("""
    Please help me design a novel {language_capitalized} algorithm.
    Here is an example algorithm function implementation:
    """).strip()

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
    """).strip()

_IMPROVEMENT_REQUEST_WITH_IDEA_TEMPLATE = dedent("""
    Please generate an improved version of the algorithm.
    Think outside the box. Do not modify the function signature (i.e., function name, args, ...)

    Return your answer in exactly the following format:

    ### Idea
    <algorithm idea>

    ### Code
    ```{language}
    ...
    ```

    Do not output anything before or after these two sections.
    """).strip()


class PromptAdapter:
    """Constructs prompts for the LLM to generate new algorithm variations."""

    def _wrap_markdown_code_block(self, code: str, language: str = "python") -> str:
        return f"```{language}\n{code.strip()}\n```"

    def _render_prompt(self, template: str, **kwargs: str) -> str:
        return template.format(**kwargs).strip()

    def construct_prompt(
        self,
        sorted_programs: List[AlgoProto],
        task_description: str = "",
        idea_prompt: bool = False,
    ) -> str:
        """Constructs a few-shot prompt to guide the LLM."""
        language = sorted_programs[0].language if sorted_programs else "python"
        language_capitalized = language.capitalize()

        prompt = ""
        if task_description:
            prompt += self._render_prompt(
                _TASK_DESCRIPTION_TEMPLATE,
                task_description=task_description,
            )
            prompt += "\n\n"

        prompt += self._render_prompt(
            _INTRO_TEMPLATE,
            language_capitalized=language_capitalized,
        )
        if len(sorted_programs) > 1:
            prompt += "\n\n"
            prompt += self._render_prompt(
                _VERSION_TEMPLATE,
                version="1",
                code_block=self._wrap_markdown_code_block(
                    str(sorted_programs[0].program), language
                ),
            )
        else:
            prompt += "\n"
            prompt += self._wrap_markdown_code_block(
                str(sorted_programs[0].program), language
            )

        for i in range(1, len(sorted_programs)):
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
                    str(sorted_programs[i].program), language
                ),
            )
        prompt += "\n\n"
        prompt += self._render_prompt(
            (
                _IMPROVEMENT_REQUEST_WITH_IDEA_TEMPLATE
                if idea_prompt
                else _IMPROVEMENT_REQUEST_TEMPLATE
            ),
            language=language,
        )
        return prompt

    def extract_code(self, response: str, language: str = "python") -> Optional[str]:
        """Extracts the code block from the LLM's response."""
        return extract_code_from_response(response, language)


if __name__ == "__main__":
    adapter = PromptAdapter()
    programs = [
        AlgoProto(program="def solve(n):\n    return n + 1", language="python"),
        AlgoProto(program="def solve(n):\n    return n * 2", language="python"),
        AlgoProto(program="def solve(n):\n    return n * n", language="python"),
    ]

    print("=== prompt ===")
    print(
        adapter.construct_prompt(programs, "Design a stronger transformation function.")
    )
    print("\n=== prompt with idea ===")
    print(
        adapter.construct_prompt(
            programs, "Design a stronger transformation function.", idea_prompt=True
        )
    )
    print("\n=== extracted code ===")
    print(
        adapter.extract_code(
            "### Idea\nUse a reverse affine transform.\n\n### Code\n```python\ndef solve(n):\n    return n - 1\n```"
        )
    )
