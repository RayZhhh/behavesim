# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import sys
import re
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from algodisco.base.algo import AlgoProto
from algodisco.toolkit.program_parser.utils import extract_code_from_response

_OUTPUT_INSTRUCTIONS_TEMPLATE = dedent("""
    Return your answer in exactly the following format:

    ### Idea
    <algorithm idea>

    ### Code
    ```{language}
    ...
    ```

    Do not output anything before or after these two sections.
    """).strip()

_INDIV_TEMPLATE = dedent("""
    No. {index} algorithm idea and the corresponding code are:
    ### Idea
    {idea}

    ### Code
    {code}
    """).strip()

_ALGO_TEMPLATE = dedent("""
    ### Idea
    {idea}

    ### Code
    {code}
    """).strip()

_PROMPT_TEMPLATE_I1 = dedent("""
    {task_description}

    Please help me design a novel {language_capitalized} algorithm.

    1. First, describe your new algorithm idea and main steps under `### Idea`.
    2. Next, implement the following {language_capitalized} function under `### Code`:
    {template_program}

    {output_instructions}
    """).strip()

_PROMPT_TEMPLATE_E1 = dedent("""
    {task_description}

    I have {num_indivs} existing algorithms with their codes as follows:

    {indivs_prompt}

    Please help me create a new algorithm that has a totally different form from the given ones.

    1. First, describe your new algorithm idea and main steps under `### Idea`.
    2. Next, implement the following {language_capitalized} function under `### Code`:
    {template_program}

    {output_instructions}
    """).strip()

_PROMPT_TEMPLATE_E2 = dedent("""
    {task_description}

    I have {num_indivs} existing algorithms with their codes as follows:

    {indivs_prompt}

    Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.

    1. First, identify the common backbone idea in the provided algorithms internally.
    2. Next, based on that backbone idea, describe your new algorithm idea under `### Idea`.
    3. Then, implement the following {language_capitalized} function under `### Code`:
    {template_program}

    {output_instructions}
    """).strip()

_PROMPT_TEMPLATE_M1 = dedent("""
    {task_description}

    I have one algorithm with its code as follows.

    {algo_prompt}

    Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.

    1. First, describe your new algorithm idea and main steps under `### Idea`.
    2. Next, implement the following {language_capitalized} function under `### Code`:
    {template_program}

    {output_instructions}
    """).strip()

_PROMPT_TEMPLATE_M2 = dedent("""
    {task_description}

    I have one algorithm with its code as follows.

    {algo_prompt}

    Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.

    1. First, describe your new algorithm idea and main steps under `### Idea`.
    2. Next, implement the following {language_capitalized} function under `### Code`:
    {template_program}

    {output_instructions}
    """).strip()


class EoHPromptAdapter:
    """Constructs prompts for EoH operators: i1, e1, e2, m1, m2."""

    def _wrap_markdown_code_block(self, code: str, language: str = "python") -> str:
        return f"```{language}\n{code.strip()}\n```"

    def _format_output_instructions(self, language: str) -> str:
        return _OUTPUT_INSTRUCTIONS_TEMPLATE.format(language=language)

    def _format_indiv(self, index: int, indi: AlgoProto, language: str) -> str:
        idea = indi.get("idea", "No description")
        code = self._wrap_markdown_code_block(str(indi.program), language)
        return _INDIV_TEMPLATE.format(
            index=index,
            idea=idea,
            code=code,
        )

    def _format_algo(self, indi: AlgoProto, language: str) -> str:
        idea = indi.get("idea", "No description")
        code = self._wrap_markdown_code_block(str(indi.program), language)
        return _ALGO_TEMPLATE.format(
            idea=idea,
            code=code,
        )

    def _render_prompt(self, template: str, **kwargs: str) -> str:
        return template.format(**kwargs).strip()

    def construct_prompt_i1(
        self, task_description: str, template_program: str, language: str = "python"
    ) -> str:
        return self._render_prompt(
            _PROMPT_TEMPLATE_I1,
            task_description=task_description.strip(),
            language_capitalized=language.capitalize(),
            template_program=template_program.strip(),
            output_instructions=self._format_output_instructions(language),
        )

    def construct_prompt_e1(
        self,
        task_description: str,
        indivs: List[AlgoProto],
        template_program: str,
        language: str = "python",
    ) -> str:
        indivs_prompt = "\n\n".join(
            self._format_indiv(i + 1, indi, language) for i, indi in enumerate(indivs)
        )
        return self._render_prompt(
            _PROMPT_TEMPLATE_E1,
            task_description=task_description.strip(),
            num_indivs=str(len(indivs)),
            indivs_prompt=indivs_prompt,
            language_capitalized=language.capitalize(),
            template_program=template_program.strip(),
            output_instructions=self._format_output_instructions(language),
        )

    def construct_prompt_e2(
        self,
        task_description: str,
        indivs: List[AlgoProto],
        template_program: str,
        language: str = "python",
    ) -> str:
        indivs_prompt = "\n\n".join(
            self._format_indiv(i + 1, indi, language) for i, indi in enumerate(indivs)
        )
        return self._render_prompt(
            _PROMPT_TEMPLATE_E2,
            task_description=task_description.strip(),
            num_indivs=str(len(indivs)),
            indivs_prompt=indivs_prompt,
            language_capitalized=language.capitalize(),
            template_program=template_program.strip(),
            output_instructions=self._format_output_instructions(language),
        )

    def construct_prompt_m1(
        self,
        task_description: str,
        indi: AlgoProto,
        template_program: str,
        language: str = "python",
    ) -> str:
        return self._render_prompt(
            _PROMPT_TEMPLATE_M1,
            task_description=task_description.strip(),
            algo_prompt=self._format_algo(indi, language),
            language_capitalized=language.capitalize(),
            template_program=template_program.strip(),
            output_instructions=self._format_output_instructions(language),
        )

    def construct_prompt_m2(
        self,
        task_description: str,
        indi: AlgoProto,
        template_program: str,
        language: str = "python",
    ) -> str:
        return self._render_prompt(
            _PROMPT_TEMPLATE_M2,
            task_description=task_description.strip(),
            algo_prompt=self._format_algo(indi, language),
            language_capitalized=language.capitalize(),
            template_program=template_program.strip(),
            output_instructions=self._format_output_instructions(language),
        )

    def _extract_section(
        self,
        response: str,
        section_name: str,
        next_sections: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not response:
            return None
        next_sections = next_sections or []
        if next_sections:
            next_pattern = "|".join(re.escape(section) for section in next_sections)
            pattern = (
                rf"^\s*###\s*{re.escape(section_name)}\s*$\s*"
                rf"(.*?)(?=^\s*###\s*(?:{next_pattern})\s*$|\Z)"
            )
        else:
            pattern = rf"^\s*###\s*{re.escape(section_name)}\s*$\s*(.*)\Z"

        match = re.search(pattern, response, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def extract_idea(self, response: str) -> Optional[str]:
        section_idea = self._extract_section(response, "Idea", next_sections=["Code"])
        return section_idea.strip() if section_idea else None

    def extract_code(self, response: str, language: str = "python") -> Optional[str]:
        code_section = self._extract_section(response, "Code")
        if not code_section:
            return None
        return extract_code_from_response(code_section, language)


if __name__ == "__main__":
    adapter = EoHPromptAdapter()
    task_description = "Design an algorithm to return the maximum value in a list."
    template_program = dedent("""
        def solve(values: list[int]) -> int:
            pass
        """).strip()

    parent_a = AlgoProto(
        program=dedent("""
            def solve(values: list[int]) -> int:
                return max(values)
            """).strip(),
        language="python",
    )
    parent_a["idea"] = "Return the largest element directly with a built-in reduction."

    parent_b = AlgoProto(
        program=dedent("""
            def solve(values: list[int]) -> int:
                best = values[0]
                for value in values[1:]:
                    if value > best:
                        best = value
                return best
            """).strip(),
        language="python",
    )
    parent_b["idea"] = "Scan once and keep the running maximum."

    print("=== i1 ===")
    print(adapter.construct_prompt_i1(task_description, template_program))
    print("\n=== e1 ===")
    print(
        adapter.construct_prompt_e1(
            task_description, [parent_a, parent_b], template_program
        )
    )
    print("\n=== e2 ===")
    print(
        adapter.construct_prompt_e2(
            task_description, [parent_a, parent_b], template_program
        )
    )
    print("\n=== m1 ===")
    print(adapter.construct_prompt_m1(task_description, parent_a, template_program))
    print("\n=== m2 ===")
    print(adapter.construct_prompt_m2(task_description, parent_b, template_program))

    sample_response = dedent("""
        ### Idea
        Partition the input into chunks, compute chunk maxima, then combine them.

        ### Code
        ```python
        def solve(values: list[int]) -> int:
            chunk_size = max(1, len(values) // 4)
            best = values[0]
            for i in range(0, len(values), chunk_size):
                best = max(best, max(values[i : i + chunk_size]))
            return best
        ```
        """).strip()

    print("\n=== extracted idea ===")
    print(adapter.extract_idea(sample_response))
    print("\n=== extracted code ===")
    print(adapter.extract_code(sample_response))
