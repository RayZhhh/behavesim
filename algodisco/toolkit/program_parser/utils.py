import re
from typing import Optional


def extract_code_from_response(
    response: str, language: str = "python"
) -> Optional[str]:
    try:
        code = extract_code_from_markdown_block(response, language)
        if code is not None:
            return code
        return extract_code_by_bottom_up(response, language)
    except Exception:
        return None


def extract_code_from_markdown_block(
    response: str, language: str = "python"
) -> Optional[str]:
    """Extracts the code block from the LLM's response."""
    if not response:
        return None
    match = re.search(rf"```(?:{language})?\s*\n(.*?)\n```", response, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_code_by_bottom_up(response: str, language: str = "python") -> Optional[str]:
    """
    Code extractor based on "bottom-up deletion" idea, supporting multiple programming languages.

    Args:
        response: Raw output from LLM, may contain comments, explanations, etc.
        language: Programming language name, supports: python, javascript, java, cpp, go

    Returns:
        Extracted clean code string, or None if extraction fails
    """
    from algodisco.toolkit.program_parser.program_parser import has_syntax_error

    # Get keywords for the specified language
    lang_config = _LANGUAGE_KEYWORDS.get(language.lower())
    if lang_config is None:
        # Fallback to python if language not supported
        lang_config = _LANGUAGE_KEYWORDS["python"]
        language = "python"

    import_keywords = lang_config["import_keywords"]
    function_keywords = lang_config["function_keywords"]

    lines = response.splitlines()

    # 1. Find starting point: locate first import or function keyword
    start_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        for keyword in import_keywords + function_keywords:
            if stripped.startswith(keyword):
                start_idx = i
                break
        if start_idx != -1:
            break

    if start_idx == -1:
        return None

    # Take all content after the starting point
    candidate_lines = lines[start_idx:]

    # 2. Core logic: repeatedly parse, delete last line if error, until success
    syntax_check_available = True
    while candidate_lines:
        code_str = "\n".join(candidate_lines)
        try:
            if syntax_check_available:
                if has_syntax_error(code_str, language):
                    # Has syntax error, delete last line and retry
                    candidate_lines.pop()
                else:
                    # Parse success! Junk has been removed
                    break
            else:
                # Syntax check unavailable, try to clean up trailing comments/empty lines
                # Delete lines that are empty or start with common comment markers
                while candidate_lines:
                    last_line = candidate_lines[-1].strip()
                    if (
                        not last_line
                        or last_line.startswith("#")
                        or last_line.startswith("//")
                        or last_line.startswith("/*")
                        or last_line.startswith("*")
                    ):
                        candidate_lines.pop()
                    else:
                        break
                break
        except ImportError:
            # Language library not installed, fallback: clean up trailing comments
            syntax_check_available = False
            continue
        except Exception:
            # Other errors, exit
            break

    if not candidate_lines:
        return None

    # 3. Return extracted code
    return "\n".join(candidate_lines)


_LANGUAGE_KEYWORDS = {
    "python": {
        "import_keywords": ["import ", "from "],
        "function_keywords": ["def ", "async def ", "class "],
    },
    "javascript": {
        "import_keywords": ["import ", "require(", "from "],
        "function_keywords": ["function ", "async function ", "=>", "class "],
    },
    "java": {
        "import_keywords": ["import ", "package "],
        "function_keywords": [
            "public ",
            "private ",
            "protected ",
            "void ",
            "static ",
            "class ",
        ],
    },
    "cpp": {
        "import_keywords": ["#include", "import "],
        "function_keywords": [
            "void ",
            "int ",
            "char ",
            "auto ",
            "template<",
            "class ",
            "struct ",
        ],
    },
    "go": {
        "import_keywords": ["import ", "package "],
        "function_keywords": ["func ", "type ", "struct {"],
    },
}

if __name__ == "__main__":
    # Test cases
    print("=" * 60)
    print("Test 1: Python with trailing comments")
    print("=" * 60)
    python_test = """
Here is a Python function to solve the problem:

import os
from typing import List

def find_max(arr: List[int]) -> int:
    if not arr:
        return 0
    return max(arr)

This is additional explanation that should be filtered out.
And some more commentary here.
"""
    result = extract_code_by_bottom_up(python_test, "python")
    print(result)
    print()

    print("=" * 60)
    print("Test 2: Python with no code")
    print("=" * 60)
    no_code_test = "This is just some explanation with no code."
    result = extract_code_by_bottom_up(no_code_test, "python")
    print(result)
    print()

    print("=" * 60)
    print("Test 3: JavaScript (fallback without tree-sitter)")
    print("=" * 60)
    js_test = """
Here is a JavaScript solution:

function findMax(arr) {
    if (!arr || arr.length === 0) return 0;
    return Math.max(...arr);
}

This is extra commentary.
"""
    result = extract_code_by_bottom_up(js_test, "javascript")
    print(result)
    print()

    print("=" * 60)
    print("Test 4: Go (fallback without tree-sitter)")
    print("=" * 60)
    go_test = """
Here is a Go solution:

package main

import "fmt"

func findMax(arr []int) int {
    if len(arr) == 0 {
        return 0
    }
    max := arr[0]
    for _, v := range arr {
        if v > max {
            max = v
        }
    }
    return max
}
"""
    result = extract_code_by_bottom_up(go_test, "go")
    print(result)
    print()

    print("=" * 60)
    print("Test 5: Default language (Python)")
    print("=" * 60)
    default_test = """
import json

def parse_data(data):
    return json.loads(data)
"""
    result = extract_code_by_bottom_up(default_test)
    print(result)
    print()
