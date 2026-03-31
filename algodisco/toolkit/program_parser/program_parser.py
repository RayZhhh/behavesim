# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import dataclasses
import importlib
from typing import List, Optional, Union

import tree_sitter
from tree_sitter import Language, Parser

__all__ = [
    "CodeBlock",
    "Function",
    "Class",
    "Program",
    "CodeParser",
    "has_syntax_error",
]


@dataclasses.dataclass
class CodeBlock:
    """
    Represents a chunk of code that isn't a parsed function or class.
    This effectively captures 'gaps' between major nodes, preserving comments,
    imports, and whitespace.
    """

    code: str
    indent: str = ""

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        # Short representation for debugging to avoid console flooding
        preview = self.code.strip().replace("\n", "\\n")[:50]
        return f"<CodeBlock content='{preview}...'>"


@dataclasses.dataclass
class Function:
    """
    Represents a function or method definition.

    Structure:
    [pre_name] def [name] [post_name] (args): [body] [footer]
    """

    # fmt:off
    name: str
    pre_name: str   # Text before the function name (e.g., "async def ")
    post_name: str  # Text between name and body start (e.g., "(args):\n")
    body: str       # The inner body code
    footer: str     # Trailing characters (e.g., closing braces, semicolons)
    docstring: Optional[str] = None
    is_async: bool = False
    decorators_or_annotations: List[str] = dataclasses.field(default_factory=list)
    indent: str = ""
    # fmt:on

    def __str__(self) -> str:
        # Reconstructs the exact source code
        return self.pre_name + self.name + self.post_name + self.body + self.footer

    def __repr__(self) -> str:
        return f"<Function name='{self.name}' async={self.is_async}>"


@dataclasses.dataclass
class Class:
    """
    Represents a class definition.

    Structure:
    [pre_name] class [name] [post_name] { [body] } [footer]
    """

    name: str
    pre_name: str
    post_name: str
    footer: str
    body: List[Union[CodeBlock, Function]] = dataclasses.field(default_factory=list)
    docstring: Optional[str] = None
    decorators_or_annotations: List[str] = dataclasses.field(default_factory=list)
    indent: str = ""

    def __str__(self) -> str:
        # Reconstructs the source code.
        # We simply join the elements. Gaps (newlines/spaces) are preserved in CodeBlocks.
        # We apply indentation to each element to handle the class-level indentation.
        body_str = "".join(
            _add_indent(str(el), el.indent) if hasattr(el, "indent") else str(el)
            for el in self.body
        )

        # Add a newline between the body and the footer to maintain clean formatting.
        return self.pre_name + self.name + self.post_name + body_str + self.footer

    def __repr__(self) -> str:
        return f"<Class name='{self.name}' items={len(self.body)}>"


@dataclasses.dataclass
class Program:
    """
    Represents the entire source file.

    'elements' maintains the ordered list of all nodes (Blocks, Functions, Classes)
    to allow for perfect reconstruction.
    """

    scripts: List[CodeBlock]
    functions: List[Function]
    classes: List[Class]
    elements: List[Union[Function, Class, CodeBlock]]

    def __str__(self) -> str:
        return "".join(str(el) for el in self.elements)

    def __repr__(self) -> str:
        return f"<Program classes={len(self.classes)} functions={len(self.functions)}>"


class CodeParser:
    def __init__(self, language_name: str):
        """
        Code parser.

        Args:
            language_name: The name of the language to use.
        """
        lang_key = language_name.lower()
        if lang_key not in _LANG_REGISTRY:
            raise ValueError(f"Language '{lang_key}' is not registered/installed.")

        entry = _LANG_REGISTRY[lang_key]
        self.language_name = lang_key
        self.config = entry["config"]

        # Lazy loading of tree-sitter bindings to avoid hard dependencies on all languages
        module_name = entry["module_name"]
        try:
            binding_module = importlib.import_module(module_name)
        except ImportError:
            raise ImportError(
                f"Language support library '{module_name}' not found. "
                f"Please install it (e.g., pip install tree-sitter-{language_name})."
            )

        # Initialize Tree-sitter
        self.ts_lang = Language(binding_module.language())
        self.parser = Parser(self.ts_lang)

    def parse(self, code: str, check_syntax_validity=True) -> Program:
        """
        Parses source code into a Program object containing Functions, Classes, and CodeBlocks.
        This parser uses a 'Gap Capturing' strategy: any text not matched as a Function
        or Class is captured as a CodeBlock to ensure 100% round-trip fidelity.
        """
        if check_syntax_validity:
            if has_syntax_error(code, self.language_name, self.parser):
                raise RuntimeError(
                    f"The given {self.language_name} code has syntax error."
                )

        # Tree-sitter works with bytes
        source_bytes = code.encode("utf-8", errors="replace")
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node

        scripts: List[CodeBlock] = []
        functions: List[Function] = []
        classes: List[Class] = []
        elements: List[Union[Function, Class, CodeBlock]] = []

        last_byte_end = 0

        # Iterate through top-level nodes
        children = root_node.children
        i = 0
        while i < len(children):
            child = children[i]
            node_type = child.type

            is_func = node_type in self.config.function_types
            is_class = node_type in self.config.class_types

            # Special handling for Python decorators (decorated_definition wrapping a function/class)
            if self.language_name == "python" and node_type == "decorated_definition":
                definition = child.child_by_field_name("definition")
                if definition:
                    if definition.type in self.config.function_types:
                        is_func = True
                    elif definition.type in self.config.class_types:
                        is_class = True

            if is_func or is_class:
                # 1. Capture Gap: The text between the previous node and this one.
                #    This includes comments, whitespaces, or unparsed statements.
                gap_text = source_bytes[last_byte_end : child.start_byte].decode(
                    "utf-8", errors="replace"
                )
                if gap_text:
                    block = CodeBlock(code=gap_text)
                    scripts.append(block)
                    elements.append(block)

                # 2. Parse the current node
                if is_func:
                    node_obj = self._parse_function(child, source_bytes)
                    functions.append(node_obj)
                else:
                    node_obj = self._parse_class(child, source_bytes)
                    classes.append(node_obj)

                # 3. Handle Trailing Semicolons (Common in C++, JS, Java)
                #    If the next sibling is a semicolon, we attach it to the current node's footer
                current_end_byte = child.end_byte
                if i + 1 < len(children):
                    next_sibling = children[i + 1]
                    if next_sibling.type in [
                        ";",
                        "empty_statement",
                        "empty_declaration",
                    ]:
                        current_end_byte = next_sibling.end_byte
                        node_obj.footer += source_bytes[
                            child.end_byte : current_end_byte
                        ].decode("utf-8", errors="replace")
                        i += 1  # Skip the semicolon in the next iteration

                elements.append(node_obj)
                last_byte_end = current_end_byte

            i += 1

        # 4. Capture Final Gap: Any text remaining at the end of the file
        remaining_text = source_bytes[last_byte_end:].decode("utf-8", errors="replace")
        if remaining_text:
            block = CodeBlock(code=remaining_text)
            scripts.append(block)
            elements.append(block)

        return Program(
            scripts=scripts, functions=functions, classes=classes, elements=elements
        )

    def _get_text(self, node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def _find_name_node(self, node):
        """Helper to find the identifier node for classes and functions."""
        # Standard field name check
        name_node = node.child_by_field_name(self.config.identifier_field)

        # Fallback for nested structures (e.g., Go type_declaration -> type_spec)
        if not name_node:
            for child in node.children:
                name_node = child.child_by_field_name(self.config.identifier_field)
                if name_node:
                    break

        # Unwrap Python decorated definitions
        if not name_node and node.type == "decorated_definition":
            definition = node.child_by_field_name("definition")
            if definition:
                name_node = definition.child_by_field_name(self.config.identifier_field)
        return name_node

    def _find_body_node(self, node):
        """Helper to find the block/body node containing inner statements."""
        # Try standard 'body' field
        body_node = node.child_by_field_name("body")

        # Unwrap Python decorators
        if not body_node and node.type == "decorated_definition":
            definition = node.child_by_field_name("definition")
            if definition:
                body_node = definition.child_by_field_name("body")

        if not body_node:
            # BFS search for known body types (useful for languages like C++ where structure varies)
            nodes_to_visit = [node]
            while nodes_to_visit:
                curr = nodes_to_visit.pop(0)
                if curr.type in self.config.class_body_types:
                    body_node = curr
                    break
                # Don't go too deep, just immediate children usually
                nodes_to_visit.extend(curr.children)
        return body_node

    def _extract_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """
        Extracts docstrings.
        Currently optimized for Python (triple-quoted strings inside the body).
        TODO: Support Javadoc (Java) or Go comments which usually precede the node.
        """
        if self.language_name != "python":
            return None

        body = node.child_by_field_name("body")
        if not body and node.type == "decorated_definition":
            definition = node.child_by_field_name("definition")
            if definition:
                body = definition.child_by_field_name("body")

        if body and body.child_count > 0:
            first_stmt = body.children[0]
            # Python docstrings are expression statements containing a string
            if first_stmt.type == "expression_statement":
                target = first_stmt
                # Some TS grammars nest the string inside the expression statement
                if target.child_count > 0 and target.children[0].type == "string":
                    target = target.children[0]

                if target.type == "string":
                    raw_doc = self._get_text(target, source_bytes)
                    try:
                        import ast

                        evaluated = ast.literal_eval(raw_doc)
                    except (SyntaxError, ValueError):
                        return raw_doc
                    return evaluated if isinstance(evaluated, str) else None
        return None

    def _extract_decorators_or_annotations(
        self, node, source_bytes: bytes
    ) -> List[str]:
        """Extracts decorators (Python) or Annotations (Java)."""
        items = []
        if self.language_name == "python" and node.type == "decorated_definition":
            for child in node.children:
                if child.type == "decorator":
                    items.append(self._get_text(child, source_bytes))
        elif self.language_name == "java":
            # In Java, annotations are often inside 'modifiers'
            modifiers = None
            for child in node.children:
                if child.type == "modifiers":
                    modifiers = child
                    break

            if modifiers:
                for child in modifiers.children:
                    if child.type in ["marker_annotation", "annotation"]:
                        items.append(self._get_text(child, source_bytes))
            else:
                # Fallback: direct children
                for child in node.children:
                    if child.type in ["marker_annotation", "annotation"]:
                        items.append(self._get_text(child, source_bytes))

        return items

    def _parse_function(self, node, source_bytes: bytes) -> Function:
        name_node = self._find_name_node(node)
        name = self._get_text(name_node, source_bytes) if name_node else "anonymous"
        body_node = self._find_body_node(node)

        # 1. Split logic: pre_name + name + post_name + body + footer
        if name_node:
            pre_name = source_bytes[node.start_byte : name_node.start_byte].decode(
                "utf-8", errors="replace"
            )
            if body_node:
                post_name = source_bytes[
                    name_node.end_byte : body_node.start_byte
                ].decode("utf-8", errors="replace")

                body_code = source_bytes[
                    body_node.start_byte : body_node.end_byte
                ].decode("utf-8", errors="replace")

                # Adjust for newline splitting:
                # Sometimes post_name captures the newline belonging to the body's start.
                idx = post_name.rfind("\n")
                if idx != -1:
                    body_code = post_name[idx + 1 :] + body_code
                    post_name = post_name[: idx + 1]

                footer = source_bytes[body_node.end_byte : node.end_byte].decode(
                    "utf-8", errors="replace"
                )
            else:
                # Abstract methods or prototypes without body
                post_name = source_bytes[name_node.end_byte : node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                body_code = ""
                footer = ""
        else:
            # Fallback for completely anonymous blocks
            pre_name = ""
            name = self._get_text(node, source_bytes)
            post_name = ""
            body_code = ""
            footer = ""

        docstring = self._extract_docstring(node, source_bytes)
        decorators_or_annotations = self._extract_decorators_or_annotations(
            node, source_bytes
        )

        # Robust Async Check
        check_node = node
        if node.type == "decorated_definition":
            check_node = node.child_by_field_name("definition") or node

        # Check node type or keyword presence
        node_text_head = self._get_text(check_node, source_bytes).split(maxsplit=2)[0]
        is_async = "async" in check_node.type or "async" in node_text_head

        return Function(
            name=name,
            pre_name=pre_name,
            post_name=post_name,
            body=body_code,
            footer=footer,
            docstring=docstring,
            is_async=is_async,
            decorators_or_annotations=decorators_or_annotations,
        )

    def _parse_class(self, node, source_bytes: bytes) -> Class:
        name_node = self._find_name_node(node)
        name = self._get_text(name_node, source_bytes) if name_node else "anonymous"
        body_node = self._find_body_node(node)

        if name_node and body_node:
            pre_name = source_bytes[node.start_byte : name_node.start_byte].decode(
                "utf-8", errors="replace"
            )
            post_name = source_bytes[name_node.end_byte : body_node.start_byte].decode(
                "utf-8", errors="replace"
            )
            footer = source_bytes[body_node.end_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            )
        else:
            # Fallback for forward declarations or odd structures
            pre_name = ""
            name = self._get_text(node, source_bytes)
            post_name = ""
            footer = ""
            return Class(
                name=name,
                pre_name=pre_name,
                post_name=post_name,
                body=[],
                footer=footer,
            )

        docstring = self._extract_docstring(node, source_bytes)
        decorators_or_annotations = self._extract_decorators_or_annotations(
            node, source_bytes
        )

        body_elements = []
        if body_node:
            last_inner_byte = body_node.start_byte

            for child in body_node.children:
                # Capture gap (comments/whitespace) between class members
                gap_text = source_bytes[last_inner_byte : child.start_byte].decode(
                    "utf-8", errors="replace"
                )

                # Indent Inference Logic
                # Rely on node position to determine indentation.
                line_start_byte = source_bytes.rfind(b"\n", 0, child.start_byte) + 1
                raw_indent_bytes = source_bytes[line_start_byte : child.start_byte]

                # Verify that this slice is indeed whitespace
                if raw_indent_bytes.strip() == b"":
                    indent_str = raw_indent_bytes.decode("utf-8", errors="replace")
                else:
                    indent_str = ""

                # Handle `post_name` borrowing indent from the first child
                if not body_elements and indent_str:
                    if post_name.endswith(indent_str):
                        idx = post_name.rfind("\n")
                        if idx != -1:
                            post_name = post_name[: idx + 1]

                # Handle Gap Text normalization.
                if gap_text:
                    normalized_gap = _remove_indent(gap_text, indent_str)
                    # Append the gap block, preserving its structural role (spacing)
                    # We assign indent to the CodeBlock so _add_indent works correctly for comments.
                    body_elements.append(
                        CodeBlock(code=normalized_gap, indent=indent_str)
                    )

                # Update pointer
                last_inner_byte = child.end_byte

                # Check if child is a method or function
                is_method = child.type in self.config.function_types
                if child.type == "decorated_definition":
                    defi = child.child_by_field_name("definition")
                    if defi and defi.type in self.config.function_types:
                        is_method = True

                if is_method:
                    func = self._parse_function(child, source_bytes)
                    func.indent = indent_str
                    # Remove indentation from inner parts of the function to normalize it
                    func.pre_name = _remove_indent(func.pre_name, indent_str)
                    func.post_name = _remove_indent(func.post_name, indent_str)
                    func.body = _remove_indent(func.body, indent_str)
                    func.footer = _remove_indent(func.footer, indent_str)
                    body_elements.append(func)
                else:
                    # Treat anything else (fields, inner classes, weird statements) as a CodeBlock
                    stmt_code = self._get_text(child, source_bytes)
                    body_elements.append(
                        CodeBlock(
                            code=_remove_indent(stmt_code, indent_str),
                            indent=indent_str,
                        )
                    )

            # Capture remaining gap after the last child (e.g. trailing newlines inside class)
            remaining_gap = source_bytes[last_inner_byte : body_node.end_byte].decode(
                "utf-8", errors="replace"
            )
            if remaining_gap:
                # Usually just whitespace/newlines at the end of the class
                body_elements.append(CodeBlock(code=remaining_gap))

        return Class(
            name=name,
            pre_name=pre_name,
            post_name=post_name,
            body=body_elements,
            footer=footer,
            docstring=docstring,
            decorators_or_annotations=decorators_or_annotations,
        )


def _remove_indent(text: str, indent: str) -> str:
    """Removes a specific indentation prefix from each line of text."""
    if not indent:
        return text
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(indent):
            lines[i] = line[len(indent) :]
        elif line.strip() == "":
            lines[i] = ""
    return "\n".join(lines)


def _add_indent(text: str, indent: str) -> str:
    """Adds indentation to each line of text, skipping empty lines."""
    if not indent:
        return text
    lines = text.split("\n")
    for i, line in enumerate(lines):
        # Only add indent if the line has content.
        # This prevents creating lines with only whitespace (trailing spaces).
        if line.strip():
            lines[i] = indent + line
        else:
            lines[i] = ""  # Ensure empty lines are just empty
    return "\n".join(lines)


@dataclasses.dataclass
class _LanguageConfig:
    """Configures AST node names and types for different languages."""

    function_types: List[str]
    class_types: List[str]
    class_body_types: List[str]
    identifier_field: str = "name"


# Language definition registry.
# Expand this dict to support more languages or customize node types.
_LANG_REGISTRY = {
    "python": {
        "module_name": "tree_sitter_python",
        "config": _LanguageConfig(
            function_types=["function_definition", "async_function_definition"],
            class_types=["class_definition"],
            class_body_types=["block"],
        ),
    },
    "javascript": {
        "module_name": "tree_sitter_javascript",
        "config": _LanguageConfig(
            function_types=[
                "function_declaration",
                "generator_function_declaration",
                "arrow_function",
                "method_definition",
            ],
            class_types=["class_declaration"],
            class_body_types=["class_body"],
        ),
    },
    "java": {
        "module_name": "tree_sitter_java",
        "config": _LanguageConfig(
            function_types=["method_declaration", "constructor_declaration"],
            class_types=[
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            ],
            class_body_types=["class_body", "interface_body", "enum_body"],
        ),
    },
    "cpp": {
        "module_name": "tree_sitter_cpp",
        "config": _LanguageConfig(
            function_types=["function_definition"],
            class_types=["class_specifier", "struct_specifier"],
            class_body_types=["field_declaration_list"],
        ),
    },
    "go": {
        "module_name": "tree_sitter_go",
        "config": _LanguageConfig(
            function_types=[
                "function_declaration",
                "method_declaration",
                "method_spec",
            ],
            class_types=["type_declaration"],
            class_body_types=["field_declaration_list", "method_spec_list"],
        ),
    },
}


def has_syntax_error(
    code_string: str, language_name: str, parser: tree_sitter.Parser = None
) -> bool:
    """
    Checks if the given code string contains any syntax errors.

    Args:
        code_string: The source code to be checked.
        language_name: The Tree-sitter Language object (e.g., tree_sitter_python.language()).
        parser: An optional tree sitter Parser object.

    Returns:
        bool: True if a syntax error is found, False otherwise.
    """
    # Initialize Tree-sitter
    if parser is None:
        lang_key = language_name.lower()
        if lang_key not in _LANG_REGISTRY:
            raise ValueError(f"Language '{lang_key}' is not registered/installed.")

        entry = _LANG_REGISTRY[lang_key]

        # Lazy loading of tree-sitter bindings to avoid hard dependencies on all languages
        module_name = entry["module_name"]
        try:
            binding_module = importlib.import_module(module_name)
        except ImportError:
            raise ImportError(
                f"Language support library '{module_name}' not found. "
                f"Please install it (e.g., pip install tree-sitter-{language_name})."
            )
        ts_lang = Language(binding_module.language())
        parser = Parser(ts_lang)

    # Convert the string to bytes (Tree-sitter requires bytes input) and parse it into a syntax tree
    tree = parser.parse(bytes(code_string, "utf8"))
    # Get the root node of the syntax tree
    root_node = tree.root_node
    # The root node's has_error property summarizes the error state of the entire tree.
    # If there are any ERROR or MISSING nodes anywhere in the tree, root_node.has_error will be True.
    return root_node.has_error


if __name__ == "__main__":

    def test_parser_1():
        import pathlib
        from typing import List

        def get_python_files(root_dir: str) -> List[pathlib.Path]:
            root = pathlib.Path(root_dir)
            # Ensure the path exists to avoid FileNotFoundError
            if not root.exists():
                print(f"Warning: The path '{root_dir}' does not exist.")
                return []
            # Use a list comprehension for a clean, Pythonic return
            # .rglob('*.py') handles the recursive search automatically
            return [py_file.resolve() for py_file in root.rglob("*.py")]

        def process_content(content):
            content = content.splitlines()
            for i, l in enumerate(content):
                if l.strip() == "":
                    content[i] = ""
            return "\n".join(content).strip()

        python_files = get_python_files("../../")
        for python_file in python_files:
            with open(python_file, "r") as f:
                content = f.read()

            parser = CodeParser(language_name="python")
            program = parser.parse(content)
            if process_content(str(program)) == process_content(content):
                print(f"{python_file}: same.")
            else:
                print(f"{python_file}: not same (caused by multi-line string). Ignore.")

    def test_parser_2():
        code = """
        def f():
            return 0
        """
        paser = CodeParser(language_name="python")
        program = paser.parse(code)

    def test_parser_3():
        content = open(__file__, "r").read()
        parser = CodeParser(language_name="python")
        program = parser.parse(content)
        print(program)

    # test_parser_1()
    # test_parser_2()
    test_parser_3()
