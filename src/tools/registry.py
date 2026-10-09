"""Register documented vault functions for model declarations and dispatch."""

import inspect
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

from src.tools.get_backlinks import get_backlinks
from src.tools.get_outgoing_links import get_outgoing_links
from src.tools.list_notes import list_notes
from src.tools.read_note import read_note
from src.tools.search_notes import SearchTimeoutError, search_notes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArgumentSpec:
    """Describe a required model argument and its accepted JSON type."""

    nullable: bool
    description: str


@dataclass(frozen=True)
class ToolSpec:
    """Pair a documented vault function with its derived public arguments."""

    tool: Callable[..., str | list[str]]
    description: str
    arguments: Mapping[str, ArgumentSpec]


class Registry:
    """Expose one collection of documented vault tools to models and callers.

    The supplied functions must have distinct names and valid public signatures.
    Their specifications govern both strict declarations and dispatch validation.

    Raises:
        ValueError: If a function is undocumented, unsupported, or duplicated.
    """

    @staticmethod
    def _argument_docs(doc: str) -> dict[str, str]:
        """Return parameter descriptions from the docstring's Args section.

        Each entry starts with ``name:`` and may continue on indented lines.

        Raises:
            ValueError: If Args is missing, malformed, or contains duplicate names.
        """
        lines = doc.splitlines()
        try:
            start = lines.index("Args:") + 1
        except ValueError as error:
            raise ValueError("Tool docstring must include an Args: section") from error
        descriptions: dict[str, str] = {}
        name: str | None = None
        for line in lines[start:]:
            if line and not line.startswith(" "):
                break
            if not line.strip():
                continue
            match = re.fullmatch(r"    ([a-zA-Z_]\w*):\s*(.*)", line)
            if match:
                name, detail = match.groups()
                if name in descriptions or not detail:
                    raise ValueError(f"Duplicate or empty Args entry: {name}")
                descriptions[name] = detail
            elif name is not None and line.startswith("        "):
                descriptions[name] += " " + line.strip()
            else:
                raise ValueError(f"Malformed Args entry: {line.strip()}")
        return descriptions

    @staticmethod
    def _register(tool: Callable[..., str | list[str]]) -> ToolSpec:
        """Return a validated specification for a documented vault function.

        The first parameter must be ``vault_path: str``. Public parameters accept
        strings or nullable strings and require matching Args descriptions. Defaults
        remain available to direct Python callers but not to model callers.

        Raises:
            ValueError: If documentation, signature, or annotations are unsupported
                or inconsistent.
        """
        name = tool.__name__
        doc = inspect.getdoc(tool)
        if not doc:
            raise ValueError(f"{name}: missing docstring")
        try:
            signature = inspect.signature(tool)
            hints = get_type_hints(tool)
        except (TypeError, ValueError, NameError) as error:
            raise ValueError(f"{name}: unsupported signature or annotations") from error
        parameters = list(signature.parameters.values())
        if (
            not parameters
            or parameters[0].name != "vault_path"
            or parameters[0].kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            or parameters[0].default is not inspect.Parameter.empty
            or hints.get("vault_path") is not str
        ):
            raise ValueError(f"{name}: first parameter must be vault_path: str")
        if hints.get("return") not in (str, list[str]):
            raise ValueError(f"{name}: unsupported return annotation")
        docs = Registry._argument_docs(doc)
        arguments: dict[str, ArgumentSpec] = {}
        for parameter in parameters[1:]:
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise ValueError(f"{name}: unsupported parameter {parameter.name}")
            annotation = hints.get(parameter.name)
            nullable = get_origin(annotation) is not None and set(
                get_args(annotation)
            ) == {str, type(None)}
            if annotation is not str and not nullable:
                raise ValueError(f"{name}: unsupported annotation for {parameter.name}")
            default = parameter.default
            if default is not inspect.Parameter.empty and not (
                isinstance(default, str) or (nullable and default is None)
            ):
                raise ValueError(f"{name}: invalid default for {parameter.name}")
            if parameter.name not in docs:
                raise ValueError(f"{name}: missing Args entry for {parameter.name}")
            arguments[parameter.name] = ArgumentSpec(nullable, docs[parameter.name])
        if set(docs) != set(arguments):
            raise ValueError(f"{name}: Args entries do not match public parameters")
        return ToolSpec(tool, doc, arguments)

    def __init__(self, tools: Iterable[Callable[..., str | list[str]]]) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            spec = self._register(tool)
            name = tool.__name__
            if name in self._tools:
                raise ValueError(f"Duplicate tool name: {name}")
            self._tools[name] = spec

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return LangChain-compatible definitions for all registered tools.

        Every public argument is required, including nullable arguments. Schemas
        reject unknown properties and do not expose vault_path. Each call returns
        fresh schema dictionaries so model binding cannot affect registration.
        """
        return [
            {
                "name": name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        argument_name: {
                            "type": (
                                ["string", "null"] if argument.nullable else "string"
                            ),
                            "description": argument.description,
                        }
                        for argument_name, argument in spec.arguments.items()
                    },
                    "required": list(spec.arguments),
                    "additionalProperties": False,
                },
            }
            for name, spec in self._tools.items()
        ]

    @staticmethod
    def _validate_arguments(spec: ToolSpec, arguments: Mapping[str, object]) -> None:
        """Accept an argument object matching a registered tool's strict schema.

        All public fields must be present. Strings and declared nullable values
        are accepted; unknown fields and ``vault_path`` are rejected.

        Raises:
            ValueError: If a field is missing, unexpected, or has an invalid type.
        """
        if set(arguments) != set(spec.arguments):
            raise ValueError("Missing or unexpected tool arguments.")
        for name, value in arguments.items():
            argument = spec.arguments[name]
            if isinstance(value, str) or (value is None and argument.nullable):
                continue
            expected = "a string or null" if argument.nullable else "a string"
            raise ValueError(f"{name} must be {expected}")

    @staticmethod
    def _error(code: str, message: str) -> str:
        """Return a JSON error envelope with the supplied code and message."""
        return json.dumps({"ok": False, "error": {"code": code, "message": message}})

    def dispatch(self, name: str, arguments_json: str, vault_path: str) -> str:
        """Run a registered tool and return a JSON success or error envelope.

        Arguments must encode an object matching the tool's required string and
        nullable fields. A successful result has ``ok: true`` and ``result``.
        Unknown names use ``unknown_tool``; malformed or invalid arguments use
        ``invalid_arguments``; search timeouts use ``search_timeout``; missing
        notes or directories use ``not_found``; decoding and filesystem errors
        use ``io_error``; unexpected failures use ``internal_error``. Details of
        I/O and unexpected failures are logged but omitted from the envelope.
        """
        spec = self._tools.get(name)
        if spec is None:
            return self._error("unknown_tool", f"Unknown tool: {name}")

        try:
            arguments: object = json.loads(arguments_json)
        except json.JSONDecodeError:
            return self._error(
                "invalid_arguments", "Tool arguments must be valid JSON."
            )
        if not isinstance(arguments, dict) or not all(
            isinstance(key, str) for key in arguments
        ):
            return self._error(
                "invalid_arguments", "Tool arguments must be a JSON object."
            )

        try:
            self._validate_arguments(spec, arguments)
            result = spec.tool(vault_path, **arguments)
        except SearchTimeoutError:
            return self._error("search_timeout", "Search exceeded its time limit.")
        except FileNotFoundError:
            return self._error("not_found", "Note or directory not found.")
        except UnicodeError:
            logger.exception("Vault tool could not decode a note")
            return self._error("io_error", "Could not read the vault.")
        except ValueError as error:
            return self._error("invalid_arguments", str(error))
        except OSError:
            logger.exception("Vault tool failed due to an I/O error")
            return self._error("io_error", "Could not read the vault.")
        except Exception:
            logger.exception("Unexpected vault tool failure")
            return self._error("internal_error", "Tool failed unexpectedly.")

        return json.dumps({"ok": True, "result": result}, ensure_ascii=False)


registry = Registry(
    (list_notes, read_note, search_notes, get_backlinks, get_outgoing_links)
)
