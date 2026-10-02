"""Convert assistant tool calls into vault operations and JSON results."""

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from src.tools.get_backlinks import get_backlinks
from src.tools.get_outgoing_links import get_outgoing_links
from src.tools.list_notes import list_notes
from src.tools.read_note import read_note
from src.tools.search_notes import SearchTimeoutError, search_notes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArgumentSpec:
    """Define presence and nullability rules for a string tool argument.

    Arguments are required and reject null by default. ``required=False`` allows
    omission so the tool's default applies; ``nullable=True`` accepts an explicit
    JSON null as well as a string.
    """

    required: bool = True
    nullable: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """Pair a vault tool with the rules for its model-provided arguments.

    ``tool`` receives the vault path as its first positional argument and
    validated arguments by keyword. ``arguments`` maps accepted names to their
    presence and nullability rules; omitted optional arguments use tool defaults.
    """

    tool: Callable[..., str | list[str]]
    arguments: Mapping[str, ArgumentSpec]


_TOOLS: dict[str, ToolSpec] = {
    "list_notes": ToolSpec(
        list_notes, {"path": ArgumentSpec(required=False, nullable=True)}
    ),
    "read_note": ToolSpec(read_note, {"notepath": ArgumentSpec()}),
    "search_notes": ToolSpec(search_notes, {"query": ArgumentSpec()}),
    "get_backlinks": ToolSpec(get_backlinks, {"notepath": ArgumentSpec()}),
    "get_outgoing_links": ToolSpec(get_outgoing_links, {"notepath": ArgumentSpec()}),
}


def _validate_arguments(spec: ToolSpec, arguments: Mapping[str, object]) -> None:
    """Validate supplied arguments against the selected tool's rules.

    Required names must be present and unknown names are rejected. Values must
    be strings, or null when the corresponding rule allows it. Omitted optional
    arguments are left absent so the tool's defaults apply.

    Raises:
        ValueError: If an argument is missing, unexpected, or has an invalid type.
    """
    required = {name for name, argument in spec.arguments.items() if argument.required}
    if not required.issubset(arguments) or not set(arguments).issubset(spec.arguments):
        raise ValueError("Missing or unexpected tool arguments.")
    for name, value in arguments.items():
        argument = spec.arguments[name]
        if isinstance(value, str) or (value is None and argument.nullable):
            continue
        expected = "a string or null" if argument.nullable else "a string"
        raise ValueError(f"{name} must be {expected}")


def _error(code: str, message: str) -> str:
    """Return a JSON error envelope containing ``code`` and ``message``.

    The envelope contains ``ok: false`` and an ``error`` object with the supplied
    code and message, matching the format returned for failed tool calls.
    """
    return json.dumps({"ok": False, "error": {"code": code, "message": message}})


def dispatch_tool_call(name: str, arguments_json: str, vault_path: str) -> str:
    """Run a named vault tool and return its JSON success or error envelope.

    ``arguments_json`` must encode an object matching the registered tool's
    required, optional, and nullable string arguments. Validated arguments are
    passed by keyword with ``vault_path`` as the first positional argument.
    Successful calls return ``ok: true`` and the tool's ``result``.

    Failed calls return ``ok: false`` with an ``error`` code and message. Unknown
    names use ``unknown_tool``; malformed JSON, invalid arguments, and tool
    validation failures use ``invalid_arguments``. Search timeouts use
    ``search_timeout``, missing notes or directories use ``not_found``, decoding
    and filesystem failures use ``io_error``, and unexpected tool failures use
    ``internal_error``. Decoding, filesystem, and unexpected failure details are
    logged while their returned messages remain generic.
    """
    spec = _TOOLS.get(name)
    if spec is None:
        return _error("unknown_tool", f"Unknown tool: {name}")

    try:
        arguments: object = json.loads(arguments_json)
    except json.JSONDecodeError:
        return _error("invalid_arguments", "Tool arguments must be valid JSON.")
    if not isinstance(arguments, dict) or not all(
        isinstance(key, str) for key in arguments
    ):
        return _error("invalid_arguments", "Tool arguments must be a JSON object.")

    try:
        _validate_arguments(spec, arguments)
        result = spec.tool(vault_path, **arguments)
    except SearchTimeoutError:
        return _error("search_timeout", "Search exceeded its time limit.")
    except FileNotFoundError:
        return _error("not_found", "Note or directory not found.")
    except UnicodeError:
        logger.exception("Vault tool could not decode a note")
        return _error("io_error", "Could not read the vault.")
    except ValueError as error:
        return _error("invalid_arguments", str(error))
    except OSError:
        logger.exception("Vault tool failed due to an I/O error")
        return _error("io_error", "Could not read the vault.")
    except Exception:
        logger.exception("Unexpected vault tool failure")
        return _error("internal_error", "Tool failed unexpectedly.")

    return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
