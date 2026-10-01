"""Convert assistant tool calls into vault operations and JSON results."""

import json
import logging

from src.tools.get_backlinks import get_backlinks
from src.tools.get_outgoing_links import get_outgoing_links
from src.tools.list_notes import list_notes
from src.tools.read_note import read_note
from src.tools.search_notes import search_notes

logger = logging.getLogger(__name__)

_ARGUMENTS: dict[str, tuple[set[str], set[str]]] = {
    "list_notes": (set(), {"path"}),
    "read_note": ({"notepath"}, {"notepath"}),
    "search_notes": ({"query"}, {"query"}),
    "get_backlinks": ({"notepath"}, {"notepath"}),
    "get_outgoing_links": ({"notepath"}, {"notepath"}),
}


def _error(code: str, message: str) -> str:
    """Serialize a failed tool call using the dispatcher's error envelope."""
    return json.dumps({"ok": False, "error": {"code": code, "message": message}})


def dispatch_tool_call(name: str, arguments_json: str, vault_path: str) -> str:
    """Run a named vault tool and return its JSON success or error envelope.

    The model-provided arguments must be a JSON object matching the selected
    tool's schema. Expected validation, filesystem, decoding, and unexpected
    failures are converted into stable error codes; implementation details are
    logged rather than exposed in the returned message.
    """
    if name not in _ARGUMENTS:
        return _error("unknown_tool", f"Unknown tool: {name}")

    try:
        arguments: object = json.loads(arguments_json)
    except json.JSONDecodeError:
        return _error("invalid_arguments", "Tool arguments must be valid JSON.")
    if not isinstance(arguments, dict) or not all(
        isinstance(key, str) for key in arguments
    ):
        return _error("invalid_arguments", "Tool arguments must be a JSON object.")

    required, allowed = _ARGUMENTS[name]
    if not required.issubset(arguments) or not set(arguments).issubset(allowed):
        return _error("invalid_arguments", "Missing or unexpected tool arguments.")

    result: str | list[str]
    try:
        if name == "list_notes":
            path = arguments.get("path")
            if path is not None and not isinstance(path, str):
                raise ValueError("path must be a string or null")
            result = list_notes(vault_path, path)
        elif name == "search_notes":
            query = arguments["query"]
            if not isinstance(query, str):
                raise ValueError("query must be a string")
            result = search_notes(vault_path, query)
        else:
            notepath = arguments["notepath"]
            if not isinstance(notepath, str):
                raise ValueError("notepath must be a string")
            if name == "read_note":
                result = read_note(vault_path, notepath)
            elif name == "get_backlinks":
                result = get_backlinks(vault_path, notepath)
            else:
                result = get_outgoing_links(vault_path, notepath)
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
