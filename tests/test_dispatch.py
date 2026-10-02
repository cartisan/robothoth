import json
from dataclasses import replace
from pathlib import Path

import pytest

import src.tools.dispatch as dispatch


@pytest.fixture
def vault(tmp_path: Path) -> str:
    """Create a small vault covering nested notes and a wikilink."""
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "Alpha.md").write_text("Python [[Beta]]", encoding="utf-8")
    (tmp_path / "Beta.md").write_text("Second note", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("list_notes", {}, ["Beta.md", "folder/Alpha.md"]),
        ("list_notes", {"path": None}, ["Beta.md", "folder/Alpha.md"]),
        ("list_notes", {"path": "folder"}, ["folder/Alpha.md"]),
        ("read_note", {"notepath": "folder/Alpha.md"}, "Python [[Beta]]"),
        ("search_notes", {"query": "pyth.n"}, ["folder/Alpha.md"]),
        ("get_backlinks", {"notepath": "Beta.md"}, ["folder/Alpha.md"]),
        ("get_outgoing_links", {"notepath": "folder/Alpha.md"}, ["Beta.md"]),
    ],
)
def test_dispatches_each_tool(
    vault: str, name: str, arguments: dict[str, str | None], expected: str | list[str]
) -> None:
    """Dispatch each supported tool and return its successful JSON result."""
    output = dispatch.dispatch_tool_call(name, json.dumps(arguments), vault)

    assert json.loads(output) == {"ok": True, "result": expected}


@pytest.mark.parametrize(
    ("name", "arguments_json"),
    [
        ("list_notes", "{"),
        ("list_notes", "[]"),
        ("list_notes", '{"path": 3}'),
        ("list_notes", '{"vault_path": "/tmp"}'),
        ("read_note", "{}"),
        ("read_note", '{"notepath": null}'),
        ("read_note", '{"notepath": "Beta.md", "extra": true}'),
        ("search_notes", '{"query": 7}'),
        ("search_notes", '{"query": null}'),
        ("search_notes", '{}'),
        ("get_backlinks", '{"notepath": null}'),
        ("get_backlinks", '{}'),
        ("get_outgoing_links", '{"notepath": false}'),
        ("get_outgoing_links", '{}'),
    ],
)
def test_rejects_bad_model_arguments(
    vault: str, name: str, arguments_json: str
) -> None:
    """Reject malformed, incomplete, mistyped, and unexpected arguments."""
    output = json.loads(dispatch.dispatch_tool_call(name, arguments_json, vault))

    assert output["ok"] is False
    assert output["error"]["code"] == "invalid_arguments"


def test_rejects_unknown_tool(vault: str) -> None:
    """Return an unknown_tool error for an unsupported tool name."""
    output = json.loads(dispatch.dispatch_tool_call("delete_note", "{}", vault))

    assert output["error"]["code"] == "unknown_tool"


@pytest.mark.parametrize(
    ("name", "arguments_json", "code"),
    [
        ("read_note", '{"notepath": "Missing.md"}', "not_found"),
        ("read_note", '{"notepath": "../outside.md"}', "invalid_arguments"),
        ("search_notes", '{"query": "["}', "invalid_arguments"),
    ],
)
def test_translates_expected_tool_errors(
    vault: str, name: str, arguments_json: str, code: str
) -> None:
    """Translate expected tool failures into stable public error codes."""
    output = json.loads(dispatch.dispatch_tool_call(name, arguments_json, vault))

    assert output["ok"] is False
    assert output["error"]["code"] == code
    assert output["error"]["message"]


def test_translates_io_error_without_exposing_path(
    vault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hide filesystem error details while preserving the public I/O code."""
    def fail_read(_vault_path: str, notepath: str) -> str:
        raise PermissionError("private path")

    monkeypatch.setitem(
        dispatch._TOOLS,
        "read_note",
        replace(dispatch._TOOLS["read_note"], tool=fail_read),
    )
    output = json.loads(
        dispatch.dispatch_tool_call("read_note", '{"notepath": "Beta.md"}', vault)
    )

    assert output == {
        "ok": False,
        "error": {"code": "io_error", "message": "Could not read the vault."},
    }


def test_invalid_note_encoding_is_an_io_error(tmp_path: Path) -> None:
    """Translate invalid note encoding into a public I/O error."""
    (tmp_path / "broken.md").write_bytes(b"\xff")

    output = json.loads(
        dispatch.dispatch_tool_call(
            "read_note", '{"notepath": "broken.md"}', str(tmp_path)
        )
    )

    assert output["error"]["code"] == "io_error"


def test_translates_unexpected_error_without_exposing_details(
    vault: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Hide unexpected exception details from callers while logging them."""
    def fail_read(_vault_path: str, notepath: str) -> str:
        raise RuntimeError("private detail")

    monkeypatch.setitem(
        dispatch._TOOLS,
        "read_note",
        replace(dispatch._TOOLS["read_note"], tool=fail_read),
    )
    output = json.loads(
        dispatch.dispatch_tool_call("read_note", '{"notepath": "Beta.md"}', vault)
    )

    assert output == {
        "ok": False,
        "error": {"code": "internal_error", "message": "Tool failed unexpectedly."},
    }
    assert "private detail" in caplog.text


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [({"text": "hello"}, "hello!"), ({"text": "hello", "suffix": None}, "hello")],
)
def test_dispatches_registered_tool_with_shared_validation(
    vault: str,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, str | None],
    expected: str,
) -> None:
    """A registered handler works without adding a tool-specific dispatch branch."""
    def echo(_vault_path: str, text: str, *, suffix: str | None = "!") -> str:
        return text + (suffix or "")

    monkeypatch.setitem(
        dispatch._TOOLS,
        "echo",
        dispatch.ToolSpec(
            echo,
            {
                "text": dispatch.ArgumentSpec(),
                "suffix": dispatch.ArgumentSpec(required=False, nullable=True),
            },
        ),
    )

    assert json.loads(
        dispatch.dispatch_tool_call("echo", json.dumps(arguments), vault)
    ) == {"ok": True, "result": expected}
    assert json.loads(
        dispatch.dispatch_tool_call("echo", '{"text": null}', vault)
    )["error"]["code"] == "invalid_arguments"
