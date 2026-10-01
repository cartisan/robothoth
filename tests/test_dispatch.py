import json
from pathlib import Path

import pytest

import src.tools.dispatch as dispatch


@pytest.fixture
def vault(tmp_path: Path) -> str:
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "Alpha.md").write_text("Python [[Beta]]", encoding="utf-8")
    (tmp_path / "Beta.md").write_text("Second note", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("list_notes", {}, ["Beta.md", "folder/Alpha.md"]),
        ("list_notes", {"path": "folder"}, ["folder/Alpha.md"]),
        ("read_note", {"notepath": "folder/Alpha.md"}, "Python [[Beta]]"),
        ("search_notes", {"query": "pyth.n"}, ["folder/Alpha.md"]),
        ("get_backlinks", {"notepath": "Beta.md"}, ["folder/Alpha.md"]),
        ("get_outgoing_links", {"notepath": "folder/Alpha.md"}, ["Beta.md"]),
    ],
)
def test_dispatches_each_tool(
    vault: str, name: str, arguments: dict[str, str], expected: str | list[str]
) -> None:
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
    ],
)
def test_rejects_bad_model_arguments(
    vault: str, name: str, arguments_json: str
) -> None:
    output = json.loads(dispatch.dispatch_tool_call(name, arguments_json, vault))

    assert output["ok"] is False
    assert output["error"]["code"] == "invalid_arguments"


def test_rejects_unknown_tool(vault: str) -> None:
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
    output = json.loads(dispatch.dispatch_tool_call(name, arguments_json, vault))

    assert output["ok"] is False
    assert output["error"]["code"] == code
    assert output["error"]["message"]


def test_translates_io_error_without_exposing_path(
    vault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_read(_vault_path: str, _notepath: str) -> str:
        raise PermissionError("private path")

    monkeypatch.setattr(dispatch, "read_note", fail_read)
    output = json.loads(
        dispatch.dispatch_tool_call("read_note", '{"notepath": "Beta.md"}', vault)
    )

    assert output == {
        "ok": False,
        "error": {"code": "io_error", "message": "Could not read the vault."},
    }


def test_invalid_note_encoding_is_an_io_error(tmp_path: Path) -> None:
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
    def fail_read(_vault_path: str, _notepath: str) -> str:
        raise RuntimeError("private detail")

    monkeypatch.setattr(dispatch, "read_note", fail_read)
    output = json.loads(
        dispatch.dispatch_tool_call("read_note", '{"notepath": "Beta.md"}', vault)
    )

    assert output == {
        "ok": False,
        "error": {"code": "internal_error", "message": "Tool failed unexpectedly."},
    }
    assert "private detail" in caplog.text
