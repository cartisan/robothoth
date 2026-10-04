import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.tools.registry import Registry, registry


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
    output = registry.dispatch(name, json.dumps(arguments), vault)

    assert json.loads(output) == {"ok": True, "result": expected}


@pytest.mark.parametrize(
    ("name", "arguments_json"),
    [
        ("list_notes", "{"),
        ("list_notes", "[]"),
        ("list_notes", "{}"),
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
    output = json.loads(registry.dispatch(name, arguments_json, vault))

    assert output["ok"] is False
    assert output["error"]["code"] == "invalid_arguments"


def test_rejects_unknown_tool(vault: str) -> None:
    """Return an unknown_tool error for an unsupported tool name."""
    output = json.loads(registry.dispatch("delete_note", "{}", vault))

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
    output = json.loads(registry.dispatch(name, arguments_json, vault))

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
        registry._tools,
        "read_note",
        replace(registry._tools["read_note"], tool=fail_read),
    )
    output = json.loads(
        registry.dispatch("read_note", '{"notepath": "Beta.md"}', vault)
    )

    assert output == {
        "ok": False,
        "error": {"code": "io_error", "message": "Could not read the vault."},
    }


def test_invalid_note_encoding_is_an_io_error(tmp_path: Path) -> None:
    """Translate invalid note encoding into a public I/O error."""
    (tmp_path / "broken.md").write_bytes(b"\xff")

    output = json.loads(
        registry.dispatch(
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
        registry._tools,
        "read_note",
        replace(registry._tools["read_note"], tool=fail_read),
    )
    output = json.loads(
        registry.dispatch("read_note", '{"notepath": "Beta.md"}', vault)
    )

    assert output == {
        "ok": False,
        "error": {"code": "internal_error", "message": "Tool failed unexpectedly."},
    }
    assert "private detail" in caplog.text


def test_declarations_and_dispatch_agree(vault: str) -> None:
    """Use each public schema's fields and types for dispatch validation."""
    declarations = registry.declarations()
    assert {item["name"] for item in declarations} == set(registry._tools)
    for declaration in declarations:
        name = declaration["name"]
        spec = registry._tools[name]
        parameters = declaration["parameters"]
        properties = parameters["properties"]
        assert declaration["type"] == "function"
        assert declaration["strict"] is True
        assert declaration["description"] == spec.description
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(properties) == set(spec.arguments)
        assert "vault_path" not in properties
        valid = {field: "Beta.md" for field in properties}
        for field, schema in properties.items():
            assert schema["description"] == spec.arguments[field].description
            assert schema["type"] == (
                ["string", "null"] if spec.arguments[field].nullable else "string"
            )
            registry._validate_arguments(spec, valid)
            for bad in (42, False, list[str](), dict[str, str]()):
                with pytest.raises(ValueError):
                    registry._validate_arguments(spec, {**valid, field: bad})
            if spec.arguments[field].nullable:
                registry._validate_arguments(spec, {**valid, field: None})
            else:
                with pytest.raises(ValueError):
                    registry._validate_arguments(spec, {**valid, field: None})
            with pytest.raises(ValueError):
                registry._validate_arguments(
                    spec, {key: value for key, value in valid.items() if key != field}
                )
        with pytest.raises(ValueError):
            registry._validate_arguments(spec, {**valid, "unexpected": "value"})
        output = json.loads(registry.dispatch(name, json.dumps(valid), vault))
        if not output["ok"]:
            assert output["error"]["code"] != "invalid_arguments"


def test_registration_rejects_bad_documentation_and_signatures() -> None:
    """Reject missing or mismatched Args and unsupported call shapes."""
    def missing(vault_path: str, value: str) -> str:
        return value

    def undocumented(vault_path: str, value: str) -> str:
        """Return a value.

        Args:
            other: A different argument.
        """
        return value

    def unsupported(vault_path: str, count: int) -> str:
        """Return a count.

        Args:
            count: Number to return.
        """
        return str(count)

    def variadic(vault_path: str, *values: str) -> str:
        """Return values.

        Args:
            values: Values to join.
        """
        return "".join(values)

    for tool in (missing, undocumented, unsupported, variadic):
        with pytest.raises(ValueError):
            Registry((tool,))


def test_isolated_registry_uses_its_own_tools(vault: str) -> None:
    """Declare and dispatch a documented tool from an independent registry."""

    def echo(vault_path: str, text: str, suffix: str | None = None) -> str:
        """Return text with an optional suffix.

        Args:
            text: Text to echo.
            suffix: Suffix to append, or null for none.
        """
        return text + (suffix or "")

    isolated = Registry((echo,))
    assert [item["name"] for item in isolated.declarations()] == ["echo"]
    output = json.loads(
        isolated.dispatch("echo", '{"text": "hi", "suffix": null}', vault)
    )
    assert output == {
        "ok": True,
        "result": "hi",
    }
    missing = json.loads(isolated.dispatch("echo", '{"text": "hi"}', vault))
    assert missing["error"]["code"] == "invalid_arguments"
    unknown = json.loads(isolated.dispatch("read_note", "{}", vault))
    assert unknown["error"]["code"] == "unknown_tool"
    with pytest.raises(ValueError, match="Duplicate tool name"):
        Registry((echo, echo))
