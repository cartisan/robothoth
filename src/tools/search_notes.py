import multiprocessing
import re
import time
from multiprocessing.connection import Connection
from pathlib import Path

SEARCH_TIMEOUT_SECONDS = 5.0


class SearchTimeoutError(TimeoutError):
    """A vault search exceeded its execution budget."""


def _search_worker(vault_root: Path, query: str, output: Connection) -> None:
    """Run Python's regex engine in a process the caller can terminate."""
    try:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"Invalid search regex: {error}") from error

        matches: list[str] = []
        for note in vault_root.rglob("*.md"):
            if not note.is_file() or not note.resolve().is_relative_to(vault_root):
                continue
            if pattern.search(note.read_text(encoding="utf-8")):
                matches.append(note.relative_to(vault_root).as_posix())
        output.send(sorted(set(matches)))
    except Exception as error:
        output.send(error)
    finally:
        output.close()


def search_notes(vault_path: str, query: str) -> list[str]:
    """Return sorted Markdown paths whose UTF-8 content matches ``query``.

    ``query`` is compiled as a case-insensitive Python regular expression and
    each matching note is returned once, relative to the vault root. Compilation
    and scanning run in an isolated process with a total execution budget of
    ``SEARCH_TIMEOUT_SECONDS``, including worker startup.

    Args:
        query: Case-insensitive Python regular expression to match note content.

    Raises:
        FileNotFoundError: If ``vault_path`` is not a directory.
        ValueError: If ``query`` is empty or is not a valid regular expression.
        UnicodeDecodeError: If a matching candidate cannot be decoded as UTF-8.
        SearchTimeoutError: If the search exceeds its execution budget.
    """
    if not query:
        raise ValueError("Search query must not be empty")

    vault_root = Path(vault_path).resolve()
    if not vault_root.is_dir():
        raise FileNotFoundError(vault_path)

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_search_worker, args=(vault_root, query, sender))
    deadline = time.monotonic() + SEARCH_TIMEOUT_SECONDS
    try:
        worker.start()
        sender.close()
        if not receiver.poll(max(0.0, deadline - time.monotonic())):
            raise SearchTimeoutError("Search exceeded its time limit.")
        result = receiver.recv()
        if isinstance(result, Exception):
            raise result
        return result
    finally:
        sender.close()
        receiver.close()
        if worker.pid is not None:
            if worker.is_alive():
                worker.terminate()
            worker.join()
            worker.close()
