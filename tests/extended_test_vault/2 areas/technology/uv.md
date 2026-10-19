## summarize
 - A single tool to replace `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `virtualenv`
	 - supports `pyproject.toml`
 - 10x-100x faster than pip

## usage
- `uv python install 3.10 3.11` - installs multiple python versions
- `uv add "fastapi>=0.112"` - installs a package and saves in toml
- `uv lock` - creats a lock file
- `uv run` - execute command in project venv
- `uv tool install ruff` - installs ruff, can manage installed tools
- `uvx ruff` - executes installed ruff

## source
https://github.com/astral-sh/uv

---
Associations: [[z> MOC Infrastructure]], [[pyenv]], [[pipenv]]
Tags: #🌲
