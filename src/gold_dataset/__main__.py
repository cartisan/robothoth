"""Validate a gold dataset file from the command line."""

import argparse
import sys

from src.gold_dataset import DatasetValidationError, load_dataset


def main() -> int:
    """Print the validated entry count, or a diagnostic and failure status."""
    parser = argparse.ArgumentParser(description="Validate a gold dataset JSON file")
    parser.add_argument(
        "path", nargs="?", help="dataset JSON path (default: bundled file)"
    )
    arguments = parser.parse_args()
    try:
        entries = load_dataset(arguments.path)
    except (DatasetValidationError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Validated {len(entries)} gold dataset entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
