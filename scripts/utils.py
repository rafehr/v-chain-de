from pathlib import Path
from typing import Any, Dict, Generator, Tuple

import orjson


def stream_jsonl(file_path: Path) -> Generator[Tuple[Dict[str, Any], int], None, None]:
    """
    Reads a JSON lines file line by line.

    Args:
        The path to the .jsonl file.

    Yields:
        A decoded JSON object as dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        orjson.JSONDecodeError: If a line is not valid JSON.
    """
    try:
        with open(file_path, "rb") as f:
            for line in f:
                length = len(line)
                if line.strip():
                    yield orjson.loads(line), length
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
