from pathlib import Path
from .base import convert_file
METHOD = "NaiveRAG"
def convert(path: str | Path) -> list[dict]:
    return convert_file(path, method=METHOD)
