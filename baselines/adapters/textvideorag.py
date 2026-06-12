from pathlib import Path
from .base import convert_file
METHOD = "TextVideoRAG"
def convert(path: str | Path) -> list[dict]:
    return convert_file(path, method=METHOD)
