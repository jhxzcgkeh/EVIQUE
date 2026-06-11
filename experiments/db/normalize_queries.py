#!/usr/bin/env python3
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT/'normalize_db_rag_queries.py'),run_name='__main__')
