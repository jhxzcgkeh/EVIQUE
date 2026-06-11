# Validation Report

Generated: 2026-06-11

| Check | Command | Result | Notes | Blocks release |
|---|---|---|---|---|
| Python compile | `python -m compileall -q .` | pass | all release Python files compile | no |
| Pytest smoke | `python -m pytest -q` | pass | 4 passed | no |
| Standalone package | `python -m evique.cli.check_standalone` | pass | package imports and version label validated | no |
| Script smoke | `python scripts/smoke/smoke_check.py` | pass | wrote temporary smoke marker, then excluded from release cleanup | no |
| Legacy-name full scan | `rg` legacy-name patterns | review pass | matches are confined to migration, rename, and inventory provenance files | no |
| Legacy-name public scan | same scan excluding provenance docs | pass | zero matches | no |
| Secret/path broad scan | placeholder and identifier audit regex | review pass | broad matches are empty env placeholders, variable names, and audit metadata | no |
| High-confidence secret scan | strict credential regex | pass | zero matches | no |
| Machine path scan | local/server path regex | pass | zero matches | no |
| Generated/cache files | filesystem cleanup plus `.gitignore` | pass | pycache, pytest cache, smoke outputs, logs, weights, full third-party repos, and generated results are excluded | no |
| Project license | manual author decision | blocked | top-level LICENSE is a placeholder notice until authors choose a license | yes |

## Scan Summary

- Full legacy-name scan counts: `RENAMING_MAP.md` 5, `RENAME_REPORT.md` 97, `REPOSITORY_FILE_INVENTORY.csv` 85.
- Public legacy-name scan excluding provenance files: zero matches.
- High-confidence secret/path scan: zero matches.
- Machine-specific path scan: zero matches.
- Broad audit scan intentionally matches terms such as empty API-key placeholders and code identifiers named token.
