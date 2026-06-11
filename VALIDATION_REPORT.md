# Validation Report

| Check | Command | Result | Failure reason | Blocks release | Suggested fix |
|---|---|---|---|---|---|
| generation | generator script | pass |  | no |  |
| compileall | `python -m compileall .` | pending |  | yes if fail | run after generation |
| pytest | `python -m pytest -q` | pending |  | yes if fail | run after generation |
| old-name scan | `rg -n -i "evigraph|evi-graph|evigraph[-_]?v[0-9]+|EVIGRAPH_"` | pending |  | review | only migration/audit provenance allowed |
| secret/path scan | credential/path regex | pending |  | yes if real secret | remove real credentials |
| license | manual author decision | blocked | project license not selected | yes | choose project license |
