# Baseline Fidelity Audit

- Generated: 2026-06-04 11:20:00
- Rule: proxy and unvalidated reimplementation methods are not eligible for main paper figures.

| Method | Class | Adapter status | Main figure | Eligible now | Needs | Blockers |
|---|---|---|---:|---:|---|---|
| LOVO | local_reproduction | integrated | 1 | 1 |  |  |
| SIEVE | ours | integrated | 1 | 1 |  |  |
| VOCAL | official | local_scene_graph_proxy_v2_official_not_integrated | 0 | 0 | official EQUI-VOCAL environment; PostgreSQL schema conversion; official query_str execution |  |
| MIRIS | official | official_not_integrated_proxy_exists | 0 | 0 | Go toolchain; MIRIS-format frames/json tracks; trained RNN/GNN models or valid plan | Go not installed |
| UMT | official | official_not_integrated_proxy_exists | 0 | 0 | separate PyTorch 1.11/torchvision 0.12 environment; official checkpoint; QVHighlights feature files |  |
| VISA | official | official_not_integrated_proxy_exists | 0 | 0 | separate VideoLISA environment; VideoLISA-3.8B checkpoint; video segmentation output adapter |  |
| FiGO | reimpl | reimpl_integrated_needs_validation | 0 | 0 | runtime/accuracy sanity check |  |
| ZELDA | reimpl | reimpl_integrated_needs_validation | 0 | 0 | runtime/accuracy sanity check |  |
| VOCALExplore | reimpl | not_integrated | 0 | 0 | paper-step validation report; runtime/accuracy sanity check |  |
| OTIF | official | local_tracking_preprocess_proxy_v1_official_not_integrated | 0 | 0 | official OTIF dataset/runtime; local detector-track conversion; official speed-accuracy executor |  |
| GroundingDINO | official | support_only | 0 | 0 |  |  |

## Immediate Integration Priorities

1. MIRIS official: install Go, convert local detections to MIRIS JSON, then run official planner/executor on a compatible track query.
2. FiGO reimplementation: validate predicate planner against paper steps and keep it out of the main figure until validation passes.
3. ZELDA reimplementation: validate frame-level VLM retrieval and keep it out of the main figure until validation passes.
4. UMT/VideoLISA official: isolate dependencies in separate environments before adapter work.
