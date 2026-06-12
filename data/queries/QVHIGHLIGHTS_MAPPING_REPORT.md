# QVHighlights Mapping Report

## Outcome

No verified 20/20 query-to-video mapping was found. The normalized QVHighlights file remains unresolved: no `video_id` values were filled, and the manifest status must remain `missing_video_id`.

## Search Scope

Searched:

- Desktop query files, experiment folders, handoff notes, result folders, and QVHighlights archives.
- The read-only reference repository supplied with the workspace.
- Codex attachment text files.
- QV-related tar/gzip archives without extracting them into source directories.

Search terms included `QVHighlights`, `qvhighlight`, `video_id`, `vid`, `clip_id`, `source_video`, `relevant_windows`, `saliency_scores`, `query_id`, and `qid`. The 20 current query strings from `data/queries/qvhighlights.json` were also searched one by one.

## Candidate Files

| Candidate | Structure | Match Result | Decision |
| --- | --- | --- | --- |
| `qvhighlight-20-query.txt` and duplicated final query text files | Plain text, 20 numbered query lines | All 20 query texts found | No video fields; not a mapping source |
| `Experiment/1_RAG/C-qvhighlight/queries.jsonl` | JSONL records with `dataset`, `query_id`, `question`, `type`, generic `video_id`, empty `video_path`, `video_dir` | All 20 exact query texts found | `video_id` is the dataset label `qvhighlights`, not a per-query source video |
| `QVHighlights_routeA/qvhighlights_auto_candidate_videos.json` | 40 official-style annotation records with `category`, `score`, `split`, `qid`, `vid`, `duration`, `query`, `relevant_windows` | 1 strict full-query match, 2 normalized embedded-annotation matches, 1 fuzzy match | Partial candidate only; does not cover 20 queries |
| `qvhighlights_15vid_autoquery_v3.1_4models_singlepass2/questions.json` | 15 old autoquery records with `id`, `question`, `source_vid`, `source_annotation_query`, `video_path`, `uploaded_filename`, `original_video_path` | Covers Q07/Q08 under normalized annotation matching | Partial old run metadata only; not the final 20-query mapping |
| `workdir-backup-lovo_qvhighlights_15videos_7models_v4_20q_visual_strong_20260607_154148.tar.gz` | Final run archive with `questions.json`, `qid_mapping_22q_to_20q.json`, `comparison_config.json`, answer JSON, and workdirs | Final `questions.json` contains all 20 queries; answer JSON has empty query-level `video_id`, `source_vid`, `video_path` | Not a mapping source; retrieval evidence contains multiple videos per query |
| `db_rag_qvhighlights_20q_9methods_v1_final_20260607_190612.tar.gz` | DB-RAG query/results archive with `queries.jsonl`, method result JSONL, evidence JSON | All 20 query texts found | Query/result `video_id` is generic `qvhighlights`; method retrieval hits are not source-video mapping |
| `qvhighlights_videos_reupload.tar.gz` | Video archive only | 15 video files found | No query metadata |
| UMT QVHighlights config/loader in the reference repo | Code expects official label files such as `highlight_train_release.jsonl` with `qid` and `vid` | Confirms standard structure | Actual official annotation JSONL files were not present locally |

## Standard Annotation Format

The local official-style candidate and the reference UMT loader confirm that standard QVHighlights annotations use fields such as:

- `qid`
- `query`
- `vid`
- `relevant_clip_ids`
- `relevant_windows`
- `saliency_scores`
- `duration`

However, no local official release file was found that contains the current 20 final query texts with per-query `vid` values.

## Match Counts

Counts below refer to candidate files that contain per-video fields such as `vid`, `source_vid`, or `video_path`, excluding the current normalized EVIQUE output.

- Exact full current-query text match: 1/20, Q08.
- Query ID match to official `qid`: 0/20.
- Normalized embedded annotation match: 2/20, Q07 and Q08.
- Fuzzy match: 1/20, Q09 matched an old typo variant: `Woman talks to camera wearing a carrying her dog.`
- Final accepted mappings: 0/20.
- Unresolved records for release: 20/20.

The partial matches are not sufficient to update `data/queries/qvhighlights.json`, because the final dataset must not mix verified and guessed video IDs.

## Multiple Video Evidence

The final answer/evidence JSON files contain retrieval evidence from multiple videos for individual queries. Their query-level fields are empty:

- `video_id`: empty
- `source_vid`: empty
- `video_path`: empty

Therefore, retrieval hits were treated as answer evidence, not as authoritative query-to-video mapping.

## Video Archive Contents

The supplied video archive contains these 15 files:

- `01_VKKH07K1zbI_210.0_360.0.mp4`
- `02_bdKv6l0PkBY_360.0_510.0.mp4`
- `03_t_9gaLdfc_Y_360.0_510.0.mp4`
- `04_iy6kh6tBCmI_360.0_510.0.mp4`
- `05_jCTGB0sHy8o_60.0_210.0.mp4`
- `06_S1MHHgJNSuY_360.0_510.0.mp4`
- `07_GPZ4So5mepU_210.0_360.0.mp4`
- `08_v9bV5ERmcCk_60.0_210.0.mp4`
- `09_S4z8QOxZisc_60.0_210.0.mp4`
- `10_FaF3OJ5e_vE_510.0_660.0.mp4`
- `11_fm5i4fqWkqU_210.0_360.0.mp4`
- `12_SsjijAXVkBE_360.0_510.0.mp4`
- `13_9WA9GGpqxQY_60.0_210.0.mp4`
- `14_UdiR9BWQKew_660.0_810.0.mp4`
- `15_5wdsBwOSh78_360.0_510.0.mp4`

The archive has no query-to-video mapping file, and the 20 queries were not paired to videos by archive order.

## Candidate SHA-256

No mapping file was adopted, so the adopted mapping SHA-256 is `N/A`.

Audited candidate hashes:

- `qvhighlights_auto_candidate_videos.json`: `a6097aa45df153eb9a1ab39e45411d8fc9403f76d58ee6b48d993230696434b4`
- Old autoquery `questions.json`: `dbf61322f523e257126083ec31f0573d6e21faedc3d989c29a26e70fd9f57c7c`
- Final workdir backup archive: `88a33bc77d3862462764687310f8c0813a7203608d22425535649ceb6c45ffac`
- DB-RAG final archive: `6ca3faa5112d0a53fc4134378f95fd7bc35ae96e60fcc1b7f3688f579b5af968`
- Video reupload archive: `58f245d895d1610d4fece5dfc0bd1792764d3e7cc155de639c779f86ea390a7f`

## Unresolved Records

All 20 normalized QVHighlights records remain unresolved:

1. `qvhighlights_001`
2. `qvhighlights_002`
3. `qvhighlights_003`
4. `qvhighlights_004`
5. `qvhighlights_005`
6. `qvhighlights_006`
7. `qvhighlights_007`
8. `qvhighlights_008`
9. `qvhighlights_009`
10. `qvhighlights_010`
11. `qvhighlights_011`
12. `qvhighlights_012`
13. `qvhighlights_013`
14. `qvhighlights_014`
15. `qvhighlights_015`
16. `qvhighlights_016`
17. `qvhighlights_017`
18. `qvhighlights_018`
19. `qvhighlights_019`
20. `qvhighlights_020`

## Final Decision

Do not update `data/queries/qvhighlights.json` with video IDs. Keep `results/paper/query_manifest.csv` status as `missing_video_id`. The repository can remain an initial code release with Warsaw, Bellevue, and Beach ready, but it cannot claim full runnable QVHighlights experiments until a verified 20-query source-video mapping is supplied.
