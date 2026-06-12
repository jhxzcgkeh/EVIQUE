# Query Conversion Report

## Warsaw
- Source file: `warsaw-30-query.txt`
- Detected format: text
- Encoding: utf-8
- Top-level structure: list
- Original fields: original_index, query, query_id, raw_line
- Raw records: 30
- Converted records: 30
- Original query IDs present: no
- Generated query IDs: 30
- Public query ID rule: generated IDs use dataset prefix; original IDs are dataset-prefixed for global uniqueness and preserved as `metadata.original_query_id`.
- Video ID source: dataset_default
- Duplicate IDs: 0
- Duplicate query texts: 0
- Empty queries: 0
- Missing video IDs: 0
- SHA-256: `9b71cf8e0809062e890434bedd89cd266213e35c5525bf69184ce2afd9bc3dd2`
- Unresolved issue: none.

## Bellevue
- Source file: `bellevue-25-query.txt`
- Detected format: JSON
- Encoding: utf-8
- Top-level structure: list
- Original fields: id, question
- Raw records: 25
- Converted records: 25
- Original query IDs present: yes
- Generated query IDs: 0
- Public query ID rule: generated IDs use dataset prefix; original IDs are dataset-prefixed for global uniqueness and preserved as `metadata.original_query_id`.
- Video ID source: dataset_default
- Duplicate IDs: 0
- Duplicate query texts: 0
- Empty queries: 0
- Missing video IDs: 0
- SHA-256: `664936b17ff2be7d87d06a81a847325d3ca17ae22958c22558721c2454432614`
- Unresolved issue: none.

## QVHighlights
- Source file: `qvhighlight-20-query.txt`
- Detected format: text
- Encoding: utf-8
- Top-level structure: list
- Original fields: original_index, query, query_id, raw_line
- Raw records: 20
- Converted records: 20
- Original query IDs present: no
- Generated query IDs: 20
- Public query ID rule: generated IDs use dataset prefix; original IDs are dataset-prefixed for global uniqueness and preserved as `metadata.original_query_id`.
- Video ID source: missing in source file
- Duplicate IDs: 0
- Duplicate query texts: 0
- Empty queries: 0
- Missing video IDs: 20
- SHA-256: `40b86cf6cc92c5945709f19f6e76a7730f70c1a60a90f9e0aed2ae769a7324a4`
- Unresolved issue: the supplied query text file has no `video_id`, `source_video`, or `clip_id` fields. The separate video archive lists 15 files, but no query-to-video mapping is present, so per-query video IDs were not guessed.

## Beach
- Source file: `beach-21-query.txt`
- Detected format: text
- Encoding: utf-8
- Top-level structure: list
- Original fields: original_index, query, query_id, raw_line
- Raw records: 21
- Converted records: 21
- Original query IDs present: yes
- Generated query IDs: 0
- Public query ID rule: generated IDs use dataset prefix; original IDs are dataset-prefixed for global uniqueness and preserved as `metadata.original_query_id`.
- Video ID source: dataset_default
- Duplicate IDs: 0
- Duplicate query texts: 0
- Empty queries: 0
- Missing video IDs: 0
- SHA-256: `0df471c88d18c82e8ff5b4d0e10a632ba10165ee77fa39e0d76fda8e08b884b2`
- Unresolved issue: none.

