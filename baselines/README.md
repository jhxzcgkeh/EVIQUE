# Baselines

Third-party baseline implementations are not redistributed in this repository.
This directory only contains result adapters, prompts, and compact published outputs.

| Method | Official repository | Source redistributed here | Included artifact |
|---|---|---:|---|
| VideoRAG | https://github.com/HKUDS/VideoRAG | No | `adapters/videorag.py` |
| NaiveRAG | Local comparison helper in original experiments | No full runner | `adapters/naiverag.py` |
| TextVideoRAG | Local comparison helper in original experiments | No full runner | `adapters/textvideorag.py` |
| GraphRAG-l / GraphRAG-g | https://github.com/microsoft/graphrag | No | `adapters/graphrag.py` |

Adapters map raw baseline rows into `dataset, query_id, video_id, method, answer, evidence, runtime, input_tokens, metadata`. Missing fields remain `null`; adapters do not fabricate evidence, timings, or token counts.
