# RAG Metrics Explanation

本文说明当前仓库中普通 RAG comparison 的指标来源、计算方式和使用边界。依据的代码包括 `run_rag_comparison.py` 和 `RAG_Baselines/NaiveRAG.py`。普通 RAG 这里指 `EVIQUE`、`VideoRAG`、`NaiveRAG`、`TextVideoRAG`、`LightRAG`、`GraphRAG-l`、`GraphRAG-g` 等方法在同一 video/text base 上的 answer-quality 和系统统计比较。

## 0. 与 DB-RAG 的核心差异

普通 RAG 的 quantitative 默认 baseline 是 `NaiveRAG`，不是 EVIQUE。`run_rag_comparison.py` 的 `--quant-baseline` 默认值为 `NAIVE_MODEL_NAME`，也就是 `NaiveRAG`。在没有 reference answer 时，`RAG_Baselines/NaiveRAG.py` 会把 baseline method 自身在每个 query、每个 run、每个 quantitative 维度上固定为 `3`，其它 method 由 LLM judge 相对 baseline answer 打 1 到 5 分。

因此：

- 普通 RAG 默认是 `NaiveRAG = 3` 作为 quantitative 中线基准。
- DB-RAG 默认是 `EVIQUE = 3` 作为 quantitative 中线基准。
- 两者都使用同一套 LLM judge prompt 和六个 quantitative metrics。
- win-rate 都是 pairwise winner 统计，没有 3 分概念。

如果提供 `--reference-answers` 并成功把 reference answer 绑定到 QueryRecord，则 quantitative 会切换为 reference-based 评分。此时所有 method 都由 LLM judge 相对 reference answer 打 1 到 5 分，baseline 自身固定 3 的逻辑不再适用于有 reference 的 query。

## 1. 普通 RAG 实验阶段和产物

普通 RAG comparison 主要由 `run_rag_comparison.py` 驱动。

1. 数据和基础索引准备
   读取视频、segment、caption、ASR 或 EVIQUE standalone base。VideoRAG、NaiveRAG、TextVideoRAG、LightRAG 和 GraphRAG 共享或适配同一 grounded text base。

2. 各方法 answer generation
   每个 method 生成 answer 文件，并尽量写出 `generation_metrics.json` 或 `all_query_results.json`。这些文件记录 retrieval count、used count、query time、evidence size、token estimate、index size 等系统统计。

3. LLM judge evaluation
   调用 `RAG_Baselines/NaiveRAG.py` 中的 `run_winrate_eval` 和 `run_quantitative_eval`，写出 `evaluation/winrate_judgements.json`、`evaluation/winrate_table.csv`、`evaluation/quantitative_judgements.json`、`evaluation/quantitative_table.csv`。

4. Summary 写出
   `write_summary` 读取 generation metrics 和 quantitative table，写出根目录下的 `comparison_summary.csv` 和 `comparison_summary.md`。

5. Config 写出
   `comparison_config.json` 记录 paper protocol、selected models、judge model、frame count、chunking、EVIQUE base mode 等运行元数据。

## 2. Answer-quality Quantitative Metrics

对应文件：`evaluation/quantitative_table.csv`。表头由 `run_quantitative_eval` 生成：

`Model,Comprehensiveness,Empowerment,Trustworthiness,Depth,Density,Overall Score,Queries`

指标来自 `QUANT_METRICS`：

- `Comprehensiveness`: 答案覆盖问题各方面和细节的程度。
- `Empowerment`: 答案帮助读者理解、判断和使用信息的程度。
- `Trustworthiness`: 答案是否足够可信、具体，并与证据或常识一致。
- `Depth`: 答案是否有深入分析，而不是只给表层信息。
- `Density`: 答案是否信息密度高，并避免无关或冗余内容。
- `Overall Score`: 整体答案质量分，是独立 judge 输出，不是前五项的算术平均。

没有 reference answer 时，prompt 使用 baseline-relative 1-5 评分：

- 1: strongly worse than the baseline answer
- 2: weakly worse than the baseline answer
- 3: comparable to the baseline answer
- 4: weakly better than the baseline answer
- 5: strongly better than the baseline answer

普通 RAG 默认 `baseline_model = NaiveRAG`。当 `model_name == baseline_model` 时，代码直接把所有 `QUANT_METRICS` 写为 `Score = 3`，并保存到 raw JSON。其它 method 的 answer 会与同一 query 的 NaiveRAG answer 组成 prompt，由 LLM judge 返回六个 metric 的 Score 和 Explanation。

聚合方式：

1. raw key 形如 `query_id::model_name::runN`。
2. 每个 method、每个 metric 收集所有 query 和所有 eval run 的 score。
3. 表格值为 `sum(scores) / len(scores)`，格式化为两位小数。
4. `Queries` 是该 method 每个 metric score 数量的最大值。因为默认 `--eval-runs=5`，所以它更接近 judgement count，不一定是去重 query 数。

校验和失败行为：

- `validate_quant_result` 要求六个 metric 都存在，且 Score 可转为 1 到 5 的整数。
- 普通 RAG 的 `run_quantitative_eval` 没有像 DB-RAG wrapper 那样逐 task 捕获异常并 resume。LLM 调用或 JSON 校验失败通常会中断当前 evaluation。
- `save_csv` 在 rows 为空时不会写 CSV。

## 3. Win-rate Metrics

对应文件：`evaluation/winrate_table.csv`。列名由 pair 动态生成，常见形式为：

`Comparison,Metric,<model_a> Win Rate (%),<model_b> Win Rate (%),<model_a> Wins,<model_b> Wins,Judgements`

指标来自 `WINRATE_METRICS`：

- `Comprehensiveness`
- `Empowerment`
- `Trustworthiness`
- `Depth`
- `Density`
- `Overall Winner`

默认 pair 由 `choose_default_pairs` 生成。如果没有指定 anchor/reference model，默认优先比较：

- `EVIQUE vs VideoRAG`
- `EVIQUE vs NaiveRAG`
- `EVIQUE vs TextVideoRAG`
- `VideoRAG vs NaiveRAG`
- `VideoRAG vs TextVideoRAG`

如果指定 `--anchor-winrate-model` 或 `--winrate-reference-model`，则只比较该 anchor/reference method 与其它 selected models。

默认 win-rate 是双向比较，除非传入 `--single-pass-winrate`。双向比较中，同一 query、pair、run 会出现：

- `ori`: Answer 1 = model_a，Answer 2 = model_b
- `rev`: Answer 1 = model_b，Answer 2 = model_a

聚合时根据 order 把 `Answer 1` 或 `Answer 2` 映射回真实 model。胜率计算为：

`model win rate = model wins / Judgements * 100`

`Judgements` 是该 pair 和 metric 下有效 winner 总数，包含所有 query、run 和 answer order。默认 `--eval-runs=5`，双向时每个 query/pair/metric 理论上最多贡献 10 个 judgement。

## 4. Comparison Summary Metrics

对应文件：根目录 `comparison_summary.csv`。列名由 `SUMMARY_COLUMNS` 定义，可能在没有可靠基础索引构建耗时时省略 `数据集基础索引构建时间 (秒)`。完整列集合为：

`数据集,视频时长 (min),数据集大小 (MB),模型名称,数据集基础索引构建时间 (秒),基础索引大小 (MB),方法增量索引大小 (MB),端到端索引大小 (MB),平均 query 临时索引大小 (MB),平均 query 临时索引时间 (秒),平均查询时间 (秒),平均准确率得分,检索到的片段或项目数,使用的支持片段或项目数,使用 / 检索,最终证据包平均大小 (字符),LLM 输入 tokens 平均估计值,答案文件,准确率文件`

这些字段不是 LLM judge 的逐维评分，而是系统运行和输出统计。

字段含义：

- `数据集`: summary 中显示的数据集名称，来自 `--dataset-name` 或运行推断。
- `视频时长 (min)`: 输入视频总时长，单位分钟。
- `数据集大小 (MB)`: 输入视频文件大小总和。
- `模型名称`: method/model 名称。
- `数据集基础索引构建时间 (秒)`: 共享基础索引或 EVIQUE standalone base 的构建时间。若历史运行无法可靠拆分，该列可能被省略。
- `基础索引大小 (MB)`: shared dependency index 大小，例如 VideoRAG compatible base 或 EVIQUE base。
- `方法增量索引大小 (MB)`: 当前 method 自己额外构建或保存的索引大小。
- `端到端索引大小 (MB)`: 基础索引大小加方法增量索引大小。
- `平均 query 临时索引大小 (MB)`: query-time 临时索引平均大小。
- `平均 query 临时索引时间 (秒)`: query-time 临时索引平均构建时间。
- `平均查询时间 (秒)`: retrieval/query 阶段平均耗时，来自 generation metrics，不等于 LLM judge 时间。
- `平均准确率得分`: 优先使用 `quantitative_table.csv` 的 `Overall Score`。如果缺失，则使用运行时返回的 `accuracy_scores`。都没有则为 `N/A`。
- `检索到的片段或项目数`: 平均 retrieved count。
- `使用的支持片段或项目数`: 平均 used count。
- `使用 / 检索`: used / retrieved 的平均或派生 ratio。
- `最终证据包平均大小 (字符)`: answer prompt 中 evidence/context 的平均字符数。
- `LLM 输入 tokens 平均估计值`: answer LLM 输入 tokens 的平均估计值，不是 judge tokens。
- `答案文件`: 该 method answer 目录。
- `准确率文件`: quantitative table 文件路径，通常是 `evaluation/quantitative_table.csv`。

这些统计主要来自每个 method 的 `generation_metrics.json`。如果没有该文件，`load_generation_metrics` 会尝试从 `all_query_results.json` 重新派生 result metrics；LightRAG 和 GraphRAG 还包含针对 structured context 或 citation 格式的修复逻辑。

## 5. Generation Metrics

`aggregate_generation_metrics` 会为每个 method 汇总 per-query retrieval metrics，核心字段包括：

- `avg_query_time_seconds`: query/retrieval 平均耗时。
- `avg_evidence_chars`: 最终 evidence/context 平均字符数。
- `avg_llm_input_tokens_estimate`: answer LLM 输入 token 估计均值。
- `avg_retrieved_count`: 平均检索项目数。
- `avg_used_count`: 平均使用项目数。
- `avg_support_ratio`: used/retrieved ratio 均值。
- `metric_source_counts`: retrieval metric 来源计数，例如 structured context、answer citation estimate、unavailable。
- `index_build_time_seconds`: 方法索引构建耗时。
- `index_size_mb`: legacy 或 method index size。
- `method_specific_index_size_mb`: 方法增量索引大小。
- `shared_dependency_index_size_mb`: 共享基础依赖索引大小。
- `end_to_end_query_index_size_mb`: 端到端 query index size。

EVIQUE 还可能附加 visual retrieval 相关字段，例如 visual relation、event supplement、temporal-aware packing、budget fill ratio 等。这些用于诊断检索机制，不直接进入 answer-quality scoring。

LightRAG 的 retrieved/used count 可从 structured context 中的 entities、relationships、chunks 等计数派生。GraphRAG 的 retrieved/used count 可从 answer 中 `[Data: ...]` citation 解析派生。若旧运行把不可用 metric 记录为 0，repair 逻辑会将其修正为 unavailable，避免误把缺失统计当成真实 0。

## 6. Raw Judgement Files

`evaluation/quantitative_judgements.json` 保存 quantitative raw judgement。key 形如：

`query_id::model_name::runN`

value 是六个 metric 的 `Score` 和 `Explanation`。普通 RAG 默认 baseline NaiveRAG 的 value 由脚本直接生成，六个 Score 都是 3。其它 method 由 LLM judge 生成。

`evaluation/winrate_judgements.json` 保存 pairwise raw judgement。key 形如：

`query_id::model_a::vs::model_b::runN::ori`

或：

`query_id::model_a::vs::model_b::runN::rev`

value 是六个 metric 的 `Winner` 和 `Explanation`。`Winner` 只能是 `Answer 1` 或 `Answer 2`。

保留 raw judgement 的意义：

- 可追踪每个 query、pair、run、order 的 judge 原始判断。
- 可检查双向顺序是否完整、run 数是否符合预期。
- 可支持 case study 和 error analysis。
- 可在后续调整聚合脚本时避免重跑昂贵 LLM judge，但普通 RAG 当前评估函数本身不提供 per-key resume。

## 7. 与论文表格解读相关的注意点

- `Overall Score` 是 judge 独立输出，不应手工改写为五维平均。
- quantitative 的 3 分表示“与 baseline 可比”，不是绝对质量中等。普通 RAG 默认 baseline 是 NaiveRAG。
- win-rate 的 `Overall Winner` 是 pairwise overall winner，不是 `Overall Score`。
- `平均准确率得分` 在 summary 中来自 quantitative `Overall Score`，名字中有“准确率”，但它不是 ground-truth accuracy。
- `平均查询时间 (秒)` 是方法 query/retrieval 阶段，不包含 judge 时间。
- `LLM 输入 tokens 平均估计值` 是 answer generation 输入估计，不是 evaluation prompt 的输入 token。
- index size policy 排除了 raw video、model weights、answers、eval cache 等文件，目的是让方法索引大小更可比。

## 8. 本地验证命令

```bash
python -m compileall run_rag_comparison.py RAG_Baselines/NaiveRAG.py
ls -lh RAG_METRICS_EXPLANATION.md RAG_METRIC_DICTIONARY.csv
head -5 RAG_METRIC_DICTIONARY.csv
```

Windows PowerShell 等价命令：

```powershell
python -m compileall run_rag_comparison.py RAG_Baselines\NaiveRAG.py
Get-Item RAG_METRICS_EXPLANATION.md, RAG_METRIC_DICTIONARY.csv | Select-Object Name,Length
Get-Content -TotalCount 5 RAG_METRIC_DICTIONARY.csv
```
