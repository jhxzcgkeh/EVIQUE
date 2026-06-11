# DB-RAG Metrics Explanation

本文说明当前仓库中 DB-RAG answer-quality comparison 的指标来源、计算方式和使用边界。依据的代码包括 `evaluate_db_rag_answers.py`、`merge_db_rag_4dataset_results.py`、`db_rag_pipeline_common.py`、`run_db_baseline_evidence_retrieval.py`、`generate_db_rag_answers.py`。当前本地工作区可见的是 smoke 结果目录，未包含粘贴文本中提到的正式四数据集合并目录 `comparison_runs/db_rag_4datasets_96q_9methods_summary_v1`，所以本文的表头核对基于本地可见 CSV header 和代码生成逻辑。

## 0. 最关键结论

DB-RAG 的 quantitative 评分默认以 `EVIQUE` 作为 baseline。代码参数 `--quant-baseline` 默认值是 `EVIQUE`。在没有 reference answer 的 DB-RAG 路径中，脚本不会调用 LLM judge 去给 EVIQUE 自己打分，而是直接把 EVIQUE 在每个 query、每个 run、每个 quantitative 维度上的分数固定写成 `3`，解释为 `Baseline answer; score fixed at 3 by definition.`。

因此，回答“DB-RAG 的实验中是不是以 EVIQUE 都打 3 分作为评判标准来的？”：

是，限于 quantitative answer-quality 评分，当前 DB-RAG 代码默认把 EVIQUE 作为 1-5 相对评分的中线基准，EVIQUE 自身固定为 3 分。其它方法由 LLM judge 相对同一 query 的 EVIQUE answer 打 1 到 5 分：1/2 表示弱于 EVIQUE，3 表示与 EVIQUE 可比，4/5 表示优于 EVIQUE。这个 3 分不是 judge 对 EVIQUE 的质量判断，而是 baseline 定义。

这不适用于 win-rate。DB-RAG win-rate 默认以 EVIQUE 为 anchor，与每个 baseline 成对比较，LLM judge 直接选择 Answer 1 或 Answer 2 的 winner，没有 3 分概念。

## 1. 实验阶段和产物

DB-RAG pipeline 分成四个主要阶段。

1. Evidence retrieval: `run_db_baseline_evidence_retrieval.py`
   对每个 method 和 query 生成 evidence JSON、context 文本，并写出 `evidence_summary.csv`。EVIQUE 走原生 retriever；其它 DB baseline 通过 adapter 输出统一格式的 evidence rows。

2. Answer generation: `generate_db_rag_answers.py`
   把 evidence context 放入统一 DB-RAG answer prompt，生成 `answers-<method>/answer_<query>.md`，并写出 `answer_metadata.jsonl`。dry-run 时写占位 answer。

3. Evaluation: `evaluate_db_rag_answers.py`
   调用普通 RAG 的 judge protocol，生成 `quantitative_judgements.json`、`winrate_judgements.json`，再聚合为 `quantitative_table.csv`、`winrate_table.csv`、`per_query_summary.csv`、`comparison_summary.csv`。

4. Merge: `merge_db_rag_4dataset_results.py`
   读取多个单数据集 run 的 `evaluation/` 输出，合并为一个 summary root 下的四个 CSV 和两个 raw JSON。合并 quantitative 时使用 per-query 行重新聚合，不是简单平均每个 dataset 的表格分数。

## 2. Answer-quality Quantitative Metrics

对应文件：`quantitative_table.csv`，本地可见 header 为：

`Model,Comprehensiveness,Empowerment,Trustworthiness,Depth,Density,Overall Score,Queries`

这些指标来自 `QUANT_METRICS`：

- `Comprehensiveness`: LLM judge 对答案覆盖问题各方面细节程度的评分。
- `Empowerment`: LLM judge 对答案帮助读者理解、判断和使用信息能力的评分。
- `Trustworthiness`: LLM judge 对答案可信度、细节充分性、与常识一致性的评分。
- `Depth`: LLM judge 对答案分析深度和非表层程度的评分。
- `Density`: LLM judge 对答案相关信息密度、避免冗余程度的评分。
- `Overall Score`: LLM judge 对整体答案质量的独立评分，不是前五个维度的算术平均。

评分范围是 1 到 5。没有 reference answer 时，prompt 明确要求 evaluated answer 相对 baseline answer 评分：

- 1: strongly worse than the baseline answer
- 2: weakly worse than the baseline answer
- 3: comparable to the baseline answer
- 4: weakly better than the baseline answer
- 5: strongly better than the baseline answer

DB-RAG 默认 `baseline_model = EVIQUE`。当 `method == EVIQUE` 时，脚本直接构造 JSON，把所有 `QUANT_METRICS` 的 `Score` 写为 3。其它 method 则把 EVIQUE answer 放入 `Baseline Answer`，把当前 method answer 放入 `Evaluation Answer`，调用 LLM judge 并通过 `validate_quant_result` 检查六个指标都存在且分数在 1 到 5。

`quantitative_table.csv` 的每一行是一个 method 的聚合均值。计算方式：

1. raw key 形如 `query_id::method::runN`。
2. 对每个 method、每个 metric，收集所有 query 和所有 eval run 的整数分数。
3. 表格中的 metric 值为 `sum(scores) / len(scores)`，格式化为两位小数。
4. `Queries` 实际是该 method 在各 metric 下可聚合分数数量的最大值。它更准确地说是 judgement count 或 score count，不一定等于去重 query 数，因为 eval runs 会把同一个 query 计入多次。

当前 DB-RAG 的 `LoadedQuery` 转成普通 RAG `QueryRecord` 时没有填 `reference_answer`，所以正常 DB-RAG 路径会走 EVIQUE baseline 逻辑。如果未来显式为 QueryRecord 提供 reference answer，则有 reference 的 query 会改用 reference-based prompt，对所有 method 调 LLM judge，而不是固定 baseline 自身为 3。

缺失和失败处理：

- 如果 judgement JSON 已存在同一个 key 且未设置 `--overwrite-eval`，DB-RAG evaluation 会跳过该 judgement。`--resume` 参数说明了这种行为，但当前实现默认就是 resume-style skip。
- 如果 LLM 调用、JSON 解析或校验失败，该 task 会记录 failed count，不保存该 key，后续聚合会忽略缺失 judgement。
- 如果没有任何可聚合 score，`_quantitative_table_from_scores` 会把对应 metric 输出为 `0.00`。本地 smoke 的 `quantitative_judgements.json` 为空，所以 smoke 表中的 0.00 是空 judgement 的兜底输出，不是正式质量分数。

## 3. Win-rate Metrics

对应文件：`winrate_table.csv`。本地单 run fallback header 为：

`Comparison,Metric,EVIQUE Win Rate (%),Baseline Win Rate (%),EVIQUE Wins,Baseline Wins,Judgements`

实际有 judgement rows 时，代码会根据 pair 动态生成列名，例如 `EVIQUE Win Rate (%)`、`LOVO Win Rate (%)`、`EVIQUE Wins`、`LOVO Wins`。合并表也沿用这种动态列名。

Win-rate 使用 `WINRATE_METRICS`：

- `Comprehensiveness`
- `Empowerment`
- `Trustworthiness`
- `Depth`
- `Density`
- `Overall Winner`

DB-RAG 默认 `--anchor-winrate-model=EVIQUE`，`choose_default_pairs` 会生成 `EVIQUE vs <each other method>`。每个 query 上，judge 会看到 Answer 1 和 Answer 2，逐维选择 winner，然后选 overall winner。返回 JSON 只允许 `Winner` 为 `Answer 1` 或 `Answer 2`。

默认是双向比较，除非传入 `--single-pass-winrate`。双向比较会对同一 pair、同一 query、同一 run 做两次：

- `ori`: Answer 1 = model_a，Answer 2 = model_b
- `rev`: Answer 1 = model_b，Answer 2 = model_a

这样做是为了降低 answer order bias。聚合时，脚本会根据 key 中的 `ori`/`rev` 把 `Answer 1` 或 `Answer 2` 映射回真实 method，再计入 win count。

计算方式：

1. raw key 形如 `query_id::model_a::vs::model_b::runN::ori` 或 `...::rev`。
2. 对每个 pair 和每个 metric 统计 model_a wins、model_b wins 和总 judgement 数。
3. `model_a Win Rate (%) = model_a wins / total * 100`。
4. `Judgements` 是进入该 pair、metric 聚合的有效 winner 数，包含所有 query、run 和 order。

Win-rate 与 quantitative 的区别：

- quantitative 是绝对表格形式的 1-5 相对 baseline 分数。DB-RAG 默认 baseline 是 EVIQUE。
- win-rate 是 pairwise 相对偏好比较。judge 不输出 1-5 分，只输出 Answer 1 或 Answer 2 winner。
- quantitative 中 EVIQUE 默认固定为 3；win-rate 中 EVIQUE 不固定分数，而是作为 pair anchor 参与胜负统计。

## 4. Per-query Metrics

对应文件：`per_query_summary.csv`。单 run header 为：

`dataset,query_id,method,method_fidelity,question,type,difficulty,Comprehensiveness,Empowerment,Trustworthiness,Depth,Density,Overall Score,evidence chars,LLM input token estimate,retrieved evidence count,used evidence count,avg query time,index size,answer path`

merge 后会在最前面增加 `source_run`：

`source_run,dataset,query_id,method,method_fidelity,question,type,difficulty,...`

字段来源：

- `dataset`, `query_id`, `question`, `type`, `difficulty`: 来自 normalized queries。
- `method`: 当前方法名。
- `method_fidelity`: 来自 answer metadata，缺失时使用 `DB_RAG_METHOD_FIDELITY`。
- 六个 quantitative metric: 来自 `aggregate_quantitative_judgements` 对 raw quantitative judgement 的 per method/per query 均值。
- `evidence chars`: answer metadata 中传入 LLM 的 evidence context 字符数。
- `LLM input token estimate`: answer prompt 总输入 token 估计值，包含 system prompt、user prompt、question 和 evidence context。
- `retrieved evidence count`: evidence retrieval 阶段返回的可用 evidence 数。
- `used evidence count`: 最终塞进 evidence context 的 evidence 数。
- `avg query time`: evidence retrieval 或 adapter query 阶段耗时，不等于 answer generation 时间，也不等于 judge 时间。
- `index size`: 当前 method 使用的索引目录大小估计，单位 MB。
- `answer path`: 当前 query/method 的 answer 文件路径。

Per-query 表的作用：

- 定位 EVIQUE 或 baseline 在哪些 query 上胜出、失败或缺失 judgement。
- 按 query type、difficulty、dataset 做分组分析。
- 支持论文 case study、error analysis 和按 query 粒度追踪 evidence/answer/judge 的链路。

当前本地 smoke 的 per-query metric 为空，是因为 raw judgement JSON 为空； evidence/system 字段仍来自 evidence 和 answer metadata。

## 5. Evidence and System Statistics

对应文件：`comparison_summary.csv`。header 为：

`method,method_fidelity,query_count,answer_count,Comprehensiveness,Empowerment,Trustworthiness,Depth,Density,Overall Score,evidence chars,LLM input token estimate,retrieved evidence count,used evidence count,avg query time,index size,answer path`

该表由 `build_comparison_summary` 从 `per_query_summary.csv` 聚合而来，粒度是 per method。

字段含义：

- `method`: 方法名，如 EVIQUE、LOVO、VOCAL、MIRIS、OTIF、UMT、VISA、FiGO、ZELDA。
- `method_fidelity`: 方法接入保真度说明，不是性能指标，不参与评分。
- `query_count`: 该 method 覆盖的去重 query 数。
- `answer_count`: 对应 answer path 存在的数量。
- 六个 quantitative metric: 对 per-query 表中数值求均值。
- `evidence chars`: 每个 query 传入 answer LLM 的 evidence context 字符数均值。
- `LLM input token estimate`: 每个 query answer prompt 的 token 估计均值。
- `retrieved evidence count`: 每个 query 检索到的 evidence 数均值。
- `used evidence count`: 每个 query 最终使用的 evidence 数均值。
- `avg query time`: evidence retrieval 或 adapter query 阶段耗时均值。
- `index size`: method index size 均值。
- `answer path`: 该 method answer 目录。

`method_fidelity` 的取值：

- `native`: EVIQUE 原生实现。
- `local_reproduction`: 按论文流程或已有代码做的本地复现。
- `third_party_proxy`: 为了统一自然语言 query 和 LLM answer pipeline 编写的 proxy adapter。
- `local_reimplementation`: 本地重实现版本。

`method_fidelity` 只用于实验透明性说明，不参与排序、评分或 win-rate。

`adapter_status` 会出现在 evidence JSON 或 adapter row 中，用于表示 adapter 在当前 pipeline 中的接入状态。当前 `comparison_summary.csv` 没有把它作为独立列输出。如果需要按 adapter status 分析，应回到 evidence JSON、DB benchmark result JSONL 或 `evidence_summary.csv`。

`avg query time` 只统计 retrieval/adapter 查询阶段，不包含 answer generation 的 LLM 调用时间，也不包含 LLM judge 时间。

## 6. Raw Judgement Files

`quantitative_judgements.json` 保存 quantitative raw judge 结果。key 形如：

`query_id::method::runN`

value 是每个 metric 的对象，例如：

`{"Comprehensiveness": {"Score": 4, "Explanation": "..."}, ...}`

对于 DB-RAG 默认 baseline EVIQUE，EVIQUE 的 value 由脚本直接生成，六个 metric 的 `Score` 都是 3。其它 method 的 value 来自 LLM judge 并通过校验。

`winrate_judgements.json` 保存 pairwise raw judge 结果。key 形如：

`query_id::model_a::vs::model_b::runN::ori`

或：

`query_id::model_a::vs::model_b::runN::rev`

value 是每个 metric 的 winner 和 explanation，例如：

`{"Comprehensiveness": {"Winner": "Answer 1", "Explanation": "..."}, ...}`

保留 raw judgement 的原因：

- 可以追溯每个 query、method、run、order 的 judge 原始输出。
- 可以在不重跑 LLM judge 的情况下重新聚合 CSV。
- 可以检查 run 数、双向顺序、缺失 judgement、parse failure 和 resume skip。
- 可以为 case study 直接引用具体 query 的 explanation。

resume 逻辑：

- DB-RAG evaluation 读取已有 JSON。如果 key 已存在且未设置 `--overwrite-eval`，该 judgement 会被跳过。
- 中断后继续跑时，已存在 key 不会重复调用 LLM judge。
- 缺失 key 会继续执行。失败且未写入 key 的 judgement 也会在下一次运行中重试。

## 7. Merge Metrics

`merge_db_rag_4dataset_results.py` 的合并流程：

1. 对每个 `--run-root` 读取 `evaluation/per_query_summary.csv`，给每行增加 `source_run`，追加到合并 per-query 表。
2. 读取每个 run 的 `evaluation/quantitative_judgements.json`，用 `run_name::raw_key` 前缀写入合并后的 `quantitative_judgements.json`。
3. 收集每个 run 的 `evaluation/winrate_judgements.json` path，用 `aggregate_winrate_judgements(paths)` 跨 run 统计 win-rate。
4. 由合并后的 per-query rows 调用 `build_comparison_summary` 生成 merged `comparison_summary.csv`。
5. 由合并后的 per-query rows 调用 `aggregate_quant_table_from_per_query` 生成 merged `quantitative_table.csv`。
6. 写出 merged `per_query_summary.csv`、`comparison_summary.csv`、`quantitative_table.csv`、`winrate_table.csv`、`quantitative_judgements.json`、`winrate_judgements.json`。

注意：合并 quantitative 时不是简单平均四个 dataset 的 `quantitative_table.csv` 分数，而是把所有 per-query rows 放到一起，再按 method 和 metric 求均值。因为 per-query rows 中的 score 已经是该 query/method 跨 eval run 的均值，所以 merged quantitative 是 query-level 聚合，而不是 dataset-level unweighted average。

Win-rate merge 直接从所有 raw `winrate_judgements.json` 重新计数，因此会自然包含所有 dataset、query、run 和 order 的 judgement。

## 8. 本地验证命令

```bash
python -m compileall evaluate_db_rag_answers.py merge_db_rag_4dataset_results.py db_rag_pipeline_common.py run_db_baseline_evidence_retrieval.py generate_db_rag_answers.py
ls -lh DB_RAG_METRICS_EXPLANATION.md DB_RAG_METRIC_DICTIONARY.csv
head -5 DB_RAG_METRIC_DICTIONARY.csv
```

Windows PowerShell 等价命令：

```powershell
python -m compileall evaluate_db_rag_answers.py merge_db_rag_4dataset_results.py db_rag_pipeline_common.py run_db_baseline_evidence_retrieval.py generate_db_rag_answers.py
Get-Item DB_RAG_METRICS_EXPLANATION.md, DB_RAG_METRIC_DICTIONARY.csv | Select-Object Name,Length
Get-Content -TotalCount 5 DB_RAG_METRIC_DICTIONARY.csv
```
