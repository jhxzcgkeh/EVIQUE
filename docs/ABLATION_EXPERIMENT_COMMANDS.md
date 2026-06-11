# EVIQUE Ablation Experiment Commands

本文件只描述 EVIQUE 内部消融实验，不重跑 DB baselines，不修改已有 `comparison_runs/db_rag_*` 主实验结果目录。

## 1. 环境变量设置

```bash
cd <repo>
source conda activate <conda-env>

export OPENAI_BASE_URL='${OPENAI_BASE_URL}'
export OPENAI_MODEL='deepseek-ai/DeepSeek-V3.2'
export OPENAI_EMBEDDING_MODEL='BAAI/bge-m3'
read -s -p "OPENAI_API_KEY: " OPENAI_API_KEY
echo
export OPENAI_API_KEY
```

常用消融开关由 config 自动注入，不需要手工 export。默认不设置这些变量时，EVIQUE 行为保持不变。

## 2. 检查 Warsaw / Beach Workdir

```bash
ls -lh <repo>/reused_evique_workdirs/warsaw
ls -lh <repo>/reused_evique_workdirs/beach

for D in warsaw beach; do
  W=<repo>/reused_evique_workdirs/$D
  test -f "$W/scope_view.jsonl" && \
  test -f "$W/target_view.jsonl" && \
  test -f "$W/track_view.jsonl" && \
  test -f "$W/event_view.jsonl" && \
  test -f "$W/adaptive_event_view.jsonl" && \
  echo "$D workdir ok"
done
```

消融 runner 只读取已有 EVIQUE workdir，不重建 index，不抽帧，不检测，不 caption，不构图。

## 3. Warsaw 3-query Smoke

```bash
RUN_ROOT=<repo>/comparison_runs/ablation_warsaw_smoke_3q_v1
QUERY_FILE=<repo>/comparison_runs/db_rag_warsaw_30q_9methods_v1/queries.jsonl
EVIQUE_WORKDIR=<repo>/reused_evique_workdirs/warsaw

mkdir -p "$RUN_ROOT"

python run_evique_ablation_db_rag.py \
  --config <repo>/ablation_configs/core_planner_packaging.json \
  --queries "$QUERY_FILE" \
  --output-root "$RUN_ROOT" \
  --evique-workdir "$EVIQUE_WORKDIR" \
  --max-evidence-tokens 3200 \
  --limit-queries 3 \
  --stage all \
  --eval-runs 1 \
  --judge-model deepseek-ai/DeepSeek-V3.2 \
  --answer-model deepseek-ai/DeepSeek-V3.2 \
  --progress \
  --resume \
  2>&1 | tee "$RUN_ROOT/smoke_all.log"
```

只先跑这个 smoke。确认结果正常后再跑 Warsaw + Beach 全量。

## 4. Core Ablation Warsaw Full

```bash
RUN_ROOT=<repo>/comparison_runs/ablation_warsaw_core_planner_packaging_v1
QUERY_FILE=<repo>/comparison_runs/db_rag_warsaw_30q_9methods_v1/queries.jsonl
EVIQUE_WORKDIR=<repo>/reused_evique_workdirs/warsaw

mkdir -p "$RUN_ROOT"

python run_evique_ablation_db_rag.py \
  --config <repo>/ablation_configs/core_planner_packaging.json \
  --queries "$QUERY_FILE" \
  --output-root "$RUN_ROOT" \
  --evique-workdir "$EVIQUE_WORKDIR" \
  --max-evidence-tokens 3200 \
  --stage all \
  --eval-runs 2 \
  --judge-model deepseek-ai/DeepSeek-V3.2 \
  --answer-model deepseek-ai/DeepSeek-V3.2 \
  --progress \
  --resume \
  2>&1 | tee "$RUN_ROOT/stage_all_core_2runs.log"
```

## 5. Core Ablation Beach Full

```bash
RUN_ROOT=<repo>/comparison_runs/ablation_beach_core_planner_packaging_v1
QUERY_FILE=<repo>/comparison_runs/db_rag_beach_21q_9methods_v1/queries.jsonl
EVIQUE_WORKDIR=<repo>/reused_evique_workdirs/beach

mkdir -p "$RUN_ROOT"

python run_evique_ablation_db_rag.py \
  --config <repo>/ablation_configs/core_planner_packaging.json \
  --queries "$QUERY_FILE" \
  --output-root "$RUN_ROOT" \
  --evique-workdir "$EVIQUE_WORKDIR" \
  --max-evidence-tokens 3200 \
  --stage all \
  --eval-runs 2 \
  --judge-model deepseek-ai/DeepSeek-V3.2 \
  --answer-model deepseek-ai/DeepSeek-V3.2 \
  --progress \
  --resume \
  2>&1 | tee "$RUN_ROOT/stage_all_core_2runs.log"
```

## 6. Merge Core Ablation

```bash
python merge_evique_ablation_results.py \
  --output-root <repo>/comparison_runs/ablation_core_planner_packaging_2datasets_summary_v1 \
  --run-root <repo>/comparison_runs/ablation_warsaw_core_planner_packaging_v1 \
  --run-root <repo>/comparison_runs/ablation_beach_core_planner_packaging_v1
```

查看合并结果：

```bash
OUT=<repo>/comparison_runs/ablation_core_planner_packaging_2datasets_summary_v1

cat "$OUT/quantitative_table.csv"
cat "$OUT/winrate_table.csv"
cat "$OUT/comparison_summary.csv"
head -20 "$OUT/per_query_summary.csv"
```

## 7. View Leave-One-Out

Warsaw:

```bash
RUN_ROOT=<repo>/comparison_runs/ablation_warsaw_view_leave_one_out_v1
QUERY_FILE=<repo>/comparison_runs/db_rag_warsaw_30q_9methods_v1/queries.jsonl
EVIQUE_WORKDIR=<repo>/reused_evique_workdirs/warsaw

mkdir -p "$RUN_ROOT"

python run_evique_ablation_db_rag.py \
  --config <repo>/ablation_configs/view_leave_one_out.json \
  --queries "$QUERY_FILE" \
  --output-root "$RUN_ROOT" \
  --evique-workdir "$EVIQUE_WORKDIR" \
  --max-evidence-tokens 3200 \
  --stage all \
  --eval-runs 2 \
  --judge-model deepseek-ai/DeepSeek-V3.2 \
  --answer-model deepseek-ai/DeepSeek-V3.2 \
  --progress \
  --resume \
  2>&1 | tee "$RUN_ROOT/stage_all_view_2runs.log"
```

Beach 同理，把 `QUERY_FILE` 和 `EVIQUE_WORKDIR` 换成 beach，并修改 `RUN_ROOT`。

## 8. 查看进度

runner 会输出：

- 当前 stage: `evidence` / `answers` / `eval`
- 当前 variant
- 当前 query_id
- 当前 run 和 order: evaluation 阶段会打印 run/order
- completed / total
- elapsed、avg time、ETA
- skipped / done / failed 计数

也可以单独查看日志：

```bash
tail -f "$RUN_ROOT/smoke_all.log"
find "$RUN_ROOT" -maxdepth 2 -type f | sort | head -100
find "$RUN_ROOT" -type f -path "*/answers-*/*" -name "answer_*.md" | wc -l
```

## 9. 断点继续

默认就是 resume-style 行为：

- evidence: 已存在 `evidence-<variant>/evidence_<query_id>.json` 和 `context_<query_id>.md` 就跳过。
- answers: 已存在 `answers-<variant>/answer_<query_id>.md` 就跳过。
- evaluation: 已存在 raw judgement key 且未设置 `--overwrite-eval` 就跳过。

继续跑同一命令即可：

```bash
python run_evique_ablation_db_rag.py ... --progress --resume
```

只补 answer：

```bash
python run_evique_ablation_db_rag.py ... --stage answers --progress --resume
```

只补 evaluation：

```bash
python run_evique_ablation_db_rag.py ... --stage eval --progress --resume
```

## 10. 覆盖重跑

重写 evidence 和 answers：

```bash
python run_evique_ablation_db_rag.py ... --stage all --overwrite
```

只重写 judgement：

```bash
python run_evique_ablation_db_rag.py ... --stage eval --overwrite-eval
```

不要把 ablation 输出写进 `comparison_runs/db_rag_*` 主实验目录。

## 11. 打包备份

```bash
RUN_ROOT=<repo>/comparison_runs/ablation_warsaw_smoke_3q_v1
tar -czf "$RUN_ROOT.tar.gz" -C "$(dirname "$RUN_ROOT")" "$(basename "$RUN_ROOT")"
ls -lh "$RUN_ROOT.tar.gz"
```

合并结果同理：

```bash
OUT=<repo>/comparison_runs/ablation_core_planner_packaging_2datasets_summary_v1
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
```

## 12. 参数含义

- `--config`: 消融配置 JSON，包含 group、anchor 和 variants。
- `--queries`: normalized `queries.jsonl`。
- `--output-root`: 本次消融输出目录，必须是新的 ablation 目录。
- `--evique-workdir`: 已存在的 EVIQUE workdir/index。
- `--max-evidence-tokens`: evidence context token budget，同时注入 `EVIQUE_EVIDENCE_TOKEN_BUDGET`。
- `--max-evidence`: 每个 query 最多保留 evidence item 数。
- `--limit-queries`: smoke 用，限制 query 数。
- `--stage`: `evidence`、`answers`、`eval` 或 `all`。
- `--eval-runs`: LLM judge 重复次数。
- `--judge-model`: judge LLM。
- `--answer-model`: answer generation LLM。
- `--single-pass-winrate`: 关闭双向 answer order 交换。
- `--dry-run`: answer 阶段写占位答案，不调用 answer LLM；仅用于本地调试。
- `--resume`: 断点继续，默认就是跳过已有完整文件。
- `--overwrite`: 重写 evidence 和 answers。
- `--overwrite-eval`: 重写 evaluation judgements。
- `--progress` / `--no-progress`: 开关进度输出。

## 13. 常见错误和解决方案

`EVIQUE workdir does not exist`

检查 `--evique-workdir` 是否指向已有 workdir。runner 不会重建 index。

`missing per-query summary`

merge 的某个 `--run-root` 没有完成 evaluation。先跑该 run 的 `--stage eval`。

`anchor has no answers`

config 的 `anchor` 必须等于某个 variant 的 `safe_name`，且该 variant 的 answer 文件必须存在。

`OPENAI_API_KEY` 或 API 连接错误

检查环境变量、base URL 和模型名。可先用 `--stage evidence` 验证非 LLM 部分。

`winrate_table.csv` 缺少某个 ablation pair

确认 anchor answer 和该 ablation answer 都存在；win-rate 只比较 `config.anchor` vs each ablation variant，不做所有 variants 两两比较。

## 14. Smoke 完成后汇报命令

```bash
RUN_ROOT=<repo>/comparison_runs/ablation_warsaw_smoke_3q_v1

find "$RUN_ROOT" -maxdepth 2 -type f | sort | head -100
find "$RUN_ROOT" -type f -path "*/answers-*/*" -name "answer_*.md" | wc -l
cat "$RUN_ROOT/evaluation/quantitative_table.csv"
cat "$RUN_ROOT/evaluation/winrate_table.csv"
cat "$RUN_ROOT/evaluation/comparison_summary.csv"
```

汇报时检查：

- 生成了哪些 `evidence-*` 目录。
- 生成了哪些 `answers-*` 目录。
- `quantitative_table.csv` 是否包含所有 variants。
- `winrate_table.csv` 是否只比较 Full vs ablation variants。
- `comparison_summary.csv` 的 evidence chars / tokens / counts / query time 是否正常。
- 是否存在 error / skipped / parse_failed。
- 下一步 Warsaw + Beach 全量命令是否准备好。
