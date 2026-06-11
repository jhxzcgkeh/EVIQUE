# EVIQUE Evidence Planning and Packing Architecture

本文整理 EVIQUE 模型中“证据规划”和“证据打包”两部分的架构实现逻辑。文档只描述现有实现的模块职责、数据流、决策规则和输出结构，不引入新的代码逻辑。

相关核心文件：

- `evique/planner.py`: 查询意图识别、查询类型分类、证据视图规划。
- `evique/cost_planner.py`: 基于成本和收益的视图选择与预算分配。
- `evique/retriever.py`: 证据视图加载、检索执行、融合、补充和最终 evidence package 生成。
- `evique/evidence_packer.py`: 候选证据去重、分层、预算控制、时序感知打包和 metadata 记录。

## 1. 总体目标

EVIQUE 的证据规划和打包层位于“索引视图”和“答案生成模型”之间，目标不是简单返回 top-k 文本片段，而是为下游 LLM 组织一组结构化、可追溯、预算受控的证据包。

其核心职责包括：

1. 理解用户 query 属于什么类型，例如对象可见性、空间关系、时序变化、轨迹、交互、事件定位、语音或通用描述。
2. 判断哪些索引视图最适合回答该 query，例如 scope、target、track、event、adaptive_event、visual_object、visual_track、visual_event、visual_relation。
3. 在多个视图之间分配检索预算，优先查询高收益、低噪声、高匹配度的视图。
4. 对候选证据进行统一归一化、打分、去重、分层和裁剪。
5. 在 token 和字符预算内保留核心证据、支撑证据和上下文证据。
6. 对空间关系、时序事件、运动轨迹等 query 保留必要的结构化证据，不让普通文本上下文压过关键视觉证据。
7. 输出包含证据、规划信息、预算信息、过滤信息、打包信息和回答约束的 evidence package。

整体流程可以概括为：

```text
user query
  -> QueryPlanner
  -> optional CostBasedViewPlanner
  -> EvidenceRetriever view execution
  -> evidence ranking and fallback
  -> visual / graph / event evidence fusion
  -> EvidencePacker
  -> final evidence package for LLM answer generation
```

## 2. 主要模块分工

### 2.1 QueryPlanner

`QueryPlanner` 是第一层规划器，负责把自然语言 query 转换为一个结构化 `QueryPlan`。

`QueryPlan` 的核心字段包括：

- `query`: 原始 query 文本。
- `query_type`: 查询类型，例如 `spatial_relation`、`trajectory`、`before_after`、`event_localization`。
- `selected_views`: 规划后需要检索的视图集合。
- `dependency_order`: 视图执行顺序。
- `required_views`: 必须保留的核心视图。
- `optional_views`: 可作为补充的视图。
- `evidence_roles`: 每类视图在回答中的角色，例如 primary、supporting、context。
- `query_terms`: 从 query 中抽取的关键词。
- `view_weights`: 不同视图的重要性权重。
- `constraints`: 检索与证据组织约束。
- `guidance`: 给下游回答生成阶段的自然语言提示。
- `query_intents`: 细粒度意图布尔标记。
- `route_reason`: 为什么把 query 路由到当前 query type。
- `visual_trigger_reason`: 为什么触发视觉证据链。

### 2.2 CostBasedViewPlanner

`CostBasedViewPlanner` 是第二层规划器。它不会改变 query 的语义，而是在 `QueryPlan` 的基础上，根据索引统计信息决定：

- 哪些视图最值得查。
- 查询顺序是什么。
- 每个视图最多拿多少行。
- 总候选证据预算是多少。
- 何时可以停止检索。

该模块使用 view stats、query 意图、关键词匹配、视频过滤条件和视图噪声风险来计算每个视图的综合分数。

### 2.3 EvidenceRetriever

`EvidenceRetriever` 是证据检索和组装的中枢。它负责：

- 从 workdir 加载各类索引视图。
- 调用 `QueryPlanner` 和 `CostBasedViewPlanner`。
- 对不同视图执行实际检索。
- 处理 video filter。
- 执行 fallback 检索。
- 合并、排序和筛选候选证据。
- 注入视觉链证据、时序证据、关系补充证据。
- 调用 `EvidencePacker` 完成最终打包。

### 2.4 EvidencePacker

`EvidencePacker` 是最终证据包的压缩与组织层。它负责在有限 token budget 内决定：

- 哪些证据必须保留。
- 哪些证据只是支撑信息。
- 哪些证据可以作为背景上下文。
- 哪些证据重复、跨视频不匹配、低价值或超预算，需要丢弃。
- 时序 query 是否需要 before、focal、after 三类证据。
- 空间 query 是否需要保留 visual_relation 或 nearby context。

## 3. 查询规划逻辑

### 3.1 query term 抽取

`QueryPlanner` 会从 query 文本中抽取关键词。其处理逻辑包括：

- 使用正则和 tokenizer 提取 token。
- 过滤停用词。
- 保留长度足够的实体词、动作词、关系词。
- 对复数词加入简单单数形式，例如 `cars` 同时加入 `car`。
- 对短语类 query 保留具有检索意义的词组。

这些 `query_terms` 后续会参与：

- 视图选择。
- 视图打分。
- 证据 relevance 计算。
- 打包阶段的 query token overlap 计算。

### 3.2 query intent 检测

规划器会检测多个细粒度 query intent。常见 intent 包括：

- `spatial_relation`: 是否询问 near、beside、behind、left of、right of、in front of 等空间关系。
- `temporal_trajectory`: 是否询问 move、cross、enter、leave、path、trajectory 等轨迹行为。
- `temporal_ordering`: 是否询问 before、after、then、during、when 等时序顺序。
- `temporal_interaction`: 是否询问对象之间的动态交互。
- `transition`: 是否询问状态变化，例如 appear、disappear、start、stop、change。
- `multi_object_interaction`: 是否涉及多个对象之间的关系或共同出现。
- `event_localization`: 是否需要定位事件发生时段。
- `evidence_grounded`: 是否要求回答基于具体证据。

这些 intent 允许系统在 query type 较粗时仍然触发特定证据链。例如一个 query 即使被归为 event，也可能因为包含 near/cross 等词触发 visual relation 或 track 证据。

### 3.3 query type 分类

`QueryPlanner` 会将 query 归入一个主 `query_type`。常见类型包括：

- `general_description`: 视频整体描述。
- `instance_spatial_temporal`: 同时包含对象、空间和时间约束的实例级 query。
- `temporal`: 普通时序 query。
- `interaction`: 对象交互 query。
- `trajectory`: 轨迹或移动路径 query。
- `spatial_relation`: 空间关系 query。
- `event_localization`: 事件定位 query。
- `speech`: 语音、说话、转录相关 query。
- `state_change`: 状态变化 query。
- `before_after`: 前后顺序 query。
- `event`: 事件级 query。
- `object_list`: 对象枚举 query。
- `object_grounding`: 对象定位或对象证据 query。
- `default`: 无法明确归类时的默认类型。

分类结果会直接影响后续：

- required views。
- optional views。
- evidence roles。
- guidance。
- 打包阶段的 mandatory floor。
- temporal-aware packing 是否启用。

### 3.4 视图选择与 evidence roles

规划器会根据 query type 和 intent 选择不同的索引视图。

典型视图角色如下：

| 视图 | 主要作用 |
| --- | --- |
| `scope` | 视频片段级文本或整体上下文 |
| `target` | 对象、实体、目标节点 |
| `track` | 目标跨帧轨迹与时序支持 |
| `event` | 事件级证据 |
| `adaptive_event` | 动态窗口事件证据 |
| `visual_object` | 检测对象、bbox、类别、置信度 |
| `visual_track` | 视觉轨迹、运动摘要、跨帧支持 |
| `visual_event` | 视觉事件、运动或状态变化摘要 |
| `visual_relation` | 对象间空间关系或近邻关系 |

常见 query 类型和视图规划关系如下：

| Query 类型 | required views | optional views | 规划意图 |
| --- | --- | --- | --- |
| object / existence | `visual_object`, `target` | `scope`, `visual_track` | 找到对象出现证据 |
| counting / multiple objects | `visual_object`, `visual_track` | `visual_event`, `scope` | 保留多个对象或多条 track |
| trajectory / motion | `visual_track`, `track` | `visual_event`, `event`, `scope` | 优先轨迹和运动摘要 |
| spatial_relation | `visual_relation`, `visual_object` | `visual_track`, `scope` | 优先对象关系和空间上下文 |
| event_localization | `adaptive_event`, `event` | `visual_event`, `scope` | 定位事件时段 |
| before_after | `adaptive_event`, `event` | `track`, `scope` | 保留前后时序证据 |
| interaction | `visual_relation`, `visual_event` | `track`, `scope` | 保留对象关系和动态事件 |
| general_description | `scope` | `event`, `target` | 汇总场景和主要活动 |
| speech | `scope` | `event` | 保留文本/语音上下文 |

如果 query 触发视觉链，规划器会优先把视觉视图加入 `selected_views`。视觉链通常包括：

```text
visual_object -> visual_track -> visual_event -> visual_relation
```

其中 `visual_relation` 会受配置控制。如果当前实验关闭 visual relations，规划器会将其从 selected views 和 optional views 中移除。

### 3.5 dependency order

EVIQUE 不把所有视图视为平级。它有一个固定依赖顺序，用于让低层视觉证据先于高层事件或文本上下文进入检索和打包流程。

常见顺序为：

```text
visual_object
visual_track
visual_event
visual_relation
scope
target
track
event
```

该顺序用于：

- 控制检索执行顺序。
- 控制 cost planner 的 view_order。
- 控制打包阶段的 planner priority。
- 避免通用 scope 文本过早压过结构化视觉证据。

## 4. 成本感知视图规划

### 4.1 启用条件

成本规划器默认启用，主要由以下配置控制：

- `EVIQUE_COST_PLANNER`
- `EVIQUE_COST_PLANNER_DEBUG`
- `EVIQUE_COST_PLANNER_MAX_VIEWS`
- `EVIQUE_COST_PLANNER_MIN_CONFIDENCE`
- `EVIQUE_COST_PLANNER_MAX_ROWS_TOTAL`

当成本规划器关闭时，系统会退回到 `QueryPlan.selected_views` 的固定顺序。

### 4.2 候选视图范围

成本规划器会在可规划视图中选择：

```text
scope
target
track
event
adaptive_event
visual_object
visual_track
visual_event
visual_relation
```

如果 visual relations 被关闭，则 `visual_relation` 不参与候选。

### 4.3 成本估计

每个视图的成本估计通常由以下因素组成：

- row count: 视图行数越多，扫描成本越高。
- avg token / avg text length: 文本越长，LLM 后续处理成本越高。
- expansion cost: 某些视图会引入额外邻居、关系或上下文扩展。
- noise risk: 低置信、高冗余或泛化上下文视图会有更高噪声风险。

成本估计的作用不是精确模拟数据库执行器，而是让 planner 避免盲目读取大量低价值证据。

### 4.4 收益估计

每个视图的收益估计通常来自：

- query type 与视图能力是否匹配。
- query intent 是否需要该视图。
- query terms 与视图统计信息是否匹配。
- 视图是否支持视觉、时序、对象、关系或事件证据。
- 是否有 video filter，且该视图能高效响应。
- 该视图是否是 base plan 中的 required view。

例如：

- motion query 会提高 `visual_track` 和 `track` 的收益。
- spatial query 会提高 `visual_relation` 和 `visual_object` 的收益。
- event localization query 会提高 `adaptive_event`、`event`、`visual_event` 的收益。
- object query 会提高 `visual_object` 和 `target` 的收益。

### 4.5 综合 planner score

成本规划器通过“收益除以成本”的方式得到综合分数：

```text
planner_score = expected_benefit / adjusted_cost
```

然后选择得分最高的若干视图，数量受 `max_views` 控制。

输出的 cost plan 通常包括：

- `view_order`: 实际查询顺序。
- `anchor_view`: 最核心的视图。
- `max_rows_per_view`: 每个视图行预算。
- `max_rows_total`: 总候选行预算。
- `stop_condition`: 停止条件。
- `estimated_costs`: 每个视图的成本、收益、分数。
- `reason`: 规划原因摘要。
- `debug_trace`: debug 模式下的详细评分轨迹。

### 4.6 query-specific refinement

在 `EvidenceRetriever` 中，cost plan 还会被进一步修正。例如：

- 对 temporal query，确保 event 或 adaptive_event 不会被完全遗漏。
- 对 pedestrian / crosswalk / yielding 等场景，确保 relation、track 或 event 证据能进入候选池。
- 对 fixed-window ablation，强制使用固定窗口 event 视图。
- 对 disabled views ablation，从 view_order 中移除禁用视图。

这一步保证 cost planner 的“性价比选择”不会破坏 query 本身必需的结构化证据。

## 5. 检索执行与候选证据组装

### 5.1 索引视图加载

`EvidenceRetriever` 初始化时会从 workdir 中加载多类索引文件。

文本和图结构视图包括：

- `scope_view.jsonl`
- `target_view.jsonl`
- `track_view.jsonl`
- `event_view.jsonl`
- `adaptive_event_view.jsonl`
- `keyframe_view.jsonl`
- `evidence_nodes.jsonl`
- `evidence_relations.jsonl`

视觉视图包括：

- `visual_object_view.jsonl`
- `visual_track_view.jsonl`
- `visual_event_view.jsonl`
- `visual_relation_view.jsonl`

辅助信息包括：

- `index_manifest.json`
- `graph_stats.json`
- `view_stats.json`

加载后，retriever 会构建多种内存索引，例如：

- segment 到 scope 的映射。
- node id 到 evidence node 的映射。
- node 到 relation 的映射。
- visual object id 到对象记录的映射。
- visual track id 到轨迹记录的映射。
- visual event id 到事件记录的映射。
- track 和 object 之间的关联映射。

这些内存结构用于后续快速补充上下文、关联邻居对象和构建时序序列。

### 5.2 video filter

检索阶段支持 query metadata 中的视频过滤条件。video filter 的作用是避免多视频索引场景下跨视频取证。

当前逻辑区分：

- explicit video filter: 明确由 query metadata 或运行器传入，严格过滤。
- inferred video filter: 从 metadata 推断，约束较弱。
- disabled video filter: 不使用视频过滤。

当 strict filter 启用时，EvidencePacker 也会继续检查候选证据的 video identity，丢弃跨视频不匹配证据。

### 5.3 view execution

根据 planner 输出，retriever 会执行对应视图检索。典型检索函数包括：

- scope 检索。
- target 检索。
- track 检索。
- event 检索。
- adaptive event 检索。
- visual object / visual track / visual event / visual relation 检索。
- instance spatial temporal 检索。
- visual intent evidence 检索。

检索结果统一表示为 evidence item，通常带有：

- `view` 或 `type`
- `text` / `summary` / `motion_summary`
- `score` 或 `weighted_score`
- `segment_id`
- `start_time` / `end_time`
- `timestamp`
- `video_id` / `source_vid`
- `object_id` / `track_id`
- `bbox`
- `provenance`

### 5.4 fallback 机制

如果某些 required view 没有返回足够结果，retriever 会尝试 fallback。fallback 的目标是避免 answer 阶段完全无证据，同时尽量不破坏 query 的证据优先级。

典型 fallback 包括：

- required visual view 为空时补充相关 visual evidence。
- temporal query 缺少 event 时补充 event 或 adaptive_event。
- object query 缺少 target 时补充 target label coverage。
- 所有视图为空时退回 scope。

fallback 证据仍会进入后续排序、融合和打包阶段，不会绕过 EvidencePacker。

### 5.5 排序与 required / optional 过滤

检索得到的候选证据会先合并，再经过 `_prioritize` 排序。排序会考虑：

- 原始 score。
- 视图优先级。
- query term 匹配。
- 结构化视觉证据是否匹配 query intent。
- 时序或空间证据是否能支撑 query。

之后系统会区分 required 与 optional 证据：

- required views 的证据优先保留。
- optional views 的证据需要满足 score threshold。
- optional evidence 如果与 required evidence 在 segment 上相邻，更容易被保留。
- cost plan 模式下会更多遵循 cost planner 的预算与 view_order。

### 5.6 融合与自然化

在打包前，retriever 会进行 evidence fusion 和 naturalization。该阶段的目标是把结构化证据变得更适合 LLM 使用。

常见处理包括：

- 把视觉对象、轨迹、事件和关系合并成更可读的描述。
- 为视觉证据补充附近对象上下文。
- 对 caption fallback、nearby object context、temporal sequence 等信息做统一组织。
- 去掉重复或明显低质量的视觉证据。
- 记录 fusion metrics，例如是否使用 caption fallback、附近对象上下文数量等。

这一步是“检索证据”到“可回答证据”的重要转换层。

### 5.7 时序上下文增强

对 before / after、trajectory、state change、event localization 等 query，retriever 会补充 temporal context。

典型增强包括：

- 根据 segment index 或 timestamps 找到前后邻近证据。
- 保留 focal event 附近的 before / after 证据。
- 为移动或交互 query 补充 track sequence。
- 为事件定位 query 补充 adaptive event 或 fixed window event。

这些补充证据会带着 provenance 和时间信息进入打包阶段。

## 6. EvidencePacker 打包逻辑

### 6.1 打包配置

EvidencePacker 受一组环境变量控制。默认配置大致为：

- token budget: 3200。
- char budget: 12000。
- core / support / context 比例: 0.55 / 0.30 / 0.15。
- 最少 core items: 3。
- 最少 packed items: 6。
- 最多 packed items: 12。
- 文本去重阈值: 0.75。
- spatial relation 最少保留数: 2。
- temporal event 最少保留数: 2。
- temporal-aware packing 默认开启。
- temporal bucket 包含 before / focal / after。

主要配置项包括：

- `EVIQUE_EVIDENCE_PACKER`
- `EVIQUE_EVIDENCE_PACKER_DEBUG`
- `EVIQUE_EVIDENCE_TOKEN_BUDGET`
- `EVIQUE_EVIDENCE_CHAR_BUDGET`
- `EVIQUE_EVIDENCE_CORE_RATIO`
- `EVIQUE_EVIDENCE_SUPPORT_RATIO`
- `EVIQUE_EVIDENCE_CONTEXT_RATIO`
- `EVIQUE_EVIDENCE_MIN_CORE_ITEMS`
- `EVIQUE_EVIDENCE_MIN_PACKED_ITEMS`
- `EVIQUE_EVIDENCE_MAX_ITEMS`
- `EVIQUE_EVIDENCE_DEDUP_THRESHOLD`
- `EVIQUE_EVIDENCE_SPATIAL_RELATION_MIN_ITEMS`
- `EVIQUE_EVIDENCE_TEMPORAL_EVENT_MIN_ITEMS`
- `EVIQUE_TEMPORAL_AWARE_PACKING`
- `EVIQUE_TEMPORAL_WINDOW_SEGMENTS`
- `EVIQUE_TEMPORAL_MIN_BEFORE`
- `EVIQUE_TEMPORAL_MIN_FOCAL`
- `EVIQUE_TEMPORAL_MIN_AFTER`
- `EVIQUE_TEMPORAL_MAX_SUPPLEMENT`

### 6.2 候选证据归一化

打包器首先把来自不同视图的原始 evidence item 归一化为统一内部结构。

归一化字段包括：

- `raw`: 原始证据。
- `view`: 证据来源视图。
- `text`: 可用于 LLM 的文本表示。
- `stable_id`: 稳定证据 id。
- `text_tokens`: 证据文本 token。
- `token_cost`: 估计 token 成本。
- `char_cost`: 字符成本。
- `video_values`: 证据中抽取到的视频身份。
- `cross_video_mismatch`: 是否与 strict video filter 冲突。
- `score_parts`: 重要性分数的组成部分。
- `importance_score`: 综合重要性分数。

如果 strict video filter 启用，且证据的 video identity 与 query filter 不匹配，该证据会被标记并丢弃。

### 6.3 importance score

每条候选证据都会被计算综合重要性分数。该分数由正向收益和负向惩罚构成。

正向因素包括：

- relevance score: 证据文本与 query tokens 的重合程度。
- planner priority score: 是否来自 cost plan 的 anchor view 或高优先级 view。
- confidence: 原始 score、weighted_score、confidence、relation_confidence、detection_confidence 等。
- query intent match: 证据视图是否匹配空间、时序、运动、对象、场景等意图。
- temporal alignment: 是否有时间戳、segment、event 或 track 信息。
- video alignment: 是否匹配 query video filter。
- coverage gain: 是否带来新的 object、track、relation、event、segment 或时间桶覆盖。

负向因素包括：

- redundancy penalty: 与已选或候选证据语义重复。
- token cost penalty: 文本过长。
- noise risk penalty: 低置信、泛上下文、缺少 provenance 或缺少时间信息。

这个分数是后续去重、mandatory 标记、分层和 MMR 选择的核心依据。

### 6.4 去重逻辑

EvidencePacker 会先按 `importance_score` 降序排列候选，再执行去重。

去重依据包括：

- stable key 重复，例如同一 view 下同一 object_id、track_id、event_id、segment_id。
- visual relation 的目标对象、相关对象、关系类型和时间桶重复。
- 文本 Jaccard 相似度超过阈值。
- 相似度比较只在合理 scope 内进行，避免误删不同视频或不同视图的关键证据。

被去重的证据不会直接消失，而是进入 `dropped_evidence`，并记录 drop reason，例如：

- `duplicate_key`
- `duplicate_text`
- `cross_video_mismatch`

### 6.5 mandatory evidence floor

为避免关键证据被预算裁剪掉，打包器会为某些 query 和视图设置 mandatory floor。

常见 mandatory 规则包括：

- anchor view 的 top evidence 会被标记为 core。
- spatial query 至少保留一定数量的 `visual_relation` 或 nearby object context。
- temporal query 至少保留一定数量的 `event`、`adaptive_event` 或 `visual_event`。
- motion query 至少保留一定数量的 `visual_track` 或 `track`。
- 如果 core evidence 太少，会用高 importance 的候选补齐 `min_core_items`。

mandatory evidence 在预算选择阶段可以 force add。这样即使 token budget 紧张，核心视觉或时序证据也不会轻易被普通上下文挤掉。

### 6.6 evidence layer 分类

每条候选证据会被划分为以下层级：

| Layer | 含义 |
| --- | --- |
| `core` | 直接回答 query 的核心证据 |
| `supporting` | 支撑 core 的补充证据 |
| `context` | 背景上下文或场景描述 |
| `low_value` | 价值较低，仅在预算充足时使用 |

分层规则包括：

- mandatory evidence 一定是 `core`。
- anchor view 的高分证据倾向于 `core`。
- spatial query 中的 relation evidence 倾向于 `core`。
- motion query 中的 track evidence 倾向于 `core`。
- temporal query 中的 event evidence 倾向于 `core`。
- 大段 scope 或 caption context 更倾向于 `context`。
- 泛化、低置信或缺少结构字段的证据更可能成为 `low_value`。

### 6.7 时序感知打包

对时序类 query，EvidencePacker 会启用 temporal-aware packing。

触发条件包括：

- query type 属于 `before_after`、`event_localization`、`interaction`、`instance_spatial_temporal`、`state_change`、`temporal`、`trajectory`。
- query intent 中包含 temporal ordering、temporal interaction 或 transition。
- query tokens 中出现明显时序词或运动词。

启用后，packer 会：

1. 选择一个 temporal anchor item。
2. 根据 segment index、start/end time 或 timestamp 把证据分到 before、focal、after。
3. 为 before、focal、after 设置最小保留数量。
4. 优先选择能覆盖时序链条的证据，而不是只选最高分的同一时刻证据。

这对于回答“之前发生了什么”“之后发生了什么”“对象如何移动”“事件何时开始/结束”非常重要。

### 6.8 预算内选择策略

如果所有去重后的候选证据都能放入 token 和字符预算，packer 会保留有效证据，并按层级和重要性排序。

如果候选证据超出预算，则执行预算选择。选择顺序大致为：

1. 先加入 temporal floor evidence。
2. 再加入 mandatory core evidence。
3. 再按 supporting、context、low_value 分层选择。
4. 每一层内部使用 marginal value 选择证据。

marginal value 由三部分组成：

```text
marginal_value =
  importance_score
  + dynamic_coverage_gain
  - redundancy_penalty
  - token_cost_penalty
```

其中：

- `dynamic_coverage_gain`: 该证据是否带来新的 object、track、relation、event、segment 或 time bucket。
- `redundancy_penalty`: 与已选证据的文本重复度和同视图重复度。
- `token_cost_penalty`: 当前证据消耗的 token 成本。

该策略接近一个 utility-aware MMR 打包过程：既保留高分证据，也鼓励覆盖更多对象、时间段和关系，避免证据包被同质内容占满。

### 6.9 文本压缩

进入最终 package 前，packer 会根据 evidence layer 调整文本长度：

- `core`: 尽量保留完整内容。
- `supporting`: 可压缩到较短文本。
- `context`: 更强压缩。
- `low_value`: 只保留短摘要。

非 core 证据被压缩时，会添加类似视图、时间、视频身份的前缀，保证压缩后仍保留基本 provenance。

### 6.10 dropped evidence

未进入最终 evidence 的候选不会被完全忽略。packer 会在 `dropped_evidence` 中记录：

- stable id。
- 原始 id。
- view。
- drop reason。
- packing layer。
- importance score。
- estimated tokens。
- text preview。

这使得调试时可以回答：

- 哪些证据因为重复被丢弃。
- 哪些证据因为超预算被丢弃。
- 哪些证据因为低 marginal value 被丢弃。
- 哪些证据因为跨视频不匹配被丢弃。

## 7. 最终 evidence package 结构

`EvidenceRetriever.retrieve()` 返回的最终 package 通常包含以下信息。

### 7.1 query and planning

- `query`: 原始 query。
- `query_type`: query 类型。
- `query_terms`: query 关键词。
- `query_intents`: 细粒度 intent。
- `plan`: QueryPlan 的结构化表示。
- `guidance`: 对 answer generation 的提示。
- `answer_constraints`: 回答约束，例如必须基于证据、证据不足时说明不足。

### 7.2 cost planning

- `cost_plan`: 成本规划结果。
- `views_queried`: 实际查询过的视图。
- `views_skipped`: 因禁用、成本、filter 或无数据跳过的视图。
- `native_window_mode` 或 ablation 相关字段。

### 7.3 evidence

- `evidence`: 最终打包后交给 LLM 的证据列表。
- `dropped_evidence`: 未进入最终包的候选证据及原因。
- `candidate_evidence_count`: 打包前候选数量。
- `packed_evidence_count`: 打包后证据数量。
- `retrieved_evidence_count`: 检索得到的证据数量。
- `used_evidence_count`: 实际用于回答上下文的证据数量。

### 7.4 packing metadata

`pack_metadata` 通常包含：

- packer 是否启用。
- token budget 和 char budget。
- estimated packed tokens。
- estimated packed chars。
- evidence layer counts。
- view counts。
- mandatory item ids。
- budget 是否耗尽。
- video filter source。
- strict video filter 是否启用。
- cross-video dropped count。
- temporal-aware packing 是否启用。
- temporal anchor。
- before / focal / after bucket 数量。
- supplement counts。
- debug trace。

### 7.5 fusion and temporal context

- `fusion_metrics`: 视觉证据融合和自然化统计。
- `temporal_context`: 时序上下文增强结果。
- `graph_stats`: 索引图统计信息。

这些字段共同让 evidence package 不只是一个 flat top-k 列表，而是一个可解释、可诊断、可控预算的回答上下文。

## 8. 重要设计特征

### 8.1 规划优先于纯相似度检索

EVIQUE 不仅根据文本相似度取 top-k。它先判断 query 类型，再决定应该优先查哪类证据。这样可以避免：

- motion query 被普通 caption 抢占。
- spatial query 缺少 relation 证据。
- event localization query 缺少时间窗口证据。
- object query 被泛场景 scope 覆盖。

### 8.2 结构化视觉证据优先

在对象、轨迹、空间关系和运动 query 中，系统会优先保留：

- visual object。
- visual track。
- visual event。
- visual relation。

这些结构化证据带有 bbox、track、timestamp、relation 或 event 信息，比纯文本 summary 更适合支撑可验证回答。

### 8.3 检索和打包分离

retriever 负责“找候选”，packer 负责“组织候选”。两者分离带来几个好处：

- 可以让 retriever 尽量召回多源证据。
- 可以让 packer 在统一预算下做最终裁剪。
- 可以记录 dropped evidence，方便诊断。
- 可以在不重建索引的情况下调整打包预算。

### 8.4 required evidence 不被轻易裁掉

mandatory floor 是 EVIQUE 打包逻辑的重要保护机制。它确保某些 query 必需的证据不会因为：

- 文本短但无关的 scope 得分较高。
- 某一类 evidence 数量太多。
- token budget 紧张。

而被挤出最终证据包。

### 8.5 时序 query 保留时间结构

对时序 query，EVIQUE 不只选择最高分片段，还会尽量保持 before、focal、after 的证据结构。这使答案能够描述事件发展，而不仅是指出某个孤立时刻。

### 8.6 严格记录 provenance

最终 evidence package 中每条证据都尽量保留：

- 来源 view。
- 时间戳或时间窗口。
- video identity。
- object / track / event id。
- 原始 score。
- packing rank。
- packing reason。

这使答案生成和后续审计可以追踪证据来源。

## 9. 与普通 top-k retrieval 的区别

普通 retrieval pipeline 往往是：

```text
query -> embedding / keyword search -> top-k chunks -> LLM
```

EVIQUE 的 pipeline 更接近：

```text
query
  -> query intent and type planning
  -> cost-aware structured view selection
  -> multi-view evidence retrieval
  -> visual / temporal / relation supplementation
  -> utility-aware evidence packing
  -> provenance-rich evidence package
  -> LLM
```

核心区别在于：

- 它查询的是多种结构化视图，而不是单一文本 chunk。
- 它保留对象、轨迹、事件、关系和时间窗口。
- 它区分 core、supporting、context。
- 它会为时序和空间 query 保留必要证据下限。
- 它会记录为什么选择或丢弃证据。
- 它可以在相同 evidence token budget 下减少冗余，提高证据密度。

## 10. 运行时开关与实验控制

EVIQUE 的规划和打包层提供了一些实验开关，便于 ablation 和诊断。

### 10.1 planner 相关

- `EVIQUE_COST_PLANNER`: 是否启用成本规划器。
- `EVIQUE_COST_PLANNER_DEBUG`: 是否输出成本规划 debug 信息。
- `EVIQUE_COST_PLANNER_MAX_VIEWS`: 最大查询视图数。
- `EVIQUE_COST_PLANNER_MAX_ROWS_TOTAL`: 总候选行预算。

### 10.2 packer 相关

- `EVIQUE_EVIDENCE_PACKER`: 是否启用 EvidencePacker。
- `EVIQUE_EVIDENCE_PACKER_DEBUG`: 是否输出打包 debug trace。
- `EVIQUE_EVIDENCE_TOKEN_BUDGET`: 证据 token 预算。
- `EVIQUE_EVIDENCE_CHAR_BUDGET`: 证据字符预算。
- `EVIQUE_EVIDENCE_MAX_ITEMS`: 最多打包证据条数。
- `EVIQUE_EVIDENCE_DEDUP_THRESHOLD`: 文本去重阈值。

### 10.3 temporal packing 相关

- `EVIQUE_TEMPORAL_AWARE_PACKING`: 是否启用时序感知打包。
- `EVIQUE_TEMPORAL_WINDOW_SEGMENTS`: 时序邻域窗口。
- `EVIQUE_TEMPORAL_MIN_BEFORE`: before bucket 最小证据数。
- `EVIQUE_TEMPORAL_MIN_FOCAL`: focal bucket 最小证据数。
- `EVIQUE_TEMPORAL_MIN_AFTER`: after bucket 最小证据数。
- `EVIQUE_TEMPORAL_MAX_SUPPLEMENT`: 最大时序补充证据数。

### 10.4 ablation 相关

- `EVIQUE_ABLATION_DISABLE_PLANNER`: 关闭 planner。
- `EVIQUE_ABLATION_DISABLE_PACKAGING`: 关闭 packer，改用 raw top-k truncate。
- `EVIQUE_ABLATION_DISABLED_VIEWS`: 禁用指定视图。
- `EVIQUE_ABLATION_FIXED_VIEW_ORDER`: 使用固定视图顺序。
- `EVIQUE_ABLATION_EVENT_MODE`: 控制 event 检索模式。
- `EVIQUE_ABLATION_FIXED_WINDOW_SIZE`: 固定事件窗口大小。
- `EVIQUE_ABLATION_FIXED_WINDOW_STRIDE`: 固定事件窗口步长。
- `EVIQUE_ABLATION_DISABLE_STRICT_VIDEO_FILTER`: ablation 中关闭 strict video filter。

这些开关主要用于实验分析，不改变 EVIQUE 的核心索引结构。

## 11. 典型 query 的端到端行为

### 11.1 对象可见性 query

示例：

```text
Find moments where a boat is visible.
```

典型流程：

1. Planner 将 query 识别为 object / existence 类。
2. 视觉链触发，优先选择 `visual_object`，并补充 `target`、`visual_track` 或 `scope`。
3. Cost planner 评估对象视图的收益较高，通常把 `visual_object` 作为 anchor view。
4. Retriever 从 object view 找到类别匹配的对象证据。
5. 如果有 track 支持，会补充 `visual_track`。
6. Packer 将 label-matched object evidence 作为 core，scope 作为 context。
7. 输出包含对象类别、时间、可能的 bbox/track 和文本描述的 evidence package。

### 11.2 空间关系 query

示例：

```text
Find moments where a bicycle is near a car.
```

典型流程：

1. Planner 检测 `spatial_relation` intent。
2. 优先选择 `visual_relation` 和 `visual_object`。
3. Retriever 查找对象关系证据，并补充相关对象上下文。
4. Packer 设置 spatial relation mandatory floor。
5. `visual_relation` 或 nearby object context 被优先保留为 core。
6. 普通 scope 文本只能作为 supporting 或 context。

### 11.3 运动轨迹 query

示例：

```text
Find moments where a truck moves across the frame.
```

典型流程：

1. Planner 检测 `temporal_trajectory` intent。
2. 优先选择 `visual_track` 和 `track`。
3. Cost planner 提高轨迹视图收益。
4. Retriever 查询 track motion summary、compact points 或时间序列。
5. Packer 设置 motion track floor。
6. 如果启用 temporal-aware packing，会保留运动前后或 focal 段证据。
7. 最终 evidence package 更偏向轨迹和运动摘要，而不是静态对象出现。

### 11.4 事件定位 query

示例：

```text
When does the vehicle stop near the intersection?
```

典型流程：

1. Planner 识别 event localization、state change 或 temporal intent。
2. 优先选择 `adaptive_event`、`event`、`visual_event`。
3. Retriever 查询事件窗口，并补充相关 track 或 object evidence。
4. Packer 启用 temporal-aware packing。
5. before / focal / after 证据被尽量保留。
6. 输出包含事件发生窗口和支撑对象/轨迹证据的 package。

## 12. 常见失败模式与诊断入口

### 12.1 candidate evidence 为空

可能原因：

- strict video filter 与 workdir 中 video identity 不匹配。
- query type 触发的视图在当前索引中不存在。
- label 或对象类别与索引标签不一致。
- ablation 禁用了关键视图。

诊断字段：

- `candidate_evidence_count`
- `views_queried`
- `views_skipped`
- `query_video_filter`
- `strict_video_filter_enabled`
- `cost_plan`

### 12.2 evidence 有候选但最终很少

可能原因：

- token budget 太小。
- 去重阈值过低。
- 大量候选被判定为 low marginal value。
- strict video filter 在 packer 阶段丢弃跨视频证据。

诊断字段：

- `dropped_evidence`
- `pack_metadata.dropped_count`
- `pack_metadata.budget_exhausted`
- `pack_metadata.layer_counts`
- `pack_metadata.cross_video_dropped_count`

### 12.3 回答缺少关系或时序细节

可能原因：

- query 没有触发 spatial / temporal intent。
- visual relation view 被关闭。
- temporal-aware packing 被关闭。
- cost planner 没有把 event/track 视图排入 top views。

诊断字段：

- `query_intents`
- `plan.selected_views`
- `cost_plan.view_order`
- `pack_metadata.temporal_aware_packing`
- `pack_metadata.mandatory_item_ids`

## 13. 小结

EVIQUE 的证据规划和打包功能可以理解为一个面向视频问答的结构化证据编排层。它不是单纯的文本检索器，而是把 query 理解、视图规划、成本预算、视觉证据、图关系、时序上下文和 LLM 输入压缩统一到一个 pipeline 中。

其关键架构思想是：

1. 先理解 query，再决定查什么视图。
2. 用成本规划避免低价值视图占用预算。
3. 用多视图检索召回对象、轨迹、事件、关系和文本上下文。
4. 用 fusion 把结构化证据转成 LLM 可读证据。
5. 用 EvidencePacker 在预算内保留核心、支撑和上下文证据。
6. 用 mandatory floor 保护空间、时序和运动 query 的关键证据。
7. 用 metadata 记录每一步选择、过滤、丢弃和压缩原因。

因此，最终 evidence package 是一个可解释的、结构化的、预算受控的证据集合，服务于后续的 DB-RAG 或视频问答答案生成。
