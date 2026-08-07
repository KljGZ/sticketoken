# Sticky / Attractor V2 实现与验收规范

V1 已固定在 GitHub 标签 `v1.0.0`，标签指向提交
`29424d8e4d71d53dbec3a84f6f6144dcf9f51987`。V2 使用独立入口、配置和结果目录，
不会改变 V1 的代码路径或注册结果。

## 1. 研究对象

V2 严格区分三种对象：

| 模式 | 正式名称 | 搜索对象 | 主实验中的插入语义 |
|---|---|---|---|
| 1 | Single-token Mean Sticky | 单个 token `t` | 重复 `t`，形成剂量 `t^n` |
| 2 | Minimal Monotone Similarity Booster | 多 token 组合 `x` | 整个 `x` 只插入一次 |
| 3 | Minimal Cluster-Escape Compact Attractor | 多 token 组合 `x` | 整个 `x` 只插入一次 |

模式 2、3 的长度 1 是全合法词表穷举基线；组合搜索按用户指定的
`2,4,...,30` 运行。长度大于 1 的组合空间不是穷举，因此最短性结论固定写为
“预注册预算下发现的最短组合”，不得写成全局最短。

V1 的 whole-string repetition 图只属于压力测试，不参与 V2 主结论。V2 模式 2、3
的横轴统一为 `Number of tokens in optimized combination`。

## 2. 数据、长度和测试隔离

句子身份依次经过 Unicode NFC、首尾空白删除、连续空白压缩，保留大小写与标点，
最后计算 SHA-256。句子是图节点，句对是无向边；连通分量整体进入
search/validation/test。分配目标为 0.6/0.2/0.2，并平衡句对数量、search 原始
相似度分箱、文本长度分箱和数据来源。输出 `split_audit.json`，三个句子集合的交集
必须全为 0。

为防止较长触发串在 suffix 位置被静默截断，三种模式在编码前使用固定源文本预算。
预算从模型最大长度中一次性减去 40 个触发 token 和特殊 token；所有长度共享同一原文，
因此长度前沿不会因不同源文本造成伪差异。

每个候选同时记录：

- `component_length`；
- `standalone_realized_length`；
- `context_realized_length_min`；
- `context_realized_length_max`；
- `sentinel_location_rate`；
- `realizability_rate`。

上下文实现性使用唯一左右哨兵和 tokenizer offset span 定位；不能稳定定位或不能包含
预期组件 token ID 的候选不能认证。模式 2、3 默认排除特殊 token；特殊 token 辅助实验
只能作为单独消融。

## 3. 模式 1

`single_sticky_paper_replication` 使用 `n=8`、`k=5`、Top 2%、当前候选分布的
IQR 自适应 epsilon 和同一 `P_f` 做论文式验证，输出：

- `paper_validated`；
- `paper_epsilon`；
- `paper_GE_max`；
- `paper_GE_q95`；
- `paper_pass_rate`。

`single_sticky_holdout_certification` 重新在 search split 做全词表打分；论文协议的
候选排序不能流入 holdout。validation 冻结唯一 token，test 只评估该 token。认证分为：

- 覆盖率至少 0.95；
- GE 的 95% 分位不超过 epsilon；
- GE 最大值不超过 epsilon。

剂量曲线在 validation 上运行 `n=0..30`，输出覆盖率、GE q95、GE max 和
`minimum_effective_repeat_count`；test 仍只评估冻结的 `n=8` 候选一次。只有模式 1
生成重复次数曲线。

## 4. 模式 2

低、高阈值只由 search 原始相似度的 30% 和 70% 分位决定，并在 validation/test 冻结。
核心认证为：

1. 低端 `Δ>=0.02` 的覆盖率至少 0.80；
2. 高端 `Δ` 的 q05 至少为 -0.02；
3. 高端最终状态保持率至少 0.95；
4. 全局 `Δ<-0.02` 的比例不超过 0.05。

动态范围比至少 0.70 且 Spearman 至少 0.80 是第二级结构认证，不能用加权目标掩盖
核心约束失败。逐前缀指标只用于解释，不参与认证。冻结候选后在 validation 上输出：

- 每个 token 单独使用；
- 每个前缀；
- 留一消融；
- 随机排列；
- 相对最佳单 token 的 synergy；
- 每个位置的 leave-one-out contribution。

## 5. 模式 3

模式 3 使用每个 split 的去重句子集合。球面 K-Means 仅在 search 拟合，候选
`K=[8,16,32,64]`，依据 cosine silhouette、最小簇大小和多初始化 ARI 稳定性选择。
validation/test 只分配至冻结中心。

核心认证为：

1. 到自身源簇 95% 半径之外的绝对间隔 q05 至少 0.02；
2. 相对原表示进一步向外移动的距离 q05 至少 0.02；
3. 源簇逃逸覆盖率至少 0.95；
4. 到触发后归一化公共中心的半径 q95 不超过 0.40；
5. 文本实现率至少 0.95。

旧版 `displacement_q05 - compact_radius_q95`、低/高句对目标和逐前缀约束不再参与
模式 3 认证。全部良性簇边界间隔和 benign kNN density ratio 是后验强诊断；只有
通过相应诊断时，才可使用“位于已知良性簇之外”或“低密度空域”的表述。

模式 3 的全词表单 token 筛选独立于模式 2，候选池由逃逸、紧致、Pareto、普通
sticky 和随机探索 token 的并集构成。

每个长度都比较随机合法串、高频普通串、自然短语、最佳单 sticky 重复、Top-L
escape 拼接、Top-L compact 拼接、V1 串长度匹配和 V2 优化串。降维图把所有阶段
联合拟合一次；高维指标是结论依据，图仅用于解释。

## 6. 搜索器和统计协议

长度 1 穷举全部合法词表。长度大于 1 使用多重重启 CEM，并包含：

- 固定均匀探索混合；
- 位置熵下限和自适应混合；
- 停滞后随机重置一半位置；
- 精英最小 Hamming 距离；
- 每轮动态 search 子集；
- 每 5 轮完整 search 复评；
- 4 个独立重启；
- 从前一个已运行长度生长 50% warm-start 个体，其余随机初始化。

搜索排序首先比较可行性，再比较归一化违反量，最后才比较质量余量。第一版长度前沿
为 15 个组合长度、每种模式 4 个重启。全词表筛选也采用确定性分片：模式 2 在
GPU 0/1/3 上分成 3 片，模式 3 在 GPU 4/5/6/7 上分成 4 片；合并时验证 token ID
全集完全一致且无重复。8 张 GPU 的主搜索分配为：GPU 0–3 运行模式 2 的四个重启链，
GPU 4–7 运行模式 3 的四个重启链。

每个长度从四个 search archive 中取候选，在完整 validation 上评估。validation 决定
该长度唯一候选和最短可行长度。随后写入 `frozen_candidate.json`，test 只对这一条
冻结候选调用一次。test 不得改变候选、长度、阈值、插入模式或算法。

## 7. 要求到产物的追踪表

| 要求 | 实现 | 主要测试/产物 |
|---|---|---|
| 句子级隔离 | `sticky_lab/v2_data.py` | `split_audit.json`, `prepared_pairs.csv` |
| 单次组合语义 | `sticky_lab/v2.py` | config `repeat_count: 1` 强校验 |
| 四种长度 | `sticky_lab/tokens.py` | validation CSV 四个长度字段 |
| test 不选模 | `finalize()` / `_test_once()` | `frozen_candidate.json`, 单行 `test_result.csv` |
| 模式 1 双协议 | `run_mode1()` | paper/holdout 两套 token score 和 validation 文件 |
| 模式 1 三层认证 | `dose_certification()` | `dose_curve.csv`, `test_result.json` |
| 模式 2 核心/结构分层 | `mode2_metrics()` | `length_frontier.csv`, `test_result.csv` |
| 模式 2 组合消融 | `_mode2_ablations()` | 四类消融 CSV 和 synergy JSON |
| 模式 3 球面聚类 | `_fit_spherical_clusters()` | centers/radii/assignments/selection |
| 模式 3 独立筛选 | `prepare()` / `screen_shard()` mode 3 branch | 独立 `single_token_screen.csv` |
| 模式 3 逃逸/紧致 | `mode3_metrics()` | validation、frontier、test 指标 |
| 模式 3 空域诊断 | `_mode3_evaluate()` | clearance、density ratio |
| 等长度对照 | `_mode3_baselines_and_projection()` | `equal_length_baselines.csv` |
| CEM 防早熟 | `sticky_lab/v2_search.py` | 每重启/长度 history CSV |
| 长度/seed 前沿 | `finalize()` | `length_frontier.csv/.png` |
| 联合投影 | `sticky_lab/v2_visualization.py` | 两张 progression 图和 metadata |
| 8 GPU 调度 | `scripts/run_v2_remote.sh` | 独立日志与 phase summaries |

## 8. 命令

完整远程实验：

```bash
bash scripts/run_v2_remote.sh
```

单阶段调试：

```bash
python -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase prepare
python -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase prepare-common
python -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase screen-shard --shard-index 0 --shard-count 3
python -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase merge-prepare --shard-count 3
python -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase search --restart 0
python -m sticky_lab.v2 --config configs/v2_multi_booster.yaml --phase finalize
```

所有 scientific run 必须保存 resolved config、Git commit、模型 revision、依赖版本、CUDA
设备、运行时和各 phase summary。`--smoke` 只用于管线检查，不能进入论文结果。
