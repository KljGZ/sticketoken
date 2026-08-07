# Sticky / Attractor V2 全量实验结果与改进路线

## 1. 结论摘要

本次实验在 `sentence-transformers/sentence-t5-base` 上完成了三种对象的统一、隔离评估。正式代码固定在提交 `54c2e5b20a82bb098627b3299d30dba1b4a3e494`，远程产物记录在提交 `cfb39e90d33bfe1a744754e5e9efdb73a3bbcaff`。

| 模式 | 冻结对象 | validation | 一次性 test | 严格结论 |
|---|---|---:|---:|---|
| 1. Single-token Mean Sticky | ` <extra_id_27>`，ID 32072，重复 8 次 | 通过覆盖率与 q95；最大误差未通过 | 三层均通过 | 复现出均值黏附 token；最小有效重复次数为 6 |
| 2. Minimal Monotone Similarity Booster | `lucrarea`，ID 30332，组件长度 1 | 核心与结构约束均通过 | 核心与结构约束均通过 | 当前模型上单 token 已足够；不需要多 token 才能满足目标 |
| 3. Minimal Cluster-Escape Compact Attractor | 注册回退串，组件长度 18 | 未通过 | 未通过 | 在长度 1,2,4,...,30 的注册预算内没有找到可认证 attractor |

模式 3 的失败不是“没有紧致效应”。冻结串在 test 上使触发后表示的平均两两余弦相似度达到 0.88343，但没有稳定越过所有源良性簇边界。因此当前主要瓶颈是最坏插入位置、最难源簇上的绝对逃逸，而不是紧致性本身。

## 2. 可复现实验边界

- 模型：`sentence-transformers/sentence-t5-base`。
- 模型 revision：`fc5d4628481afbbaaacd7af6bb07cf9d3865f781`。
- Python 3.11.15，PyTorch 2.6.0+cu124，Transformers 5.14.1，Sentence-Transformers 5.6.1。
- 计算资源：8 张 NVIDIA GeForce RTX 4090；模式 2 使用 GPU 0–3，模式 3 使用 GPU 4–7，模式 1 使用 GPU 2。
- 模式 2/3 全词表合法候选数：31,994；各自独立筛选并构造上限 1,200 的候选池。
- 模式 2/3 长度 1 做全合法词表穷举；组合长度为 `2,4,...,30`，每个长度 4 个独立 CEM 重启，每次 20 轮、种群 128。
- validation 对每个长度从四个 search archive 合并后选择唯一候选；test 不参与阈值、候选或长度选择，只对冻结对象评估一次。
- 模式 2/3 的字符串整体只插入一次。只有模式 1 重复同一个 token。

因此，模式 2/3 的长度结论应表述为“注册搜索预算内发现的最短组合”，不能宣称对长度大于 1 的组合空间进行了全局穷举。模式 2 的最终长度为 1，而长度 1 已全词表穷举，所以它在本协议的允许长度中确实是最短的。

## 3. 数据隔离与审计

句子先经过 Unicode NFC、首尾空白删除和连续空白压缩，再用规范化文本的 SHA-256 标识身份。句子是图节点，句对是无向边。由于原数据存在一个包含 888 个句对的超大连通分量，无法在保留全部边的同时达到 0.6/0.2/0.2 且保持句子级互斥，协议采用固定的 `unique_sentence_groups_drop_cross_split_pairs` 回退：先分配唯一句子，再删除跨 split 句对。

| 项目 | 数值 |
|---|---:|
| 原图连通分量数 | 51 |
| 最大分量句对数 | 888 |
| 删除的跨 split 句对 | 634 |
| 删除率 | 64.959% |
| search / validation / test 句对 | 206 / 69 / 67 |
| search / validation / test 唯一句子 | 350 / 205 / 203 |
| 三组两两句子重叠 | 全部为 0 |
| 源文本截断率 | 0.410% |

句子泄漏被严格消除，但 64.959% 的删边率意味着有效样本量明显下降。这是本次结果最重要的数据限制，后续不能只扩大搜索预算而忽略数据构造。

## 4. 模式 1：论文单 token Mean Sticky

### 4.1 论文协议复现

- 模型词元表示平均两两相似度 `u = 0.7959119365`。
- 重复次数 `n = 8`、稀疏句对数 `k = 5`、Top 2% 候选。
- 当前候选分布 IQR 得到论文式自适应 `epsilon = 0.3150300533`。
- 论文式验证通过 642 个 token。

该数字复现的是原论文筛选/验证定义，不是独立 holdout 认证。V2 另行在 search 上打分、validation 冻结唯一 token、test 只评估一次。

### 4.2 Holdout 认证

冻结 token 为 ` <extra_id_27>`（ID 32072），固定认证阈值 `epsilon = 0.1106`。

| 指标 | validation | test |
|---|---:|---:|
| 覆盖率 | 0.991667 | 1.000000 |
| GE q95 | 0.088459 | 0.090563 |
| GE max | 0.114168 | 0.095434 |
| 覆盖率认证 | 通过 | 通过 |
| q95 认证 | 通过 | 通过 |
| 严格最大值认证 | 未通过 | 通过 |

validation 剂量曲线覆盖 `n=0..30`，最小达到覆盖率认证的重复次数为 6。图文件为 `results/sticky_lab/sentence_t5_base/single_sticky_v2/dose_similarity_curves.png`。它对应论文图中的 “Inserted number of sticky token” 语义。

需要注意，冻结对象属于 T5 sentinel special token。它证明模型/tokenizer 中存在强均值黏附效应，但若研究目标要求普通可见字符串，应在下一版增加“排除全部 special/control token”的主实验，而把本结果保留为论文协议对照。

## 5. 模式 2：单调相似度增强串

低/高相似度阈值只由 search 原始相似度的 30%/70% 分位得到，并冻结为：

- low threshold：0.6292823553；
- high threshold：0.8198487759。

全词表穷举的长度 1 候选 `lucrarea`（ID 30332）在 validation 排名第一，因此按最短长度优先规则冻结。它在独立 test 上得到：

| 约束或诊断 | test 数值 | 要求 | 结果 |
|---|---:|---:|---:|
| 低端增益 q10 | 0.044586 | 至少 0.02 | 通过 |
| 低端增益覆盖率 | 1.000000 | 至少 0.80 | 通过 |
| 高端增益 q05 | -0.019605 | 至少 -0.02 | 通过 |
| 高端状态保持率 | 1.000000 | 至少 0.95 | 通过 |
| 全局显著下降率 | 0.024876 | 至多 0.05 | 通过 |
| 动态范围比 | 0.826754 | 至少 0.70 | 通过 |
| Spearman | 0.989903 | 至少 0.80 | 通过 |
| 文本实现率 | 1.000000 | 至少 0.95 | 通过 |

这说明它不是简单把所有句对压到一个常数附近：动态范围和原始排序大部分得到保留，同时低相似度尾部被稳定提升，高相似度尾部仅允许非常小的下降。

长度 2–30 的优化前沿也产生了大量可行组合，但它们不能推翻长度 1 的最短性。冻结对象只有一个组件，所以 leave-one-out 为空，组合相对最佳单 token 的 synergy 为约 `-1.12e-7`，不支持“多 token 协同是成功原因”的说法。

模式 2/3 的横轴是 `Number of tokens in optimized combination`，而不是重复插入次数。对应前沿图为 `results/sticky_lab/sentence_t5_base/multi_booster_v2/length_frontier.png`。

## 6. 模式 3：原句排斥的紧致 attractor

### 6.1 良性几何

球面 K-Means 只在 search 唯一句子上拟合。`K=[8,16,32,64]` 中，K=8 在最小簇大小约束下获得最高注册选择分数：

- cosine silhouette：0.101445；
- 最小簇大小：12；
- 多重初始化稳定性 ARI：0.582071。

validation/test 只分配到冻结的 8 个中心，不重新拟合聚类。

### 6.2 注册搜索结果

长度 1 全词表穷举以及长度 `2,4,...,30` 的 4×CEM 搜索均未产生通过全部核心约束的 validation 候选。按注册规则冻结最小约束违背回退串：

```text
 Sultan never prochain strategically incentives consistent incentives Communitiesachi never $50,000supervised subsidies. farms NEVER remix Cape
```

组件长度为 18，token IDs 为：

```text
30356,470,14597,25178,16844,4700,16844,28086,11015,470,29788,23313,26994,5,16537,28218,26063,9702
```

| 核心指标 | validation | test | 要求 |
|---|---:|---:|---:|
| 绝对逃逸 q05 | 0.010238 | -0.022719 | 至少 0.02 |
| 相对外移 q05 | 0.055961 | 0.054644 | 至少 0.02 |
| 源簇逃逸率 | 0.961165 | 0.902174 | 至少 0.95 |
| 紧致半径 q95 | 0.450149 | 0.422308 | 至多 0.40 |
| 触发后两两相似度均值 | 0.877677 | 0.883432 | 诊断 |
| 触发后两两相似度 q05 | 0.821497 | 0.831344 | 诊断 |
| 文本实现率 | 1.000000 | 1.000000 | 至少 0.95 |

validation 上相对外移和逃逸覆盖率通过，但绝对逃逸和紧致半径失败；test 上绝对逃逸、覆盖率和紧致半径都失败。`all_cluster_clearance_q05 < 0` 且 kNN density ratio 分别约 0.982/0.975，所以不能声称该串进入了已知良性簇之外的低密度空白区域。

等长 18 的对照显示，V2 优化串显著优于随机合法串、自然短语、频繁普通串、最佳 sticky 重复串以及 V1 串：其 validation 绝对逃逸约 0.00944、相对外移约 0.06048、逃逸率约 0.97087。与此同时，它的紧致半径约 0.43863，仍未达到 0.40。也就是说，V2 的方向有效，但当前约束和搜索预算下没有越过认证边界。

降维使用所有阶段联合拟合。由于远程环境没有可用 UMAP 实现，实际记录的是确定性 joint-fit t-SNE fallback；高维指标才是结论依据，投影只用于解释。产物为：

- `embedding_length_progression.png`；
- `embedding_search_iteration_progression.png`；
- `projection_metadata.json`。

## 7. 下一步改进路线

### P0：针对模式 3 的真实瓶颈改造目标

当前串已经能让样本相互聚集并整体外移，但失败由最坏插入模式和最难源簇决定。下一版应把三种插入方式 × 八个源簇的约束保留为显式向量，不再只用一个聚合标量驱动 CEM：

1. 先按字典序最小化 `absolute_escape` 违背，再优化逃逸覆盖率和紧致半径；
2. 使用 worst-group/CVaR 目标，动态提高当前最差插入模式和最差源簇的采样比例；
3. 使用增广拉格朗日或 epsilon-constraint 更新约束乘子，避免紧致性较容易优化时掩盖绝对逃逸失败；
4. 为 prefix/suffix/random 分别维护 elite archive，只在周期性全模式复评时合并 Pareto 前沿。

这是最高优先级，因为继续单纯扩大长度或奖励两两相似度，预计只会得到“更紧但仍留在原簇内”的串。

### P1：扩展离散组合搜索，而不是仅增加 CEM 轮数

1. 从模式 3 的独立单 token 屏幕构造 token-pair 交互矩阵，加入高逃逸×高紧致的互补 pair 种子；
2. 使用 beam search 保留跨位置条件依赖，再用 CEM 做大范围分布探索；
3. 对长度 2–8 做更高覆盖率的 pair/beam 搜索，优先确认短串不存在还是当前优化器漏检；
4. 白盒场景增加梯度引导的候选替换，黑盒场景保留当前 score-only 接口，并分别报告查询预算；
5. 对停滞位置使用受约束 mutation/crossover，而不是只重置一半位置。

### P1：扩大严格隔离的数据，而不是复用高连通句对图

本次 64.959% 的删边率显著降低了统计功效。应重新构造天然句子互斥的 search/validation/test 数据：

1. 从文档或语义簇层面先划分，再生成组内句对；
2. 每组扩大到足以给 q05/q95 提供稳定估计的样本量；
3. 对核心指标报告 bootstrap 置信区间，而不只报告点估计；
4. 用至少 5 个数据划分种子重复，候选仍必须在每个种子的 test 上只评估一次；
5. 加入域外语料，检验 trigger 是否只是利用当前句对集合的词汇偏差。

### P2：增加模型与 tokenizer 真实性

1. 在多个编码器家族上重复，并报告同模型发现、跨模型迁移和跨 tokenizer 实现率；
2. 模式 1 增加 ordinary-token-only 主结果，special token 结果降为机制对照；
3. 对自然分隔符、大小写、Unicode 正规化、标点邻接和真实检索模板分别验证；
4. 对每个候选报告 standalone 与上下文 token IDs，继续禁止只凭 decode 字符串认定组件长度。

### P2：进入真实 RAG/检索任务

几何认证不是攻击成功率。下一阶段应在冻结 trigger 后测量：

- top-k 检索命中率、排名变化和攻击成功率；
- 查询侧、文档侧以及双侧插入的差异；
- 语义相关性、可见性和困惑度代价；
- 中心化、异常 token 过滤、输入规范化和多编码器一致性防御。

只有在不再用 test 选模的前提下，几何效应稳定转化为检索行为，才能把模式 2/3 从表示空间现象推进到 RAG 攻击结论。

### P3：工程性能

1. 缓存每种组件序列的 tokenizer 实现与三种插入位置文本；
2. 将候选×句子的编码请求按长度桶批量化，减少 Python 字符串构造和 tokenizer 开销；
3. validation 按长度或候选分片到空闲 GPU，保持冻结规则不变；
4. 等长基线和投影在核心 test 文件写出后独立调度，并保存阶段 checkpoint；
5. 为搜索 archive 增加可恢复 checkpoint，使中断后不重复已完成长度。

## 8. 可核查产物

- 总览：`results/sticky_lab/sentence_t5_base/comparison_v2/README.md`。
- 三模式汇总：`comparison_v2/three_mode_summary.csv`。
- 模式 1：`single_sticky_v2/full_summary.json`、`dose_curve.csv`、`dose_similarity_curves.png`。
- 模式 2：`multi_booster_v2/frozen_candidate.json`、`test_result.json`、`length_frontier.csv/.png` 和消融文件。
- 模式 3：`repulsive_attractor_v2/frozen_candidate.json`、`test_result.json`、`equal_length_baselines.csv`、前沿图和两张联合投影图。
- 每个模式均保存 resolved config、split audit、环境信息、阶段摘要与 SHA-256 artifact manifest。

本报告中的“通过”均指当前冻结模型、数据分布、阈值和有限 test 集上的经验认证，不是对所有自然语言输入的数学全称证明。
