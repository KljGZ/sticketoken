# Sticky / Attractor V3 全量实验结果与改进路线

## 1. 结论先行

本轮在固定的 Sentence-T5-base 修订版本上，完整执行了模式 2 与模式 3 的注册长度表：长度 1 额外穷举，组合长度按 2、4、…、30 搜索。模式 2 和模式 3 的组合都只插入一次，不把整个组合重复多次作为主实验。

结果支持三个彼此独立的结论：

1. 模式 2 找到最短的单组件相似度增强串 `lucrarea`，冻结后的 test 同时通过核心约束与结构保持约束。
2. 模式 3A 在 prefix、suffix、random 以及独立搜索的 universal 协议中都找到长度 1 的触发分割串，并在 validation、test、OOD 上泛化。
3. 模式 3B 没有找到严格认证的良性支持集外空域吸引串。长串能产生大位移、很高的触发后两两相似度、较小半径和正的有限样本空白间隔，但良性簇包络间隔与 kNN 密度间隔仍为负。

因此，Sentence-T5-base 在当前数据分布与长度预算下支持：

\[
\text{稳定触发分割区域}
\]

但现有证据不支持：

\[
\text{紧致且位于良性支持集之外的空域吸引子}.
\]

这正是 V3 分层理论的作用：3A 成立不能被自动升级为 3B；“远离自己的原句并彼此紧致”也不能替代“远离全部良性支持”。

## 2. V3 相对 V2 的理论与实现改造

V2 的模式 3 主要使用源簇逃逸与紧致性。它不能排除触发表示离开自己的源簇后落入另一个良性簇。V3 将模式 3 重构为 Trigger-Induced Representation Region，并拆成两个强度层级：

- 3A Minimal Trigger-Separating String：要求表示位移成立，且触发表示与良性表示之间的硬分位数分割间隔显著为正。
- 3B Minimal Compact Blank-Region Attractor：在 3A 之上，要求触发簇紧致、与有限良性样本不相交，并通过良性簇包络或 kNN 低密度边界中的至少一个。

正式 3B 证书为：

\[
\text{separator}
\land
\text{compact}
\land
\text{sample-blank}
\land
(\text{cluster-blank}\lor\text{density-blank}).
\]

代码层面完成了以下变化：

- 真正迭代到收敛的 spherical K-Means，而不是普通 K-Means 后做一次余弦重分配；
- 样本、簇包络、kNN 密度三套良性支持边界；
- 500 次 95% grouped bootstrap，所有硬证书由 CI 上下界决定；
- prefix、suffix、固定种子的 random 分开搜索，再独立搜索 universal 候选；
- 长度 1 全合法词表穷举，长度 2..30 使用 4 个 restart 的 CEM，并在注册长度上加入 HotFlip beam；
- 连续 soft prompt 只作为可行性上界，不能作为可部署文本结果；
- validation 冻结候选后，test 与 OOD 各只用于一次泛化评估；
- 实际 tokenizer 长度、精确 round-trip 与 48 个上下文中的实现率进入正式验证；
- PCA/UMAP、高维真实中心、长度 frontier、best-up-to-length 与冻结串前缀增长分别保存。

## 3. 实验注册与数据审计

| 项目 | 注册值 |
|---|---|
| 模型 | `sentence-transformers/sentence-t5-base` |
| 模型修订 | `fc5d4628481afbbaaacd7af6bb07cf9d3865f781` |
| 嵌入维数 | 768 |
| 数值精度 | float32 hard-text evaluation |
| search / validation / test / OOD | 3000 / 1000 / 1000 / 1000 |
| 长度表 | 1，以及 2,4,…,30 |
| 搜索预算 | 4 restarts；population 128；20 iterations |
| 候选池 | 1422 个合法 token |
| bootstrap | 500 次；95% CI |
| 随机基线 | 每长度 64 条；要求超过 q99 |

原始 38 个 CSV 共 38,000 行。全局去重与独立长度过滤后有 17,889 个唯一句子；search、validation、test 之间的句子与 group 重叠均为 0。两个排除于 IID 的文件提供 OOD 1000 句。

必须保留的限制是：原始 CSV 没有可验证的文档 ID。实验没有把模型目录名伪装成文档来源，而是明确记录 `document_provenance_available=false`，并退化为一个唯一句子一个 group。因此当前结果保证句子级隔离，但不能宣称真实文档级隔离；grouped bootstrap 在这批数据上等价于唯一句子重采样。

良性支持模型只在 search 上拟合。spherical K-Means 在 K=8、16、32、64 中选择 K=8；该解收敛于 26 次迭代，最小簇为 214，cosine silhouette 为 0.09037。K=64 因最小簇只有 2 个样本而不合格。

## 4. 模式 2：Minimal Monotone Similarity Booster

冻结串为 `lucrarea`，搜索组件长度为 1。其 test 结果如下：

| 指标 | 数值 | 注册要求 | 结果 |
|---|---:|---:|---|
| low gain mean | 0.057191 | — | 提升 |
| low gain q10 | 0.044586 | ≥ 0.02 | 通过 |
| low coverage | 1.000000 | ≥ 0.80 | 通过 |
| high gain q05 | -0.019605 | ≥ -0.02 | 通过，接近边界 |
| high state retention | 1.000000 | ≥ 0.95 | 通过 |
| global drop rate | 0.024876 | ≤ 0.05 | 通过 |
| range ratio | 0.826754 | ≥ 0.70 | 通过 |
| Spearman | 0.989903 | ≥ 0.80 | 通过 |

`test_core_certified=true` 且 `test_structure_certified=true`。选择只使用 validation，`test_used_for_selection=false`。

tokenizer 口径必须严谨区分：`lucrarea` 是一个搜索组件，独立文字重编码得到两个实际 ID `3,30332`。因此模式 2 的“长度 1”指 V2 候选组件数，不应冒充 V3 的精确实际 tokenizer 长度。模式 3 V3 已将“实际长度等于组件长度且精确 round-trip”升级为硬条件。

## 5. 模式 3A：最短触发分割串

下表是冻结 test 结果；universal 行的连续指标取三种注册位置中的最坏值。

| 协议 | 最短串 | L | validation | test | OOD | 最坏 test M_sep | 最坏 test rho95 |
|---|---|---:|---|---|---|---:|---:|
| prefix | `LGBTQ` | 1 | 通过 | 通过 | 通过 | 0.167380 | 0.562169 |
| suffix | `Diabetes` | 1 | 通过 | 通过 | 通过 | 0.175870 | 0.556911 |
| random | `Minecraft` | 1 | 通过 | 通过 | 通过 | 0.124393 | 0.582951 |
| universal | `Minecraft` | 1 | 三位置通过 | 三位置通过 | 三位置通过 | 0.124393 | 0.582951 |

四个 3A 任务均有 `full_generalized=true`。其中 universal 候选来自独立 universal 搜索，而不是事后拿某个单位置冠军冒充位置普适结果。

这些结果证明短文本能够在 Sentence-T5-base 中建立稳定的触发/良性分割方向，但其触发半径约 0.56–0.58，未达到 `rho95≤0.40` 的紧致阈值；有限样本、簇包络与密度间隔也不构成 3B 证书。AUC 很高只能作为辅助量，主结论由硬分位数间隔及其 CI 给出。

## 6. 模式 3B：紧致空域吸引串

三个单位置任务均无 validation 可行串，下面报告注册回退串在 test 上的结果。回退结果用于描述搜索达到的几何前沿，不能称为认证成功。

| 协议 | 回退 L | M_sep | rho95 | M_sample | M_cluster | M_density | 严格 3B |
|---|---:|---:|---:|---:|---:|---:|---|
| prefix | 30 | 0.444882 | 0.166812 | 0.509190 | -0.186245 | -0.170302 | 否 |
| suffix | 30 | 0.356151 | 0.199099 | 0.413637 | -0.315315 | -0.243282 | 否 |
| random | 30 | 0.360445 | 0.218397 | 0.410247 | -0.326565 | -0.255108 | 否 |
| universal | 30 | 0.334200 | 0.225382 | 0.369312 | -0.373254 | -0.279882 | 否 |

单位置长度 30 回退串全文：

```text
prefix:
MORE celebrities useless TexasMostly Trump Muslim boycott Ballet concerts:// Constanța besuchen Alberta dimineata Reynolds lunette (30stream $10 Suceava Constanța chose humorousetapa Rit Timișoara („ Direktnam

suffix:
București strawberries rapper abortion reușit JoomlaAsociația Minecraftlucrarea tent autism crowdfunding feedinglucrarea Minecraftconsists rude Cause uniquely $5,000 placebo Scientist celebrities100 impossible Strawberry cars Aucklandputeti Diabetes

random:
Punjab hormone asleep breastfeeding scientists facutlucrarea cineva Romaniei chiropractor laugh Sony autism outrageous Weitere toothpaste Stay Beijing cookiesDatorita20around incercat celebrities roastsait (" Bruno ignored Minecraft

universal:
chiropractic Christmas României rudelucrareaterrorism Comedy YouTube Vatican harmless cineva Biblical thieves Spotify PHP toothpaste smoothie cineva prevalent broccolilucrarea Massachusetts humorous limbi poultry University Minecraft Pirate Vietnam Chicken
```

几何解释必须分两层：

- `rho95` 很小且 `M_sample` 明显为正，说明至少约 95% 的触发表示进入一个紧致球，并且该球不与已采样良性点相交。
- `M_cluster` 和 `M_density` 均为负，说明该球仍与经验良性簇包络或正常密度区域重叠，不能外推为“良性表示空间中的新空白区域”。

连续 soft-prompt 上界得到相同诊断。长度 8/16 的连续提示能让 separator、compact 与 sample-blank 同时成立，但 cluster-blank 和 density-blank 始终不成立。因此离散 3B 失败不能简单归因于 CEM 没有找到词；当前模型与良性支持定义本身就显示出结构性冲突。

## 7. 可复现性与完整性

最终审计器检查：

- 8 个搜索任务 × 4 restarts × 15 个组合长度，共 480 个正式候选文件；
- 32 个搜索摘要及其开始 commit；其中 24 个记录结束 commit，8 个在结束 commit 字段加入前已经完成，依靠源 commit 与候选文件内容哈希追溯；
- 8 个 validation frontier 是否完整覆盖注册长度表；
- 冻结串的实际 tokenizer 长度、精确 round-trip、上下文实现率；
- validation 选择与 test/OOD 隔离；
- 每个 split 上的 `3B => 3A` 逻辑不变量；
- 配置、数据切分、候选池、支持模型和 480 个搜索候选文件的 SHA-256 清单摘要。

大体积的搜索 history、嵌入矩阵和临时缓存保留在远程实验目录，不提交到 Git；Git 保存注册配置、数据/支持审计、候选池、冻结 frontier、test/OOD、图、摘要和哈希审计。这样既保留结论证据，也避免用二百余 MB 可重建缓存污染版本历史。

## 8. 下一步改进路线

### P0：先补强实验有效性

1. 使用含真实 document ID、domain、timestamp 的语料重新做 group split 与 grouped bootstrap，消除当前句子级分组退化。
2. 将相同注册协议扩展到不同结构的嵌入模型；当前结论只能归属于固定 Sentence-T5-base 修订与当前语料。
3. 固化 Docker/Conda lock、模型快照校验和与输入 CSV 校验和，使远程环境之外也能逐位复核。

### P1：针对 3B 的目标改造

1. 不再主要奖励平均位移；直接优化最邻近良性支持的软最小间隔、簇包络最坏间隔和 kNN 密度间隔。
2. 使用约束延续：先满足 separator，再逐步收紧 `rho95`，最后逐步提高 cluster/density blank margin，避免加权和用一个容易指标补偿失败硬约束。
3. 为 prefix、suffix、random 使用独立课程与候选池，再在 universal 阶段做最坏位置鲁棒优化；不从一开始强迫一个目标兼顾所有位置。
4. 对 continuous prompt 与最近词表投影的损失做分解，量化“模型几何不可行”与“离散化损失”，再决定是否增加长度或搜索预算。

### P2：面向真实 RAG 的有效性

1. 把当前 representation certificate 接到实际 ANN 索引，报告 Top-k margin、命中率、跨索引参数稳定性和端到端回答影响。
2. 区分两类威胁模型：查询与文档共享触发串，以及只有文档含触发串。后者不能盲目远离所有良性查询，需要额外优化目标查询覆盖锚点。
3. 增加自然性、可见性、跨分词器稳定性和输入清洗后的存活率，避免只得到几何上强但不可部署的字符串。

### P3：工程效率

1. 将 grouped bootstrap 的支持距离批量矩阵化，避免每个 replicate 重复扫描同一良性 memory；用数值等价测试锁定 CI 不变。
2. 为 universal validation 增加按长度分片、原子写入与只读合并阶段，使独立长度能够并行，同时保持最短长度选择和一次 test 的注册语义。
3. 对正式 archive、embedding cache 与结果文件做内容寻址，支持安全断点续跑并避免重复前向。

## 9. 允许与不允许的论文式表述

可以表述：

> 在固定 Sentence-T5-base 与注册数据上，长度 1 的文本 token 可使触发表示与良性表示形成跨 IID/OOD、跨插入位置的稳定分割；长度 30 的组合可产生紧致并与有限良性样本分离的触发簇。

不可以表述：

> 已找到位于全部自然文本嵌入空间之外的 universal 空域吸引子。

原因是当前所有 3B 候选的良性簇包络或 kNN 密度间隔没有通过 CI 认证，而且数据没有真实文档级 provenance。V3 的正确贡献是精确定位证据停留在何处，而不是强行把强位移或高两两相似度包装成更高层的几何结论。
