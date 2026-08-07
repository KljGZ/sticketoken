# Sticky / Attractor V3：模式 3 注册实验协议

V3 将 V2 的“原句排斥紧致吸引子”重构为 **Trigger-Induced Representation Region**。它不再用单个加权分数宣称一个新区域，而是将表示位移、线性可分、紧致性、样本空白、簇空白和局部密度空白分别测量、分别认证。

本文件描述实现于 `sticky_lab/mode3_v3/`、配置于 `configs/v3_mode3.yaml`、由 `scripts/run_v3_remote.sh` 调度的注册协议。模式 2 继续使用 V2 定义，但以相同长度表 `2,4,...,30` 重跑；模式 3 也搜索至 30，步长 2，并额外穷举长度 1。

## 1. 研究对象与两级主张

对归一化编码器表示

\[
z(s)=E(s)/\lVert E(s)\rVert_2
\]

和只插入一次的多 token 串 `x=(t1,...,tL)`，定义触发表示 `z_x(s)=z(I(s,x))`。V3 分离两个不等价的子协议：

1. **3A Minimal Trigger-Separating String**：触发表示与良性表示形成稳定可分的区域；
2. **3B Minimal Compact Blank-Region Attractor**：触发表示先形成紧致区域，且该区域在有限良性支持集、球面簇支持和局部密度支持下处于空白区域。

3B 的证据层级强于 3A，但实验不把 3A 自动升级为 3B。搜索长度 1 为全合法词表穷举；长度大于 1 是预注册预算内搜索。因此“最短”只表示：在已穷举长度 1、并按 `2,4,...,30` 完成预算搜索的长度中，validation 首次认证的最短串，不表示全局组合空间最短。

## 2. 数据、隔离与支持模型

数据首先对文本执行 Unicode NFC、首尾空白删除和连续空白压缩，保留大小写和标点，再计算稳定 SHA-256 句子 ID。两侧文本独立做 5–160 tokenizer token 的长度过滤，全局去重后先分 search/validation/test，再定种子抽样为 3000/1000/1000。另从完全排除于 IID 输入的两个文件生成去重 OOD 1000。

原始 CSV 没有可验证的文档 ID，代码不会把模型目录名冒充文档来源。审计明确记录 `document_provenance_available=false`，退化为“一条唯一句子一个 group”。这保证句子级零重叠，但不能声称真实文档级独立。bootstrap group 使用冻结的 search 球面语义簇。

良性支持模型只在 search 上拟合：

- 对 `K in {8,16,32,64}` 运行真正迭代的 spherical K-Means；
- 每个 K 使用 3 次初始化；
- 根据 cosine silhouette、最小簇大小和稳定性选择冻结 K；
- 保存归一化中心、每簇 95% 半径、search 样本和 kNN 距离基线；
- validation、test、OOD 只能分配到冻结中心，不能重新拟合支持边界。

## 3. 硬指标与几何含义

令触发中心

\[
c_x=\frac{\sum_i z_x(s_i)}{\lVert\sum_i z_x(s_i)\rVert_2},
\]

触发半径 `rho95` 为 `||z_x(s_i)-c_x||` 的 95% 分位。

### 3.1 位移

报告 `||z_x(s_i)-z(s_i)||` 的 q05、median、q95。`displacement_q05 >= 0.02` 只认证普遍表示变化，不认证新区域。

### 3.2 线性分离

方向

\[
w=\frac{\mu_x-\mu_b}{\lVert\mu_x-\mu_b\rVert_2}
\]

和硬间隔

\[
M_{sep}=Q_{.05}(w^Tz_x)-Q_{.95}(w^Tz_b)
\]

是 3A 核心指标。另报告 mean separation、ROC-AUC、balanced accuracy、FPR@95TPR，但这些辅助量不替代硬间隔。

### 3.3 紧致性

报告 `rho95`、触发后两两余弦均值和 q05。3B 要求 `rho95 <= 0.40`；pairwise 指标只用于解释。

### 3.4 三种空白边界

有限样本边界：

\[
M_{sample}=\min_{b\in B}\lVert c_x-b\rVert_2-\rho_{95}.
\]

球面簇边界：

\[
M_{cluster}=\min_k(\lVert c_x-\mu_k\rVert_2-R_k)-\rho_{95}.
\]

局部密度边界：

\[
M_{density}=r_k(c_x,B)-\rho_{95}-Q_{.95}(r_k(b,B)).
\]

`source_escape_q05` 只诊断样本是否离开各自原始语义簇，不是 3B 证书。

## 4. 分层认证

所有核心硬指标在 validation/test 上使用 500 次 grouped bootstrap、95% 双侧区间。下界约束用 CI lower，上界约束用 CI upper。

- Level 1 `shift_certified`：位移 q05 的 CI lower 至少 0.02；
- Level 2 `separator_certified`：`M_sep` 的 CI lower 大于 0；
- Level 3 `compact_certified`：`rho95` 的 CI upper 不超过 0.40；
- Level 4 分别输出 `sample_blank_certified`、`cluster_blank_certified`、`density_blank_certified`。

3B 的组合证书为：separator AND compact AND sample-blank AND (cluster-blank OR density-blank)，因此代码层面保证 `3B => 3A`。每个布尔量独立保存，不能只报告最终 AND。validation 的正式认证还要求：

- token ID 与字符串精确 round-trip；
- 在 48 个上下文中的文本实现率至少 0.95；
- 主目标严格超过同长度 64 条随机合法串的 q99。

每个长度先对 32 个候选计算完整 validation 点估计并冻结排序，再只对点估计第一名运行 500 次 CI 认证。由于研究目标是“最短认证串”，某个长度一旦通过，后续更长长度仍保留点估计前沿，但不再消耗 CI 预算或产生新的认证主张。这样不会降低更短长度的检验强度，也避免把不参与候选冻结的 AUC、FPR、pairwise 统计重复计算 500 次。

validation 冻结候选和长度后，test 只调用一次。test 不参与 fallback、调参或长度选择；`generalized=true` 仅在 validation 与 test 的核心证书同时成立时给出。冻结候选随后还在 OOD-source 1000 上独立评估，OOD 同样不得回流选择；`full_generalized=true` 要求 validation、test 与 OOD 三者的核心证书全部成立。

## 5. 插入位置协议

首先分别搜索和认证 `prefix`、`suffix`、固定种子的 token 边界 `random`。随后独立搜索 `universal` 候选，并要求同一个冻结串在上述三个位置全部通过；禁止把某个单位置胜者事后称为 universal。

每个组合只插入一次，禁止把整串重复 30 次作为主实验。长度来自 tokenizer ID 个数；同时记录字符串独立编码长度、上下文实现率和 round-trip，防止字符串拼接导致的 token 合并偷换长度。

## 6. 搜索与连续可行性上界

### 6.1 全词表筛选

长度 1 对所有合法、非特殊、可稳定解码 token 做穷举。8 个 GPU 按 token ID 确定性分片；合并时校验全集无缺失且无重复。筛选同时测 prefix/suffix/random 的 3A/3B 指标，构造 separator、blank、compact、普通 sticky 与随机 token 的并集候选池。

### 6.2 连续 soft prompt

长度 `1,2,4,8,16` 先优化连续 prompt，作为离散搜索的可行性上界。它只回答编码器局部几何上是否存在更强区域，不可直接当作可部署文本结果。连续上界还用于检验离散化损失。

### 6.3 混合离散搜索

长度 `2,4,...,30`、每长度 4 个独立 restart。主体 CEM 使用 population 128、20 轮、elite ratio 0.10，并具备：均匀探索混合、概率下限、熵保护、停滞位置重置、elite 最小 Hamming 距离、跨长度 warm start。

关键防偏差约束：动态 mini-batch 结果只进入临时 archive；每 5 轮在固定完整 search 子集复评，完整评估 elites 反向更新 CEM 分布；正式 archive 与临时 archive 分离；累计 full champion 不丢失；每个长度结束时对完整正式 archive 强制复评。

对长度 2/8/16/30 的 restart 0 额外使用多坐标 HotFlip beam：梯度只提出 token 替换，所有入选结果仍经过硬文本前向评估。最终候选由 CEM、HotFlip、单 token 穷举和控制基线共同进入 validation。

## 7. 对照、可视化与检索桥

每个长度至少比较：随机合法串、频繁普通 token、自然短语、重复 sticky、V2 串长度调整、separator-only、blank-only、compact-only、AgentPoison mean-distance proxy、连续 upper bound。正式候选须超过随机合法串 q99。

图包括三条互不混淆的长度关系：同长度最优结果、截至该长度的 envelope、最终冻结 trigger 的逐前缀增长。PCA 与 UMAP 对良性点、各搜索阶段和中心联合拟合；图中的中心是高维真实中心的投影，不是二维点的再平均。高维指标写入图注，图只解释，不替代认证。

检索桥在 validation 上优化连续 anchor，并在 test 报告 anchor margin、oracle coverage 和 `test_retrieval_anchor_certified`。它验证区域能否被检索方向利用，不回流改变 trigger。

## 8. 可复现产物

正式目录为 `results/sticky_lab/sentence_t5_base/mode3_v3/`，至少保存：resolved config、Git commit、依赖和设备清单、数据审计、冻结支持模型、分片筛选结果、候选池、连续上界、每 restart/length 的 history 与 formal candidates、validation frontier、等长基线、冻结候选、单次 test、检索 anchor、前缀增长和联合降维图。

完整远程命令：

```bash
bash scripts/run_v3_remote.sh
```

`--smoke` 只用于流水线和依赖检查，任何 smoke 结果不得进入研究结论。
