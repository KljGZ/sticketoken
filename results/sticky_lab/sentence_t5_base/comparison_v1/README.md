# Sentence-T5-base 三模式第一版实验汇总

> `certified` 只表示各模式注册约束在独立 test 分区上通过；三种模式的认证定义不同，数量不能直接当作同一指标横比。

| mode | best_trigger_json | certified_count | best_certified | best_constraint_violation | runtime_seconds |
| --- | --- | --- | --- | --- | --- |
| single_sticky | " <extra_id_27>" | 0 | False | 0.0 | 735.5820434093475 |
| multi_booster | "lucrareatial senzati sportingenţă" | 0 | False | 1.0425912141799927 | 1004.6469066143036 |
| repulsive_attractor | "lucrarea earthquake Smartphoneărălucrarea" | 0 | False | 2.905104182368986 | 165.26105213165283 |

详细条件见 `docs/three_mode_experiments.md`，逐候选证据见各模式目录中的 CSV 和 `run_summary.json`。

完整解释与失败项分析见 [`docs/first_three_mode_results.md`](../../../../docs/first_three_mode_results.md)。
