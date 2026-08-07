# Sentence-T5-base Sticky / Attractor V2 实验汇总

模式 1 的长度表示一个 token；其剂量由重复次数曲线单独报告。模式 2、3 的长度表示只插入一次的组合组件数。

| mode | selected_length | selected_trigger_json | validation_core_certified | test_core_certified | runtime_seconds |
| --- | --- | --- | --- | --- | --- |
| single_sticky | 1 | " <extra_id_27>" | True | True | 1239.5580315589905 |
| multi_booster | 1 | "lucrarea" | True | True | 319.66423749923706 |
| repulsive_attractor | 18 | " Sultan never prochain strategically incentives consistent incentives Communitiesachi never $50,000supervised subsidies. farms NEVER remix Cape" | False | False | 3868.0353679656982 |

`test_core_certified` 只来自 validation 冻结后的一次 test 评估；test 未参与候选或长度选择。
