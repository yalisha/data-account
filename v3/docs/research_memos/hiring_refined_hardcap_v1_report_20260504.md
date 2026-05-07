# 招聘细化版硬能力试跑报告

日期：2026-05-04

## 1. 本轮目的

本轮不是再造一个“披露强度”变量，而是检验招聘大数据能不能把 X 做细：

```text
年报数据要素叙事 - 企业级可验证硬能力
```

其中招聘侧代表“企业是否真的在配置数据/AI/算法/数据工程人力资本”。这比泛泛的数字人力投入更像硬证据。

本轮所有回归和筛选使用 Stata MCP；Python 只用于把 1.7G 招聘原始表流式压缩成 firm-year 小表。

## 2. 新增脚本和输出

招聘原始数据构造：

```text
src/python/build_hiring_custom_v1.py
results/data/hiring_custom_v1_firm_year.csv
results/data/hiring_custom_v1_for_stata.csv
results/stata/hiring_custom_v1_summary.csv
results/stata/hiring_custom_v1_by_year.csv
```

招聘加入综合硬能力：

```text
src/stata/hardcap_hiring_custom_v1_screen.do
src/stata/hardcap_hiring_custom_v1_compare.do
results/data/hardcap_hiring_custom_v1_panel.dta
results/stata/hardcap_hiring_custom_v1_x_summary.csv
results/stata/hardcap_hiring_custom_v1_y_screen.csv
results/stata/hardcap_hiring_custom_v1_correlations.csv
```

纯招聘硬能力：

```text
src/stata/hiring_pure_mismatch_v1_screen.do
src/stata/hiring_pure_mismatch_v1_compare.do
results/data/hiring_pure_mismatch_v1_panel.dta
results/stata/hiring_pure_mismatch_v1_x_summary.csv
results/stata/hiring_pure_mismatch_v1_y_screen.csv
results/stata/hiring_pure_mismatch_v1_correlations.csv
```

对应 Stata MCP 日志已经保存到：

```text
results/logs/
```

## 3. 招聘数据覆盖

原始招聘表：

```text
/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司其他/上市公司招聘大数据2014-2026.3.rar
```

脚本流式读取 RAR 内 CSV，没有完整解压 1.7G 文件。

处理结果：

| 项目 | 数值 |
|---|---:|
| 原始招聘行数 | 8,996,146 |
| 输出 firm-year 行数 | 53,063 |
| 覆盖公司数 | 5,643 |
| 覆盖年份 | 2014-2026 |

进入当前回归面板后的招聘变量分布：

| 变量 | N | mean | median | positive |
|---|---:|---:|---:|---:|
| `HirePostsAll` | 37,372 | 45.8335 | 0 | 8,841 |
| `DataHireBroadShare` | 37,372 | 0.0704 | 0 | 8,138 |
| `DataHireTitleShare` | 37,372 | 0.0028 | 0 | 2,637 |
| `AIHireShare` | 37,372 | 0.0047 | 0 | 3,249 |
| `DataEngHireShare` | 37,372 | 0.0092 | 0 | 4,661 |

判断：

```text
Broad text 太宽，只能稳健性；岗位标题、数据工程、AI 招聘更适合主写。
```

原因是 broad text 会把职位描述里泛泛出现“数据”的岗位都算进去，概念上不够硬。

## 4. 版本 A：招聘加入综合 hard capability

构造：

```text
HCap_hire_title =
    rowmean(
        z(DigInvPatAut_ln),
        z(DataHireTitleShare),
        z(AIInvestLevel),
        z(DigCapexAmount_ln),
        z(PatentTitleDataApp_ln)
    )
```

对应 X：

```text
NHGap_hire_title  = PctYear(DU_kw) - PctYear(HCap_hire_title)
NHWash_hire_title = high narrative, low hard capability
NHHush_hire_title = high hard capability, low narrative
NHVer_hire_title  = high narrative, high hard capability
```

还构造了 engineer、posts、broad 三个版本。

### 4.1 关键回归结果

回归设定：

```text
Y_t = L.X + controls + firm FE + year FE, cluster(IndYear_num)
```

`NHGap_hire_title`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | -0.1725 | -9.21 | 37,294 |
| `Analyst` | 0.0687 | 3.14 | 37,294 |
| `AuditFee` | 0.0177 | 3.06 | 37,211 |
| `ForecastDisp` | -0.0121 | -2.21 | 19,126 |
| `PriceDelay` | -0.0090 | -3.90 | 37,294 |
| `TFP` | 0.0399 | 2.35 | 5,951 |

`NHWash_hire_title`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | -0.0776 | -6.25 | 37,294 |
| `Analyst` | 0.0641 | 4.48 | 37,294 |
| `AuditFee` | 0.0152 | 4.02 | 37,211 |
| `PriceDelay` | -0.0061 | -3.99 | 37,294 |

strict `NHWash_hire_title_s`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | -0.0724 | -4.28 | 37,294 |
| `Analyst` | 0.0952 | 5.28 | 37,294 |
| `RatingDisp` | 0.0160 | 2.60 | 22,842 |
| `ReportFreq` | 0.0572 | 2.37 | 24,046 |
| `AuditFee` | 0.0169 | 3.07 | 37,211 |

表面上很好，但这个版本不宜当主 X。

### 4.2 为什么不宜主打

和 direct v1 的相关性太高：

| pair | rho |
|---|---:|
| `HCap_hire_title` vs `HardCap_direct_v1` | 0.9174 |
| `NHGap_hire_title` vs `NarrHardGap_direct_v1` | 0.9838 |
| `NHWash_hire_title` vs `NarrHardWashing_direct_v1` | 0.9927 |
| `NHWash_hire_title_s` vs `NHWash_direct_v1_s` | 0.9948 |

判断：

```text
综合 hard capability 加入招聘后，结果稳，但增量不大；它更像 direct v1 的稳健性，不是新的主变量。
```

## 5. 版本 B：纯招聘 hard capability

为了真正“做细”，本轮又构造了只基于招聘的 hard capability。

核心变量：

```text
HireCap_title    = z(DataHireTitleShare)
HireCap_engineer = rowmean(z(DataEngHireShare), z(AIHireShare))
HireCap_posts    = rowmean(z(DataHireTitlePostsLn), z(DataEngHirePostsLn), z(AIHirePostsLn))
HireCap_broad    = z(DataHireBroadShare)
```

对应 X：

```text
NarrHireGap_title  = PctYear(DU_kw) - PctYear(HireCap_title)
NarrHireWash_title = high narrative, low hiring evidence
NarrHireHush_title = high hiring evidence, low narrative
NarrHireVer_title  = high narrative, high hiring evidence
```

注意：

```text
纯招聘版本只从 2014 年开始构造，因此 L.X 回归主要使用 2015 年以后样本。
```

### 5.1 纯招聘 X 分布

| X | N | mean | positive |
|---|---:|---:|---:|
| `HireCap_title` | 37,372 | 0.0000 | 2,499 |
| `HireCap_engineer` | 37,372 | -0.0000 | 4,567 |
| `HireCap_posts` | 37,372 | 0.0000 | 5,250 |
| `NarrHireGap_title` | 37,372 | 0.4317 | 33,991 |
| `NarrHireWash_title` | 37,372 | 0.4634 | 17,318 |
| `NarrHireHush_title` | 37,372 | 0.0339 | 1,268 |
| `NarrHireVer_title` | 37,372 | 0.0366 | 1,369 |
| `NHireWash_title_s` | 37,372 | 0.2320 | 8,670 |

判断：

```text
纯招聘 hard evidence 稀疏，但这正好适合识别“年报喊数据要素、招聘上没有数据岗位证据”的错配。
```

### 5.2 与 direct v1 的相关性

纯招聘能力本身和 direct v1 不是一回事：

| pair | rho |
|---|---:|
| `HireCap_title` vs `HardCap_direct_v1` | 0.1654 |
| `HireCap_engineer` vs `HardCap_direct_v1` | 0.2552 |
| `HireCap_posts` vs `HardCap_direct_v1` | 0.3639 |
| `HireCap_broad` vs `HardCap_direct_v1` | 0.2690 |

但 X 仍然包含同一个年报叙事侧，所以和 direct v1 的 X 有中等到较高重合：

| pair | rho |
|---|---:|
| `NarrHireGap_title` vs `NarrHardGap_direct_v1` | 0.7006 |
| `NarrHireWash_title` vs `NarrHardWashing_direct_v1` | 0.8381 |
| `NarrHireHush_title` vs `NarrHardHushing_direct_v1` | 0.4627 |
| `NarrHireVer_title` vs `NarrHardVerified_direct_v1` | 0.4860 |

判断：

```text
纯招聘版有增量，但它不应该替代综合 hard capability；更适合作为人力资本硬证据分支。
```

### 5.3 纯招聘主结果

`NarrHireGap_title`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | -0.1077 | -5.16 | 31,190 |
| `Analyst` | 0.0646 | 2.87 | 31,190 |
| `InvestIneff` | -0.0044 | -2.17 | 26,759 |
| `SCConc` | -0.7031 | -3.28 | 30,475 |
| `SuppConc` | -0.7268 | -2.68 | 30,243 |

`NarrHireWash_title`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | -0.0562 | -4.25 | 31,190 |
| `Analyst` | 0.0598 | 3.97 | 31,190 |
| `InvestIneff` | -0.0044 | -3.17 | 26,759 |
| `PriceDelay` | -0.0041 | -2.55 | 31,190 |

strict `NHireWash_title_s`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | -0.0602 | -3.25 | 31,190 |
| `Analyst` | 0.0758 | 4.02 | 31,190 |
| `RatingDisp` | 0.0169 | 2.68 | 18,507 |
| `ReportFreq` | 0.0704 | 2.99 | 19,342 |
| `InvestIneff` | -0.0049 | -2.47 | 26,759 |

engineer/posts 版本方向一致：

| X | Y | coef | t |
|---|---|---:|---:|
| `NarrHireWash_engineer` | `TobinQ` | -0.0618 | -4.49 |
| `NarrHireWash_engineer` | `Analyst` | 0.0498 | 3.33 |
| `NarrHireWash_posts` | `TobinQ` | -0.0600 | -4.36 |
| `NarrHireWash_posts` | `Analyst` | 0.0506 | 3.39 |

### 5.4 纯招聘版的弱点

1. `AuditFee` 不显著。纯招聘错配不像综合 hardcap 那样自然连接审计验证成本。
2. `PriceDelay` 只有部分 washing 版本显著，且主线不宜回到定价效率。
3. `Hushing` 样本很小，title 版只有约 1,268 个正样本，不适合主写。
4. 招聘数据从 2014 年开始，回归样本变成 2015 年以后。

## 6. 本轮结论

当前最合理的论文变量体系是两层：

### 第一层：主 X

继续用综合企业级硬能力：

```text
NarrHardGap_direct_v1
NarrHardWashing_direct_v1
NarrHardHushing_direct_v1
NarrHardVerified_direct_v1
NHWash_direct_v1_s
```

理由：

1. 全样本更长，2011-2024。
2. 结果最稳，尤其是 `TobinQ`、`Analyst`、`AuditFee`。
3. 概念上覆盖专利、投资、资本开支、企业级数据/AI能力证据。

### 第二层：细化/机制 X

使用纯招聘版作为人力资本硬证据分支：

```text
NarrHireGap_title
NarrHireWash_title
NHireWash_title_s
```

理由：

1. `HireCap_title` 与 `HardCap_direct_v1` 相关性只有 0.1654，有明显增量。
2. `NarrHireWash_title` 对 `TobinQ` 和 `Analyst` 仍显著。
3. strict washing 能同时对应估值折价、分析师关注、分歧和研报频率。

不建议主打：

```text
HCap_hire_title
NHGap_hire_title
NHWash_hire_title
```

因为它们与 direct v1 几乎重合，只能作为“加入招聘后结果不变”的稳健性。

## 7. 对题目的影响

这轮结果支持把题目做成：

```text
数据要素叙事-可验证硬能力错配、估值折价与信息中介反应
```

而不是：

```text
年报数据要素披露与企业经济后果
```

更细一点可以写成：

```text
When Data Narratives Outrun Hard Capabilities:
Evidence from Data-Element Disclosures and Data-Talent Hiring
```

中文可以压成：

```text
数据要素叙事是否超过企业真实数据能力？
来自年报披露与数据岗位招聘的证据
```

但投稿主文不宜把“招聘”放到标题最前面。招聘是强补充，不是唯一硬能力。

## 8. 下一步

最值得继续做的是三件事：

1. 固定主 X：`NarrHardGap_direct_v1`、`NarrHardWashing_direct_v1`、`NHWash_direct_v1_s`。
2. 把纯招聘版放进 mechanism / robustness：`NarrHireWash_title`、`NHireWash_title_s`。
3. 再补一个企业级硬证据：软件著作权或 20G 全量专利摘要/IPC/主权项。

如果补到软件著作权，最终 `HardCap_core` 可以写成：

```text
HardCap_core =
    rowmean(
        z(DataPatent_ln),
        z(SoftwareCopyright_ln),
        z(DataHiringTitleShare),
        z(AIInvestLevel),
        z(DataCapex_ln)
    )
```

然后主文变量变为：

```text
NarrHardGap_core
DataWashing_core
DataHushing_core
DataVerified_core
```

这会比当前 direct v1 更像正式投稿版。
