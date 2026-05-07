# X 构造：事前 data-washing 暴露

更新日期：2026-05-06

## 1. X 的经济含义

本文的 X 不是普通“数据要素披露强度”。真正的异质性暴露是：

```text
企业在数据交易所设立之前，是否存在“数据要素叙事强、可验证硬能力弱”的错配。
```

这个设定服务于主问题：数据交易所出现后，最可能调整披露结构的不是所有企业，而是原先容易进行泛化数据叙事、但缺少硬证据支撑的企业。

## 2. 基础错配变量

基础变量来自：

```text
results/data/hardcap_direct_v1_panel.dta
src/python/build_hardcap_direct_v1.py
src/stata/hardcap_direct_v1_screen.do
```

核心定义：

```text
NarrHardGap_direct_v1
  = percentile(data narrative) - percentile(hard data capability)

NarrHardWashing_direct_v1
  = 1{data narrative 高, hard data capability 低}

NHWash_direct_v1_s
  = 更严格口径的 high narrative / low hard capability
```

其中 data narrative 主要由年报数据要素叙事强度捕捉，hard data capability 来自企业级可验证硬能力。

## 3. HardCap_direct_v1 的构成

`HardCap_direct_v1` 是综合硬能力指标，不是行业/地区潜力。当前组件包括：

```text
DigInvPatAut          数字发明专利授权
DigHumanDemand        数字人力需求
DigHumanRecruit       数字招聘
DigCapexAmount        数字资本开支
AIInvestLevel         AI 投资水平
PatentTitleDataApp    专利标题中的数据/AI/软件关键词
```

当前覆盖特征：

| 组件 | N | 正值数 |
|---|---:|---:|
| `DigInvPatAut` | 43,735 | 313 |
| `DigHumanDemand` | 13,599 | 1,586 |
| `DigHumanRecruit` | 13,599 | 1,582 |
| `DigCapexAmount` | 43,735 | 77 |
| `AIInvestLevel` | 43,735 | 9,258 |
| `PatentTitleDataApp` | 43,735 | 59 |
| `HardCap_direct_v1` | 43,735 | 4,042 |

解释：硬能力本身较稀疏，但这正是识别 data-washing 暴露的原因。大量企业有数据叙事，但缺少可验证硬证据。

## 4. 事前窗口

主窗口使用 2016-2017 年：

```text
PreGap1617
  = 2016-2017 年 NarrHardGap_direct_v1 均值

PreWashAny1617
  = 1{2016 或 2017 年存在 NarrHardWashing_direct_v1}

PreStrictWashAny1617
  = 1{2016 或 2017 年存在 NHWash_direct_v1_s}
```

使用 2016-2017 的原因：

1. 避免和 2018-2024 年数据交易所事件窗口重叠；
2. 保证 exposure 是事前特征，而不是政策后调整结果；
3. 使 DDD 更接近“哪些企业在政策前更可能 data-washing”的分组。

## 5. 暴露变量分布

在 2018 年进入 DDD 样本的企业中：

| exposure | N | mean | positives |
|---|---:|---:|---:|
| `PreGap1617` | 2,851 | 0.344 | 2,328 |
| `PreHighGap1617` | 2,851 | 0.838 | 2,390 |
| `PreWashAny1617` | 2,851 | 0.474 | 1,350 |
| `PreStrictWashAny1617` | 2,851 | 0.241 | 686 |
| `PreHardCap1617` | 2,851 | 0.006 | 192 |
| `PreDU1617` | 2,851 | 0.926 | 2,784 |

`PreWashAny1617` 覆盖更广，适合主表。`PreStrictWashAny1617` 更窄，适合严格暴露和机制补充。

## 6. 处理变量

处理变量来自既有数据交易所项目：

```text
DID_city_ct = 1{企业注册城市 c 在年份 t 已设立数据交易所}
```

主交互项：

```text
dx_wany17 = DID_city_ct x PreWashAny1617
dx_wstr17 = DID_city_ct x PreStrictWashAny1617
```

## 7. 为什么不用 DataAsset 构造事前 X

真实 `DataAsset` / `BookEntry` 很干净，但正值集中在 2024 年。刷新后原始全库 48,217 个公司年中，正值 89 个，全部在 2024 年。并入当前 DDD 面板后，2024 年有 84 个正值，2018-2023 年没有正值。

所以：

```text
DataAsset 不适合构造 2016-2017 年事前 hard capability。
```

它应该用于 Y 的验证层：

```text
BookEntry / DataAsset_ln
```

## 8. 当前写法

建议在论文中称为：

```text
prior data-washing exposure
```

中文：

```text
事前数据要素叙事-硬能力错配暴露
```

不要简单叫“高数据披露企业”。真正识别的是“叙事与硬证据错配”的企业，而不是数据披露高低本身。
