# 本地新增数据盘点：是否可用于 NarrHardGap 的硬能力侧

日期：2026-05-04

检查路径：

```text
/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司其他
```

检查对象：

```text
上市公司招聘大数据2014-2026.3.rar    1.7G
分年份保存数据.rar                    20G
```

只做只读检查，没有解压全量、没有移动原始文件。

## 1. 结论

这两个都是需要的，但优先级不同。

| 文件 | 判断 | 最适合用途 | 是否直接可并入 firm-year |
|---|---|---|---|
| `上市公司招聘大数据2014-2026.3.rar` | 很需要 | 构造 `DataHiringShare / DataHiringPosts / DataHiringRecruits` | 是，有 `关联股票代码` |
| `分年份保存数据.rar` | 需要，但不应第一步啃全量 | 构造自定义 `DataPatent / AIDataPatent / SoftwarePatent` | 否，需要匹配上市公司 |

当前最推荐的处理顺序：

```text
1. 先用招聘大数据构造 DataHiring
2. 再用同级目录里已有的“小表”构造 DigPatent / DigHuman / DigCapex
3. 最后再决定是否动 20G 全量专利库
```

## 2. 招聘大数据

文件：

```text
/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司其他/上市公司招聘大数据2014-2026.3.rar
```

压缩包内：

```text
上市公司招聘大数据2014-2026.3.csv
```

表头字段：

```text
企业名称
股票简称
关联股票代码
与上市公司关系
上市公司行业
招聘发布年份
招聘结束年份
招聘岗位
工作城市
工作区域
最低月薪
最高月薪
职位描述
学历要求
要求经验
招聘人数
招聘类别
初级分类
来源平台
公司地点
工作地点
招聘发布日期
招聘结束日期
来源
```

可构造变量：

```text
HiringPosts_all          firm-year 全部招聘条数
HiringRecruits_all       firm-year 招聘人数合计
DataHiringPosts          数据/AI/算法/软件/数据库相关岗位条数
DataHiringRecruits       数据/AI/算法/软件/数据库相关招聘人数
DataHiringShare          DataHiringPosts / HiringPosts_all
DataHiringRecruitShare   DataHiringRecruits / HiringRecruits_all
DataHiringSalary         数据岗位平均薪资或薪资中位数
```

推荐关键词：

```text
数据
大数据
数据库
数据仓库
数据治理
数据安全
数据分析
数据挖掘
数据工程师
算法
机器学习
深度学习
人工智能
AI
NLP
自然语言
计算机视觉
云计算
云平台
软件开发
软件工程师
平台开发
后端开发
前端开发
信息系统
```

注意：

1. `关联股票代码` 已有，可直接标准化成 6 位股票代码。
2. `招聘人数` 里可能有“若干”、空值，要统一规则。
3. 这张表看起来是全招聘，不是数字岗位专表，所以必须做岗位/描述关键词筛选。
4. 最好同时保留分母，避免只用岗位数量导致企业规模偏误。

## 3. 全量专利库

文件：

```text
/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司其他/分年份保存数据.rar
```

压缩包内按年份保存：

```text
分年份保存数据/中国全量专利数据库1985年.csv
...
分年份保存数据/中国全量专利数据库2025年.csv
```

2024 年样例字段：

```text
专利名称
专利类型
申请人
申请人类型
申请人地址
申请人地区
申请人城市
申请人区县
申请号
申请日
申请年份
公开公告号
公开公告日
公开公告年份
授权公告号
授权公告日
授权公告年份
IPC分类号
IPC主分类号
发明人
摘要文本
主权项内容
当前权利人
当前专利权人地址
专利权人类型
统一社会信用代码
引证次数
被引证次数
自引次数
他引次数
被自引次数
被他引次数
家族引证次数
家族被引证次数
```

可构造变量：

```text
Patent_all_count
Patent_invention_count
Patent_granted_count
DataPatent_count
DataPatent_invention_count
DataPatent_granted_count
AIDataPatent_count
SoftwarePatent_count
DataPatent_citations
DataPatent_family_citations
```

匹配逻辑：

```text
优先：统一社会信用代码 -> STK_LISTEDCOINFOANL.xlsx 的 SocialCreditCode
兜底：申请人/当前权利人 -> FullName / ShortName 标准化匹配
```

本地已确认存在上市公司映射表：

```text
/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司财务信息/STK_LISTEDCOINFOANL.xlsx
```

其中有：

```text
Symbol
ShortName
FullName
SocialCreditCode
EndDate
LISTINGSTATE
```

注意：

1. 专利表是全量，不是上市公司专表，20G 不应直接进 Stata。
2. 先按年份流式读取，只保留能匹配到上市公司或疑似上市公司的记录。
3. 先做 2014-2024，与当前主样本年份对齐。
4. 2024 年样例里第一行后面又重复了一次表头，正式处理时要剔除重复表头行。
5. 同一申请号可能有发明申请/发明授权重复记录，必须去重或分别定义申请口径/授权口径。

## 4. 还有更优先的小表

在同级目录：

```text
/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司数据相关信息
```

已发现几张直接按证券代码整理好的小表。它们比 20G 全量专利库更适合第一轮跑通。

### 4.1 数字发明专利授权

文件：

```text
数字发明专利授权情况表（年）190317870.zip
```

字段说明：

```text
sgnyear
symbol
shortname
diginvpatautnum
```

含义：

```text
上市公司本身当年申请的发明专利已授权数量
```

建议变量：

```text
DigInvPatAut_ln = ln(1 + diginvpatautnum)
```

用途：

```text
HardCap_core 第一批组件
```

### 4.2 数字人力投入计划

文件：

```text
数字人力投入计划统计表（月）185606015.zip
```

字段说明：

```text
sgnmonth
symbol
shortname
releasedemandtimes
recruitmentsnumber
```

含义：

```text
数字职位招聘信息条数与招聘人数。
限定职能类别：计算机/互联网/通信/电子、软件研发、人工智能、数据工程师。
```

建议变量：

```text
DigHumanDemand_y = sum(releasedemandtimes)
DigHumanRecruit_y = sum(recruitmentsnumber)
DigHumanDemand_ln = ln(1 + DigHumanDemand_y)
DigHumanRecruit_ln = ln(1 + DigHumanRecruit_y)
```

用途：

```text
HardCap_core 第一批组件
```

### 4.3 数字资本投入计划

文件：

```text
数字资本投入计划统计表（年）185545091.zip
```

字段说明：

```text
sgnyear
symbol
shortname
investitemnum
investtotalamount
```

含义：

```text
按数字化关键词筛选募集资金项目名称后的计划投资项目数和计划投资总额。
```

建议变量：

```text
DigCapexItem_ln = ln(1 + investitemnum)
DigCapexAmount_ln = ln(1 + investtotalamount)
```

用途：

```text
HardCap_extended / data investment 机制
```

### 4.4 人工智能投资水平

文件：

```text
人工智能投资水平131920031(仅供沪江大学使用).zip
```

关键字段：

```text
AISoftInvest
AISoftInvestValueAdd
AIHardInvest
AIHardInvestValueAdd
AIInvestTotal
AIInvestTotalValueAdd
AIInvestLevel
```

含义：

```text
从财务报告附注中筛选人工智能软件/硬件投资，含软件、系统、数据、数字、云计算、服务器、算力等关键词。
```

建议变量：

```text
AIInvestTotal_ln
AIInvestLevel
AIInvestValueAdd_ln
```

用途：

```text
HardCap_extended，且很贴 special issue 的 data investment。
```

### 4.5 专利明细

文件：

```text
专利明细情况093018090(仅供哈佛大学使用).zip
```

字段：

```text
Symbol
EndDate
PatentName
PatentType
ApplicationDate
GrantDate
LegalStatus
Applicant
```

用途：

```text
上市公司专利基础控制、或用 PatentName 做数据/AI/软件关键词筛选。
```

限制：

```text
字段比 20G 全量专利库少，没有摘要和 IPC，所以做精细数据专利时不如全量库。
```

## 5. 推荐第一轮 HardCap 构造

第一轮不要直接从 20G 全量专利库开始。先用小表跑通：

```text
HardCap_direct_v1 =
    rowmean(
        z(DigInvPatAut_ln),
        z(DigHumanDemand_ln),
        z(AIInvestTotal_ln)
    )
```

如果人工智能投资表覆盖不足：

```text
HardCap_direct_v1 =
    rowmean(
        z(DigInvPatAut_ln),
        z(DigHumanDemand_ln),
        z(DigCapexAmount_ln)
    )
```

对应 X：

```text
NarrHardGap_direct_v1 =
    PctYear(DU_kw) - PctYear(HardCap_direct_v1)
```

第一轮目标：

```text
先看用直接小表构造的硬能力，是否能复现或增强 NarrHardGap_base 的 TobinQ / Analyst / RatingDisp 信号。
```

## 6. 第二轮再用两个大文件

### 6.1 招聘大数据

用于替代或验证 `数字人力投入计划统计表（月）`：

```text
DataHiringShare_custom
DataHiringPosts_custom
DataHiringRecruits_custom
```

优势：

```text
可以自己定义“数据岗位”，更贴数据要素，不局限于泛数字岗位。
```

### 6.2 全量专利库

用于替代或验证 `数字发明专利授权情况表（年）`：

```text
DataPatent_custom
AIDataPatent_custom
SoftwarePatent_custom
```

优势：

```text
有标题、摘要、主权项、IPC、统一社会信用代码，可做更精细的 data/AI/software patent。
```

但成本更高：

```text
需要流式读取、匹配上市公司、去重、关键词/IPC 分类。
```

## 7. 是否需要这两个大文件

最终判断：

```text
上市公司招聘大数据2014-2026.3.rar：需要，优先级高。
分年份保存数据.rar：需要，但先不急着全量处理。
```

如果现在要最快进入实证，应先处理：

```text
数字发明专利授权情况表（年）
数字人力投入计划统计表（月）
人工智能投资水平 / 数字资本投入计划
```

然后再用两个大文件做自定义版和稳健性。
