# CHANGELOG v14 -> v15

## 内容层面

- 保留 `DU_kw -> PriceDelay` 作为主结果，不改变主因变量和基准处理变量。
- 新增增强测度：
  - `DUevent`
  - `DUchain_count`
  - `DUclosedloop`
  - `DUcore`
  - `DUkw_mda`
- 主文新增“增强测度与章节限定口径”部分，用于说明更完整、更嵌入的数据利用表述同样与更低股价延迟相关。
- 明确写出：
  - `DUevent` 当前不稳，不升主文核心结论。
  - `DUclosedloop` 是当前最值得保留的增强测度。
  - `DUchain_count` 和 `DUkw_mda` 提供支持性证据。
  - `DUcore` 负向但不替代 `DU_kw`。

## 写法层面

- 摘要、Highlights、引言、研究设计、实证结果和结论全部同步到 `v15` 口径。
- 继续避免以下过度表述：
  - “与一般数字叙事完全分离”
  - “强因果识别”
  - “信息供给效应占据主导”
- 将构念升级写成：
  - “更接近数据资源实际使用”
  - “更完整、更嵌入的数据利用表述”
  - “支持性增强测度”

## LaTeX 层面

- Elsevier `elsarticle` 工程继续沿用。
- 前言区摘要和 Highlights 已同步为 `v15` 口径。
- 后续构建将基于更新后的 Markdown 章节自动重生 `sections/*.tex`。
