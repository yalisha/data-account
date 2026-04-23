# OPEN QUESTIONS

1. `Semantic Scholar` 的 `snippet_search` 在本机 MCP 上对主题关键词和精确题名查询均出现超时；本轮检索将改用 `search_authors_by_name` + `get_paper` + `get_citations` 的组合完成主题扩展与 citation 排序。
2. 主题 2 指定的 `Farboodi & Veldkamp (2022, Review of Economic Studies)` 在 spec 中已注明“需核对刊名”；本轮会先核对其正式发表状态、年份与 DOI，再决定是否放入核心清单。
3. Zotero MCP 的 collection/item API 在 Zotero 未启动时返回 `Connection refused`；现已启动本机 Zotero，后续若 collection 写入或 BibTeX 导出仍受限，将在对应条目处标记并给出本地 `.bib` 回退方案。
4. 主题 7 要求中文 CSSCI 文献同时带 DOI + citation count，但部分中文期刊在 Semantic Scholar 的 DOI 或 citation 数据可能缺失；若无法通过 MCP 完整核验，将按 spec 标注 `[需人工核对]`。
5. spec 要求 citation count 截止 `2025-12`，但当前可访问的 MCP/API 仅返回当前累计引用，无法直接回溯历史快照；本轮将记录“当前可核验 citation count”，并在正文说明这一限制。
6. 主题 1 的已知必命中文献存在刊名口径冲突：`Crouzet et al. (2022) The Economics of Intangible Capital` 当前可核验来源显示为 `Journal of Economic Perspectives`，不是 spec 中写的 `QJE`。
7. 主题 1 的 `Ewens, Peters, Wang, Measuring Intangible Capital with Market Prices` 当前先核到的是 `2019 NBER working paper` 版本；是否已有 `RFS` 正式发表版本仍需继续核对。

# v2-1 现代文献清单（2015-2024）

## 按主题分组

## 按假说分配

## Zotero 操作
