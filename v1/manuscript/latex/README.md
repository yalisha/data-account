# LaTeX Manuscript

这个目录用于维护论文的 Elsevier `elsarticle` LaTeX 版本。

## 当前状态

- 版式使用 Elsevier 官方 `elsarticle` 模板。
- 主文件是 `main.tex`。
- 各章节从 `../` 下的 Markdown 源文件自动转换到 `sections/*.tex`。
- 当前参考文献仍保留为正文中的手工参考文献章节，`references.bib` 已同步到本目录，后续可再逐步切换为 `\cite` + BibTeX 工作流。

## 使用方式

```bash
cd "/Users/mac/computerscience/0做完了/15会计研究/manuscript/latex"
./build.sh
```

## 目录说明

- `main.tex`：Elsevier 主文档。
- `regen_sections.sh`：把当前 Markdown 章节重生成为 LaTeX 小节文件。
- `sections/`：由脚本生成的章节文件。
- `vendor/elsarticle/`：下载的官方模板原包。
