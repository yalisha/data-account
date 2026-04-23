"""
Generate manuscript v10 Word document from markdown chapters.
Creates fresh document with 宋体/黑体/Times New Roman styling.
"""
import re, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

BASE = Path("/Users/mac/computerscience/15会计研究")
OUTPUT = BASE / "manuscript/word/数据要素利用与资产定价效率v10.docx"

MD_FILES = [
    BASE / "manuscript/00_摘要.md",
    BASE / "manuscript/01_引言.md",
    BASE / "manuscript/02_理论与假说.md",
    BASE / "manuscript/03_研究设计.md",
    BASE / "manuscript/04_实证检验.md",
    BASE / "manuscript/05_进一步分析.md",
    BASE / "manuscript/06_异质性分析.md",
    BASE / "manuscript/07_结论.md",
    BASE / "manuscript/09_参考文献.md",
    BASE / "manuscript/附录.md",
]


def setup_styles(doc):
    """Set up document styles to match v9 format."""
    # Page setup
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(3.0)

    # Normal style
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    pf = style.paragraph_format
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)
    pf.line_spacing = Pt(20)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # East Asian font
    rPr = style.element.find(qn("w:rPr"))
    if rPr is None:
        rPr = style.element.makeelement(qn("w:rPr"), {})
        style.element.append(rPr)
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "宋体")
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")


def _set_run_font(run, cn="宋体", en="Times New Roman", size=12, bold=False):
    """Set fonts on a run."""
    run.font.name = en
    run.font.size = Pt(size)
    run.bold = bold
    rPr = run._element.find(qn("w:rPr"))
    if rPr is None:
        rPr = run._element.makeelement(qn("w:rPr"), {})
        run._element.insert(0, rPr)
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), cn)
    rFonts.set(qn("w:ascii"), en)
    rFonts.set(qn("w:hAnsi"), en)


def _add_text_bold(p, text, cn="宋体", en="Times New Roman", size=12):
    """Add text to paragraph, handling **bold** markers."""
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2])
            _set_run_font(run, cn, en, size, bold=True)
        else:
            clean = part.replace("\\*", "*")
            if clean:
                run = p.add_run(clean)
                _set_run_font(run, cn, en, size)


def add_heading1(doc, text):
    """Title-level heading: 黑体 15pt bold centered."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, "黑体", "Times New Roman", 15, bold=True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    return p


def add_heading2(doc, text):
    """Chapter heading: 黑体 14pt bold."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, "黑体", "Times New Roman", 14, bold=True)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    return p


def add_heading3(doc, text):
    """Subsection heading: 黑体 12pt bold."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, "黑体", "Times New Roman", 12, bold=True)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    return p


def add_normal(doc, text, indent=True, size=12, cn="宋体"):
    """Normal paragraph."""
    p = doc.add_paragraph()
    _add_text_bold(p, text, cn, "Times New Roman", size)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)
    p.paragraph_format.line_spacing = Pt(20)
    return p


def add_formula(doc, text):
    """Centered italic formula."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, "宋体", "Times New Roman", 11)
    run.italic = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    return p


def parse_md_table(lines):
    """Parse markdown table lines."""
    rows = []
    for line in lines:
        if re.match(r"^\s*\|?\s*[-:|]+\s*\|", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)
    return rows


def add_table(doc, rows):
    """Add a formatted Word table."""
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < ncols:
            r.append("")

    table = doc.add_table(rows=len(rows), cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    try:
        table.style = "Table Grid"
    except Exception:
        pass

    for i, row_data in enumerate(rows):
        for j, cell_text in enumerate(row_data):
            cell = table.cell(i, j)
            p = cell.paragraphs[0]
            p.clear()
            bold = (i == 0)
            run = p.add_run(cell_text)
            _set_run_font(run, "宋体", "Times New Roman", 10, bold=bold)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = None
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
    return table


def add_note(doc, text):
    """Table note: 10pt, no indent."""
    return add_normal(doc, text, indent=False, size=10)


def process_abstract(doc, fp):
    """Process 00_摘要.md."""
    text = fp.read_text("utf-8")
    lines = [l for l in text.split("\n")]

    # Extract parts
    title = ""
    abstract = ""
    keywords = ""
    highlights = []
    mode = None

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("# ") and "Highlights" not in s and "摘要" not in s:
            title = s.lstrip("# ").strip()
            mode = None
            continue
        if "## Highlights" in s:
            mode = "hl"
            continue
        if "## 摘要" in s:
            mode = "abs"
            continue
        if s.startswith("**关键词**"):
            keywords = s.replace("**关键词**：", "").replace("**关键词**:", "").strip()
            mode = None
            continue
        if mode == "hl" and s.startswith("- "):
            highlights.append(s[2:])
        elif mode == "hl" and s.startswith("#"):
            mode = None
        elif mode == "abs" or (mode is None and not s.startswith("#") and not s.startswith("-")):
            if not title:
                title = s
            else:
                abstract += s

    # Write
    add_heading1(doc, "Highlights")
    for h in highlights:
        p = add_normal(doc, "- " + h, indent=False, size=12)

    doc.add_paragraph()
    add_heading1(doc, title)

    # 摘要 label
    p = doc.add_paragraph()
    run = p.add_run("摘要")
    _set_run_font(run, "黑体", "Times New Roman", 12, bold=True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_normal(doc, abstract)

    p = doc.add_paragraph()
    run = p.add_run("关键词：")
    _set_run_font(run, "黑体", "Times New Roman", 12, bold=True)
    run2 = p.add_run(keywords)
    _set_run_font(run2, "宋体", "Times New Roman", 12)
    doc.add_paragraph()


def process_chapter(doc, fp):
    """Process a chapter markdown file."""
    text = fp.read_text("utf-8")
    lines = text.split("\n")
    n = len(lines)
    i = 0

    while i < n:
        s = lines[i].strip()

        if not s:
            i += 1
            continue

        # ## heading
        if s.startswith("## ") and not s.startswith("### "):
            add_heading2(doc, s.lstrip("# ").strip())
            i += 1
            continue

        # ### heading
        if s.startswith("### "):
            add_heading3(doc, s.lstrip("# ").strip())
            i += 1
            continue

        # **bold heading** (standalone)
        if s.startswith("**") and s.endswith("**") and len(s) < 100:
            p = doc.add_paragraph()
            run = p.add_run(s[2:-2])
            _set_run_font(run, "黑体", "Times New Roman", 12, bold=True)
            i += 1
            continue

        # Table: line with | and next line has ---
        if "|" in s:
            next_s = lines[i + 1].strip() if i + 1 < n else ""
            if "---" in next_s or ("---" in s):
                tbl_lines = []
                while i < n and "|" in lines[i]:
                    tbl_lines.append(lines[i])
                    i += 1
                rows = parse_md_table(tbl_lines)
                if rows:
                    add_table(doc, rows)
                continue

        # 注：
        if s.startswith("注：") or s.startswith("注:"):
            add_note(doc, s)
            i += 1
            continue

        # $$ formula
        if s.startswith("$$"):
            formula = s.replace("$$", "").strip()
            if not formula:
                i += 1
                parts = []
                while i < n and not lines[i].strip().startswith("$$"):
                    parts.append(lines[i].strip())
                    i += 1
                formula = " ".join(parts)
                if i < n:
                    i += 1
            add_formula(doc, formula)
            i += 1 if not s.endswith("$$") or formula else 0
            # We already advanced i correctly for single-line formulas
            if s.count("$$") >= 2 and formula:
                # single-line formula, i already at next line from earlier i+=1
                pass
            continue

        # Normal paragraph
        add_normal(doc, s)
        i += 1


def process_references(doc, fp):
    """Process 09_参考文献.md."""
    text = fp.read_text("utf-8")
    add_heading2(doc, "参考文献")
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", s)
        if m:
            p = doc.add_paragraph()
            run = p.add_run(f"[{m.group(1)}] {m.group(2)}")
            _set_run_font(run, "宋体", "Times New Roman", 10.5)
            p.paragraph_format.first_line_indent = None
        else:
            add_normal(doc, s, indent=False, size=10.5)


def process_appendix(doc, fp):
    """Process 附录.md."""
    text = fp.read_text("utf-8")
    lines = text.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("## "):
            add_heading2(doc, s.lstrip("# ").strip())
            i += 1
            continue
        if "|" in s:
            next_s = lines[i + 1].strip() if i + 1 < n else ""
            if "---" in next_s:
                tbl = []
                while i < n and "|" in lines[i]:
                    tbl.append(lines[i])
                    i += 1
                rows = parse_md_table(tbl)
                if rows:
                    add_table(doc, rows)
                continue
        if s.startswith("注：") or s.startswith("注:"):
            add_note(doc, s)
            i += 1
            continue
        add_normal(doc, s, indent=False)
        i += 1


def main():
    print("Creating document...", flush=True)
    doc = Document()
    setup_styles(doc)

    # Remove default empty paragraph
    if doc.paragraphs:
        doc.paragraphs[0]._element.getparent().remove(doc.paragraphs[0]._element)

    print("Processing 00_摘要...", flush=True)
    process_abstract(doc, MD_FILES[0])

    for fp in MD_FILES[1:8]:
        print(f"Processing {fp.stem}...", flush=True)
        process_chapter(doc, fp)

    print("Processing 09_参考文献...", flush=True)
    process_references(doc, MD_FILES[8])

    print("Processing 附录...", flush=True)
    process_appendix(doc, MD_FILES[9])

    print(f"Saving {OUTPUT.name}...", flush=True)
    doc.save(str(OUTPUT))

    # Stats
    d2 = Document(str(OUTPUT))
    print(f"Done: {len(d2.paragraphs)} paragraphs, {len(d2.tables)} tables")


if __name__ == "__main__":
    main()
