"""
重建v7.docx中的所有表格，使其与v10 PDF表格完全一致。
"""

from docx import Document
from docx.shared import Pt, Cm, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
from copy import deepcopy
import re

INPUT = 'manuscript/word/数据要素利用与资产定价效率v7.docx'
OUTPUT = 'manuscript/word/数据要素利用与资产定价效率v7.docx'

doc = Document(INPUT)

# ============================================================
# Helper functions
# ============================================================

def set_cell_text(cell, text, bold=False, italic=False, font_size=9,
                  alignment=WD_ALIGN_PARAGRAPH.CENTER, font_name='Times New Roman',
                  cn_font='宋体'):
    """Set cell text with formatting."""
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = alignment
    # Set paragraph spacing
    pf = p.paragraph_format
    pf.space_before = Pt(1)
    pf.space_after = Pt(1)
    pf.line_spacing = Pt(12)

    # Parse text for superscript stars
    # Split on *** ** * patterns
    parts = re.split(r'(\*{1,3})', text)
    for part in parts:
        if not part:
            continue
        run = p.add_run(part)
        run.font.size = Pt(font_size)
        run.bold = bold
        # Detect if text contains Chinese
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in part)
        if has_chinese:
            run.font.name = cn_font
            run._element.rPr.rFonts.set(qn('w:eastAsia'), cn_font)
        else:
            run.font.name = font_name
        run.italic = italic
        if part in ('*', '**', '***'):
            run.font.superscript = True


def set_cell_border(cell, **kwargs):
    """Set cell borders. kwargs: top, bottom, left, right, each is dict with sz, val, color."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}></w:tcBorders>')
    for edge, attrs in kwargs.items():
        element = parse_xml(
            f'<w:{edge} {nsdecls("w")} w:val="{attrs.get("val", "single")}" '
            f'w:sz="{attrs.get("sz", "4")}" w:space="0" '
            f'w:color="{attrs.get("color", "000000")}"/>'
        )
        tcBorders.append(element)
    tcPr.append(tcBorders)


def apply_three_line_style(table, header_rows=1):
    """Apply three-line table style (三线表)."""
    # Remove all default borders first
    tbl = table._tbl
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = parse_xml(f'<w:tblPr {nsdecls("w")}></w:tblPr>')
        tbl.insert(0, tblPr)

    # Set no borders by default
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'  <w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f'</w:tblBorders>'
    )
    # Remove existing borders
    for old_borders in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(old_borders)
    tblPr.append(borders)

    n_rows = len(table.rows)
    n_cols = len(table.columns)

    # Top border on first row (thick)
    for cell in table.rows[0].cells:
        set_cell_border(cell, top={"sz": "12", "val": "single", "color": "000000"})

    # Bottom border on header row
    for cell in table.rows[header_rows - 1].cells:
        set_cell_border(cell, bottom={"sz": "6", "val": "single", "color": "000000"})

    # Bottom border on last row (thick)
    for cell in table.rows[n_rows - 1].cells:
        set_cell_border(cell, bottom={"sz": "12", "val": "single", "color": "000000"})


def clear_table(table):
    """Remove all rows from a table."""
    tbl = table._tbl
    for tr in tbl.findall(qn('w:tr')):
        tbl.remove(tr)


def add_row(table, n_cols):
    """Add a row to the table and return cells."""
    row = table.add_row()
    return row.cells


def build_table_from_data(table, data, header_rows=1, col_widths=None,
                          italic_col0=True, bold_header=True):
    """Build table from 2D data list. First `header_rows` rows are headers."""
    clear_table(table)

    for i, row_data in enumerate(data):
        cells = add_row(table, len(row_data))
        for j, val in enumerate(row_data):
            is_header = i < header_rows
            is_first_col = j == 0
            alignment = WD_ALIGN_PARAGRAPH.LEFT if is_first_col else WD_ALIGN_PARAGRAPH.CENTER
            # Variable names in first col should be italic (for English var names)
            should_italic = (italic_col0 and is_first_col and not is_header
                           and any(c.isalpha() for c in str(val))
                           and not any('\u4e00' <= c <= '\u9fff' for c in str(val)))
            set_cell_text(cells[j], str(val),
                         bold=is_header and bold_header,
                         italic=should_italic,
                         alignment=alignment)

    apply_three_line_style(table, header_rows)

    # Set column widths if provided
    if col_widths:
        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                if j < len(col_widths):
                    cell.width = Cm(col_widths[j])


def insert_table_after(doc, ref_element, n_rows, n_cols):
    """Insert a new table after a given element in the document body."""
    body = doc.element.body
    new_tbl = doc.add_table(rows=n_rows, cols=n_cols)
    # Move it after ref_element
    tbl_element = new_tbl._tbl
    body.remove(tbl_element)
    ref_element.addnext(tbl_element)
    return new_tbl


def add_panel_separator(table, text, n_cols, bold=True):
    """Add a panel header row (e.g., 'Panel A: xxx')."""
    cells = add_row(table, n_cols)
    # Merge all cells
    cells[0].merge(cells[n_cols - 1])
    set_cell_text(cells[0], text, bold=bold, alignment=WD_ALIGN_PARAGRAPH.LEFT)


# ============================================================
# Table 1: Descriptive Statistics (PDF Table 1)
# ============================================================

print("Rebuilding Table 1: Descriptive Statistics...")

table1_data = [
    ['变量', 'N', 'Mean', 'SD', 'Min', 'Median', 'Max'],
    ['PriceDelay', '43,843', '0.111', '0.123', '0.005', '0.068', '0.665'],
    ['DUkw', '43,843', '1.223', '1.863', '0.000', '0.575', '11.064'],
    ['ln(1+DUkw)', '43,843', '2.552', '1.272', '0.000', '2.565', '5.568'],
    ['Size', '43,843', '22.595', '0.969', '20.948', '22.433', '25.608'],
    ['Lev', '43,843', '0.424', '0.207', '0.055', '0.416', '0.916'],
    ['ROA', '43,843', '0.030', '0.067', '-0.280', '0.033', '0.191'],
    ['TobinQ', '43,843', '2.007', '1.289', '0.829', '1.589', '8.451'],
    ['Age', '43,843', '2.154', '0.739', '0.693', '2.303', '3.219'],
    ['Growth', '43,843', '0.134', '0.370', '-0.591', '0.083', '2.157'],
    ['BoardSize', '43,843', '2.110', '0.198', '1.609', '2.197', '2.639'],
    ['IndepRatio', '43,843', '0.378', '0.054', '0.333', '0.364', '0.571'],
    ['Dual', '43,843', '0.298', '0.457', '0.000', '0.000', '1.000'],
    ['Top1Share', '43,843', '33.432', '14.832', '8.126', '31.010', '74.000'],
    ['SOE', '43,843', '0.329', '0.470', '0.000', '0.000', '1.000'],
    ['InstHold', '43,843', '42.311', '24.601', '0.350', '43.462', '90.577'],
    ['Amihud', '43,843', '0.046', '0.048', '0.002', '0.032', '0.302'],
    ['Analyst', '43,843', '2.464', '1.711', '0.000', '2.639', '5.656'],
    ['AuditType', '43,843', '0.966', '0.182', '0.000', '1.000', '1.000'],
]

build_table_from_data(doc.tables[1], table1_data, header_rows=1,
                      col_widths=[2.5, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8])

# ============================================================
# Table 2: Baseline Regression (PDF Table 2)
# ============================================================

print("Rebuilding Table 2: Baseline Regression...")

# Build the complex regression table
table2_data = [
    # Header rows
    ['', '(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)'],
    ['被解释变量=', 'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay', 'SYNCH', 'PriceDelay'],
    # DUkw
    ['DUkw', '-0.005***', '-0.004***', '', '', '-0.002***', '0.011*', '-0.004***'],
    ['', '(0.001)', '(0.001)', '', '', '(0.0005)', '(0.006)', '(0.001)'],
    # ln(1+DUkw)
    ['ln(1+DUkw)', '', '', '-0.004***', '', '', '', ''],
    ['', '', '', '(0.001)', '', '', '', ''],
    # DUsub(ln)
    ['DUsub(ln)', '', '', '', '-0.003***', '', '', ''],
    ['', '', '', '', '(0.001)', '', '', ''],
    # FinAsset
    ['FinAsset', '', '', '', '', '', '', '-0.015'],
    ['', '', '', '', '', '', '', '(0.010)'],
    # Size
    ['Size', '', '0.020***', '0.020***', '0.020***', '0.009***', '-0.318***', '0.020***'],
    ['', '', '(0.003)', '(0.003)', '(0.003)', '(0.001)', '(0.020)', '(0.003)'],
    # Lev
    ['Lev', '', '0.023***', '0.023***', '0.023***', '0.018***', '-0.174***', '0.022***'],
    ['', '', '(0.006)', '(0.006)', '(0.006)', '(0.003)', '(0.040)', '(0.006)'],
    # ROA
    ['ROA', '', '-0.037**', '-0.035**', '-0.035**', '-0.069***', '0.240**', '-0.038**'],
    ['', '', '(0.015)', '(0.015)', '(0.015)', '(0.013)', '(0.100)', '(0.015)'],
    # TobinQ
    ['TobinQ', '', '0.010***', '0.010***', '0.010***', '0.011***', '-0.077***', '0.010***'],
    ['', '', '(0.001)', '(0.001)', '(0.001)', '(0.001)', '(0.007)', '(0.001)'],
    # Age
    ['Age', '', '-0.026***', '-0.028***', '-0.027***', '-0.014***', '0.045', '-0.026***'],
    ['', '', '(0.004)', '(0.004)', '(0.004)', '(0.001)', '(0.028)', '(0.004)'],
    # Growth
    ['Growth', '', '0.010***', '0.010***', '0.010***', '0.015***', '-0.069***', '0.010***'],
    ['', '', '(0.002)', '(0.002)', '(0.002)', '(0.002)', '(0.012)', '(0.002)'],
    # BoardSize
    ['BoardSize', '', '-0.014**', '-0.013**', '-0.013**', '-0.011***', '0.133***', '-0.014**'],
    ['', '', '(0.006)', '(0.006)', '(0.006)', '(0.003)', '(0.038)', '(0.006)'],
    # IndepRatio
    ['IndepRatio', '', '-0.036**', '-0.034*', '-0.034*', '-0.031***', '0.241**', '-0.036**'],
    ['', '', '(0.018)', '(0.018)', '(0.018)', '(0.011)', '(0.121)', '(0.018)'],
    # Dual
    ['Dual', '', '-0.002', '-0.002', '-0.002', '-0.002*', '0.014', '-0.002'],
    ['', '', '(0.002)', '(0.002)', '(0.002)', '(0.001)', '(0.012)', '(0.002)'],
    # Top1Share
    ['Top1Share', '', '-0.0001', '-0.0001', '-0.0001', '-0.0001*', '0.0020***', '-0.0001'],
    ['', '', '(0.0001)', '(0.0001)', '(0.0001)', '(0.0000)', '(0.0007)', '(0.0001)'],
    # SOE
    ['SOE', '', '0.0002', '0.0000', '0.0001', '-0.0097***', '0.0580**', '0.0002'],
    ['', '', '(0.0040)', '(0.0040)', '(0.0040)', '(0.0015)', '(0.0254)', '(0.0039)'],
    # InstHold
    ['InstHold', '', '0.0002**', '0.0002***', '0.0002***', '0.0001***', '-0.0027***', '0.0002**'],
    ['', '', '(0.0001)', '(0.0001)', '(0.0001)', '(0.0000)', '(0.0006)', '(0.0001)'],
    # Amihud
    ['Amihud', '', '0.146***', '0.148***', '0.149***', '0.179***', '-1.325***', '0.146***'],
    ['', '', '(0.023)', '(0.022)', '(0.022)', '(0.022)', '(0.144)', '(0.023)'],
    # Analyst
    ['Analyst', '', '-0.010***', '-0.010***', '-0.010***', '-0.009***', '0.089***', '-0.010***'],
    ['', '', '(0.001)', '(0.001)', '(0.001)', '(0.001)', '(0.006)', '(0.001)'],
    # AuditType
    ['AuditType', '', '-0.014***', '-0.014***', '-0.014***', '-0.023***', '0.085***', '-0.014***'],
    ['', '', '(0.004)', '(0.004)', '(0.004)', '(0.004)', '(0.023)', '(0.004)'],
    # Bottom rows
    ['Controls', 'NO', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES'],
    ['Firm/Year FE', 'YES', 'YES', 'YES', 'YES', 'NO', 'YES', 'YES'],
    ['Ind/Year FE', 'NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO'],
    ['N', '43,843', '43,843', '43,843', '43,843', '43,856', '34,062', '43,843'],
    ['R²', '0.409', '0.425', '0.425', '0.425', '0.318', '0.616', '0.425'],
]

# Table 2 needs 8 columns (currently has 6). Need to rebuild.
# Remove old table and insert new one
body = doc.element.body
old_tbl2 = doc.tables[2]._tbl
# Create new table
new_tbl2 = doc.add_table(rows=0, cols=8)
new_tbl2_element = new_tbl2._tbl
# Move new table to where old table was
old_tbl2.addnext(new_tbl2_element)
body.remove(old_tbl2)
# Remove the auto-added table at end
# (it was added at the end of doc, we moved it)

build_table_from_data(new_tbl2, table2_data, header_rows=2,
                      col_widths=[2.2, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8])

# ============================================================
# Table 3: Robustness (PDF Table 3)
# Need to replace existing Table 3 (which was simplified)
# ============================================================

print("Rebuilding Table 3: Robustness...")

table3_data = [
    # Header row 1
    ['', '(1) 控制\nInd×Year', '(2) 控制\nProv×Year', '(3) 行业均\n值调整',
     '(4) 剔除\n2024年', '(5) 剔除信息\n技术业', '(6) 仅\n主板',
     '(7) 倾向得\n分匹配', '(8) 双向\n聚类SE', '(9) 前导项\n检验'],
    # Header row 2
    ['被解释变量=', 'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay',
     'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay'],
    # Variable row
    ['', 'DUkw', 'DUkw', 'DUkw,adj', 'DUkw', 'DUkw', 'DUkw', 'DUkw', 'DUkw', 'DUkw'],
    # Coefficients
    ['', '-0.0022***', '-0.004***', '-0.0020***', '-0.005***', '-0.004***',
     '-0.003**', '-0.004***', '-0.004***', '-0.005***'],
    # SE
    ['', '(0.0006)', '(0.001)', '(0.0006)', '(0.001)', '(0.001)',
     '(0.001)', '(0.001)', '(0.001)', '(0.001)'],
    # DUkw,lead (only model 9)
    ['', '', '', '', '', '', '', '', '', 'DUkw,lead'],
    ['', '', '', '', '', '', '', '', '', '0.0005'],
    ['', '', '', '', '', '', '', '', '', '(0.0010)'],
    # Bottom rows
    ['Controls', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES'],
    ['Firm FE', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES'],
    ['Year FE', 'NO', 'NO', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES'],
    ['Ind×Year FE', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
    ['Prov×Year FE', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],
    ['N', '43,778', '43,842', '43,843', '38,916', '40,720', '18,837', '34,425', '43,843', '38,567'],
    ['R²', '0.478', '0.435', '0.424', '0.434', '0.425', '0.417', '0.434', '0.425', '0.434'],
]

# Replace old Table 3 with new 10-column table
old_tbl3 = doc.tables[3]._tbl
new_tbl3 = doc.add_table(rows=0, cols=10)
new_tbl3_element = new_tbl3._tbl
old_tbl3.addnext(new_tbl3_element)
body.remove(old_tbl3)

build_table_from_data(new_tbl3, table3_data, header_rows=2,
                      col_widths=[2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

# ============================================================
# NEW: Insert Table 4 (Endogeneity) after Table 3
# ============================================================

print("Inserting Table 4: Endogeneity...")

# We need to add a new table for endogeneity.
# First, add a paragraph for the table title before the table
# Find the element after where table 3 now is
# The new table 3 is now at the position of old table 3
# We need to insert Table 4 right after new Table 3

# Create a paragraph for spacing/title
new_para = parse_xml(
    f'<w:p {nsdecls("w")}>'
    f'</w:p>'
)
new_tbl3_element.addnext(new_para)

# Create Table 4
new_tbl4 = doc.add_table(rows=0, cols=6)
new_tbl4_element = new_tbl4._tbl
new_para.addnext(new_tbl4_element)

# Build Table 4 data
# Panel A: First Stage
table4_data = [
    ['Panel A: First Stage', '', '', '', '', ''],
    ['', '', '(2) Peer IV', '', '(3) Bartik IV', '(5) Lag IV'],
    ['被解释变量=', '', 'DUkw', '', 'DUkw', 'DUkw,lag'],
    ['', '', 'DUkw,peer', 'BartikIV', '', 'Peerlag'],
    ['', '', '0.5970***', '0.2889***', '', '0.6018***'],
    ['', '', '(0.0367)', '(0.0365)', '', '(0.0444)'],
    ['Controls', '', 'YES', '', 'YES', 'YES'],
    ['Firm/Year', '', 'YES', '', 'YES', 'YES'],
    ['N', '', '43,779', '', '43,058', '37,340'],
    ['R²', '', '—', '', '—', '—'],
    # Separator
    ['Panel B: Second Stage', '', '', '', '', ''],
    ['', '(1) OLS', '(2) Peer IV', '(3) Bartik IV', '(4) Lag OLS', '(5) Lag IV'],
    ['被解释变量=', 'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay'],
    ['DUkw', '-0.0042***', '-0.0147***', '-0.0112*', '-0.0036***', '-0.0165***'],
    ['', '(0.0009)', '(0.0046)', '(0.0064)', '(0.0010)', '(0.0053)'],
    ['Controls', 'YES', 'YES', 'YES', 'YES', 'YES'],
    ['Firm/Year', 'YES', 'YES', 'YES', 'YES', 'YES'],
    ['KP F', '—', '265.1', '62.5', '—', '183.5'],
    ['DWH p', '—', '0.009', '0.268', '—', '0.004'],
    ['N', '43,843', '43,779', '43,058', '37,404', '37,340'],
    ['R²', '0.425', '—', '—', '—', '—'],
]

build_table_from_data(new_tbl4, table4_data, header_rows=1,
                      col_widths=[2.5, 2.2, 2.2, 2.2, 2.2, 2.2])

# Add extra border for Panel B separator
# Row 10 is "Panel B" header - add top border
if len(new_tbl4.rows) > 10:
    for cell in new_tbl4.rows[10].cells:
        set_cell_border(cell, top={"sz": "6", "val": "single", "color": "000000"})

# ============================================================
# Table 5 (was Table 4): Portfolio Alpha → PDF 附表1
# Now doc.tables[4] - but indices may have shifted due to insertions
# ============================================================

print("Rebuilding Portfolio Alpha table (附表1)...")

# After inserting new table, we need to re-find tables
# The portfolio table is the 5th table in the original (index 4)
# But we've added one table, so it's now at a different index
# Let's find it by position in the body

# Re-enumerate tables
tables_in_body = []
for child in body:
    tag = child.tag.split('}')[-1]
    if tag == 'tbl':
        tables_in_body.append(child)

# tables_in_body mapping:
# [0] = Table 0 (variable definitions) - keep
# [1] = Table 1 (descriptive stats) - rebuilt
# [2] = NEW Table 2 (baseline regression) - rebuilt
# [3] = NEW Table 3 (robustness) - rebuilt
# [4] = NEW Table 4 (endogeneity) - inserted
# [5] = OLD Table 4 (portfolio alpha) - need to rebuild as 附表1
# [6] = OLD Table 5 (channel analysis) - need to rebuild as Table 5
# [7] = OLD Table 6 (heterogeneity) - need to rebuild as Table 6

# Portfolio alpha table (was index 4, now at body position 5)
# Find the table object
portfolio_tbl_elem = tables_in_body[5]

# Create new portfolio table
portfolio_data = [
    ['Portfolio', '', 'CAPM', '', 'FF3', '', 'FF5+MOM', ''],
    ['', 'alpha', 't', 'alpha', 't', 'alpha', 't', ''],
    ['Q1', '0.25', '0.81', '-0.28**', '-2.08', '-0.64***', '-3.81', ''],
    ['Q2', '0.27', '0.84', '-0.17', '-1.58', '-0.55***', '-3.64', ''],
    ['Q3', '0.37', '1.16', '-0.04', '-0.43', '-0.44***', '-3.40', ''],
    ['Q4', '0.45', '1.23', '0.08', '0.81', '-0.33**', '-2.31', ''],
    ['Q5', '0.54', '1.04', '0.09', '0.38', '-0.41', '-1.60', ''],
    ['DAT', '0.29', '0.77', '0.37', '1.17', '0.23', '0.71', ''],
]

# Actually the PDF has 7 columns: Portfolio, alpha, t, alpha, t, alpha, t
portfolio_data = [
    ['Portfolio', 'CAPM', '', 'FF3', '', 'FF5+MOM', ''],
    ['', 'alpha', 't', 'alpha', 't', 'alpha', 't'],
    ['Q1', '0.25', '0.81', '-0.28**', '-2.08', '-0.64***', '-3.81'],
    ['Q2', '0.27', '0.84', '-0.17', '-1.58', '-0.55***', '-3.64'],
    ['Q3', '0.37', '1.16', '-0.04', '-0.43', '-0.44***', '-3.40'],
    ['Q4', '0.45', '1.23', '0.08', '0.81', '-0.33**', '-2.31'],
    ['Q5', '0.54', '1.04', '0.09', '0.38', '-0.41', '-1.60'],
    ['DAT', '0.29', '0.77', '0.37', '1.17', '0.23', '0.71'],
]

new_portfolio_tbl = doc.add_table(rows=0, cols=7)
new_portfolio_elem = new_portfolio_tbl._tbl
portfolio_tbl_elem.addnext(new_portfolio_elem)
body.remove(portfolio_tbl_elem)

build_table_from_data(new_portfolio_tbl, portfolio_data, header_rows=2,
                      col_widths=[2.0, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8])


# ============================================================
# Table 5 (was Table 5): Channel Analysis → PDF Table 5
# ============================================================

print("Rebuilding Table 5: Channel Analysis...")

# Re-enumerate
tables_in_body = []
for child in body:
    tag = child.tag.split('}')[-1]
    if tag == 'tbl':
        tables_in_body.append(child)

channel_tbl_elem = tables_in_body[6]

channel_data = [
    ['', '(1)', '(2)', '(3)', '(4)'],
    ['被解释变量=', 'Analystt', 'Dispt', 'Analystt+1', 'Dispt+1'],
    # Panel A
    ['Panel A: 同期回归', '', '', '', ''],
    ['DUkw', '0.0255***', '-0.0064***', '', ''],
    ['', '(0.0081)', '(0.0015)', '', ''],
    ['N', '43,843', '25,092', '', ''],
    ['R²', '0.794', '0.490', '', ''],
    # Panel B
    ['Panel B: 滞后一期', '', '', '', ''],
    ['DUkw', '', '', '0.0288***', '-0.0054***'],
    ['', '', '', '(0.0077)', '(0.0017)'],
    ['N', '', '', '37,404', '21,160'],
    ['R²', '', '', '0.824', '0.513'],
    # Bottom
    ['Controls', 'YES', 'YES', 'YES', 'YES'],
    ['Firm/Year FE', 'YES', 'YES', 'YES', 'YES'],
]

new_channel_tbl = doc.add_table(rows=0, cols=5)
new_channel_elem = new_channel_tbl._tbl
channel_tbl_elem.addnext(new_channel_elem)
body.remove(channel_tbl_elem)

build_table_from_data(new_channel_tbl, channel_data, header_rows=2,
                      col_widths=[2.5, 2.5, 2.5, 2.5, 2.5])

# Add panel borders
for cell in new_channel_tbl.rows[2].cells:
    set_cell_border(cell, top={"sz": "4", "val": "single", "color": "000000"})
for cell in new_channel_tbl.rows[7].cells:
    set_cell_border(cell, top={"sz": "4", "val": "single", "color": "000000"})


# ============================================================
# Table 6: Heterogeneity → PDF Table 6
# ============================================================

print("Rebuilding Table 6: Heterogeneity...")

tables_in_body = []
for child in body:
    tag = child.tag.split('}')[-1]
    if tag == 'tbl':
        tables_in_body.append(child)

het_tbl_elem = tables_in_body[7]

het_data = [
    # Panel A header
    ['Panel A: 调节效应', '', '', ''],
    ['', '(1)', '(2)', '(3)'],
    ['被解释变量=', 'PriceDelay', 'PriceDelay', 'PriceDelay'],
    ['DUkw', '-0.005***', '-0.004***', '-0.004***'],
    ['', '(0.001)', '(0.001)', '(0.001)'],
    ['DUkw × SOE', '0.001', '', '0.002*'],
    ['', '(0.001)', '', '(0.001)'],
    ['DUkw × Amihudc', '', '0.025**', '0.026**'],
    ['', '', '(0.010)', '(0.010)'],
    ['Controls', 'YES', 'YES', 'YES'],
    ['Firm/Year FE', 'YES', 'YES', 'YES'],
    ['N', '43,843', '43,843', '43,843'],
    ['R²', '0.425', '0.425', '0.425'],
    # Panel B
    ['Panel B: 分组回归', '', '', ''],
    ['', '(1) 国有', '(2) 非国有', '(3) 高分析师覆盖', '(4) 低分析师覆盖'],
    ['被解释变量=', 'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay'],
    ['DUkw', '-0.006***', '-0.003***', '-0.006***', '-0.003***'],
    ['', '(0.001)', '(0.001)', '(0.001)', '(0.001)'],
    ['Controls', 'YES', 'YES', 'YES', 'YES'],
    ['Firm/Year FE', 'YES', 'YES', 'YES', 'YES'],
    ['N', '14,412', '29,369', '21,836', '21,457'],
    ['R²', '0.418', '0.439', '0.422', '0.483'],
]

# Panel A has 4 cols, Panel B has 5 cols - use 5 cols total
new_het_tbl = doc.add_table(rows=0, cols=5)
new_het_elem = new_het_tbl._tbl
het_tbl_elem.addnext(new_het_elem)
body.remove(het_tbl_elem)

# Build manually since Panel A and B have different column counts
clear_table(new_het_tbl)

for i, row_data in enumerate(het_data):
    cells = add_row(new_het_tbl, 5)
    # Panel A rows (0-12) only use 4 columns, merge last 2
    if i < 13:
        # Only fill first 4 columns
        for j in range(min(len(row_data), 4)):
            is_header = (i in [0, 1, 2])
            is_first_col = j == 0
            alignment = WD_ALIGN_PARAGRAPH.LEFT if is_first_col else WD_ALIGN_PARAGRAPH.CENTER
            should_italic = (is_first_col and not is_header
                           and any(c.isalpha() for c in str(row_data[j]))
                           and not any('\u4e00' <= c <= '\u9fff' for c in str(row_data[j])))
            set_cell_text(cells[j], str(row_data[j]),
                         bold=(i in [0, 1, 2]),
                         italic=should_italic,
                         alignment=alignment)
        # Leave cell 4 empty
        set_cell_text(cells[4], '', alignment=WD_ALIGN_PARAGRAPH.CENTER)
    else:
        # Panel B rows use all 5 columns
        for j in range(min(len(row_data), 5)):
            is_header = (i in [13, 14, 15])
            is_first_col = j == 0
            alignment = WD_ALIGN_PARAGRAPH.LEFT if is_first_col else WD_ALIGN_PARAGRAPH.CENTER
            should_italic = (is_first_col and not is_header
                           and any(c.isalpha() for c in str(row_data[j]))
                           and not any('\u4e00' <= c <= '\u9fff' for c in str(row_data[j])))
            set_cell_text(cells[j], str(row_data[j]),
                         bold=(i in [13, 14, 15]),
                         italic=should_italic,
                         alignment=alignment)

apply_three_line_style(new_het_tbl, header_rows=1)

# Add panel B separator border
for cell in new_het_tbl.rows[13].cells:
    set_cell_border(cell, top={"sz": "6", "val": "single", "color": "000000"})


# ============================================================
# Save
# ============================================================

print("Saving...")
doc.save(OUTPUT)
print(f"Done! Saved to {OUTPUT}")
