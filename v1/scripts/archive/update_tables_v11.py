"""
更新v7.docx中的渠道表(Table 6)和异质性表(Table 7)
渠道: Analyst, Disp, FinAsset (3条)
异质性Panel A: SOE, Amihud, both (3列不变)
异质性Panel B: SOE/Analyst/Size/HighTech (8列)
"""

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import re, json

BASE = "/Users/mac/computerscience/15会计研究"
DOCX = f"{BASE}/manuscript/word/数据要素利用与资产定价效率v7.docx"

doc = Document(DOCX)
body = doc.element.body

# Load results
with open(f"{BASE}/results/v11_expanded/expanded_results.json") as f:
    res = json.load(f)

# ============================================================
# Helper functions (same as before)
# ============================================================

def set_cell_text(cell, text, bold=False, italic=False, font_size=9,
                  alignment=WD_ALIGN_PARAGRAPH.CENTER, font_name='Times New Roman',
                  cn_font='宋体'):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = alignment
    pf = p.paragraph_format
    pf.space_before = Pt(1)
    pf.space_after = Pt(1)
    pf.line_spacing = Pt(12)
    parts = re.split(r'(\*{1,3})', text)
    for part in parts:
        if not part: continue
        run = p.add_run(part)
        run.font.size = Pt(font_size)
        run.bold = bold
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
    tbl = table._tbl
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = parse_xml(f'<w:tblPr {nsdecls("w")}></w:tblPr>')
        tbl.insert(0, tblPr)
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
    for old_borders in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(old_borders)
    tblPr.append(borders)
    n_rows = len(table.rows)
    for cell in table.rows[0].cells:
        set_cell_border(cell, top={"sz": "12", "val": "single", "color": "000000"})
    for cell in table.rows[header_rows - 1].cells:
        set_cell_border(cell, bottom={"sz": "6", "val": "single", "color": "000000"})
    for cell in table.rows[n_rows - 1].cells:
        set_cell_border(cell, bottom={"sz": "12", "val": "single", "color": "000000"})


def clear_table(table):
    tbl = table._tbl
    for tr in tbl.findall(qn('w:tr')):
        tbl.remove(tr)


def add_row(table, n_cols):
    row = table.add_row()
    return row.cells


def fmt_coef(val, sig, decimals=4):
    return f"{val:.{decimals}f}{sig}"

def fmt_se(val, decimals=4):
    return f"({val:.{decimals}f})"


# ============================================================
# Find tables by position
# ============================================================
tables_in_body = []
for child in body:
    tag = child.tag.split('}')[-1]
    if tag == 'tbl':
        tables_in_body.append(child)

print(f"Found {len(tables_in_body)} tables in document")
# [0]=vardef, [1]=desc, [2]=baseline, [3]=robust, [4]=endog, [5]=portfolio, [6]=channel, [7]=het

# ============================================================
# Rebuild Table 6 (Channel Analysis) - now 3 channels
# ============================================================
print("Rebuilding Channel table (3 channels: Analyst, Disp, FinAsset)...")

channel_tbl_elem = tables_in_body[6]
new_channel = doc.add_table(rows=0, cols=7)
new_channel_elem = new_channel._tbl
channel_tbl_elem.addnext(new_channel_elem)
body.remove(channel_tbl_elem)

# Get results
an_c = res['mechanism']['Analyst_concurrent']
di_c = res['mechanism']['Disp_concurrent']
fa_c = res['mechanism']['FinAsset_concurrent']
an_l = res['mechanism']['Analyst_lagged']
di_l = res['mechanism']['Disp_lagged']
fa_l = res['mechanism']['FinAsset_lagged']

channel_data = [
    ['', '(1)', '(2)', '(3)', '(4)', '(5)', '(6)'],
    ['被解释变量=', 'Analystt', 'Dispt', 'FinAssett', 'Analystt+1', 'Dispt+1', 'FinAssett+1'],
    # Panel A
    ['Panel A: 同期回归', '', '', '', '', '', ''],
    ['DUkw', fmt_coef(an_c['coef'], an_c['sig']), fmt_coef(di_c['coef'], di_c['sig']),
     fmt_coef(fa_c['coef'], fa_c['sig']), '', '', ''],
    ['', fmt_se(an_c['se']), fmt_se(di_c['se']), fmt_se(fa_c['se']), '', '', ''],
    ['N', f"{an_c['N']:,}", f"{di_c['N']:,}", f"{fa_c['N']:,}", '', '', ''],
    ['R²', f"{an_c['R2']:.3f}", f"{di_c['R2']:.3f}", f"{fa_c['R2']:.3f}", '', '', ''],
    # Panel B
    ['Panel B: 滞后一期', '', '', '', '', '', ''],
    ['DUkw', '', '', '', fmt_coef(an_l['coef'], an_l['sig']),
     fmt_coef(di_l['coef'], di_l['sig']), fmt_coef(fa_l['coef'], fa_l['sig'])],
    ['', '', '', '', fmt_se(an_l['se']), fmt_se(di_l['se']), fmt_se(fa_l['se'])],
    ['N', '', '', '', f"{an_l['N']:,}", f"{di_l['N']:,}", f"{fa_l['N']:,}"],
    ['R²', '', '', '', f"{an_l['R2']:.3f}", f"{di_l['R2']:.3f}", f"{fa_l['R2']:.3f}"],
    # Bottom
    ['Controls', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES'],
    ['Firm/Year FE', 'YES', 'YES', 'YES', 'YES', 'YES', 'YES'],
]

clear_table(new_channel)
for i, row_data in enumerate(channel_data):
    cells = add_row(new_channel, 7)
    for j, val in enumerate(row_data):
        is_header = i < 2
        is_first = j == 0
        alignment = WD_ALIGN_PARAGRAPH.LEFT if is_first else WD_ALIGN_PARAGRAPH.CENTER
        should_italic = (is_first and not is_header
                        and any(c.isalpha() for c in str(val))
                        and not any('\u4e00' <= c <= '\u9fff' for c in str(val)))
        set_cell_text(cells[j], str(val), bold=is_header, italic=should_italic,
                     alignment=alignment)

apply_three_line_style(new_channel, header_rows=2)
for cell in new_channel.rows[2].cells:
    set_cell_border(cell, top={"sz": "4", "val": "single", "color": "000000"})
for cell in new_channel.rows[7].cells:
    set_cell_border(cell, top={"sz": "4", "val": "single", "color": "000000"})


# ============================================================
# Rebuild Table 7 (Heterogeneity) - expanded Panel B
# ============================================================
print("Rebuilding Heterogeneity table (expanded Panel B)...")

# Re-find tables after channel rebuild
tables_in_body = []
for child in body:
    tag = child.tag.split('}')[-1]
    if tag == 'tbl':
        tables_in_body.append(child)

het_tbl_elem = tables_in_body[7]
# Panel B now has 8 columns: SOE/NonSOE/HighAnalyst/LowAnalyst/Large/Small/HighTech/Trad
# Total columns needed = max(Panel A: 4, Panel B: 9 including label) = 9
new_het = doc.add_table(rows=0, cols=9)
new_het_elem = new_het._tbl
het_tbl_elem.addnext(new_het_elem)
body.remove(het_tbl_elem)

# Panel A interaction data (use first 4 cols + leave rest empty)
ia = res['interact']
# col1: SOE, col2: Amihud, col3: both (SOE+Amihud)
het_data = [
    # Panel A header
    ['Panel A: 调节效应', '', '', '', '', '', '', '', ''],
    ['', '(1)', '', '(2)', '', '(3)', '', '', ''],
    ['被解释变量=', 'PriceDelay', '', 'PriceDelay', '', 'PriceDelay', '', '', ''],
    ['DUkw',
     fmt_coef(ia['col1']['DU_kw']['coef'], ia['col1']['DU_kw']['sig']), '',
     fmt_coef(ia['col2']['DU_kw']['coef'], ia['col2']['DU_kw']['sig']), '',
     fmt_coef(ia['col3']['DU_kw']['coef'], ia['col3']['DU_kw']['sig']), '', '', ''],
    ['',
     fmt_se(ia['col1']['DU_kw']['se']), '',
     fmt_se(ia['col2']['DU_kw']['se']), '',
     fmt_se(ia['col3']['DU_kw']['se']), '', '', ''],
    ['DUkw × SOE',
     fmt_coef(ia['col1']['DU_SOE']['coef'], ia['col1']['DU_SOE']['sig']), '',
     '', '',
     fmt_coef(ia['col3']['DU_SOE']['coef'], ia['col3']['DU_SOE']['sig']), '', '', ''],
    ['',
     fmt_se(ia['col1']['DU_SOE']['se']), '',
     '', '',
     fmt_se(ia['col3']['DU_SOE']['se']), '', '', ''],
    ['DUkw × Amihudc',
     '', '',
     fmt_coef(ia['col2']['DU_Amihud']['coef'], ia['col2']['DU_Amihud']['sig']), '',
     fmt_coef(ia['col3b']['DU_Amihud']['coef'], ia['col3b']['DU_Amihud']['sig']), '', '', ''],
    ['',
     '', '',
     fmt_se(ia['col2']['DU_Amihud']['se']), '',
     fmt_se(ia['col3b']['DU_Amihud']['se']), '', '', ''],
    ['Controls', 'YES', '', 'YES', '', 'YES', '', '', ''],
    ['Firm/Year FE', 'YES', '', 'YES', '', 'YES', '', '', ''],
    ['N', f"{ia['col1']['N']:,}", '', f"{ia['col2']['N']:,}", '', f"{ia['col3']['N']:,}", '', '', ''],
    ['R²', f"{ia['col1']['R2']:.3f}", '', f"{ia['col2']['R2']:.3f}", '', f"{ia['col3']['R2']:.3f}", '', '', ''],
    # Panel B
    ['Panel B: 分组回归', '', '', '', '', '', '', '', ''],
    ['', '(1) 国有', '(2) 非国有', '(3) 高分析师\n覆盖', '(4) 低分析师\n覆盖',
     '(5) 大企业', '(6) 小企业', '(7) 高科技', '(8) 传统'],
    ['被解释变量=', 'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay',
     'PriceDelay', 'PriceDelay', 'PriceDelay', 'PriceDelay'],
]

# Add subgroup results
sg = res['subgroup']
groups = ['国有', '非国有', '高分析师覆盖', '低分析师覆盖', '大企业', '小企业', '高科技行业', '传统行业']
coef_row = ['DUkw']
se_row = ['']
n_row = ['N']
r2_row = ['R²']
ctrl_row = ['Controls']
fe_row = ['Firm/Year FE']
for g in groups:
    coef_row.append(fmt_coef(sg[g]['coef'], sg[g]['sig']))
    se_row.append(fmt_se(sg[g]['se']))
    n_row.append(f"{sg[g]['N']:,}")
    r2_row.append(f"{sg[g]['R2']:.3f}")
    ctrl_row.append('YES')
    fe_row.append('YES')

het_data.extend([coef_row, se_row, ctrl_row, fe_row, n_row, r2_row])

clear_table(new_het)
for i, row_data in enumerate(het_data):
    cells = add_row(new_het, 9)
    for j in range(min(len(row_data), 9)):
        is_header = i in [0, 1, 2, 13, 14, 15]
        is_first = j == 0
        alignment = WD_ALIGN_PARAGRAPH.LEFT if is_first else WD_ALIGN_PARAGRAPH.CENTER
        should_italic = (is_first and not is_header
                        and any(c.isalpha() for c in str(row_data[j]))
                        and not any('\u4e00' <= c <= '\u9fff' for c in str(row_data[j])))
        set_cell_text(cells[j], str(row_data[j]), bold=is_header, italic=should_italic,
                     alignment=alignment, font_size=8)

apply_three_line_style(new_het, header_rows=1)
# Panel B separator
for cell in new_het.rows[13].cells:
    set_cell_border(cell, top={"sz": "6", "val": "single", "color": "000000"})

# ============================================================
# Save
# ============================================================
print("Saving...")
doc.save(DOCX)
print("Done!")
