"""
Fix subscript formatting in v8 docx:
- DU_kw → DU(italic) + kw(italic subscript)  [no underscore]
- Handle all underscore variable patterns in paragraphs and table cells
- Also handle no-underscore patterns like DUkw in table cells
"""

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import copy
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = f"{BASE}/manuscript/word/数据要素利用与资产定价效率v8.docx"

# ============================================================
# Pattern definitions: (search_string, [(text, italic, subscript), ...])
# IMPORTANT: longest patterns first to avoid partial matches
# ============================================================
PATTERNS = [
    # Compound DU variants (longest first)
    ('DU_kw_strict', [('DU', True, False), ('kw,strict', True, True)]),
    ('DU_kw_adj',    [('DU', True, False), ('kw,adj', True, True)]),
    ('DU_kw_lead',   [('DU', True, False), ('kw,lead', True, True)]),
    ('DU_kw_lag',    [('DU', True, False), ('kw,lag', True, True)]),
    ('DU_kw_peer',   [('DU', True, False), ('kw,peer', True, True)]),
    ('DU_kw_ln',     [('ln(1+DU', True, False), ('kw', True, True), (')', True, False)]),
    ('DU_sub_ln',    [('DU', True, False), ('sub', True, True), ('(ln)', True, False)]),
    # Simple DU variants
    ('DU_kw',        [('DU', True, False), ('kw', True, True)]),
    ('DU_sub',       [('DU', True, False), ('sub', True, True)]),
    # No-underscore variants (for table cells)
    ('DUkw,strict',  [('DU', True, False), ('kw,strict', True, True)]),
    ('DUkw,adj',     [('DU', True, False), ('kw,adj', True, True)]),
    ('DUkw,lead',    [('DU', True, False), ('kw,lead', True, True)]),
    ('DUkw,lag',     [('DU', True, False), ('kw,lag', True, True)]),
    ('DUkw,peer',    [('DU', True, False), ('kw,peer', True, True)]),
    ('DUkw',         [('DU', True, False), ('kw', True, True)]),
    # Other variables
    ('Placebo_kw',   [('Placebo', True, False), ('kw', True, True)]),
    ('PriceDelay_post', [('PriceDelay', True, False), ('post', True, True)]),
    ('Amihud_c',     [('Amihud', True, False), ('c', True, True)]),
    ('Amihudc',      [('Amihud', True, False), ('c', True, True)]),
    ('Bartik_IV',    [('Bartik', True, False), ('IV', True, True)]),
    ('Peer_lag',     [('Peer', True, False), ('lag', True, True)]),
    ('R²_max',       [('R²', True, False), ('max', True, True)]),
    # Log transforms in table cells
    ('ln_total_chars', [('ln(TotalChars)', True, False)]),
    ('ln_mda_chars',   [('ln(MDAChars)', True, False)]),
    # Keyword dimension names
    ('data_stock',   [('data', True, False), ('stock', True, True)]),
    ('data_dev',     [('data', True, False), ('dev', True, True)]),
    ('data_app',     [('data', True, False), ('app', True, True)]),
    ('data_value',   [('data', True, False), ('value', True, True)]),
    ('data_gov',     [('data', True, False), ('gov', True, True)]),
]


def make_run_element(text, rPr_source, italic=None, subscript=False):
    """Create a new w:r element preserving source formatting, overriding italic/subscript."""
    new_run = OxmlElement('w:r')

    if rPr_source is not None:
        new_rPr = copy.deepcopy(rPr_source)
    else:
        new_rPr = OxmlElement('w:rPr')

    # Override italic
    if italic is not None:
        for tag in ('w:i', 'w:iCs'):
            existing = new_rPr.find(qn(tag))
            if italic:
                if existing is None:
                    new_rPr.append(OxmlElement(tag))
            else:
                if existing is not None:
                    new_rPr.remove(existing)

    # Override subscript via w:vertAlign
    vertAlign = new_rPr.find(qn('w:vertAlign'))
    if subscript:
        if vertAlign is not None:
            new_rPr.remove(vertAlign)
        vertAlign = OxmlElement('w:vertAlign')
        vertAlign.set(qn('w:val'), 'subscript')
        new_rPr.append(vertAlign)
    else:
        if vertAlign is not None:
            new_rPr.remove(vertAlign)

    new_run.append(new_rPr)
    t_elem = OxmlElement('w:t')
    t_elem.text = text
    t_elem.set(qn('xml:space'), 'preserve')
    new_run.append(t_elem)

    return new_run


def process_run(run_elem):
    """Replace patterns in a run element with properly formatted runs.
    Returns number of replacements made."""
    t_elem = run_elem.find(qn('w:t'))
    if t_elem is None or not t_elem.text:
        return 0

    text = t_elem.text
    rPr = run_elem.find(qn('w:rPr'))
    count = 0

    for pattern, parts in PATTERNS:
        idx = text.find(pattern)
        if idx == -1:
            continue

        before = text[:idx]
        after = text[idx + len(pattern):]
        parent = run_elem.getparent()

        # Build replacement runs
        new_runs = []
        if before:
            new_runs.append(make_run_element(before, rPr))
        for part_text, is_italic, is_subscript in parts:
            new_runs.append(make_run_element(part_text, rPr,
                                             italic=is_italic, subscript=is_subscript))
        if after:
            new_runs.append(make_run_element(after, rPr))

        # Insert new runs after original, then remove original
        for new_run in reversed(new_runs):
            run_elem.addnext(new_run)
        parent.remove(run_elem)

        count += 1

        # Recursively process ALL new plain-text runs (before & after)
        # Skip the formatted variable parts (they're already done)
        for j, new_run in enumerate(new_runs):
            t = new_run.find(qn('w:t'))
            if t is not None and t.text:
                count += process_run(new_run)

        return count

    return 0


# ============================================================
# Main
# ============================================================
doc = Document(DOC_PATH)
para_fixes = 0
table_fixes = 0

# Process paragraphs
for i, p in enumerate(doc.paragraphs):
    for run_elem in list(p._element.findall(qn('w:r'))):
        n = process_run(run_elem)
        if n:
            para_fixes += n

print(f"Paragraph fixes: {para_fixes}")

# Process table cells
for ti, table in enumerate(doc.tables):
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run_elem in list(p._element.findall(qn('w:r'))):
                    n = process_run(run_elem)
                    if n:
                        table_fixes += n

print(f"Table cell fixes: {table_fixes}")
print(f"Total: {para_fixes + table_fixes}")

doc.save(DOC_PATH)
print(f"\nSaved: {DOC_PATH}")
