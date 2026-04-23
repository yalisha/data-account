"""
1. 修复五、作用渠道分析开头：去掉问句，加过渡语+机制模型公式
2. 修复六、异质性分析开头：改善过渡语+加交互项模型公式
处理顺序：从后往前（先异质性，再机制），避免索引偏移问题
"""

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import copy
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = f"{BASE}/manuscript/word/数据要素利用与资产定价效率v8.docx"

doc = Document(DOC_PATH)


def replace_para_text(para, new_text):
    """Clear all runs, add single new run with first run's formatting."""
    runs = para.runs
    first_rPr = None
    if runs:
        rPr_elem = runs[0]._element.find(qn('w:rPr'))
        if rPr_elem is not None:
            first_rPr = copy.deepcopy(rPr_elem)

    for r in runs:
        r._element.getparent().remove(r._element)
    for r_elem in list(para._element.findall(qn('w:r'))):
        para._element.remove(r_elem)

    new_run = OxmlElement('w:r')
    if first_rPr is not None:
        new_run.append(first_rPr)
    t_elem = OxmlElement('w:t')
    t_elem.text = new_text
    t_elem.set(qn('xml:space'), 'preserve')
    new_run.append(t_elem)
    para._element.append(new_run)


def make_para_element(text, style_name='Normal', alignment=None):
    """Create a new paragraph XML element with text and style."""
    new_p = OxmlElement('w:p')

    # Paragraph properties
    pPr = OxmlElement('w:pPr')
    pStyle = OxmlElement('w:pStyle')
    pStyle.set(qn('w:val'), style_name)
    pPr.append(pStyle)

    if alignment == 'center':
        jc = OxmlElement('w:jc')
        jc.set(qn('w:val'), 'center')
        pPr.append(jc)

    new_p.append(pPr)

    # Run with text
    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    # Make formula text italic
    if alignment == 'center':
        i_elem = OxmlElement('w:i')
        rPr.append(i_elem)
        iCs = OxmlElement('w:iCs')
        rPr.append(iCs)
    new_run.append(rPr)
    t_elem = OxmlElement('w:t')
    t_elem.text = text
    t_elem.set(qn('xml:space'), 'preserve')
    new_run.append(t_elem)
    new_p.append(new_run)

    return new_p


def insert_para_after(doc, para_idx, text, style='Normal', alignment=None):
    """Insert a new paragraph after para_idx. Returns the new element."""
    ref_para = doc.paragraphs[para_idx]
    new_p = make_para_element(text, style, alignment)
    ref_para._element.addnext(new_p)
    return new_p


# ============================================================
# Step 1: 异质性分析 [120] — 先处理（索引不受后续插入影响）
# ============================================================

print("=== 六、异质性分析 ===")

# Verify location
p119 = doc.paragraphs[119]
p120 = doc.paragraphs[120]
assert '异质性' in p119.text, f"Expected 异质性 heading at [119], got: {p119.text}"
print(f"[119] {p119.style.name}: {p119.text}")
print(f"[120] before: {p120.text[:60]}...")

# New transition text for [120]
het_transition = (
    "上述渠道检验揭示了数据要素利用经由信息中介、信息质量和市场流动性三条路径改善定价效率的证据。"
    "本文进一步考察这一效应在不同企业特征和市场条件下的差异，"
    "从产权性质、市场流动性、企业规模和行业属性四个维度进行分组回归，"
    "并设定如下交互项模型对边界条件做连续化的正式检验："
)
replace_para_text(p120, het_transition)
print(f"[120] after: {het_transition[:60]}...")

# Insert formula (3) after [120] — centered, italic
het_formula = (
    "PriceDelayᵢ,ₜ = α₀ + α₁DU_kwᵢ,ₜ + α₂DU_kwᵢ,ₜ × Zᵢ,ₜ "
    "+ α₃Zᵢ,ₜ + γControlsᵢ,ₜ + μᵢ + λₜ + εᵢ,ₜ    （3）"
)
insert_para_after(doc, 120, het_formula, alignment='center')
print(f"  Inserted formula (3)")

# Insert explanation after formula (now at index 121, so insert after 121)
het_explain = (
    "其中Z分别为产权性质（SOE）和市场流动性（Amihud_c，中心化处理）。"
    "分组回归采用费舍尔组合检验判断组间系数差异的统计显著性。"
)
# The formula is now the element right after [120], find it
formula_elem = p120._element.getnext()
explain_p = make_para_element(het_explain, 'Normal')
formula_elem.addnext(explain_p)
print(f"  Inserted explanation")
print()


# ============================================================
# Step 2: 作用渠道分析 [96] — 在异质性插入之前的索引
# ============================================================

print("=== 五、作用渠道分析 ===")

p95 = doc.paragraphs[95]
p96 = doc.paragraphs[96]
assert '渠道' in p95.text, f"Expected 渠道 heading at [95], got: {p95.text}"
print(f"[95] {p95.style.name}: {p95.text}")
print(f"[96] before: {p96.text[:60]}...")

# New transition text for [96] — no question mark
mech_transition = (
    "上述分析充分证实了数据要素利用能够降低股价延迟、提升资本定价效率。"
    "本文进一步探究这一效应的传导机制。"
    "遵循江艇（2022）两步法的思路，仅估计自变量对中介变量的因果效应，"
    "中介变量对因变量的影响关系依据已有文献论证，设定如下渠道检验模型："
)
replace_para_text(p96, mech_transition)
print(f"[96] after: {mech_transition[:60]}...")

# Insert formula (2) after [96] — centered, italic
mech_formula = (
    "Mᵢ,ₜ = θ₀ + θ₁DU_kwᵢ,ₜ + θ₂Controlsᵢ,ₜ + μᵢ + λₜ + εᵢ,ₜ    （2）"
)
insert_para_after(doc, 96, mech_formula, alignment='center')
print(f"  Inserted formula (2)")

# Insert explanation after formula
mech_explain = (
    "其中M分别为Analyst（分析师覆盖对数）、Disp（预测分歧度）和Amihud（非流动性指标）。"
    "表5分别报告了同期回归和滞后一期回归的结果。"
)
formula_elem_mech = p96._element.getnext()
explain_p_mech = make_para_element(mech_explain, 'Normal')
formula_elem_mech.addnext(explain_p_mech)
print(f"  Inserted explanation")
print()


# ============================================================
# Verify final structure
# ============================================================
print("=== 验证 ===")
# Re-read to verify (paragraph indices shifted after insertions)
# Mechanism section: [95] heading, [96] transition, [97] formula, [98] explain, [99] 标题3 (一)
for i in range(95, 103):
    p = doc.paragraphs[i]
    print(f"[{i}] {p.style.name}: {p.text[:80]}")
print()

# Heterogeneity section (shifted by +2 from mechanism insertions, +2 from own insertions = +4 total)
# Original [119] is now [123], but let's find it
for i, p in enumerate(doc.paragraphs):
    if p.style.name == 'Heading 2' and '异质性' in p.text:
        print(f"异质性 heading now at [{i}]")
        for j in range(i, min(i+6, len(doc.paragraphs))):
            pp = doc.paragraphs[j]
            print(f"  [{j}] {pp.style.name}: {pp.text[:80]}")
        break

doc.save(DOC_PATH)
print(f"\nDone! Saved to {DOC_PATH}")
