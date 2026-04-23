"""
v8三渠道统一：在理论、设计、摘要、结论中统一为三条渠道
1. 理论部分：加Amihud理论段、更新标题和intro
2. 模型设定[80]：M加Amihud
3. Highlights[3]和引言[9]：两条→三条
4. 结论[124]和局限[129]：更新
"""
from docx import Document
from lxml import etree
import copy
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = f"{BASE}/manuscript/word/数据要素利用与资产定价效率v8.docx"
doc = Document(DOC_PATH)
ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

def get_run_fmt(para):
    for r in para.runs:
        if r.text and r.text.strip():
            rPr = r._element.find(f'{ns}rPr')
            if rPr is not None:
                return copy.deepcopy(rPr)
    return None

def replace_para_text(para, text, style_name=None):
    fmt = get_run_fmt(para)
    for r_elem in list(para._element.findall(f'{ns}r')):
        para._element.remove(r_elem)
    new_r = etree.SubElement(para._element, f'{ns}r')
    if fmt is not None:
        new_r.insert(0, fmt)
    new_t = etree.SubElement(new_r, f'{ns}t')
    new_t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    new_t.text = text
    if style_name:
        para.style = doc.styles[style_name]

def make_para_element(text, style_name=None, run_fmt=None):
    new_p = etree.SubElement(doc.element.body, f'{ns}p')
    if style_name:
        pPr = etree.SubElement(new_p, f'{ns}pPr')
        for s in doc.styles:
            if s.name == style_name:
                pStyle = etree.SubElement(pPr, f'{ns}pStyle')
                pStyle.set(f'{ns}val', s.style_id)
                break
    new_r = etree.SubElement(new_p, f'{ns}r')
    if run_fmt is not None:
        new_r.insert(0, copy.deepcopy(run_fmt))
    new_t = etree.SubElement(new_r, f'{ns}t')
    new_t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    new_t.text = text
    doc.element.body.remove(new_p)
    return new_p

ref_fmt = get_run_fmt(doc.paragraphs[33])  # Normal content formatting

# ============================================================
# PART 1: Insert Amihud theory paragraph after [38] (H2b)
# This shifts all subsequent indices by +1
# ============================================================
amihud_theory = (
    "第三条渠道是市场流动性渠道。信息要转化为定价效率的改善，"
    "最终需要通过市场交易来实现。Amihud和Mendelson（1986）的经典模型表明，"
    "买卖价差和交易成本构成了信息融入价格的物理摩擦。"
    "当信息不对称程度较高时，做市商面临严重的逆向选择风险，"
    "倾向于扩大买卖价差以补偿潜在损失（Kyle，1985），"
    "由此产生的流动性不足阻碍了知情交易者将私有信息传递至价格。"
    "数据要素利用通过增加企业可观测信息的供给、降低内外部信息不对称程度，"
    "可能缓解逆向选择问题，改善市场流动性。"
    "Chordia等（2008）的实证研究证实了流动性与价格发现效率之间的正向关联。"
    "与前两条渠道不同，市场流动性渠道刻画的不是信息的生产或加工，"
    "而是信息融入价格的交易环节，三者共同构成了从信息供给到定价效率改善的完整链条。"
    "本文在渠道分析中以Amihud非流动性指标作为中间变量加以检验。"
)

# Insert after [38] = before [39]
ref_elem = doc.paragraphs[39]._element  # （三）边界条件
new_p = make_para_element(amihud_theory, "Normal", ref_fmt)
ref_elem.addprevious(new_p)
print("[38+] Inserted Amihud theory paragraph")

# ============================================================
# From here, all indices >= 39 have shifted by +1
# Old [39] = New [40], Old [40] = New [41], etc.
# Old [80] = New [81], Old [124] = New [125], etc.
# ============================================================

# PART 2: Update theory section headings and intros
# [31] heading (unchanged index)
replace_para_text(doc.paragraphs[31], "（二）信息中介、信息质量与市场流动性：传导机制", "标题3")
print("[31] Updated heading to include 流动性")

# [32] intro
replace_para_text(doc.paragraphs[32],
    "H1确立了数据要素利用与定价效率之间的总体关联，"
    "但信息从企业年报向股票价格的具体传导路径尚待厘清。"
    "从信息向定价的传导机制看，存在三条主要渠道，"
    "分别对应信息生产的广度、信息加工的精度和信息融入价格的交易效率。"
)
print("[32] Updated intro: 两条→三条")

# [36] second channel: remove ending "二者分别从...两个维度"
# Need to update the ending to accommodate three channels
p36_text = doc.paragraphs[36].text
old_ending = "信息中介渠道侧重于信息生产的数量扩展，信息质量渠道则聚焦于信息加工的精度提升，二者分别从信息供给的广度和深度两个维度促进定价效率。据此提出："
new_ending = "信息中介渠道侧重于信息生产的数量扩展，信息质量渠道则聚焦于信息加工的精度提升。据此提出："
if old_ending in p36_text:
    replace_para_text(doc.paragraphs[36], p36_text.replace(old_ending, new_ending))
    print("[36] Updated: removed 二者两个维度 summary")
else:
    print(f"[36] WARNING: ending not found. Text ends with: ...{p36_text[-60:]}")

# [41] (was [40]) boundary conditions intro: update reference
p41 = doc.paragraphs[41]
old_41 = "上述H1至H2b的分析给出了数据要素利用对定价效率的总体效应及其主要传导渠道。"
new_41 = "上述分析给出了数据要素利用对定价效率的总体效应及其三条传导渠道。"
if old_41 in p41.text:
    replace_para_text(p41, p41.text.replace(old_41, new_41))
    print("[41] Updated boundary intro")
else:
    print(f"[41] WARNING: text not found")

# ============================================================
# PART 3: Model specification [81] (was [80])
# ============================================================
p81 = doc.paragraphs[81]
old_80 = "其中M分别为Analyst和Disp。"
new_80 = "其中M分别为Analyst、Disp和Amihud。"
if old_80 in p81.text:
    replace_para_text(p81, p81.text.replace(old_80, new_80).replace(
        "两个内生变量", "多个内生变量"
    ))
    print("[81] Updated M variable list")
else:
    print(f"[81] WARNING: text not found. Starts: {p81.text[:60]}")

# ============================================================
# PART 4: Highlights [3]
# ============================================================
replace_para_text(doc.paragraphs[3],
    "渠道分析揭示了数据利用经由外部信息中介影响定价效率的三条路径："
    "分析师跟踪增加、预测分歧度降低、市场流动性改善，"
    "三条路径在滞后一期回归中方向一致。"
    "按t-1期分析师覆盖分组的梯度检验与渠道解释相互印证。"
    "效应在不同企业特征下普遍存在，市场流动性和行业属性是两个显著的边界条件。"
)
print("[3] Updated Highlights")

# ============================================================
# PART 5: Introduction abstract [9]
# ============================================================
p9 = doc.paragraphs[9]
old_9a = "渠道分析提供了数据利用经由外部信息中介影响定价效率的证据：数据利用增加了分析师跟踪并降低了预测分歧度，两条路径在滞后一期回归中依然稳健。"
new_9a = "渠道分析提供了数据利用经由外部信息环境影响定价效率的证据：数据利用增加了分析师跟踪、降低了预测分歧度并改善了市场流动性，三条路径在滞后一期回归中依然稳健。"
if old_9a in p9.text:
    replace_para_text(p9, p9.text.replace(old_9a, new_9a))
    print("[9] Updated introduction")
else:
    print(f"[9] WARNING: text not found")

# ============================================================
# PART 6: Conclusion [125] (was [124])
# ============================================================
p125 = doc.paragraphs[125]
# Update 渠道分析 part
old_c1 = "渠道分析采用江艇（2022）两步法，提供了与数据利用经由外部信息中介影响定价效率一致的证据。信息中介渠道方面，数据利用与分析师跟踪正相关，滞后一期方向一致；信息质量渠道方面，数据利用降低了预测分歧度，时序证据一致。按t-1期分析师覆盖分组的梯度检验进一步提供了与渠道解释一致的模式。这一外部信息传导机制与既有文献所关注的企业内部行为渠道形成互补。"
new_c1 = "渠道分析采用江艇（2022）两步法，提供了数据利用经由外部信息环境影响定价效率的证据。信息中介渠道方面，数据利用显著增加了分析师跟踪；信息质量渠道方面，数据利用降低了预测分歧度；市场流动性渠道方面，数据利用降低了Amihud非流动性指标。三条路径在滞后一期回归中方向一致，按t-1期分析师覆盖分组的梯度检验进一步提供了与渠道解释一致的模式。"
if old_c1 in p125.text:
    replace_para_text(p125, p125.text.replace(old_c1, new_c1))
    print("[125] Updated conclusion")
else:
    print(f"[125] WARNING: old conclusion text not found")

# ============================================================
# PART 7: Limitations [130] (was [129])
# ============================================================
p130 = doc.paragraphs[130]
old_lim = "传导机制方面，本文聚焦于分析师中介渠道，社交媒体传播和ESG评级嵌入等其他路径留待后续研究。"
new_lim = "传导机制方面，本文检验了信息中介、信息质量和市场流动性三条渠道，但社交媒体传播和ESG评级嵌入等其他路径留待后续研究。"
if old_lim in p130.text:
    replace_para_text(p130, p130.text.replace(old_lim, new_lim))
    print("[130] Updated limitations")
else:
    print(f"[130] WARNING: old limitation text not found")

# ============================================================
# Clean empty runs
# ============================================================
empty_cleaned = 0
for p in doc.paragraphs:
    runs_to_remove = []
    has_content = False
    for r in p.runs:
        if r.text and r.text.strip():
            has_content = True
        elif r.text == '' or r.text is None:
            runs_to_remove.append(r)
    if has_content and runs_to_remove:
        for r in runs_to_remove:
            r._element.getparent().remove(r._element)
            empty_cleaned += 1
print(f"\nCleaned {empty_cleaned} empty runs")

# ============================================================
# Verify key paragraphs
# ============================================================
print("\n--- Verification ---")
for i in [3, 9, 31, 32, 36, 38, 39, 41, 81, 125, 130]:
    if i < len(doc.paragraphs):
        p = doc.paragraphs[i]
        print(f"[{i}] style={p.style.name}: {p.text[:80]}")

doc.save(DOC_PATH)
print(f"\nSaved!")
