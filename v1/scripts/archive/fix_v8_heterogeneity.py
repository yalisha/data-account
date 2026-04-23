"""
v8异质性分析重构：朱康风格，（一）（二）（三）（四）分点
Current: [112]heading [113]intro [114]PanelA [115]PanelB产权流动性 [116]规模行业 [117]小结 [118]脚注
Target:  [112]heading [113]intro [114]标题3产权 [115]产权content [116]标题3流动性 [117]流动性content
         + insert before 脚注: 标题3规模, 规模content, 标题3行业, 行业content
"""
from docx import Document
from lxml import etree
import copy
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = f"{BASE}/manuscript/word/数据要素利用与资产定价效率v8.docx"
doc = Document(DOC_PATH)

ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

# ---- Helper functions ----
def get_run_fmt(para):
    """Get run formatting from first content run."""
    for r in para.runs:
        if r.text and r.text.strip():
            rPr = r._element.find(f'{ns}rPr')
            if rPr is not None:
                return copy.deepcopy(rPr)
    return None

def replace_para_text(para, text, style_name=None):
    """Replace paragraph text, keeping first run's formatting."""
    fmt = get_run_fmt(para)
    # Clear all runs
    for r in list(para.runs):
        r._element.getparent().remove(r._element)
    # Also remove any remaining w:r elements
    for r_elem in list(para._element.findall(f'{ns}r')):
        para._element.remove(r_elem)
    # Add new run
    new_r = etree.SubElement(para._element, f'{ns}r')
    if fmt is not None:
        new_r.insert(0, fmt)
    new_t = etree.SubElement(new_r, f'{ns}t')
    new_t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    new_t.text = text
    if style_name:
        para.style = doc.styles[style_name]

def make_para_element(text, style_name=None, run_fmt=None):
    """Create a new w:p element with text and optional style."""
    new_p = etree.SubElement(doc.element.body, f'{ns}p')  # temp parent
    # Set style
    if style_name:
        pPr = etree.SubElement(new_p, f'{ns}pPr')
        # Find style ID
        for s in doc.styles:
            if s.name == style_name:
                pStyle = etree.SubElement(pPr, f'{ns}pStyle')
                pStyle.set(f'{ns}val', s.style_id)
                break
    # Add run
    new_r = etree.SubElement(new_p, f'{ns}r')
    if run_fmt is not None:
        new_r.insert(0, copy.deepcopy(run_fmt))
    new_t = etree.SubElement(new_r, f'{ns}t')
    new_t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    new_t.text = text
    # Remove from temp parent (body)
    doc.element.body.remove(new_p)
    return new_p

# ---- Get reference formatting from existing Normal paragraph ----
ref_fmt = get_run_fmt(doc.paragraphs[115])  # existing Normal content para

# ---- Content ----
intro_text = "渠道检验揭示了数据要素利用经由信息中介、信息质量和市场流动性三条路径改善定价效率的证据。本节进一步考察这一效应是否因企业特征和市场条件而异，从产权性质、市场流动性、企业规模和行业属性四个维度进行分组回归，采用费舍尔组合检验判断组间系数差异的统计显著性，并通过交互项回归对边界条件做连续化的正式检验。"

prop_text = "与国有企业相比，非国有企业面临的融资约束及承担的社会责任不同，可能导致其在应对数据要素发展时的行为决策存在差异。本文依据产权性质将企业分为国有企业和非国有企业，分别进行回归，结果如表6 Panel B第(1)列和第(2)列所示。可以看出，两组样本中DU_kw的系数均在1%水平上显著为负，说明数据要素利用对定价效率的改善作用在两种产权性质下均成立。其中国有企业组的系数为-0.006，约为非国有企业组（-0.003）的1.8倍。采用费舍尔组合检验抽样1000次进行组间系数差异检验，发现系数差异P值为0.094，说明组间差异性显著存在。原因可能在于国有企业因多层级委托代理结构，信息不对称程度系统性高于同规模非国有企业（任广乾等，2021），在信息透明度较低的基线条件下，数据要素利用所释放的增量信息具有更高的边际价值。此外，国有企业具备的\"国家队\"背景使其更容易获得数据基础设施和技术资源的支持，数据利用的信息供给效应更为充分。"

liq_text = "市场流动性是信息融入价格的物理条件。流动性越低的股票，知情交易者的交易成本越高，信息反映到价格中的速度越慢（Amihud和Mendelson，1986），数据要素利用所释放的增量信息可能无法充分定价。本文按Amihud非流动性指标的全样本中位数将企业分为低流动性组和高流动性组，结果如表6 Panel B第(3)列和第(4)列所示。两组样本的DU_kw系数均在1%水平上显著为负：低流动性组为-0.003，高流动性组为-0.004，费舍尔组合检验的P值为0.805，组间差异不具有统计意义。进一步地，Panel A的交互项回归中，DU_kw与中心化Amihud指标的交互项系数为0.025（p<0.05），说明流动性约束确实削弱了数据要素利用的定价效率改善作用。分组检验与交互项检验结果的差异源于方法论特征的不同：交互项回归在全样本中连续刻画调节效应，统计效率更高；分组回归以中位数粗略二分，可能掩盖了组内的连续变异。整体来看，市场流动性对效应具有调节作用，但效应在不同流动性环境下均稳健成立。"

size_text = "企业规模可能通过信息环境和资源禀赋两个渠道调节数据要素利用的定价效率效应。大企业的信息环境更为透明、分析师覆盖更广，数据信息的边际增量价值可能较低；但大企业同时拥有更多的数据资源和技术能力，数据利用的质量可能更高。本文按企业规模的年度中位数进行分组，结果如表6 Panel B第(5)列和第(6)列所示。大企业和小企业的DU_kw系数均为-0.005且在1%水平上显著，费舍尔组合检验的P值为0.966，两组系数几乎完全相同。这一结果表明数据要素利用的定价效率改善作用不受企业规模制约，前述两种对冲机制可能相互抵消，企业无论大小均可通过数据利用行为改善市场对其价值的认知。"

ind_text = "行业属性决定了企业信息环境的基线透明度。高科技行业由于技术导向强、信息更新频繁、市场关注度高，信息环境的基线透明度相对较高，数据要素利用的增量信息价值可能有限；传统行业的信息环境则相对模糊，市场参与者对企业数据利用行为的认知缺口更大，增量信息的边际价值因此更高。本文将信息传输业和计算机通信电子制造业归入高科技行业，其余归入传统行业，分别进行回归。结果如表6 Panel B第(7)列和第(8)列所示，高科技行业的DU_kw系数为-0.002（p<0.05，N=7,473），传统行业为-0.004（p<0.01，N=36,328）。采用费舍尔组合检验抽样1000次进行组间系数差异检验，发现系数差异P值为0.055，说明组间差异性显著存在。传统行业中效应更强的结果与理论预期一致：信息环境越模糊的行业，数据要素利用所带来的信息供给改善越大，定价效率的提升空间也越充裕（李世刚等，2025）。"

# ============================================================
# Step 1: Rewrite existing paragraphs [113]-[117]
# ============================================================
print("[112] Heading 2: 六、异质性分析 (keep)")

# [113] intro
replace_para_text(doc.paragraphs[113], intro_text)
print("[113] Rewritten: intro")

# [114] → 标题3: 产权 heading
replace_para_text(doc.paragraphs[114], "（一）考虑产权性质差异", "标题3")
print("[114] Rewritten: 标题3 产权")

# [115] → 产权 content
replace_para_text(doc.paragraphs[115], prop_text)
print("[115] Rewritten: 产权 content")

# [116] → 标题3: 流动性 heading
replace_para_text(doc.paragraphs[116], "（二）考虑市场流动性差异", "标题3")
print("[116] Rewritten: 标题3 流动性")

# [117] → 流动性 content
replace_para_text(doc.paragraphs[117], liq_text)
print("[117] Rewritten: 流动性 content")

# ============================================================
# Step 2: Insert 4 paragraphs before 脚注 [118]
# ============================================================
footnote_elem = doc.paragraphs[118]._element

# Insert in reverse order (each goes before footnote)
p_ind_content = make_para_element(ind_text, "Normal", ref_fmt)
footnote_elem.addprevious(p_ind_content)
print("Inserted: 行业 content")

p_ind_heading = make_para_element("（四）考虑行业属性差异", "标题3", None)
footnote_elem.addprevious(p_ind_heading)
# Move heading before content
p_ind_content.addprevious(p_ind_heading)
print("Inserted: 标题3 行业")

p_size_content = make_para_element(size_text, "Normal", ref_fmt)
p_ind_heading.addprevious(p_size_content)
print("Inserted: 规模 content")

p_size_heading = make_para_element("（三）考虑企业规模差异", "标题3", None)
p_size_content.addprevious(p_size_heading)
print("Inserted: 标题3 规模")

# ============================================================
# Step 3: Clean empty runs in new paragraphs
# ============================================================
empty_cleaned = 0
for i, p in enumerate(doc.paragraphs):
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
print(f"Cleaned {empty_cleaned} empty runs")

# ============================================================
# Verify
# ============================================================
print("\n--- Verification ---")
for i in range(112, 127):
    if i < len(doc.paragraphs):
        p = doc.paragraphs[i]
        print(f"[{i}] style={p.style.name}: {p.text[:60]}")

doc.save(DOC_PATH)
print(f"\nSaved to {DOC_PATH}")
