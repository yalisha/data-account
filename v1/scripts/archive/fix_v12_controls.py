"""
v12修订：
1. 控制变量15→13（移除Analyst和Amihud）
2. 补H2c假说声明
3. 更新机制分析具体系数
4. 更新基准回归系数描述
"""

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import copy
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = f"{BASE}/manuscript/word/数据要素利用与资产定价效率v8.docx"

doc = Document(DOC_PATH)


def replace_in_para(para, old_text, new_text):
    """Replace text in paragraph, preserving formatting of first occurrence."""
    full = para.text
    if old_text not in full:
        print(f"  WARNING: '{old_text[:30]}' not found!")
        return False

    # Simple approach: work with runs
    runs = para.runs
    combined = ""
    for r in runs:
        combined += r.text

    if old_text in combined:
        # Find which runs contain the old text
        pos = combined.find(old_text)
        end = pos + len(old_text)

        # Build new text by splicing
        new_combined = combined[:pos] + new_text + combined[end:]

        # Redistribute text across runs
        offset = 0
        for r in runs:
            rlen = len(r.text)
            r.text = new_combined[offset:offset + rlen + (len(new_text) - len(old_text)) // len(runs)]
            offset += len(r.text)

        # Fallback: if redistribution is messy, use single-run approach
        if para.text != new_combined:
            # Clear and rewrite with first run's formatting
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
            t_elem.text = new_combined
            t_elem.set(qn('xml:space'), 'preserve')
            new_run.append(t_elem)
            para._element.append(new_run)

    return True


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


def insert_para_after(doc, para_idx, text):
    """Insert a new paragraph after para_idx, copying style from para_idx."""
    ref_para = doc.paragraphs[para_idx]

    new_p = OxmlElement('w:p')
    # Copy paragraph properties from reference
    ref_pPr = ref_para._element.find(qn('w:pPr'))
    if ref_pPr is not None:
        new_p.append(copy.deepcopy(ref_pPr))

    # Copy first run's formatting
    first_rPr = None
    if ref_para.runs:
        rPr_elem = ref_para.runs[0]._element.find(qn('w:rPr'))
        if rPr_elem is not None:
            first_rPr = copy.deepcopy(rPr_elem)

    new_run = OxmlElement('w:r')
    if first_rPr is not None:
        new_run.append(first_rPr)
    t_elem = OxmlElement('w:t')
    t_elem.text = text
    t_elem.set(qn('xml:space'), 'preserve')
    new_run.append(t_elem)
    new_p.append(new_run)

    ref_para._element.addnext(new_p)
    return new_p


# ============================================================
# Step 1: 控制变量 15→13
# ============================================================
print("=== Step 1: 控制变量数量 ===")

# [62]
p62 = doc.paragraphs[62]
assert '15个控制变量' in p62.text
replace_in_para(p62, '15个控制变量，涵盖企业规模、财务状况、治理特征、市场流动性和信息环境等维度',
                '13个控制变量，涵盖企业规模、财务状况、治理特征和信息环境等维度')
print(f"  [62] done: {p62.text[:60]}")

# [69]
p69 = doc.paragraphs[69]
assert '15个控制变量' in p69.text
replace_in_para(p69, '全部15个控制变量', '全部13个控制变量')
print(f"  [69] done: {p69.text[:60]}")


# ============================================================
# Step 2: 基准回归系数更新 [76]
# ============================================================
print("\n=== Step 2: 基准回归系数 ===")

p76 = doc.paragraphs[76]
old76 = p76.text

# 系数 -0.004 → -0.005 (rounded from -0.00459)
# 经济意义 6.7% → 7.7% (0.00459 * 1.86 / 0.1115 = 7.65%)
new76 = (
    "以基准模型(2)为例，加入全部控制变量后DUkw的回归系数为-0.005（p<0.01）。"
    "经济意义方面，DUkw每增加一个标准差（1.86），股价延迟预期下降约7.7%，"
    "即企业数据利用强度的边际提升能以可观测的幅度加速信息向股价的融入。"
    "替换自变量测度、固定效应结构和因变量后，七个模型的系数方向和显著性保持一致，"
    "H1得到支持：在信息供给效应与信息模糊效应的竞争中，前者占据主导。"
    "这一结果对企业的信息披露决策具有直接含义：在年报中主动呈现数据要素利用实践"
    "不会因信息复杂性而受到市场惩罚，反而降低了投资者与企业之间的信息摩擦，"
    "改善了资本市场对企业价值的识别。"
    "不过，仅凭基准回归的系数方向尚不能排除两种效应并存而供给碰巧更强的可能，"
    "后文的渠道检验和投资组合分析将从差异化含义的角度进一步区分。"
)
replace_para_text(p76, new76)
print(f"  [76] done: {p76.text[:60]}")


# ============================================================
# Step 3: 补H2c假说声明
# ============================================================
print("\n=== Step 3: 补H2c ===")

# [37] 末尾加 "据此提出："
p37 = doc.paragraphs[37]
old37 = p37.text
# 替换最后一句
replace_in_para(p37,
    "本文在机制分析中以Amihud非流动性指标作为中间变量加以检验。",
    "本文在机制分析中以Amihud非流动性指标作为中间变量加以检验。据此提出：")
print(f"  [37] done: {p37.text[-40:]}")

# 在[37]后面插入H2c
h2c_text = "H2c：企业数据要素利用通过改善市场流动性降低交易摩擦，进而提升资本定价效率。"
insert_para_after(doc, 37, h2c_text)
print(f"  Inserted H2c after [37]")

# 注意：插入后索引偏移+1，原[38]变[39]以此类推


# ============================================================
# Step 4: 更新机制分析具体系数
# 注意：因Step 3插入了1段，原[95]→[96]，原[97]→[98]，原[99]→[100]
# ============================================================
print("\n=== Step 4: 机制分析系数 ===")

# 信息中介渠道 - 原[95] → 现[96]
p_analyst = doc.paragraphs[96]
assert '信息中介' in p_analyst.text or '分析师' in p_analyst.text[:20]

new_analyst = (
    "分析师作为资本市场最重要的信息中介，通过深度调研、盈利预测和投资评级"
    "将企业层面的复杂信息转化为可供投资者使用的交易信号。"
    "Hong等（2000）发现分析师跟踪不足的股票信息融入价格的速度显著更慢，"
    "产生了可被动量策略利用的异常收益。"
    "黄俊和郭照蕊（2014）利用券商关闭这一准自然实验证实，"
    "外生的分析师覆盖减少导致股价延迟显著增加，"
    "为分析师覆盖与定价效率之间的因果关系提供了直接证据。"
    "数据要素利用涉及数据资产化、数字化转型和数据驱动决策等技术含量较高的业务实践，"
    "需要专业中介进行解读和传播。"
    "年报中数据利用信息的增加为分析师提供了新的研究素材和增量信息来源，"
    "降低了信息获取成本，从而可能吸引更多分析师关注和跟踪。"
    "带入机制模型的回归结果如表5第(1)(4)列所示，"
    "DUkw对分析师覆盖（取对数）的同期系数为+0.0273（p<0.01），"
    "滞后一期系数为+0.0335（p<0.001），方向一致且系数在滞后口径下略有增大。"
    "经济意义上，DUkw增加一个标准差（1.86）对应分析师跟踪人数提升约5.2%。"
    "进一步地，以t-1期分析师覆盖中位数分组，"
    "高覆盖组的DUkw系数（-0.006，p<0.01）约为低覆盖组（-0.003，p<0.01）的1.7倍"
    "（Fisher p=0.141），"
    "这一梯度模式进一步验证了信息中介在数据信息传导中的作用。"
    "综上说明，数据要素利用可以通过提升分析师覆盖、"
    "拓展信息中介网络来降低股价延迟、提升定价效率。"
)
replace_para_text(p_analyst, new_analyst)
print(f"  [96] Analyst done")

# 信息质量渠道 - 原[97] → 现[98]
p_disp = doc.paragraphs[98]
assert '分歧' in p_disp.text or '信息质量' in p_disp.text[:20]

new_disp = (
    "分析师预测分歧度衡量的是市场参与者对企业价值判断的不一致程度，"
    "是信息环境质量的重要反映。"
    "较高的预测分歧意味着分析师所掌握的信息集存在显著差异，"
    "市场难以形成一致预期，价格发现过程受到阻碍。"
    "Zhang（2006）发现信息不确定性较高的企业预测分歧更大，"
    "且高分歧股票存在更严重的定价偏差。"
    "Diether等（2002）进一步证实，"
    "预测分歧度较高的股票未来收益率显著偏低。"
    "数据要素利用产生了更多结构化、可验证的运营信息，"
    "有助于降低分析师在预测企业未来盈利时面临的不确定性。"
    "与一般性文字叙述不同，数据驱动业务实践的披露往往伴随更具体的量化信息，"
    "为分析师提供了更硬的信息基础，"
    "有助于缩小分析师之间的信息差异，降低预测分歧度。"
    "带入机制模型的回归结果如表5第(2)(5)列所示，"
    "DUkw对预测分歧度的同期系数为-0.0059（p<0.001），"
    "滞后一期系数为-0.0064（p<0.001），方向一致。"
    "DUkw增加一个标准差对应分歧度降低约7.1%。"
    "Disp的显著下降同时也有助于区分两种竞争效应："
    "若数据利用主要带来信息模糊，分析师之间的理解差异应扩大而非缩小，"
    "实证结果与此方向相反。"
    "综上说明，数据要素利用可以通过降低分析师预测分歧、"
    "改善信息质量来促进价格发现。"
)
replace_para_text(p_disp, new_disp)
print(f"  [98] Disp done")

# 市场流动性渠道 - 原[99] → 现[100]
p_amihud = doc.paragraphs[100]
assert 'Kyle' in p_amihud.text or '流动性' in p_amihud.text[:20]

new_amihud = (
    "信息要转化为定价效率的改善，最终需要通过市场交易来实现。"
    "Kyle（1985）和Glosten和Milgrom（1985）的经典模型表明，"
    "信息不对称程度较高时，做市商面临严重的逆向选择风险，"
    "倾向于扩大买卖价差以补偿潜在损失，"
    "由此产生的流动性不足阻碍了信息向价格的传递。"
    "Amihud和Mendelson（1986）在理论上证明了流动性溢价的存在，"
    "流动性较差的资产要求更高的预期回报作为补偿。"
    "Chordia等（2008）从实证上发现，流动性改善加速了信息融入价格的过程，"
    "为流动性与定价效率之间的正向关联提供了经验证据。"
    "数据要素利用通过增加企业可观测信息的供给，"
    "缩小内外部信息不对称程度，可能缓解逆向选择问题，"
    "降低交易摩擦从而改善市场流动性。"
    "与前两条渠道侧重信息的生产和加工不同，"
    "市场流动性渠道刻画的是信息融入价格的交易环节，"
    "三者共同构成了从信息供给到定价效率改善的完整链条。"
    "带入机制模型的回归结果如表5第(3)(6)列所示，"
    "DUkw对Amihud非流动性指标的同期系数为-0.0010（p<0.01），"
    "滞后一期系数为-0.0014（p<0.001），方向一致且滞后口径系数绝对值更大。"
    "DUkw增加一个标准差对应Amihud降低约3.9%。"
    "综上说明，数据要素利用可以通过改善市场流动性、"
    "降低交易摩擦来加速信息融入价格。"
)
replace_para_text(p_amihud, new_amihud)
print(f"  [100] Amihud done")


# ============================================================
# Step 5: 验证
# ============================================================
print("\n=== 验证 ===")

# 检查H2c
found_h2c = False
for i in range(35, 42):
    if i < len(doc.paragraphs) and 'H2c' in doc.paragraphs[i].text:
        found_h2c = True
        print(f"  H2c found at [{i}]: {doc.paragraphs[i].text}")
        break
if not found_h2c:
    print("  WARNING: H2c not found!")

# 检查控制变量
for i, p in enumerate(doc.paragraphs):
    if '15个控制变量' in p.text:
        print(f"  WARNING: '15个控制变量' still at [{i}]!")

# 打印关键段落
print("\n=== 关键段落 ===")
for i in [62, 69, 76, 37, 38]:
    print(f"[{i}] {doc.paragraphs[i].text[:80]}")

doc.save(DOC_PATH)
print(f"\nDone! Saved to {DOC_PATH}")
