"""
重写理论分析章节：被解释变量导向（DV-oriented）三步结构
(一) 资本定价效率的形成机制
(二) 数据要素利用的双重效应
(三) 传导机制与研究假说

替换段落 [26]-[39] (14段) → 新内容 21段 [26]-[46]
"""
import shutil
from docx import Document
from docx.oxml.ns import qn
from copy import deepcopy
from lxml import etree

src = "/Users/mac/computerscience/15会计研究/manuscript/word/数据要素利用与资产定价效率v8.docx"
dst = "/Users/mac/computerscience/15会计研究/manuscript/word/数据要素利用与资产定价效率v9.docx"
shutil.copy2(src, dst)

doc = Document(dst)

# ============================================================
# 新内容定义
# ============================================================
# type: "subtitle" = 标题3, "body" = Normal, "h" = Normal (H statement)
new_content = [
    # --- (一) 资本定价效率的形成机制 ---
    ("subtitle", "（一）资本定价效率的形成机制"),

    ("body", "股价延迟衡量的是企业特质信息融入股票价格的速度（Hou和Moskowitz，2005）。"
     "在有效市场中，新信息应当即时反映于价格；但现实中，"
     "信息从产生到被价格充分吸收需要依次经过三个环节，每一环节的摩擦都会延缓这一过程。"),

    ("body", "第一个环节是信息生产。企业经营活动产生的特质信息需要经过分析师、媒体等信息中介的采集、解读和传播，"
     "才能转化为可定价的公开信号。当信息中介覆盖不足时，特质信息停留在企业内部或少数知情者手中，价格发现无从展开。"
     "跟踪分析师数量较少的企业，其股票的信息融入速度显著更慢（Hong等，2000）。"
     "在中国市场，信息中介活动同样对特质信息融入股价具有显著推动作用（黄俊和郭照蕊，2014）。"),

    ("body", "第二个环节是信息解读。即使公开信息供给充分，如果信息本身不确定性较高、"
     "投资者对其含义的理解存在分歧，价格发现仍然受阻。"
     "信息不确定性越高的股票，异常收益的事后持续性越强，价格向基本面的回归越迟缓（Zhang，2006）。"
     "分析师预测分歧度是信息解读摩擦的可观测指标："
     "分歧度越大，市场对企业价值的共识程度越低，价格向均衡水平的收敛也越慢。"),

    ("body", "第三个环节是交易执行。即使信息已被采集并被正确解读，信息向价格的转化仍然依赖于市场流动性。"
     "信息不对称导致知情交易者与非知情交易者之间产生逆向选择，做市商扩大买卖价差以补偿信息劣势（Kyle，1985），"
     "流动性下降使信息难以在合理时间内融入价格。"
     "Amihud（2002）的非流动性指标刻画的正是这一层面的交易摩擦。"),

    ("body", "信息生产的广度、信息解读的精度、交易执行的效率共同决定了企业特质信息融入价格的速度。"
     "三者并非独立运作：充裕的信息生产为解读提供素材，解读的收敛又通过知情交易者的订单流影响流动性。"
     "理解不同企业之间定价效率的差异，需要追溯到这三个环节中的具体摩擦来源。"),

    # --- (二) 数据要素利用的双重效应 ---
    ("subtitle", "（二）数据要素利用的双重效应"),

    ("body", "数据资产长期游离于会计体系之外，这对上述三个环节均构成影响。"
     "剩余收益估值框架假定账面净资产能准确代表企业经济价值（Ohlson，1995），"
     "但当数据资产因不满足可辨认性或可靠计量标准而被排斥在表外时，这一前提不再成立。"
     "数据资产的价值随使用场景呈非线性增长，历史成本法无法捕捉这一动态特征（黄世忠等，2023）。"
     "在此背景下，年度报告中关于数据要素利用的文本描述成为市场参与者评估企业数据能力的增量信号来源。"),

    ("body", "这类增量信号可能沿两条相反的方向作用于定价效率。"
     "信息供给效应认为，数据要素利用的信号降低了定价过程中的信息约束。"
     "在信息生产环节，数据利用涉及数据资产化、数字化转型等技术含量较高的业务实践，"
     "为分析师提供了差异化的研究素材，可能吸引更多分析师跟踪。"
     "在信息解读环节，数据驱动业务模式产生更多结构化运营信息，降低了分析师预测面临的不确定性。"
     "在交易执行环节，信息供给的增加缩小了知情与非知情交易者之间的信息差距，有望改善市场流动性。"
     "三个环节的改善共同加速信息向价格的融入，"
     "这一机制既有理论推演的支持（Diamond和Verrecchia，1991），"
     "也有信息披露提升定价效率的经验证据（Callen等，2013；Gao等，2024）。"),

    ("body", "信息模糊效应指向相反方向。数据驱动商业模式的复杂性和会计准则的估值空白"
     "可能加剧投资者面临的不确定性，使信息解读变得更加困难。"
     "无形资产密集度高的企业往往面临更大的估值分歧（Dong和Doukas，2025），"
     "数据资产作为一种更新、更难估值的无形资产类型，可能放大这种效应。"
     "当不确定性足够大时，更多的信息披露反而可能增加噪音，使价格发现变慢而非加速。"),

    ("body", "两种效应的行为假设不同。"
     "信息供给效应认为定价延迟的瓶颈在于供给侧的信息不足，数据利用通过增加可验证信号缓解这一约束；"
     "信息模糊效应认为瓶颈在于需求侧的认知摩擦，数据利用的复杂性反而加大了投资者的处理负担。"
     "净效应的方向是一个需要经验检验的问题。"
     "在中国A股市场，年报作为经审计的法定文件，信号的可验证性高于自愿性披露渠道，有利于信息供给效应的发挥。"
     "两种效应还可通过差异化含义加以区分：若信息供给效应占优，分析师预测分歧度应随数据利用增加而下降；"
     "若信息模糊效应占优，分歧度应上升。据此提出："),

    ("h", "H1：企业数据要素利用降低了资本市场的股价延迟，即提升了定价效率。"),

    # --- (三) 传导机制与研究假说 ---
    ("subtitle", "（三）传导机制与研究假说"),

    ("body", "H1描述的是数据要素利用对定价效率的总体效应，但信息从年报文本到股票价格的具体传导路径尚待厘清。"
     "沿第（一）节建立的三个环节，分别对应三条可检验的渠道。"),

    ("body", "第一条渠道对应信息生产环节。分析师是连接企业信息披露与市场定价的核心中介，"
     "其覆盖密度直接影响信息向价格的转换效率。"
     "数据要素利用涉及技术含量较高的业务实践，年报中此类信息的增加为分析师提供了差异化研究素材，"
     "可能吸引更多分析师关注，从而加速企业特质信息向价格的传导。据此提出："),

    ("h", "H2a：企业数据要素利用通过提升分析师覆盖发挥信息中介作用。"),

    ("body", "第二条渠道对应信息解读环节。数据驱动业务模式产生更多结构化运营信息，"
     "使分析师在预测未来盈利时面临的不确定性降低，从而缩小预测分歧。"
     "分歧度的下降意味着市场参与者对企业价值的判断趋于一致，价格发现过程得以加速。据此提出："),

    ("h", "H2b：企业数据要素利用通过降低分析师预测分歧改善信息质量，进而提升资本定价效率。"),

    ("body", "第三条渠道对应交易执行环节。"
     "信息不对称加剧逆向选择、扩大买卖价差、降低市场流动性，阻碍信息向价格的传递。"
     "数据要素利用通过增加可观测信息的供给、缩小信息差距，有望缓解逆向选择，改善流动性。"
     "与前两条渠道侧重信息的生产和加工不同，流动性渠道刻画的是信息融入价格的交易环节，"
     "三条渠道共同构成从信息供给到定价效率改善的完整链条。据此提出："),

    ("h", "H2c：企业数据要素利用通过改善市场流动性降低交易摩擦，进而提升资本定价效率。"),

    ("body", "上述效应的强度可能因企业特征和市场条件而异。"
     "后续分析从产权性质、市场流动性和行业属性三个维度检验效应的边界条件。"
     "产权性质反映信息环境的制度基础，市场流动性反映信息融入价格的物理约束，"
     "行业属性反映信息环境的基线透明度差异。"),
]

print(f"新内容共 {len(new_content)} 段，原内容 14 段 (26-39)，需插入 {len(new_content)-14} 段")

# ============================================================
# Helper functions
# ============================================================

def get_style_ref(doc, style_type):
    """Get reference paragraph for each style type."""
    refs = {}
    for i, p in enumerate(doc.paragraphs):
        if p.style.name == '标题3' and 'subtitle' not in refs:
            refs['subtitle'] = i
        if p.style.name == 'Normal' and p.text.startswith('H') and 'h' not in refs:
            refs['h'] = i
        if p.style.name == 'Normal' and not p.text.startswith('H') and len(p.text) > 50 and 'body' not in refs:
            refs['body'] = i
    return refs

def set_para_text(para, text):
    """Replace paragraph text, keeping first run formatting."""
    # Clear all runs
    for r in para.runs[1:]:
        para._element.remove(r._element)
    if para.runs:
        para.runs[0].text = text
    else:
        run = para.add_run(text)

def set_para_style(para, style_name, ref_para):
    """Copy paragraph style and run formatting from reference."""
    # Set paragraph style
    if style_name == 'subtitle':
        para.style = doc.styles['标题3']
    else:
        para.style = doc.styles['Normal']
    
    # Copy pPr (paragraph properties) from reference
    old_pPr = para._element.find(qn('w:pPr'))
    ref_pPr = ref_para._element.find(qn('w:pPr'))
    if old_pPr is not None:
        para._element.remove(old_pPr)
    if ref_pPr is not None:
        new_pPr = deepcopy(ref_pPr)
        para._element.insert(0, new_pPr)
    
    # Copy run formatting from reference's first run
    ref_runs = ref_para._element.findall(qn('w:r'))
    if ref_runs and para.runs:
        ref_rPr = ref_runs[0].find(qn('w:rPr'))
        cur_rPr = para.runs[0]._element.find(qn('w:rPr'))
        if cur_rPr is not None:
            para.runs[0]._element.remove(cur_rPr)
        if ref_rPr is not None:
            para.runs[0]._element.insert(0, deepcopy(ref_rPr))

def insert_para_after(ref_para, text, style_para):
    """Insert a new paragraph after ref_para, copying style from style_para."""
    # Create new paragraph element by cloning style_para
    new_p = deepcopy(style_para._element)
    
    # Clear all runs
    for r in new_p.findall(qn('w:r')):
        new_p.remove(r)
    
    # Add new run with text, copying formatting from style_para's first run
    source_runs = style_para._element.findall(qn('w:r'))
    new_r = None
    if source_runs:
        new_r = deepcopy(source_runs[0])
        for t in new_r.findall(qn('w:t')):
            new_r.remove(t)
    else:
        new_r = etree.SubElement(new_p, qn('w:r'))
    
    new_t = etree.SubElement(new_r, qn('w:t'))
    new_t.text = text
    new_t.set(qn('xml:space'), 'preserve')
    new_p.append(new_r)
    
    # Insert after reference
    ref_para._element.addnext(new_p)
    return new_p

# ============================================================
# Execute replacement
# ============================================================

# Save reference paragraphs for formatting (before any changes)
ref_subtitle = doc.paragraphs[26]  # (一) subtitle
ref_body = doc.paragraphs[27]      # body text
ref_h = doc.paragraphs[30]         # H1 statement

# Style mapping
style_refs = {
    'subtitle': ref_subtitle,
    'body': ref_body,
    'h': ref_h
}

# Step 1: Replace text and style of existing paragraphs 26-39
print("\n=== Step 1: 替换段落 26-39 的文本和样式 ===")
for i in range(14):
    para_idx = 26 + i
    stype, text = new_content[i]
    para = doc.paragraphs[para_idx]
    set_para_text(para, text)
    set_para_style(para, stype, style_refs[stype])
    print(f"  [{para_idx}] ({stype}) {text[:40]}...")

# Step 2: Insert 7 new paragraphs after paragraph 39
print("\n=== Step 2: 在段落 39 之后插入 7 段 ===")
# Insert in reverse order so positions stay correct
insert_items = new_content[14:]  # items 14-20 (7 items)
insert_items_reversed = list(reversed(insert_items))

ref_para = doc.paragraphs[39]
for stype, text in insert_items_reversed:
    new_p_elem = insert_para_after(ref_para, text, style_refs[stype])
    print(f"  inserted ({stype}) {text[:40]}...")

# Save
doc.save(dst)
print(f"\n=== 保存到 {dst} ===")

# Verify
print("\n=== 验证 ===")
doc2 = Document(dst)
for i in range(25, 48):
    p = doc2.paragraphs[i]
    style = p.style.name if p.style else 'None'
    print(f"  [{i}] ({style}) {p.text[:60]}")
