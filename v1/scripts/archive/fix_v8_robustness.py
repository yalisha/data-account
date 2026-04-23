"""
重构稳健性与内生性章节为朱康(2025)编号子项风格。
Para [89]-[101] (13段) → 同样13段但结构化为：标题 + 导言 + 10个编号子项 + 总结
所有数字沿用现有结果，不改变任何系数/t值/p值/N。
"""

from docx import Document
from docx.oxml.ns import qn
import copy
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = f"{BASE}/manuscript/word/数据要素利用与资产定价效率v8.docx"

doc = Document(DOC_PATH)

# ============================================================
# Helper: replace paragraph text preserving style
# ============================================================
def replace_para_text(para, new_text):
    """Clear all runs, add single new run with text, preserve first run's formatting."""
    runs = para.runs
    if runs:
        first_rPr = copy.deepcopy(runs[0]._element.find(qn('w:rPr')))
    else:
        first_rPr = None

    # Remove all existing runs
    for r in runs:
        r._element.getparent().remove(r._element)

    # Also remove any remaining w:r elements (e.g., field codes)
    for r_elem in list(para._element.findall(qn('w:r'))):
        para._element.remove(r_elem)

    # Add new run
    from docx.oxml import OxmlElement
    new_run = OxmlElement('w:r')
    if first_rPr is not None:
        new_run.append(copy.deepcopy(first_rPr))
    t_elem = OxmlElement('w:t')
    t_elem.text = new_text
    t_elem.set(qn('xml:space'), 'preserve')
    new_run.append(t_elem)
    para._element.append(new_run)


# ============================================================
# New content for each paragraph
# ============================================================

TITLE = "（三）稳健性检验与内生性分析"

INTRO = (
    "为确保基准回归结论的可靠性，本文从固定效应结构、变量测度方式、"
    "样本范围、工具变量和系数稳定性等维度进行检验，结果见表3和表4。"
)

ITEM_1 = (
    "1. 替换固定效应结构。考虑到时变行业特征和时变地区特征可能给实证结果带来影响，"
    "本文借鉴锁雪松等（2019）和马黎珺等（2022）的做法，在控制个体及年份固定效应基础上"
    "分别新增行业与年份交互固定效应以及地区与年份交互固定效应进行回归。"
    "结果如表3第（1）列和第（2）列所示，DU_kw的系数分别为-0.0022和-0.004，"
    "均在1%的水平上显著为负，说明基准结论不受固定效应结构选择的影响。"
)

ITEM_2 = (
    "2. 行业均值调整。采用行业年度均值调整后的DU_kw为自变量，"
    "排除行业层面数据利用趋势的干扰。"
    "结果如表3第（3）列所示，系数为-0.0020（p<0.01），与基准回归方向一致且高度显著。"
)

ITEM_3 = (
    "3. 剔除特定样本。2024年为数据入表新规实施首年，企业数据披露行为可能发生结构性变化。"
    "剔除2024年观测后（N=38,916），系数为-0.005（p<0.01），绝对值有所增大，"
    "说明结论并非由政策冲击年份驱动。"
    "剔除信息技术行业样本后（N=40,720），系数为-0.004（p<0.01），说明效应并非由IT行业主导。"
    "仅保留主板上市企业（N=18,837），系数为-0.003（p<0.05），"
    "在更为同质的样本中效应依然显著。结果分别如表3第（4）至第（6）列所示。"
)

ITEM_4 = (
    "4. 倾向得分匹配。为缓解潜在的函数形式错误设定问题（Shipman等, 2017），"
    "本文采用倾向得分匹配法（PSM）进行处理。"
    "以DU_kw中位数划分处理组，全部控制变量作为匹配协变量，"
    "经Logit倾向得分1:1近邻匹配（caliper=0.05）后在匹配样本上以连续DU_kw回归。"
    "结果如表3第（7）列所示，匹配样本（N=34,425）上DU_kw系数为-0.004（p<0.01），"
    "与基准回归高度一致。"
)

ITEM_5 = (
    "5. 替换聚类标准误。使用企业和年份水平双向聚类标准误替代行业×年份聚类。"
    "结果如表3第（8）列所示，系数为-0.004（p<0.01），"
    "结论不受聚类方式选择的影响。"
)

ITEM_6 = (
    "6. 前导项检验。为排除反向因果，同时纳入DU_kw和其前导项DU_kw(t+1)。"
    "结果如表3第（9）列所示，DU_kw系数为-0.005（p<0.01），"
    "而DU_kw(t+1)系数为0.0005且统计上不显著（t=0.57），"
    "说明是当期数据要素利用而非未来变化驱动了定价效率的改善。"
)

ITEM_7 = (
    "7. 工具变量检验。参照已有文献的做法（Leary和Roberts, 2014; 李世刚等, 2025; "
    "朱康和唐勇, 2025），以同年度同行业（证监会二级，剔除本企业）的DU_kw均值作为工具变量。"
    "一阶段回归中同行业均值系数为0.597（p<0.01），"
    "Kleibergen-Paap rk Wald F统计量为265.1，远超Stock和Yogo (2005) "
    "在10%最大偏误下的临界值16.38；二阶段系数为-0.0147（p<0.01），"
    "方向与OLS一致且绝对值约为OLS的3.5倍。"
    "另外构造Bartik型移位份额工具变量（Goldsmith-Pinkham等, 2020），"
    "一阶段F统计量为62.5，二阶段系数为-0.0112（p=0.080），方向一致。"
    "进一步以DU_kw(t-1)的同行业均值滞后值作为工具变量进行2SLS估计，"
    "一阶段F统计量为183.5，二阶段系数为-0.0165（p<0.01）。"
    "三种设定下2SLS系数（0.011至0.017）均大于OLS（0.004），"
    "与DU_kw作为关键词代理变量含有测量误差、OLS存在衰减偏误的预期一致。"
    "结果见表4。"
)

ITEM_8 = (
    "8. 系数稳定性检验。采用Oster (2019) 提出的方法评估遗漏变量的潜在影响。"
    "以R²_max=1.3R̃=0.553计算，δ*=27.8；以更保守的R²_max=min(1, 1.3R̃)计算，δ*=3.6。"
    "两种设定下δ*均远超Oster (2019) 建议的临界值1，遗漏变量不太可能推翻基准结论。"
)

ITEM_9 = (
    "9. 滞后回归与披露后窗口。将自变量替换为滞后一期DU_kw(t-1)，该口径在时序上更为干净："
    "t-1年年报在t年4月底前披露，而PriceDelay(t)覆盖t年全年日度收益率，"
    "年报披露后至少有8个月的重叠窗口。OLS估计中系数为-0.0036（p<0.01），与同期口径方向一致。"
    "进一步构造披露后窗口股价延迟PriceDelay_post（以t+1年5月至12月的日度收益率计算），"
    "此时DU_kw(t)对应年报在因变量测度窗口之前已完全公开，时序对齐无歧义。"
    "系数为-0.004（t=-2.58，p=0.010，N=42,683），确认主结论不受信息披露时点的影响。"
)

ITEM_10 = (
    "10. 构念效度检验。为排除DU_kw反映的是年报写作风格或泛化热词而非数据要素利用的可能，"
    "进行三组检验（详见附表2）。"
    "第一，在基准模型中分别加入年报总字数对数和MD&A字数对数，"
    "DU_kw系数在三种设定下均保持稳健（t值分别为-4.44、-4.11和-4.19），"
    "表明其解释力来自关键词的语义内容而非文本篇幅。"
    "第二，构造泛战略叙事词频作为安慰剂指标，与DU_kw同时纳入后安慰剂变为不显著（t=-1.18），"
    "而DU_kw系数几乎不变（t=-4.19）。"
    "第三，从关键词表中剔除8个最泛化的词后重新计算DU_kw_strict，"
    "系数仍然显著（t=-3.90，p<0.001）。"
)

SUMMARY = (
    "综合以上检验，数据要素利用对资本市场定价效率的正向影响在替换固定效应结构、"
    "行业均值调整、剔除特定样本、倾向得分匹配、替换聚类标准误、"
    "工具变量、系数稳定性和时序识别等多重检验下均保持稳健。"
)

# ============================================================
# Apply changes: Para [89]-[101]
# ============================================================

new_texts = [
    TITLE,    # [89] heading
    INTRO,    # [90]
    ITEM_1,   # [91]
    ITEM_2,   # [92]
    ITEM_3,   # [93]
    ITEM_4,   # [94]
    ITEM_5,   # [95]
    ITEM_6,   # [96]
    ITEM_7,   # [97] (was empty)
    ITEM_8,   # [98] (was empty)
    ITEM_9,   # [99]
    ITEM_10,  # [100]
    SUMMARY,  # [101]
]

for i, text in enumerate(new_texts):
    para_idx = 89 + i
    p = doc.paragraphs[para_idx]
    old_style = p.style.name
    old_text_preview = p.text[:40] if p.text else "(empty)"

    replace_para_text(p, text)

    # Ensure heading stays as 标题3, content stays as Normal
    if para_idx == 89:
        p.style = doc.styles['标题3']
    else:
        p.style = doc.styles['Normal']

    print(f"[{para_idx}] {old_style}: {old_text_preview}...")
    print(f"  → {text[:60]}...")
    print()

# Verify next paragraph is still the channel analysis heading
p102 = doc.paragraphs[102]
print(f"[102] (unchanged): {p102.style.name} = {p102.text}")

doc.save(DOC_PATH)
print(f"\nDone! Saved to {DOC_PATH}")
