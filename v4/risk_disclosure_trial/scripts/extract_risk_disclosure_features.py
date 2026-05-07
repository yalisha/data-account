import argparse
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/mac/computerscience/0做完了/15会计研究")
ZIP_DIR = Path("/Users/mac/computerscience/第三方资料/第三方数据资源/2001~2024年年报/2001-2024年A股年报TXT格式")
STATA = ROOT / "v1" / "data_stata"
OUT = ROOT / "v4" / "risk_disclosure_trial"

MDA_PAT = re.compile(r"(管理层讨论与分析|经营情况讨论与分析|董事会报告)")
TOP_HEADER_RE = re.compile(r"^第[一二三四五六七八九十百]+[章节]\s*.+$", re.MULTILINE)
TOP_HEADER_KEY_RE = re.compile(r"^(第[一二三四五六七八九十百]+[章节])")
HEADING_RE = re.compile(
    r"^\s*(第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十百]+[、.．]|[0-9]{1,2}[、.．]|[（(][一二三四五六七八九十百0-9]{1,2}[）)]).{0,80}$"
)
RISK_HEADER_RE = re.compile(
    r"(重大风险提示|风险因素|主要风险|面临的风险|面对的风险|可能面临|可能面对|风险及应对|风险和应对|风险与应对|风险管理|风险控制)"
)
EXCLUDE_HEADER_RE = re.compile(
    r"(金融工具|套期|衍生|审计报告|财务报表|附注|会计政策|公司治理|内部控制|社会责任|债券|担保情况|诉讼事项)"
)

RISK_CATEGORY_TERMS = {
    "market": ["市场竞争", "市场需求", "价格波动", "行业周期", "宏观经济", "经济下行"],
    "finance": ["融资", "资金", "现金流", "流动性", "利率", "汇率", "债务", "偿债", "授信", "信贷"],
    "credit_debt": ["应收账款", "坏账", "客户信用", "信用风险", "担保", "保证", "违约", "逾期"],
    "operation": ["原材料", "供应商", "供应链", "库存", "存货", "订单", "产能", "生产", "采购"],
    "technology": ["技术", "研发", "产品迭代", "知识产权", "专利", "核心技术", "技术更新"],
    "policy_legal": ["政策", "监管", "法律", "诉讼", "合规", "税收", "环保", "安全生产"],
    "human_capital": ["人才", "员工", "核心人员", "管理团队", "人员流失", "薪酬"],
    "international": ["海外", "国际", "贸易摩擦", "关税", "出口", "进口", "地缘"],
}

SPECIFIC_TERMS = sorted(
    {
        "客户",
        "供应商",
        "原材料",
        "汇率",
        "利率",
        "政策",
        "监管",
        "市场",
        "技术",
        "研发",
        "诉讼",
        "债务",
        "担保",
        "现金流",
        "融资",
        "信用",
        "存货",
        "应收账款",
        "海外",
        "环保",
        "安全生产",
        "合规",
        "产能",
        "订单",
        "供应链",
        "违约",
        "逾期",
    },
    key=len,
    reverse=True,
)

ACTION_TERMS = sorted(
    {
        "应对",
        "加强",
        "采取",
        "建立",
        "完善",
        "控制",
        "管理",
        "防范",
        "降低",
        "监测",
        "优化",
        "提升",
        "改善",
        "推进",
        "调整",
        "储备",
        "拓展",
    },
    key=len,
    reverse=True,
)

GENERIC_TERMS = sorted(
    {
        "可能面临",
        "存在不确定性",
        "不确定因素",
        "敬请投资者注意风险",
        "市场竞争加剧",
        "宏观经济形势",
        "行业政策变化",
        "公司将密切关注",
        "积极采取措施",
        "进一步加强",
    },
    key=len,
    reverse=True,
)

DEBT_RISK_TERMS = sorted(
    {
        "债务",
        "偿债",
        "融资",
        "授信",
        "银行贷款",
        "借款",
        "担保",
        "保证",
        "违约",
        "逾期",
        "流动性风险",
        "信用风险",
        "资金链",
        "现金流",
    },
    key=len,
    reverse=True,
)


def decode_zipname(raw_name):
    try:
        return raw_name.encode("cp437").decode("gbk")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw_name


def decode_bytes(data):
    for enc in ["utf-8", "gbk", "gb18030", "gb2312"]:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            pass
    return data.decode("utf-8", errors="replace")


def count_terms(text, terms):
    return sum(text.count(term) for term in terms)


def line_offsets(text):
    pos = 0
    for line in text.splitlines(True):
        yield pos, line
        pos += len(line)


def extract_top_section(text, pat):
    headers = []
    for m in TOP_HEADER_RE.finditer(text):
        title = m.group().strip()
        key_match = TOP_HEADER_KEY_RE.search(title)
        key = key_match.group(1) if key_match else title[:8]
        headers.append((m.start(), m.end(), title, key))
    candidates = []
    for idx, (start, end, title, key) in enumerate(headers):
        if pat.search(title):
            next_start = len(text)
            for next_start_i, _, _, next_key in headers[idx + 1 :]:
                if next_key != key and next_start_i > end + 2000:
                    next_start = next_start_i
                    break
            candidates.append(text[end:next_start])
    if not candidates:
        return ""
    return max(candidates, key=len)


def is_heading(line):
    s = line.strip()
    if len(s) < 4 or len(s) > 90:
        return False
    return bool(HEADING_RE.search(s))


def extract_risk_blocks(text):
    mda = extract_top_section(text, MDA_PAT)
    search_text = mda if len(mda) > 500 else text

    lines = list(line_offsets(search_text))
    headers = []
    for i, (pos, line) in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        strong_risk_header = bool(
            re.search(r"(重大风险提示|风险因素|主要风险|公司面临的风险|可能面临的风险|可能面对的风险|面临的风险及应对|风险及应对)", s)
        )
        if (
            RISK_HEADER_RE.search(s)
            and not EXCLUDE_HEADER_RE.search(s)
            and len(s) <= 120
            and (strong_risk_header or is_heading(s))
        ):
            headers.append((i, pos, s))

    blocks = []
    for idx, (line_i, start_pos, title) in enumerate(headers):
        content_start = start_pos + len(lines[line_i][1])
        content_end = len(search_text)
        for j in range(line_i + 1, len(lines)):
            next_pos, next_line = lines[j]
            if next_pos <= content_start + 200:
                continue
            next_s = next_line.strip()
            if TOP_HEADER_RE.search(next_s) and MDA_PAT.search(next_s):
                continue
            if is_heading(next_line):
                content_end = next_pos
                break
        block = search_text[content_start:content_end].strip()
        if len(block) >= 80:
            blocks.append((title, block[:60000]))

    return blocks, bool(mda)


def category_count(text):
    return sum(any(term in text for term in terms) for terms in RISK_CATEGORY_TERMS.values())


def process_report(zip_path, raw_name):
    decoded = decode_zipname(raw_name)
    parts = decoded.split("_")
    if len(parts) < 3:
        return None
    try:
        stkcd = int(parts[0])
        year = int(parts[1])
    except ValueError:
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            data = zf.read(raw_name)
    except Exception:
        return None

    text = decode_bytes(data)
    total_chars = len(text)
    blocks, has_mda = extract_risk_blocks(text)
    risk_text = "\n".join(block for _, block in blocks)
    risk_chars = len(risk_text)
    denom = max(risk_chars, 1)

    number_count = len(re.findall(r"\d+(?:\.\d+)?\s*(?:%|万|亿|元|吨|台|件|人|项|次)?", risk_text))
    specific_count = count_terms(risk_text, SPECIFIC_TERMS)
    action_count = count_terms(risk_text, ACTION_TERMS)
    generic_count = count_terms(risk_text, GENERIC_TERMS)
    debt_risk_count = count_terms(risk_text, DEBT_RISK_TERMS)
    cat_count = category_count(risk_text)

    detail_count = number_count + specific_count + action_count
    risk_chars_per10k = risk_chars / max(total_chars, 1) * 10000
    detail_per10k = detail_count / denom * 10000
    number_per10k = number_count / denom * 10000
    specific_per10k = specific_count / denom * 10000
    action_per10k = action_count / denom * 10000
    debt_per10k = debt_risk_count / denom * 10000
    boilerplate_ratio = generic_count / max(detail_count + generic_count, 1)

    return {
        "Stkcd": stkcd,
        "year": year,
        "filename": decoded,
        "total_chars": total_chars,
        "has_mda": int(has_mda),
        "has_risk_text": int(risk_chars > 0),
        "risk_block_count": len(blocks),
        "risk_headers": " | ".join(title for title, _ in blocks[:8]),
        "risk_chars": risk_chars,
        "risk_chars_ln": np.log1p(risk_chars),
        "risk_chars_per10k": risk_chars_per10k,
        "risk_number_count": number_count,
        "risk_specific_count": specific_count,
        "risk_action_count": action_count,
        "risk_generic_count": generic_count,
        "risk_debt_count": debt_risk_count,
        "risk_category_count": cat_count,
        "risk_detail_per10k": detail_per10k,
        "risk_number_per10k": number_per10k,
        "risk_specific_per10k": specific_per10k,
        "risk_action_per10k": action_per10k,
        "risk_debt_per10k": debt_per10k,
        "risk_boilerplate_ratio": boilerplate_ratio,
        "risk_preview": risk_text[:220].replace("\n", " "),
    }


def load_target_set(years):
    panel = pd.read_stata(STATA / "reg_sample_v18.dta", columns=["Stkcd", "year"], convert_categoricals=False)
    panel = panel[panel["year"].isin(years)].copy()
    return set((int(s), int(y)) for s, y in panel[["Stkcd", "year"]].dropna().itertuples(index=False, name=None))


def zscore(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd


def finalize(df):
    out = df.sort_values(["Stkcd", "year", "risk_chars"], ascending=[True, True, False]).drop_duplicates(
        ["Stkcd", "year"], keep="first"
    )
    for col in [
        "risk_chars_per10k",
        "risk_detail_per10k",
        "risk_number_per10k",
        "risk_specific_per10k",
        "risk_action_per10k",
        "risk_debt_per10k",
        "risk_boilerplate_ratio",
    ]:
        lo, hi = out[col].quantile([0.01, 0.99])
        out[col] = out[col].clip(lo, hi)

    out["risk_specificity_index"] = (
        zscore(out["risk_detail_per10k"].fillna(0))
        + zscore(out["risk_number_per10k"].fillna(0))
        + zscore(out["risk_category_count"].fillna(0))
        - zscore(out["risk_boilerplate_ratio"].fillna(0))
    ) / 4.0
    out["risk_quality_index"] = (
        zscore(out["risk_chars_ln"].fillna(0))
        + zscore(out["risk_specificity_index"].fillna(0))
        + zscore(out["risk_category_count"].fillna(0))
        - zscore(out["risk_boilerplate_ratio"].fillna(0))
    ) / 4.0
    return out


def parse_years(value):
    if "-" in value:
        a, b = value.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2015-2023")
    parser.add_argument("--limit-per-year", type=int, default=0)
    parser.add_argument("--all-firms", action="store_true")
    args = parser.parse_args()

    years = parse_years(args.years)
    OUT.mkdir(parents=True, exist_ok=True)
    target = None if args.all_firms else load_target_set(years)
    rows = []
    summary = []

    for year in years:
        zips = sorted([p for p in ZIP_DIR.iterdir() if p.name.startswith(f"{year}_") and p.suffix == ".zip"])
        if not zips:
            summary.append({"year": year, "status": "missing_zip", "n_zip_reports": 0, "n_extracted": 0})
            continue
        zip_path = zips[0]
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
        if args.limit_per_year:
            names = names[: args.limit_per_year]

        extracted = 0
        for raw_name in names:
            decoded = decode_zipname(raw_name)
            parts = decoded.split("_")
            if len(parts) < 2:
                continue
            try:
                key = (int(parts[0]), int(parts[1]))
            except ValueError:
                continue
            if target is not None and key not in target:
                continue
            r = process_report(zip_path, raw_name)
            if r is not None:
                rows.append(r)
                extracted += 1
        summary.append({"year": year, "status": "ok", "n_zip_reports": len(names), "n_extracted": extracted})
        print(f"{year}: extracted {extracted} from {len(names)} reports")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No reports extracted.")
    out = finalize(df)
    suffix = f"{min(years)}_{max(years)}"
    if args.limit_per_year:
        suffix += f"_limit{args.limit_per_year}"
    if args.all_firms:
        suffix += "_allfirms"
    feature_path = OUT / f"risk_disclosure_features_{suffix}.parquet"
    csv_path = OUT / f"risk_disclosure_features_{suffix}.csv"
    summary_path = OUT / f"risk_disclosure_extract_summary_{suffix}.csv"
    audit_path = OUT / f"risk_disclosure_audit_sample_{suffix}.csv"
    out.to_parquet(feature_path, index=False)
    out.to_csv(csv_path, index=False)
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    out.sort_values("risk_chars", ascending=False).head(200).to_csv(audit_path, index=False)
    print(f"wrote {feature_path}")
    print(f"wrote {summary_path}")
    print(f"obs={len(out)}, firms={out.Stkcd.nunique()}, years={sorted(out.year.unique())}")


if __name__ == "__main__":
    main()
