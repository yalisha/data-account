import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from extract_risk_disclosure_features import (
    ZIP_DIR,
    OUT,
    decode_bytes,
    decode_zipname,
    extract_risk_blocks,
    load_target_set,
    parse_years,
    zscore,
)


POLICY_TERMS = sorted(
    {
        "个人破产制度",
        "个人债务集中清理",
        "债务集中清理",
        "类个人破产",
        "自然人破产",
        "个人破产",
        "破产试点",
        "破产保护",
    },
    key=len,
    reverse=True,
)

VERIFIABLE_COMPONENTS = {
    "amount_timing": [
        "万元",
        "亿元",
        "金额",
        "余额",
        "比例",
        "占比",
        "利率",
        "期限",
        "到期",
        "报告期末",
        "一年内",
        "本期",
    ],
    "responsible_party": [
        "客户",
        "供应商",
        "银行",
        "金融机构",
        "债权人",
        "债务人",
        "担保方",
        "保证人",
        "关联方",
    ],
    "debt_distress": [
        "债务",
        "偿债",
        "融资",
        "授信",
        "银行贷款",
        "借款",
        "现金流",
        "流动性风险",
        "资金链",
        "违约",
        "逾期",
        "展期",
        "延期支付",
        "无法偿还",
        "不能按期偿付",
        "债务重组",
    ],
    "legal_process": [
        "诉讼",
        "仲裁",
        "判决",
        "裁定",
        "执行",
        "强制执行",
        "查封",
        "冻结",
        "保全",
        "被申请执行",
        "被执行人",
    ],
    "guarantee_pledge": [
        "担保",
        "保证",
        "连带责任",
        "反担保",
        "抵押",
        "质押",
        "股权质押",
        "股份质押",
        "抵质押",
    ],
}

NP_LIABILITY_COMPONENTS = {
    "np_actor": [
        "实际控制人",
        "实控人",
        "控股股东",
        "法定代表人",
        "董事长",
        "总经理",
        "董监高",
        "管理层",
        "自然人股东",
        "个人",
        "配偶",
        "亲属",
    ],
    "np_guarantee": [
        "个人担保",
        "个人保证",
        "连带保证",
        "连带责任",
        "无限连带责任",
        "个人连带责任",
        "个人反担保",
        "共同担保",
    ],
    "recourse_capacity": [
        "追偿",
        "代偿",
        "清偿",
        "偿付",
        "履约能力",
        "偿债能力",
        "资产状况",
        "信用状况",
        "财产线索",
    ],
    "control_right": [
        "股权质押",
        "股份质押",
        "平仓",
        "补仓",
        "控制权稳定",
        "控制权变更",
        "表决权",
        "司法冻结",
    ],
}


def count_terms(text, terms):
    return sum(text.count(term) for term in terms)


def remove_policy_terms(text):
    out = text
    for term in POLICY_TERMS:
        out = out.replace(term, "")
    return out


def component_counts(text, components):
    return {name: count_terms(text, terms) for name, terms in components.items()}


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
    blocks, has_mda = extract_risk_blocks(text)
    risk_text = "\n".join(block for _, block in blocks)
    risk_text_no_policy = remove_policy_terms(risk_text)
    risk_chars = len(risk_text)
    denom = max(len(risk_text_no_policy), 1)

    verif = component_counts(risk_text_no_policy, VERIFIABLE_COMPONENTS)
    np_liability = component_counts(risk_text_no_policy, NP_LIABILITY_COMPONENTS)
    number_count = len(re.findall(r"\d+(?:\.\d+)?\s*(?:%|万|亿|元|吨|台|件|人|项|次|年|月|日)?", risk_text_no_policy))
    verif_raw = sum(verif.values()) + number_count
    np_liability_raw = sum(np_liability.values())

    row = {
        "Stkcd": stkcd,
        "year": year,
        "filename": decoded,
        "has_mda": int(has_mda),
        "has_risk_text": int(risk_chars > 0),
        "risk_block_count": len(blocks),
        "risk_headers": " | ".join(title for title, _ in blocks[:8]),
        "risk_chars": risk_chars,
        "risk_text": risk_text[:120000],
        "risk_preview": risk_text[:320].replace("\n", " "),
        "risk_text_no_policy_chars": len(risk_text_no_policy),
        "risk_verif_number_count": number_count,
        "risk_verif_raw": verif_raw,
        "risk_verif_per10k": verif_raw / denom * 10000,
        "risk_verif_share": verif_raw / max(verif_raw + len(risk_text_no_policy) / 100, 1),
        "np_liability_risk_verif_raw": np_liability_raw,
        "np_liability_risk_verif_per10k": np_liability_raw / denom * 10000,
        "np_liability_risk_verif_share": np_liability_raw / max(verif_raw, 1),
    }
    for name, val in verif.items():
        row[f"risk_verif_{name}_count"] = val
        row[f"risk_verif_{name}_per10k"] = val / denom * 10000
    for name, val in np_liability.items():
        row[f"np_liability_{name}_count"] = val
        row[f"np_liability_{name}_per10k"] = val / denom * 10000
    return row


def finalize(df):
    out = df.sort_values(["Stkcd", "year", "risk_chars"], ascending=[True, True, False]).drop_duplicates(
        ["Stkcd", "year"], keep="first"
    )
    clip_cols = [c for c in out.columns if c.endswith("_per10k") or c.endswith("_share")]
    for col in clip_cols:
        x = pd.to_numeric(out[col], errors="coerce")
        if x.notna().sum() == 0:
            continue
        lo, hi = x.quantile([0.01, 0.99])
        if hi <= lo or (hi == 0 and x.max() > 0):
            continue
        out[col] = x.clip(lo, hi)

    out["risk_verifiability_index"] = (
        zscore(out["risk_verif_per10k"].fillna(0))
        + zscore(out["risk_verif_number_count"].fillna(0))
        + zscore(out["risk_verif_legal_process_per10k"].fillna(0))
        + zscore(out["risk_verif_guarantee_pledge_per10k"].fillna(0))
        + zscore(out["risk_verif_debt_distress_per10k"].fillna(0))
    ) / 5.0
    out["np_liability_verifiability_index"] = (
        zscore(out["np_liability_risk_verif_per10k"].fillna(0))
        + zscore(out["np_liability_np_actor_per10k"].fillna(0))
        + zscore(out["np_liability_np_guarantee_per10k"].fillna(0))
        + zscore(out["np_liability_recourse_capacity_per10k"].fillna(0))
        + zscore(out["np_liability_control_right_per10k"].fillna(0))
    ) / 5.0
    return out


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
            row = process_report(zip_path, raw_name)
            if row is not None:
                rows.append(row)
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

    feature_path = OUT / f"risk_disclosure_verifiable_features_{suffix}.parquet"
    csv_path = OUT / f"risk_disclosure_verifiable_features_{suffix}.csv"
    audit_path = OUT / f"risk_disclosure_verifiable_audit_sample_{suffix}.csv"
    summary_path = OUT / f"risk_disclosure_verifiable_extract_summary_{suffix}.csv"

    out.to_parquet(feature_path, index=False)
    csv_cols = [c for c in out.columns if c != "risk_text"]
    out[csv_cols].to_csv(csv_path, index=False)
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    audit_cols = [
        "Stkcd",
        "year",
        "filename",
        "has_risk_text",
        "risk_headers",
        "risk_chars",
        "risk_verifiability_index",
        "np_liability_verifiability_index",
        "risk_verif_raw",
        "np_liability_risk_verif_raw",
        "risk_preview",
        "risk_text",
    ]
    out.sort_values(["risk_verifiability_index", "risk_chars"], ascending=[False, False]).head(240)[audit_cols].to_csv(
        audit_path, index=False
    )
    print(f"wrote {feature_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {audit_path}")
    print(f"obs={len(out)}, firms={out.Stkcd.nunique()}, years={sorted(out.year.unique())}")


if __name__ == "__main__":
    main()
