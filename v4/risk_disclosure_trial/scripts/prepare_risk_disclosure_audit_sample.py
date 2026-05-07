import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial")
DEFAULT_FEATURES = ROOT / "risk_disclosure_verifiable_features_2015_2023.parquet"

SUSPECT_HEADER_TERMS = [
    "金融工具",
    "财务报表",
    "附注",
    "会计政策",
    "公司治理",
    "内部控制",
    "社会责任",
    "债券",
    "担保情况",
    "诉讼事项",
    "风险管理体系",
]


def take(df, n, label, random_state=20260506):
    if df.empty:
        return df.assign(audit_stratum=label)
    return df.sample(min(n, len(df)), random_state=random_state).assign(audit_stratum=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--n", type=int, default=220)
    args = parser.parse_args()

    feature_path = Path(args.features)
    df = pd.read_parquet(feature_path)
    for col in ["risk_text", "risk_preview", "risk_headers"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["suspect_header"] = df["risk_headers"].apply(
        lambda x: int(any(term in x for term in SUSPECT_HEADER_TERMS))
    )
    df["risk_text_len"] = df["risk_text"].str.len()

    samples = []
    samples.append(
        df[df["has_risk_text"].eq(1)]
        .sort_values(["risk_verifiability_index", "risk_chars"], ascending=[False, False])
        .head(55)
        .assign(audit_stratum="top_verifiable")
    )
    samples.append(
        df[df["has_risk_text"].eq(1)]
        .sort_values(["np_liability_verifiability_index", "risk_chars"], ascending=[False, False])
        .head(55)
        .assign(audit_stratum="top_np_liability")
    )
    samples.append(
        df[df["suspect_header"].eq(1)]
        .sort_values(["risk_chars"], ascending=False)
        .head(45)
        .assign(audit_stratum="suspect_header")
    )
    samples.append(
        take(df[df["has_risk_text"].eq(1)], 45, "random_has_risk")
    )
    samples.append(
        take(df[df["has_risk_text"].eq(0)], 20, "random_no_risk")
    )

    out = pd.concat(samples, ignore_index=True)
    out = out.drop_duplicates(["Stkcd", "year"], keep="first")
    if len(out) > args.n:
        priority = {
            "top_verifiable": 1,
            "top_np_liability": 2,
            "suspect_header": 3,
            "random_has_risk": 4,
            "random_no_risk": 5,
        }
        out["_priority"] = out["audit_stratum"].map(priority).fillna(9)
        out = out.sort_values(["_priority", "risk_verifiability_index"], ascending=[True, False]).head(args.n)
        out = out.drop(columns="_priority")

    audit_cols = [
        "audit_id",
        "audit_stratum",
        "Stkcd",
        "year",
        "filename",
        "has_risk_text",
        "suspect_header",
        "risk_headers",
        "risk_chars",
        "risk_text_len",
        "risk_verifiability_index",
        "np_liability_verifiability_index",
        "risk_verif_raw",
        "np_liability_risk_verif_raw",
        "risk_verif_legal_process_count",
        "risk_verif_guarantee_pledge_count",
        "risk_verif_debt_distress_count",
        "np_liability_np_actor_count",
        "np_liability_np_guarantee_count",
        "np_liability_recourse_capacity_count",
        "np_liability_control_right_count",
        "risk_preview",
        "risk_text",
        "manual_valid_risk_section",
        "manual_false_positive_type",
        "manual_verifiable_true",
        "manual_np_liability_true",
        "manual_notes",
    ]

    out = out.reset_index(drop=True)
    out["audit_id"] = np.arange(1, len(out) + 1)
    for col in audit_cols:
        if col not in out.columns:
            out[col] = ""
    out = out[audit_cols]

    suffix = feature_path.stem.replace("risk_disclosure_verifiable_features_", "")
    csv_path = ROOT / f"risk_disclosure_manual_audit_sheet_{suffix}.csv"
    md_path = ROOT / f"risk_disclosure_manual_audit_instructions_{suffix}.md"
    out.to_csv(csv_path, index=False)

    lines = [
        "# 风险披露抽取人工核验说明\n\n",
        f"样本文件：`{csv_path}`\n\n",
        f"样本量：{len(out)} 条。\n\n",
        "## 核验列填写规则\n\n",
        "- `manual_valid_risk_section`：1=确实是年报风险披露段，0=误抓。\n",
        "- `manual_false_positive_type`：若误抓，填 `financial_instrument`、`footnote`、`governance`、`bond_guarantee`、`litigation_note`、`other`。\n",
        "- `manual_verifiable_true`：1=确实包含可外部核验的具体风险信息，0=主要是模板化/空泛表述。\n",
        "- `manual_np_liability_true`：1=确实涉及自然人责任主体、个人担保、控制人质押、追偿/代偿/控制权风险，0=不涉及。\n",
        "- `manual_notes`：记录明显问题，如风险段太短、标题误抓、金融工具附注、只出现政策名词等。\n\n",
        "## 抽样层\n\n",
        "- `top_verifiable`：可核验披露指数最高的样本。\n",
        "- `top_np_liability`：自然人责任可核验指数最高的样本。\n",
        "- `suspect_header`：标题可能误抓的样本，用来估计误抓率。\n",
        "- `random_has_risk`：随机有风险段样本。\n",
        "- `random_no_risk`：随机无风险段样本，用来检查漏抓。\n",
    ]
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
