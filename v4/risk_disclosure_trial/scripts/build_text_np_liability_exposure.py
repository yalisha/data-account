import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from extract_risk_disclosure_features import ZIP_DIR, OUT, decode_bytes, decode_zipname, load_target_set, parse_years


NP_ACTOR = r"(实际控制人|实控人|控股股东|法定代表人|董事长|总经理|董监高|自然人股东|自然人|个人|配偶|亲属)"
GUARANTEE = r"(个人担保|个人保证|连带保证|连带责任保证|无限连带责任|个人连带责任|个人反担保|共同担保)"
PLEDGE = r"(股权质押|股份质押|质押|司法冻结|冻结|平仓|补仓|控制权稳定|控制权变更)"

NP_GUARANTEE_RE = re.compile(
    rf"("
    rf"{NP_ACTOR}.{{0,60}}(为|向|对|给|提供|承担).{{0,60}}{GUARANTEE}|"
    rf"{NP_ACTOR}.{{0,60}}{GUARANTEE}|"
    rf"{GUARANTEE}.{{0,60}}{NP_ACTOR}|"
    rf"(实际控制人|实控人|控股股东|法定代表人|董事长|总经理|自然人股东|配偶|亲属).{{0,80}}(为|对|向).{{0,40}}(本公司|公司|上市公司|子公司|借款|贷款|债务|授信|融资).{{0,60}}(担保|保证|反担保)"
    rf")"
)
CONTROLLER_PLEDGE_RE = re.compile(
    rf"((实际控制人|实控人|控股股东|法定代表人|董事长|总经理|自然人股东).{{0,70}}{PLEDGE}|"
    rf"{PLEDGE}.{{0,70}}(实际控制人|实控人|控股股东|法定代表人|董事长|总经理|自然人股东))"
)
PERSONAL_RECOURSE_RE = re.compile(
    rf"({NP_ACTOR}.{{0,80}}(追偿|代偿|清偿|偿付|履约能力|偿债能力|信用状况|财产线索)|"
    rf"(追偿|代偿|清偿|偿付|履约能力|偿债能力|信用状况|财产线索).{{0,80}}{NP_ACTOR})"
)

SECTION_EXCLUDE_RE = re.compile(
    r"(员工持股计划|股权激励|社会责任|职工|薪酬|简历|任职资格|"
    r"保证年度报告内容|保证上市公司|保证关联交易|保证不损害|保证将依照|"
    r"本公司保证|公司保证|承诺及保证|不可撤销的承诺|同业竞争|独立意见|专项说明|"
    r"不存在.{0,40}担保|没有.{0,40}担保|未向.{0,40}担保|未为.{0,40}担保|无.{0,40}担保|"
    r"诚信状况|不存在未履行法院生效判决|债务到期未清偿|"
    r"股东数量及持股情况|无质押|质押0|冻结0|未质押|不存在质押|"
    r"母公司情况|最终控制方|结构化主体|"
    r"为股东、实际控制人提供担保|其中：为股东、实际控制人|"
    r"实际控制人及其关联方提供担保|实际控制人及其关联方提供担保的余额|"
    r"实际控制人及其关联方提供担保的金额)"
)


def clean_text(text):
    return re.sub(r"\s+", "", text)


def count_and_snippets(text, pattern, max_snippets=4):
    matches = []
    count = 0
    for m in pattern.finditer(text):
        s = m.group(0)
        if SECTION_EXCLUDE_RE.search(s):
            continue
        count += 1
        if len(matches) < max_snippets:
            matches.append(s[:180])
    return count, " || ".join(matches)


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
    text = clean_text(decode_bytes(data))
    guarantee_count, guarantee_snip = count_and_snippets(text, NP_GUARANTEE_RE)
    pledge_count, pledge_snip = count_and_snippets(text, CONTROLLER_PLEDGE_RE)
    recourse_count, recourse_snip = count_and_snippets(text, PERSONAL_RECOURSE_RE)
    total = guarantee_count + pledge_count + recourse_count
    return {
        "Stkcd": stkcd,
        "year": year,
        "filename": decoded,
        "ar_np_guarantee_count": guarantee_count,
        "ar_controller_pledge_count": pledge_count,
        "ar_personal_recourse_count": recourse_count,
        "ar_np_liability_text_count": total,
        "ar_np_guarantee_dummy": int(guarantee_count > 0),
        "ar_controller_pledge_dummy": int(pledge_count > 0),
        "ar_personal_recourse_dummy": int(recourse_count > 0),
        "ar_np_liability_text_dummy": int(total > 0),
        "np_guarantee_snippet": guarantee_snip,
        "controller_pledge_snippet": pledge_snip,
        "personal_recourse_snippet": recourse_snip,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2016-2018")
    parser.add_argument("--limit-per-year", type=int, default=0)
    args = parser.parse_args()

    years = parse_years(args.years)
    target = load_target_set(years)
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
            if key not in target:
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

    df = df.sort_values(["Stkcd", "year", "ar_np_liability_text_count"], ascending=[True, True, False]).drop_duplicates(
        ["Stkcd", "year"], keep="first"
    )
    pre = df.groupby("Stkcd").agg(
        ar_np_guarantee_pre=("ar_np_guarantee_dummy", "max"),
        ar_controller_pledge_pre=("ar_controller_pledge_dummy", "max"),
        ar_personal_recourse_pre=("ar_personal_recourse_dummy", "max"),
        ar_np_liability_text_pre=("ar_np_liability_text_dummy", "max"),
        ar_np_guarantee_count_pre=("ar_np_guarantee_count", "mean"),
        ar_controller_pledge_count_pre=("ar_controller_pledge_count", "mean"),
        ar_personal_recourse_count_pre=("ar_personal_recourse_count", "mean"),
        ar_np_liability_text_count_pre=("ar_np_liability_text_count", "mean"),
    ).reset_index()
    pre["ar_np_liability_index_pre"] = (
        pre[["ar_np_guarantee_pre", "ar_controller_pledge_pre", "ar_personal_recourse_pre"]].sum(axis=1) / 3.0
    )

    suffix = f"{min(years)}_{max(years)}"
    if args.limit_per_year:
        suffix += f"_limit{args.limit_per_year}"
    fy_path = OUT / f"np_liability_text_exposure_firmyear_{suffix}.parquet"
    pre_path = OUT / f"np_liability_text_exposure_pre_{suffix}.csv"
    audit_path = OUT / f"np_liability_text_exposure_audit_sample_{suffix}.csv"
    summary_path = OUT / f"np_liability_text_exposure_summary_{suffix}.csv"
    df.to_parquet(fy_path, index=False)
    pre.to_csv(pre_path, index=False)
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    df.sort_values("ar_np_liability_text_count", ascending=False).head(240).to_csv(audit_path, index=False)

    print(f"wrote {fy_path}")
    print(f"wrote {pre_path}")
    print(f"wrote {audit_path}")
    print(
        pre[
            [
                "ar_np_guarantee_pre",
                "ar_controller_pledge_pre",
                "ar_personal_recourse_pre",
                "ar_np_liability_text_pre",
            ]
        ].mean().to_string()
    )


if __name__ == "__main__":
    main()
