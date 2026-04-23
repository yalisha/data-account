"""
Extract a 120-context stratified validation sample for v14.

Strata:
  generic / non-generic
  low-score parent firm-year (llm_score 0/1) / high-score parent firm-year (2/3)
  sparse window-hit density (1 keyword hit) / dense (>=2 hits)

Outputs:
  - results/v14/llm_validation_sample.csv
  - results/v14/llm_validation_examples.md
"""

from __future__ import annotations

import csv
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ZIP_DIR = BASE / "第三方资料/第三方数据资源/2001~2024年年报/2001-2024年A股年报TXT格式"
AR_FEAT = BASE / "data_parquet" / "annual_report_features.parquet"
REG = BASE / "data_stata" / "reg_sample_v18.dta"
OUT_DIR = BASE / "results" / "v14"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUZZWORDS = {
    "数字化转型",
    "数字化",
    "智能化",
    "信息化",
    "人工智能",
    "大数据",
    "智能制造",
    "智慧城市",
}

KEYWORDS = {
    "data_stock": [
        "大数据", "数据库", "数据中心", "数据仓库", "数据湖",
        "数据集", "数据存储", "数据采集", "数据资源", "数据积累",
        "用户数据", "客户数据", "行为数据", "交易数据", "运营数据",
    ],
    "data_dev": [
        "数据挖掘", "数据分析", "数据处理", "数据清洗", "数据建模",
        "机器学习", "深度学习", "人工智能", "算法", "自然语言处理",
        "数据科学", "数据工程", "数据平台", "数据中台", "数据架构",
        "数字化转型", "数字化", "智能化", "信息化",
    ],
    "data_app": [
        "数据驱动", "精准营销", "个性化推荐", "智能推荐", "用户画像",
        "风险控制", "风控模型", "智能决策", "智能客服", "智能制造",
        "预测模型", "需求预测", "供应链优化", "智慧物流", "智慧城市",
        "数字营销", "程序化", "数据赋能", "数据服务",
    ],
    "data_value": [
        "数据资产", "数据要素", "数据交易", "数据产品", "数据变现",
        "数据确权", "数据定价", "数据流通", "数据市场", "数据入表", "数据资源入表",
    ],
    "data_gov": [
        "数据治理", "数据安全", "数据隐私", "数据合规", "数据质量",
        "数据标准", "数据脱敏", "个人信息保护", "数据分类分级",
    ],
}
ALL_KEYWORDS = sorted({kw for kws in KEYWORDS.values() for kw in kws}, key=len, reverse=True)
YEARS = range(2010, 2025)
TARGET_PER_STRATUM = 15
RESERVOIR_SIZE = 30
WINDOW = 100
random.seed(20260412)


def decode_zipname(raw_name: str) -> str:
    try:
        return raw_name.encode("cp437").decode("gbk")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw_name


def parse_file_meta(raw_name: str):
    decoded = decode_zipname(raw_name)
    parts = decoded.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def load_text(zf: zipfile.ZipFile, raw_name: str) -> str:
    data = zf.read(raw_name)
    for enc in ["utf-8", "gbk", "gb18030", "gb2312"]:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def reservoir_add(bucket: list[dict], item: dict, seen_n: int) -> None:
    if len(bucket) < RESERVOIR_SIZE:
        bucket.append(item)
        return
    j = random.randint(0, seen_n - 1)
    if j < RESERVOIR_SIZE:
        bucket[j] = item


def build_sample():
    reg = pd.read_stata(REG, columns=["Stkcd", "year", "llm_score"])
    reg["Stkcd"] = reg["Stkcd"].astype(int)
    reg["year"] = reg["year"].astype(int)
    score_map = {(r.Stkcd, r.year): int(r.llm_score) for r in reg.itertuples()}

    reservoirs: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    seen_counter: dict[tuple[str, str, str], int] = defaultdict(int)

    zip_files = {int(p.name.split("_")[0]): p for p in ZIP_DIR.glob("*.zip") if "_" in p.name}

    for year in YEARS:
        zip_path = zip_files.get(year)
        if zip_path is None:
            continue
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".txt")]
            for raw_name in names:
                meta = parse_file_meta(raw_name)
                if meta is None:
                    continue
                stkcd, yr = meta
                parent_score = score_map.get((stkcd, yr))
                if parent_score is None:
                    continue
                score_bucket = "high" if parent_score >= 2 else "low"
                text = load_text(zf, raw_name)
                for kw in ALL_KEYWORDS:
                    start = 0
                    while True:
                        idx = text.find(kw, start)
                        if idx == -1:
                            break
                        left = max(0, idx - WINDOW)
                        right = min(len(text), idx + len(kw) + WINDOW)
                        context = text[left:right].replace("\n", " ").replace("\r", " ")
                        window_hits = sum(context.count(k) for k in ALL_KEYWORDS)
                        density_bucket = "dense" if window_hits >= 2 else "sparse"
                        generic_bucket = "generic" if kw in BUZZWORDS else "nongeneric"
                        stratum = (generic_bucket, score_bucket, density_bucket)
                        seen_counter[stratum] += 1
                        item = {
                            "Stkcd": stkcd,
                            "year": yr,
                            "parent_llm_score": parent_score,
                            "keyword": kw,
                            "generic_bucket": generic_bucket,
                            "score_bucket": score_bucket,
                            "density_bucket": density_bucket,
                            "window_hits": window_hits,
                            "context": re.sub(r"\s+", " ", context).strip(),
                        }
                        reservoir_add(reservoirs[stratum], item, seen_counter[stratum])
                        start = idx + len(kw)

    final_rows = []
    for generic_bucket in ["generic", "nongeneric"]:
        for score_bucket in ["low", "high"]:
            rows = []
            for density_bucket in ["sparse", "dense"]:
                rows.extend(reservoirs[(generic_bucket, score_bucket, density_bucket)][:TARGET_PER_STRATUM])
            sparse_rows = [r for r in rows if r["density_bucket"] == "sparse"][:TARGET_PER_STRATUM]
            dense_rows = [r for r in rows if r["density_bucket"] == "dense"][:TARGET_PER_STRATUM]
            combo = sparse_rows + dense_rows
            if len(sparse_rows) < TARGET_PER_STRATUM or len(dense_rows) < TARGET_PER_STRATUM:
                backup = rows[TARGET_PER_STRATUM * 2 :]
                needed = TARGET_PER_STRATUM * 2 - len(combo)
                combo.extend(backup[:needed])
            final_rows.extend(combo[: TARGET_PER_STRATUM * 2])

    out = pd.DataFrame(final_rows).drop_duplicates(subset=["Stkcd", "year", "keyword", "context"])
    out = out.reset_index(drop=True)
    out["sample_id"] = [f"S{i + 1:03d}" for i in range(len(out))]
    return out[
        [
            "sample_id",
            "Stkcd",
            "year",
            "parent_llm_score",
            "keyword",
            "generic_bucket",
            "score_bucket",
            "density_bucket",
            "window_hits",
            "context",
        ]
    ]


sample = build_sample()
sample.to_csv(OUT_DIR / "llm_validation_sample.csv", index=False, quoting=csv.QUOTE_MINIMAL)

examples = []
for score in [0, 1, 2, 3]:
    hit = sample[sample["parent_llm_score"] == score].head(1)
    if len(hit) == 0:
        continue
    row = hit.iloc[0]
    examples.append(
        f"### 评分{score}示例\n\n"
        f"- 关键词：`{row['keyword']}`\n"
        f"- 样本：`{int(row['Stkcd'])}`，`{int(row['year'])}`\n"
        f"- 上下文：{row['context']}\n"
    )

(OUT_DIR / "llm_validation_examples.md").write_text(
    "# v14 LLM复核样本示例\n\n" + "\n".join(examples),
    encoding="utf-8",
)

print(f"Saved {len(sample)} contexts to {OUT_DIR / 'llm_validation_sample.csv'}")
