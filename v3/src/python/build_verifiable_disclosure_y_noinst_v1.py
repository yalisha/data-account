#!/usr/bin/env python3
"""Build no-institution verifiable data-disclosure outcomes from annual reports."""

from __future__ import annotations

import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ZIP_DIR = Path(
    "/Users/mac/computerscience/第三方资料/第三方数据资源/"
    "2001~2024年年报/2001-2024年A股年报TXT格式"
)
OUT = Path(
    "/Users/mac/computerscience/0做完了/15会计研究/v3/results/data/"
    "verifiable_disclosure_y_noinst_v1.csv"
)
YEARS = set(range(2018, 2025))

MASK_TERMS = [
    "数据交易所",
    "数据交易中心",
    "数据要素市场",
    "大数据交易所",
    "大数据交易中心",
]

ASSET_TRADE_NOINST_TERMS = [
    "数据资产入表",
    "数据资源入表",
    "数据资源会计处理",
    "数据资产化",
    "数据产权",
    "数据确权",
    "数据定价",
    "数据估值",
    "数据交易",
    "数据流通",
    "数据挂牌",
    "数据产品",
    "数据资产",
    "数据入表",
]

STRICT_TERMS = [
    "数据资产入表",
    "数据资源入表",
    "数据资源会计处理",
    "数据资产化",
    "数据产权",
    "数据确权",
    "数据定价",
    "数据产品",
    "数据资产",
    "数据入表",
]

ACCOUNTING_TERMS = [
    "数据资产入表",
    "数据资源入表",
    "数据资源会计处理",
    "数据入表",
    "数据资产确认",
    "数据资源确认",
    "数据资产计量",
    "数据资源计量",
    "数据资产列示",
    "数据资源列示",
    "数据资产披露",
    "数据资源披露",
]

RIGHTS_TERMS = [
    "数据产权",
    "数据确权",
    "数据资产确权",
    "数据资源确权",
    "数据权属",
    "数据权利",
    "数据权利限制",
    "数据权属登记",
    "数据资产登记",
    "数据资源登记",
    "数据知识产权",
    "数据授权运营",
]

PRICING_TERMS = [
    "数据定价",
    "数据估值",
    "数据资产评估",
    "数据资源评估",
    "数据价值评估",
    "数据价值计量",
    "数据价值确认",
    "数据资产价值",
    "数据资源价值",
    "数据价格",
]

PRODUCT_TRANSACTION_TERMS = [
    "数据产品",
    "数据服务产品",
    "数据产品交易",
    "数据资产交易",
    "数据资源交易",
    "数据流通交易",
    "数据挂牌",
    "数据产品挂牌",
    "数据资产挂牌",
    "数据资源挂牌",
    "数据产品上架",
    "数据流通",
    "数据交易",
]


def sort_terms(terms: list[str]) -> list[str]:
    return sorted(set(terms), key=len, reverse=True)


MASK_TERMS = sort_terms(MASK_TERMS)
ASSET_TRADE_NOINST_TERMS = sort_terms(ASSET_TRADE_NOINST_TERMS)
STRICT_TERMS = sort_terms(STRICT_TERMS)
ACCOUNTING_TERMS = sort_terms(ACCOUNTING_TERMS)
RIGHTS_TERMS = sort_terms(RIGHTS_TERMS)
PRICING_TERMS = sort_terms(PRICING_TERMS)
PRODUCT_TRANSACTION_TERMS = sort_terms(PRODUCT_TRANSACTION_TERMS)


def decode_zipname(raw_name: str) -> str:
    try:
        return raw_name.encode("cp437").decode("gbk")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw_name


def read_text(zip_path: str, raw_name: str) -> str | None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            raw = zf.read(raw_name)
    except Exception:
        return None

    for enc in ("utf-8", "gb18030", "gbk", "gb2312"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def count_terms(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


def rate(raw_count: int, text_len: int) -> float:
    return round(raw_count / text_len * 10000, 6)


def process_report(args: tuple[str, str]) -> dict[str, object] | None:
    zip_path, raw_name = args
    decoded = decode_zipname(raw_name)
    parts = decoded.split("_")
    if len(parts) < 3:
        return None

    try:
        stkcd_int = int(parts[0])
        year = int(parts[1])
    except ValueError:
        return None

    if year not in YEARS:
        return None

    text = read_text(zip_path, raw_name)
    if not text:
        return None

    text_len = len(text)
    if text_len < 1000:
        return None

    masked = text
    inst_raw = 0
    for term in MASK_TERMS:
        inst_raw += masked.count(term)
        masked = masked.replace(term, " " * len(term))

    asset_trade_noinst_raw = count_terms(masked, ASSET_TRADE_NOINST_TERMS)
    strict_noinst_raw = count_terms(masked, STRICT_TERMS)
    acct_raw = count_terms(masked, ACCOUNTING_TERMS)
    rights_raw = count_terms(masked, RIGHTS_TERMS)
    pricing_raw = count_terms(masked, PRICING_TERMS)
    product_tx_noinst_raw = count_terms(masked, PRODUCT_TRANSACTION_TERMS)
    verif_noinst_raw = acct_raw + rights_raw + pricing_raw + product_tx_noinst_raw

    return {
        "stkcd": f"{stkcd_int:06d}",
        "year": year,
        "text_length": text_len,
        "inst_term_raw": inst_raw,
        "inst_term_kw": rate(inst_raw, text_len),
        "asset_trade_noinst_raw": asset_trade_noinst_raw,
        "asset_trade_noinst_kw": rate(asset_trade_noinst_raw, text_len),
        "strict_noinst_raw": strict_noinst_raw,
        "strict_noinst_kw": rate(strict_noinst_raw, text_len),
        "acct_raw": acct_raw,
        "acct_kw": rate(acct_raw, text_len),
        "rights_raw": rights_raw,
        "rights_kw": rate(rights_raw, text_len),
        "pricing_raw": pricing_raw,
        "pricing_kw": rate(pricing_raw, text_len),
        "product_tx_noinst_raw": product_tx_noinst_raw,
        "product_tx_noinst_kw": rate(product_tx_noinst_raw, text_len),
        "verif_noinst_raw": verif_noinst_raw,
        "verif_noinst_kw": rate(verif_noinst_raw, text_len),
    }


def collect_tasks() -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    for fname in sorted(os.listdir(ZIP_DIR)):
        if not fname.endswith(".zip"):
            continue
        try:
            year = int(fname.split("_")[0])
        except ValueError:
            continue
        if year not in YEARS:
            continue
        zip_path = ZIP_DIR / fname
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                tasks.extend(
                    (str(zip_path), info.filename)
                    for info in zf.infolist()
                    if info.filename.endswith(".txt")
                )
        except Exception as exc:
            print(f"skip {zip_path.name}: {exc}")
    return tasks


def main() -> None:
    tasks = collect_tasks()
    print(f"reports to process: {len(tasks)}")

    rows = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_report, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), 1):
            if idx % 5000 == 0:
                print(f"processed {idx}/{len(tasks)}")
            row = future.result()
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("text_length", ascending=False).drop_duplicates(
        subset=["stkcd", "year"], keep="first"
    )
    df = df.sort_values(["stkcd", "year"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"saved {len(df)} rows to {OUT}")
    for col in [
        "asset_trade_noinst_raw",
        "strict_noinst_raw",
        "verif_noinst_raw",
        "acct_raw",
        "rights_raw",
        "pricing_raw",
        "product_tx_noinst_raw",
        "inst_term_raw",
    ]:
        print(f"{col}: nonzero={(df[col] > 0).mean():.4f}, mean={df[col].mean():.4f}")


if __name__ == "__main__":
    main()
