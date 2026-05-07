#!/usr/bin/env python3
"""Extract balance-sheet data-resource booking fields for v3 validation."""

from __future__ import annotations

import csv
import math
import zipfile
from pathlib import Path
from typing import Iterable

try:
    from lxml import etree

    HAS_LXML = True
except Exception:  # pragma: no cover - fallback for machines without lxml
    import xml.etree.ElementTree as etree  # type: ignore

    HAS_LXML = False


SRC = Path(
    "/Users/mac/computerscience/第三方资料/第三方数据资源/"
    "上市公司财务信息/FS_Combas.xlsx"
)
OUT_PERIOD = Path(
    "/Users/mac/computerscience/0做完了/15会计研究/v3/"
    "results/data/balance_sheet_data_resource_period.csv"
)
OUT_ANNUAL = Path(
    "/Users/mac/computerscience/0做完了/15会计研究/v3/"
    "results/data/balance_sheet_data_resource_annual.csv"
)
OUT_SUMMARY = Path(
    "/Users/mac/computerscience/0做完了/15会计研究/v3/"
    "results/data/balance_sheet_data_resource_summary.csv"
)

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

COLUMNS = {
    "A": "Stkcd",
    "B": "ShortName",
    "C": "Accper",
    "D": "Typrep",
    "AI": "DataRes_inventory",
    "BS": "DataRes_intangible",
    "BU": "DataRes_devexp",
    "CC": "TotalAssets",
}

OUT_FIELDS = [
    "Stkcd",
    "ShortName",
    "Accper",
    "year_num",
    "Typrep",
    "DataRes_inventory",
    "DataRes_intangible",
    "DataRes_devexp",
    "BookDataResourceAmount",
    "BookDataResource",
    "lnBookDataResource",
    "BookDataResourceRatio",
    "TotalAssets",
]


def col_letters(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def cell_text(cell) -> str | None:
    inline = cell.find(NS + "is")
    if inline is not None:
        text = "".join(t.text or "" for t in inline.iter(NS + "t"))
        return text
    value = cell.find(NS + "v")
    return value.text if value is not None else None


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0


def as_year(accper: str | None) -> int | None:
    if not accper or len(accper) < 4:
        return None
    try:
        return int(accper[:4])
    except ValueError:
        return None


def parse_rows() -> Iterable[dict[str, object]]:
    with zipfile.ZipFile(SRC) as workbook:
        with workbook.open("xl/worksheets/sheet1.xml") as sheet:
            kwargs = {"tag": NS + "row"} if HAS_LXML else {}
            context = etree.iterparse(sheet, events=("end",), **kwargs)
            for _, row in context:
                if row.tag != NS + "row":
                    continue
                row_num = row.get("r")
                if row_num in {"1", "2", "3"}:
                    row.clear()
                    continue

                vals: dict[str, str | None] = {}
                for cell in row.iter(NS + "c"):
                    col = col_letters(cell.get("r", ""))
                    if col in COLUMNS:
                        vals[COLUMNS[col]] = cell_text(cell)

                accper = vals.get("Accper")
                year = as_year(accper)
                inv = as_float(vals.get("DataRes_inventory"))
                intangible = as_float(vals.get("DataRes_intangible"))
                devexp = as_float(vals.get("DataRes_devexp"))
                total_assets = as_float(vals.get("TotalAssets"))
                amount = inv + intangible + devexp

                out = {
                    "Stkcd": vals.get("Stkcd"),
                    "ShortName": vals.get("ShortName"),
                    "Accper": accper,
                    "year_num": year,
                    "Typrep": vals.get("Typrep"),
                    "DataRes_inventory": inv,
                    "DataRes_intangible": intangible,
                    "DataRes_devexp": devexp,
                    "BookDataResourceAmount": amount,
                    "BookDataResource": 1 if amount > 0 else 0,
                    "lnBookDataResource": math.log1p(amount),
                    "BookDataResourceRatio": amount / total_assets if total_assets > 0 else None,
                    "TotalAssets": total_assets,
                }
                yield out
                row.clear()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int | None, str | None, bool], dict[str, object]] = {}
    for row in rows:
        year = row["year_num"]
        typ = row["Typrep"]
        is_year_end = row["Accper"] == f"{year}-12-31"
        key = (year, typ, is_year_end)
        g = groups.setdefault(
            key,
            {
                "year_num": year,
                "Typrep": typ,
                "is_year_end": int(is_year_end),
                "N": 0,
                "positive": 0,
                "sum_amount": 0.0,
                "pos_inventory": 0,
                "pos_intangible": 0,
                "pos_devexp": 0,
            },
        )
        g["N"] = int(g["N"]) + 1
        if float(row["BookDataResourceAmount"]) > 0:
            g["positive"] = int(g["positive"]) + 1
            g["sum_amount"] = float(g["sum_amount"]) + float(row["BookDataResourceAmount"])
        if float(row["DataRes_inventory"]) > 0:
            g["pos_inventory"] = int(g["pos_inventory"]) + 1
        if float(row["DataRes_intangible"]) > 0:
            g["pos_intangible"] = int(g["pos_intangible"]) + 1
        if float(row["DataRes_devexp"]) > 0:
            g["pos_devexp"] = int(g["pos_devexp"]) + 1
    return sorted(groups.values(), key=lambda x: (x["year_num"] or 0, str(x["Typrep"]), x["is_year_end"]))


def main() -> None:
    period_rows = [row for row in parse_rows() if row["year_num"] is not None]
    annual_rows = [
        row
        for row in period_rows
        if row["Typrep"] == "A" and row["Accper"] == f"{row['year_num']}-12-31"
    ]

    write_csv(OUT_PERIOD, period_rows)
    write_csv(OUT_ANNUAL, annual_rows)

    summary_rows = summarize(period_rows)
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "year_num",
            "Typrep",
            "is_year_end",
            "N",
            "positive",
            "sum_amount",
            "pos_inventory",
            "pos_intangible",
            "pos_devexp",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    positive_annual = [r for r in annual_rows if float(r["BookDataResourceAmount"]) > 0]
    print(f"wrote {OUT_PERIOD}")
    print(f"wrote {OUT_ANNUAL}")
    print(f"wrote {OUT_SUMMARY}")
    print(f"period_rows={len(period_rows)} annual_rows={len(annual_rows)}")
    print(f"annual_positive={len(positive_annual)}")
    for row in positive_annual[:12]:
        print(
            row["Stkcd"],
            row["ShortName"],
            row["Accper"],
            row["DataRes_inventory"],
            row["DataRes_intangible"],
            row["DataRes_devexp"],
            row["BookDataResourceAmount"],
        )


if __name__ == "__main__":
    main()
