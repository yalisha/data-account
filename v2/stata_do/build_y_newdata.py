#!/opt/miniconda3/bin/python
from __future__ import annotations

from collections import defaultdict
from datetime import date
from html import unescape
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import math
import re

from lxml import etree as ET
import numpy as np
import pandas as pd


BASE = Path("/Users/mac/computerscience/0做完了/15会计研究")
THIRD = Path("/Users/mac/computerscience/第三方资料/第三方数据资源")
DATA = BASE / "v1/data_parquet"
REG_PATH = BASE / "v1/data_stata/reg_sample_iv_v16.dta"
OUTPUT_DTA = BASE / "v1/data_stata/reg_sample_y_newdata.dta"
SUMMARY_PATH = BASE / "v2/results/y_newdata_build_summary.csv"
EVENT_PATH = BASE / "v2/results/y_newdata_event_level.csv"

FIN_DIR = THIRD / "上市公司财务信息"
REPORT_DIR = THIRD / "上市公司年报信息"

A_SHARE_TYPES = {1, 4, 16, 32}
SAMPLE_START = 2011
SAMPLE_END = 2024

NEW_Y_VARS = [
    "CS_Spread",
    "HL_Range_year",
    "PEAD20_signed",
    "PEAD60_signed",
    "PEAD20_abs",
    "PEAD60_abs",
    "EA_CAR02",
    "EA_absCAR02",
    "SUE_price",
    "SUE_abs_price",
    "AF_Dispersion",
    "AF_Coverage",
    "AF_Error_abs",
    "AF_Optimism",
    "Bench_Accuracy",
    "Bench_Optimism",
    "Bench_BuyShare",
    "NCSKEW_off",
    "DUVOL_off",
    "CRASH_off",
]

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ROW_RE = re.compile(rb'<row r="(\d+)"')
CELL_RE = re.compile(
    rb'<c r="([A-Z]+)\d+"[^>]*>(?:<is><t[^>]*>(.*?)</t></is>|<v>(.*?)</v>)</c>'
)
DAILY_HILO_RE = re.compile(
    rb'<c r="A\d+"[^>]*><is><t[^>]*>(.*?)</t></is></c>'
    rb'<c r="B\d+"[^>]*><is><t[^>]*>(.*?)</t></is></c>'
    rb'<c r="C\d+"[^>]*>.*?</c>'
    rb'<c r="D\d+"[^>]*><v>(.*?)</v></c>'
    rb'<c r="E\d+"[^>]*><v>(.*?)</v></c>'
)


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    return series.clip(clean.quantile(lower), clean.quantile(upper))


def load_reg_sample() -> pd.DataFrame:
    reg = pd.read_stata(REG_PATH, convert_categoricals=False)
    reg["Stkcd"] = pd.to_numeric(reg["Stkcd"], errors="coerce").astype("Int64")
    reg["year"] = pd.to_numeric(reg["year"], errors="coerce").astype("Int64")
    return reg.dropna(subset=["Stkcd", "year"]).copy()


def cell_ref_to_idx(ref: str) -> int:
    out = 0
    for ch in ref.upper():
        if ch < "A" or ch > "Z":
            break
        out = out * 26 + (ord(ch) - ord("A") + 1)
    return out - 1


def load_shared_strings(book: ZipFile) -> list[str]:
    try:
        payload = book.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(payload)
    strings: list[str] = []
    for si in root.findall(".//m:si", NS):
        strings.append("".join(t.text or "" for t in si.findall(".//m:t", NS)))
    return strings


def cell_value(cell: ET.Element, shared: list[str]) -> str | None:
    ctype = cell.attrib.get("t")
    if ctype == "inlineStr":
        inline = cell.find("m:is", NS)
        return "".join(inline.itertext()) if inline is not None else None
    v = cell.find("m:v", NS)
    if v is None or v.text is None:
        return None
    if ctype == "s":
        idx = int(float(v.text))
        return shared[idx] if 0 <= idx < len(shared) else None
    return v.text


def iter_xlsx_rows(xlsx: Path | BytesIO, wanted: list[str]):
    yield from iter_xlsx_rows_fast(xlsx, wanted)


def decode_cell(raw: bytes | None) -> str | None:
    if raw is None:
        return None
    return unescape(raw.decode("utf-8", errors="replace"))


def iter_xlsx_rows_fast(xlsx: Path | BytesIO, wanted: list[str]):
    wanted_set = set(wanted)
    with ZipFile(xlsx) as book:
        wanted_letters: dict[str, str] = {}
        current_row = 0
        with book.open("xl/worksheets/sheet1.xml") as sheet:
            for line in sheet:
                row_match = ROW_RE.search(line)
                if row_match:
                    current_row = int(row_match.group(1))
                    if b"<c " not in line:
                        continue
                elif b"<c " not in line:
                    continue

                if current_row == 1:
                    for col_raw, text_raw, val_raw in CELL_RE.findall(line):
                        value = decode_cell(text_raw or val_raw)
                        if value in wanted_set:
                            wanted_letters[col_raw.decode("ascii")] = value
                elif current_row >= 4:
                    row_out: dict[str, str | None] = {}
                    for col_raw, text_raw, val_raw in CELL_RE.findall(line):
                        col = wanted_letters.get(col_raw.decode("ascii"))
                        if col is not None:
                            row_out[col] = decode_cell(text_raw or val_raw)
                    yield row_out


def iter_xlsx_rows_xml(xlsx: Path | BytesIO, wanted: list[str]):
    wanted_set = set(wanted)
    row_tag = f"{{{NS['m']}}}row"
    cell_tag = f"{{{NS['m']}}}c"
    with ZipFile(xlsx) as book:
        shared = load_shared_strings(book)
        with book.open("xl/worksheets/sheet1.xml") as sheet:
            context = ET.iterparse(sheet, events=("end",), tag=row_tag, huge_tree=True)
            wanted_idx: dict[int, str] = {}
            for _, elem in context:
                row_no = int(elem.attrib.get("r", "0"))
                if row_no == 1:
                    for cell in elem.iterchildren(tag=cell_tag):
                        ref = cell.attrib.get("r", "")
                        if not ref:
                            continue
                        value = cell_value(cell, shared)
                        if value in wanted_set:
                            wanted_idx[cell_ref_to_idx(ref)] = str(value)
                elif row_no >= 4:
                    row_out: dict[str, str | None] = {}
                    for cell in elem.iterchildren(tag=cell_tag):
                        ref = cell.attrib.get("r", "")
                        if not ref:
                            continue
                        col = wanted_idx.get(cell_ref_to_idx(ref))
                        if col is not None:
                            row_out[col] = cell_value(cell, shared)
                    yield row_out
                elem.clear()


def iter_daily_hilo_rows(xlsx: Path):
    with ZipFile(xlsx) as book:
        with book.open("xl/worksheets/sheet1.xml") as sheet:
            for line in sheet:
                if b'<c r="A' not in line:
                    continue
                match = DAILY_HILO_RE.search(line)
                if not match:
                    continue
                stk, trddt, hiprc, loprc = match.groups()
                yield (
                    decode_cell(stk),
                    decode_cell(trddt),
                    decode_cell(hiprc),
                    decode_cell(loprc),
                )


def read_xlsx_frame(xlsx: Path | BytesIO, wanted: list[str]) -> pd.DataFrame:
    rows = list(iter_xlsx_rows(xlsx, wanted))
    if not rows:
        return pd.DataFrame(columns=wanted)
    return pd.DataFrame.from_records(rows)


def read_zipped_xlsx_frame(zip_path: Path, inner_name: str, wanted: list[str]) -> pd.DataFrame:
    with ZipFile(zip_path) as outer:
        payload = BytesIO(outer.read(inner_name))
    return read_xlsx_frame(payload, wanted)


def parse_int_stock(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def parse_date_series(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    text = text.mask(text.str.contains("-00", na=False))
    return pd.to_datetime(text, errors="coerce")


def daily_files() -> list[Path]:
    files: list[Path] = []
    for folder in ["_daily_2011_2016", "_daily_2016_2021", "_daily_2021_2026"]:
        files.extend(sorted((FIN_DIR / folder).glob("TRD_Dalyr*.xlsx")))
    return files


def build_cs_spread(reg: pd.DataFrame) -> pd.DataFrame:
    stocks = set(reg["Stkcd"].dropna().astype(int))
    years = set(reg["year"].dropna().astype(int))
    agg: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: {"cs_sum": 0.0, "cs_n": 0, "hl_sum": 0.0, "hl_n": 0}
    )
    prev_by_stock: dict[int, tuple[int, float, float, float]] = {}

    denom = 3 - 2 * math.sqrt(2)
    processed = 0
    kept = 0
    for fp in daily_files():
        print(f"  parsing daily high/low: {fp}")
        for stk_raw, trddt_raw, hiprc_raw, loprc_raw in iter_daily_hilo_rows(fp):
            processed += 1
            stk = parse_int_stock(stk_raw)
            if stk is None or stk not in stocks:
                continue
            try:
                trade_date = date.fromisoformat(str(trddt_raw))
            except (TypeError, ValueError):
                continue
            date_ord = trade_date.toordinal()
            high = parse_float(hiprc_raw)
            low = parse_float(loprc_raw)
            if not (high > 0 and low > 0 and high >= low):
                continue

            prev = prev_by_stock.get(stk)
            if prev is not None and date_ord <= prev[0]:
                continue

            year = trade_date.year
            hl_log = math.log(high / low)
            if year in years:
                bucket = agg[(stk, year)]
                bucket["hl_sum"] += hl_log
                bucket["hl_n"] += 1
                kept += 1

            if prev is not None and year in years and (date_ord - prev[0]) <= 10:
                prev_hl = prev[3]
                beta = hl_log**2 + prev_hl**2
                gamma = math.log(max(high, prev[1]) / min(low, prev[2])) ** 2
                alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / denom - math.sqrt(gamma / denom)
                alpha = max(alpha, 0.0)
                spread = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
                if math.isfinite(spread):
                    bucket = agg[(stk, year)]
                    bucket["cs_sum"] += spread
                    bucket["cs_n"] += 1

            prev_by_stock[stk] = (date_ord, high, low, hl_log)

    records = []
    for (stk, year), vals in agg.items():
        records.append(
            {
                "Stkcd": stk,
                "year": year,
                "CS_Spread": vals["cs_sum"] / vals["cs_n"] if vals["cs_n"] >= 60 else np.nan,
                "HL_Range_year": vals["hl_sum"] / vals["hl_n"] if vals["hl_n"] >= 60 else np.nan,
                "n_hl_days": vals["hl_n"],
                "n_cs_pairs": vals["cs_n"],
            }
        )
    out = pd.DataFrame.from_records(records)
    print(f"  daily high/low rows processed={processed:,}; sample rows kept={kept:,}; firm-years={len(out):,}")
    return out


def read_announcement_events(reg: pd.DataFrame) -> pd.DataFrame:
    years = set(reg["year"].dropna().astype(int))
    stocks = set(reg["Stkcd"].dropna().astype(int))
    rept_zip = REPORT_DIR / "年、中、季报基本情况文件111546034(仅供四川大学使用).zip"
    fore_zip = REPORT_DIR / "年、中、季报预披露日期表111134854(仅供四川大学使用).zip"

    rept = read_zipped_xlsx_frame(
        rept_zip,
        "IAR_Rept.xlsx",
        ["Stkcd", "Reptyp", "Accper", "Annodt", "Profita", "Profitb", "Erana", "Eranb"],
    )
    rept["Stkcd"] = rept["Stkcd"].map(parse_int_stock)
    rept["Reptyp"] = pd.to_numeric(rept["Reptyp"], errors="coerce")
    rept["accper_dt"] = parse_date_series(rept["Accper"])
    rept["annodt_dt"] = parse_date_series(rept["Annodt"])
    rept["year"] = rept["accper_dt"].dt.year
    for col in ["Profita", "Profitb", "Erana", "Eranb"]:
        rept[col] = pd.to_numeric(rept[col], errors="coerce")
    rept = rept[
        rept["Stkcd"].isin(stocks)
        & rept["year"].isin(years)
        & rept["Reptyp"].eq(4)
    ].copy()

    fore = read_zipped_xlsx_frame(
        fore_zip,
        "IAR_Forecdt.xlsx",
        ["Stkcd", "Accper", "Firforecdt", "Firchangdt", "Secchangdt", "Thirchangdt", "Actudt"],
    )
    fore["Stkcd"] = fore["Stkcd"].map(parse_int_stock)
    fore["accper_dt"] = parse_date_series(fore["Accper"])
    fore["year"] = fore["accper_dt"].dt.year
    fore["actudt_dt"] = parse_date_series(fore["Actudt"])
    fore = fore[fore["Stkcd"].isin(stocks) & fore["year"].isin(years)].copy()
    fore = fore.sort_values("actudt_dt").drop_duplicates(["Stkcd", "year"], keep="last")

    events = rept.merge(fore[["Stkcd", "year", "actudt_dt"]], on=["Stkcd", "year"], how="left")
    events["event_dt"] = events["annodt_dt"].fillna(events["actudt_dt"])
    events = events.dropna(subset=["event_dt"])
    events = events.sort_values("event_dt").drop_duplicates(["Stkcd", "year"], keep="last")
    print(f"  annual announcement events={len(events):,}")
    return events[["Stkcd", "year", "event_dt", "annodt_dt", "actudt_dt", "Erana", "Eranb", "Profita", "Profitb"]]


def read_actual_eps(events: pd.DataFrame) -> pd.DataFrame:
    actual = read_xlsx_frame(FIN_DIR / "AF_Actual.xlsx", ["Stkcd", "Ddate", "Meps", "Mnetpro", "Mturnover"])
    actual["Stkcd"] = actual["Stkcd"].map(parse_int_stock)
    actual["ddate_dt"] = parse_date_series(actual["Ddate"])
    actual["year"] = actual["ddate_dt"].dt.year
    for col in ["Meps", "Mnetpro", "Mturnover"]:
        actual[col] = pd.to_numeric(actual[col], errors="coerce")
    actual = actual.dropna(subset=["Stkcd", "year"])
    actual = (
        actual.groupby(["Stkcd", "year"], observed=True)
        .agg(actual_eps=("Meps", "mean"), actual_netpro=("Mnetpro", "mean"))
        .reset_index()
    )
    fallback = events[["Stkcd", "year", "Eranb", "Erana", "Profitb", "Profita"]].copy()
    fallback["eps_from_iar"] = fallback["Eranb"].fillna(fallback["Erana"])
    fallback["profit_from_iar"] = fallback["Profitb"].fillna(fallback["Profita"])
    fallback = fallback[["Stkcd", "year", "eps_from_iar", "profit_from_iar"]]
    actual = fallback.merge(actual, on=["Stkcd", "year"], how="left")
    actual["actual_eps"] = actual["actual_eps"].fillna(actual["eps_from_iar"])
    actual["actual_netpro"] = actual["actual_netpro"].fillna(actual["profit_from_iar"])
    return actual[["Stkcd", "year", "actual_eps", "actual_netpro"]]


def forecast_files() -> list[Path]:
    files: list[Path] = []
    for folder in ["_forecast_batch1", "_forecast_batch2"]:
        files.extend(sorted((FIN_DIR / folder).glob("AF_Forecast*.xlsx")))
    return files


def read_forecasts(events: pd.DataFrame) -> pd.DataFrame:
    wanted = ["Stkcd", "Rptdt", "Fenddt", "ReportID", "DeclareDate", "Feps", "Fnetpro", "Fturnover"]
    frames = []
    for fp in forecast_files():
        print(f"  parsing forecasts: {fp}")
        frame = read_xlsx_frame(fp, wanted)
        frames.append(frame)
    fc = pd.concat(frames, ignore_index=True)
    fc["Stkcd"] = fc["Stkcd"].map(parse_int_stock)
    fc["forecast_dt"] = parse_date_series(fc["DeclareDate"]).fillna(parse_date_series(fc["Rptdt"]))
    fc["fend_dt"] = parse_date_series(fc["Fenddt"])
    fc["year"] = fc["fend_dt"].dt.year
    for col in ["Feps", "Fnetpro", "Fturnover"]:
        fc[col] = pd.to_numeric(fc[col], errors="coerce")
    fc = fc.dropna(subset=["Stkcd", "forecast_dt", "year", "Feps"])
    fc = fc[fc["fend_dt"].dt.month.eq(12) & fc["fend_dt"].dt.day.eq(31)].copy()
    fc = fc.drop_duplicates(["Stkcd", "forecast_dt", "year", "ReportID", "Feps"])

    fc = fc.merge(events[["Stkcd", "year", "event_dt"]], on=["Stkcd", "year"], how="inner")
    lower = fc["event_dt"] - pd.Timedelta(days=365)
    fc = fc[(fc["forecast_dt"] < fc["event_dt"]) & (fc["forecast_dt"] >= lower)].copy()
    out = (
        fc.groupby(["Stkcd", "year"], observed=True)
        .agg(
            af_n=("Feps", "count"),
            af_eps_mean=("Feps", "mean"),
            af_eps_median=("Feps", "median"),
            af_eps_std=("Feps", "std"),
            af_netpro_mean=("Fnetpro", "mean"),
        )
        .reset_index()
    )
    out["AF_Coverage"] = np.log1p(out["af_n"])
    out["AF_Dispersion"] = out["af_eps_std"] / out["af_eps_median"].abs()
    out.loc[out["af_n"] < 2, "AF_Dispersion"] = np.nan
    out["AF_Dispersion"] = out["AF_Dispersion"].replace([np.inf, -np.inf], np.nan)
    print(f"  forecast consensus firm-years={len(out):,}")
    return out


def load_daily_abnormal_returns(reg: pd.DataFrame) -> pd.DataFrame:
    stocks = set(reg["Stkcd"].dropna().astype(int))
    daily = pd.read_parquet(
        DATA / "daily_return.parquet",
        columns=["Stkcd", "Trddt", "Clsprc", "Dretwd", "Markettype"],
    )
    daily["Stkcd"] = pd.to_numeric(daily["Stkcd"], errors="coerce").astype("Int64")
    daily["Trddt"] = pd.to_datetime(daily["Trddt"], errors="coerce")
    for col in ["Clsprc", "Dretwd", "Markettype"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily[daily["Stkcd"].isin(stocks) & daily["Markettype"].isin(A_SHARE_TYPES)].copy()

    mkt = pd.read_parquet(DATA / "market_index.parquet", columns=["Indexcd", "Trddt", "Retindex"])
    mkt["Indexcd"] = pd.to_numeric(mkt["Indexcd"], errors="coerce")
    mkt["Trddt"] = pd.to_datetime(mkt["Trddt"], errors="coerce")
    mkt["Retindex"] = pd.to_numeric(mkt["Retindex"], errors="coerce")
    mkt = mkt[mkt["Indexcd"].eq(1)][["Trddt", "Retindex"]].drop_duplicates("Trddt")
    daily = daily.merge(mkt, on="Trddt", how="left")
    daily["abret"] = daily["Dretwd"] - daily["Retindex"]
    return daily[["Stkcd", "Trddt", "Clsprc", "Dretwd", "abret"]].dropna(subset=["Trddt"])


def car_window(abret: np.ndarray, start: int, end: int, min_obs: int) -> float:
    if start < 0 or start >= len(abret):
        return np.nan
    end = min(end, len(abret) - 1)
    vals = abret[start : end + 1]
    vals = vals[~np.isnan(vals)]
    if len(vals) < min_obs:
        return np.nan
    return float(vals.sum())


def attach_event_returns(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    for col in ["EA_CAR02", "PEAD20_raw", "PEAD60_raw", "pre_close"]:
        out[col] = np.nan

    daily = daily.sort_values(["Stkcd", "Trddt"])
    event_index = {stk: grp.index.to_numpy() for stk, grp in out.groupby("Stkcd", observed=True)}
    for stk, group in daily.groupby("Stkcd", observed=True, sort=False):
        if stk not in event_index:
            continue
        idxs = event_index[stk]
        d = group["Trddt"].to_numpy(dtype="datetime64[ns]")
        ab = group["abret"].to_numpy(dtype=float)
        close = group["Clsprc"].to_numpy(dtype=float)
        event_dates = out.loc[idxs, "event_dt"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(d, event_dates, side="left")
        vals_02 = []
        vals_20 = []
        vals_60 = []
        vals_preclose = []
        for pos in positions:
            vals_02.append(car_window(ab, pos, pos + 2, 2))
            vals_20.append(car_window(ab, pos + 2, pos + 20, 10))
            vals_60.append(car_window(ab, pos + 2, pos + 60, 30))
            vals_preclose.append(float(close[pos - 1]) if pos > 0 and math.isfinite(close[pos - 1]) else np.nan)
        out.loc[idxs, "EA_CAR02"] = vals_02
        out.loc[idxs, "PEAD20_raw"] = vals_20
        out.loc[idxs, "PEAD60_raw"] = vals_60
        out.loc[idxs, "pre_close"] = vals_preclose

    out["EA_absCAR02"] = out["EA_CAR02"].abs()
    return out


def build_pead_and_analyst(reg: pd.DataFrame) -> pd.DataFrame:
    print("  reading announcement events...")
    events = read_announcement_events(reg)
    print("  reading actual EPS...")
    actual = read_actual_eps(events)
    print("  reading analyst forecasts...")
    forecasts = read_forecasts(events)
    print("  loading daily abnormal returns...")
    daily = load_daily_abnormal_returns(reg)
    print("  attaching event-window returns...")
    events = attach_event_returns(events, daily)

    out = events.merge(actual, on=["Stkcd", "year"], how="left")
    out = out.merge(forecasts, on=["Stkcd", "year"], how="left")
    out["eps_error"] = out["actual_eps"] - out["af_eps_median"]
    out["SUE_price"] = out["eps_error"] / out["pre_close"]
    out["SUE_abs_price"] = out["SUE_price"].abs()
    out["AF_Error_abs"] = out["SUE_abs_price"]
    out["AF_Optimism"] = (out["af_eps_median"] - out["actual_eps"]) / out["pre_close"]
    surprise_sign = np.sign(out["SUE_price"])
    out["PEAD20_signed"] = surprise_sign * out["PEAD20_raw"]
    out["PEAD60_signed"] = surprise_sign * out["PEAD60_raw"]
    out["PEAD20_abs"] = out["PEAD20_raw"].abs()
    out["PEAD60_abs"] = out["PEAD60_raw"].abs()

    keep = [
        "Stkcd",
        "year",
        "event_dt",
        "pre_close",
        "actual_eps",
        "af_eps_median",
        "af_n",
        "EA_CAR02",
        "EA_absCAR02",
        "PEAD20_signed",
        "PEAD60_signed",
        "PEAD20_abs",
        "PEAD60_abs",
        "SUE_price",
        "SUE_abs_price",
        "AF_Dispersion",
        "AF_Coverage",
        "AF_Error_abs",
        "AF_Optimism",
    ]
    event_out = out[keep].copy()
    event_out.to_csv(EVENT_PATH, index=False)
    print(f"  saved event-level Y to {EVENT_PATH}")
    return event_out.drop(columns=["event_dt", "pre_close", "actual_eps", "af_eps_median", "af_n"])


def bench_files() -> list[Path]:
    files: list[Path] = []
    for folder in ["_rating_batch1", "_rating_batch2"]:
        files.extend(sorted((FIN_DIR / folder).glob("AF_Bench*.xlsx")))
    return files


def build_bench_measures(reg: pd.DataFrame) -> pd.DataFrame:
    wanted = ["Stkcd", "Accper", "Stdrank", "ForecastAccuracy", "ForecastOptimism"]
    frames = []
    for fp in bench_files():
        print(f"  parsing analyst benchmark/rating: {fp}")
        frame = read_xlsx_frame(fp, wanted)
        frames.append(frame)
    bench = pd.concat(frames, ignore_index=True)
    bench["Stkcd"] = bench["Stkcd"].map(parse_int_stock)
    bench["accper_dt"] = parse_date_series(bench["Accper"])
    bench["year"] = bench["accper_dt"].dt.year
    for col in ["ForecastAccuracy", "ForecastOptimism"]:
        bench[col] = pd.to_numeric(bench[col], errors="coerce")
    bench = bench[
        bench["Stkcd"].isin(set(reg["Stkcd"].dropna().astype(int)))
        & bench["year"].isin(set(reg["year"].dropna().astype(int)))
        & bench["accper_dt"].dt.month.eq(12)
        & bench["accper_dt"].dt.day.eq(31)
    ].copy()
    bench = bench.drop_duplicates()
    bench["is_buy"] = bench["Stdrank"].isin(["买入", "增持"]).astype(float)
    out = (
        bench.groupby(["Stkcd", "year"], observed=True)
        .agg(
            Bench_Accuracy=("ForecastAccuracy", "mean"),
            Bench_Optimism=("ForecastOptimism", "mean"),
            Bench_BuyShare=("is_buy", "mean"),
        )
        .reset_index()
    )
    print(f"  analyst benchmark firm-years={len(out):,}")
    return out


def build_official_crash(reg: pd.DataFrame) -> pd.DataFrame:
    zip_path = FIN_DIR / "股价崩盘指标表(年)102211597(仅供沪江大学使用).zip"
    wanted = [
        "Stkcd",
        "Trdynt",
        "NCSKEW_Cmdeq",
        "DUVOL_Cmdeq",
        "CRASH_Cmdeq",
    ]
    crash = read_zipped_xlsx_frame(zip_path, "BF_CRASHRISK.xlsx", wanted)
    crash["Stkcd"] = crash["Stkcd"].map(parse_int_stock)
    crash["year"] = pd.to_numeric(crash["Trdynt"], errors="coerce")
    crash = crash[
        crash["Stkcd"].isin(set(reg["Stkcd"].dropna().astype(int)))
        & crash["year"].isin(set(reg["year"].dropna().astype(int)))
    ].copy()
    for col in ["NCSKEW_Cmdeq", "DUVOL_Cmdeq", "CRASH_Cmdeq"]:
        crash[col] = pd.to_numeric(crash[col], errors="coerce")
    crash = crash.rename(
        columns={
            "NCSKEW_Cmdeq": "NCSKEW_off",
            "DUVOL_Cmdeq": "DUVOL_off",
            "CRASH_Cmdeq": "CRASH_off",
        }
    )
    crash = crash.sort_values(["Stkcd", "year"]).drop_duplicates(["Stkcd", "year"], keep="last")
    print(f"  official crash-risk firm-years={len(crash):,}")
    return crash[["Stkcd", "year", "NCSKEW_off", "DUVOL_off", "CRASH_off"]]


def summarize(merged: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in NEW_Y_VARS:
        s = merged[col] if col in merged.columns else pd.Series(dtype=float)
        clean = s.dropna()
        records.append(
            {
                "y_name": col,
                "nonmissing_reg_sample": int(clean.shape[0]),
                "mean": float(clean.mean()) if len(clean) else np.nan,
                "sd": float(clean.std()) if len(clean) else np.nan,
                "min": float(clean.min()) if len(clean) else np.nan,
                "p01": float(clean.quantile(0.01)) if len(clean) else np.nan,
                "median": float(clean.median()) if len(clean) else np.nan,
                "p99": float(clean.quantile(0.99)) if len(clean) else np.nan,
                "max": float(clean.max()) if len(clean) else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    print("Loading lagged regression sample...")
    reg = load_reg_sample()
    print(f"Regression sample shape: {reg.shape}; years={reg['year'].min()}-{reg['year'].max()}")

    print("\nBuilding CS_Spread and high-low range from local CSMAR TRD_Dalyr...")
    cs = build_cs_spread(reg)

    print("\nBuilding PEAD / SUE / analyst-forecast Y variables...")
    pead = build_pead_and_analyst(reg)

    print("\nBuilding AF_Bench analyst accuracy / optimism variables...")
    bench = build_bench_measures(reg)

    print("\nBuilding official CSMAR crash-risk variables...")
    crash = build_official_crash(reg)

    candidate = reg[["Stkcd", "year"]].drop_duplicates().copy()
    for frame in [cs, pead, bench, crash]:
        candidate = candidate.merge(frame, on=["Stkcd", "year"], how="left", validate="one_to_one")

    merged = reg.merge(candidate, on=["Stkcd", "year"], how="left", validate="one_to_one")
    for col in NEW_Y_VARS:
        if col in merged.columns:
            merged[col] = winsorize(pd.to_numeric(merged[col], errors="coerce"))
        else:
            merged[col] = np.nan

    summary = summarize(merged)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved build summary to {SUMMARY_PATH}")
    print(summary.to_string(index=False))

    out = merged.copy()
    out["Stkcd"] = out["Stkcd"].astype("int32")
    out["year"] = out["year"].astype("int16")
    out.to_stata(OUTPUT_DTA, write_index=False, version=118)
    print(f"\nSaved merged DTA to {OUTPUT_DTA}")
    print(f"Final shape: {out.shape}")


if __name__ == "__main__":
    main()
