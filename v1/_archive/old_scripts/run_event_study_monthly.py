"""
Monthly Event Study: 2024入表新规的月度动态效应

核心思路:
- 从日频数据计算滚动120交易日PriceDelay (Hou-Moskowitz 2005)
- 以此为月度因变量, 做月度事件研究
- 事件: 2024年1月 (入表新规正式实施)
- 优势: 年度版只有1个处理期(2024), 月度版有12+个月

设计:
- 因变量: 滚动PriceDelay_m (120交易日窗口, 5阶每日滞后)
- 处理组: 2017-2021年DU_kw均值 >= 中位数 (同年度版)
- 窗口: 相对月 t=-24 to t=+23 (2022M1-2025M12)
- 基期: t=-1 (2023M12)
- FE: Firm+YearMonth / Firm+Ind×YearMonth
- 聚类: Firm (月度数据用firm聚类控制序列相关)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import matplotlib.pyplot as plt
import warnings, os, time
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
RES_DIR = "/Users/mac/computerscience/15会计研究/results"
os.makedirs(RES_DIR, exist_ok=True)

# === CONFIG ===
WINDOW = 120       # rolling window (trading days)
N_LAGS = 5         # daily lags of market return
MIN_OBS = 80       # minimum obs per window (relaxed from annual's 120)
REL_LO, REL_HI = -24, 23   # relative months window
EVENT_YM = '2024-01'
BASE_REL = -1      # base period: t=-1 (2023-12)
A_SHARE_TYPES = [1, 4, 16, 32]  # same as annual version

# ================================================================
# 1. Load daily data
# ================================================================
print("=" * 60)
print("Monthly Event Study: 2024入表新规")
print("=" * 60)
print("\n1. Loading daily data...")
t0 = time.time()

# Individual stock daily returns (A-shares only)
daily = pd.read_parquet(f"{DATA_DIR}/daily_return.parquet",
                        columns=['Stkcd', 'Trddt', 'Dretwd', 'Markettype'])
daily = daily[daily['Markettype'].isin(A_SHARE_TYPES)].copy()
daily['Trddt'] = pd.to_datetime(daily['Trddt'])
daily['Dretwd'] = pd.to_numeric(daily['Dretwd'], errors='coerce')
daily = daily.dropna(subset=['Dretwd'])

# FF3 market return (P9714 = 综合A股, RiskPremium1 = equal-weighted, matching annual)
ff3 = pd.read_parquet(f"{DATA_DIR}/ff3_daily.parquet")
avail_types = sorted(ff3['MarkettypeID'].unique())
print(f"  FF3 market types: {avail_types}")

mkt_id = 'P9714' if 'P9714' in avail_types else avail_types[0]
print(f"  Using: {mkt_id}")

mkt = ff3[ff3['MarkettypeID'] == mkt_id][['TradingDate', 'RiskPremium1']].copy()
mkt.columns = ['Trddt', 'Rm']
mkt['Trddt'] = pd.to_datetime(mkt['Trddt'])
mkt = mkt.sort_values('Trddt').reset_index(drop=True)

# Create market return lags on the trading calendar (not per-firm)
lag_cols = [f'Rm_lag{i}' for i in range(1, N_LAGS + 1)]
for lag in range(1, N_LAGS + 1):
    mkt[f'Rm_lag{lag}'] = mkt['Rm'].shift(lag)
mkt = mkt.dropna()

# Merge
daily = daily.merge(mkt, on='Trddt', how='inner')
daily = daily.sort_values(['Stkcd', 'Trddt'])

print(f"  Daily panel: {len(daily):,} rows, {daily.Stkcd.nunique():,} firms")
print(f"  Date range: {daily.Trddt.min().date()} to {daily.Trddt.max().date()}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ================================================================
# 2. Compute rolling monthly PriceDelay
# ================================================================
print(f"\n2. Computing rolling PriceDelay ({WINDOW}-day window, {N_LAGS} lags)...")
t0 = time.time()

# Check if cached result exists
CACHE_FILE = f"{RES_DIR}/monthly_pricedelay.parquet"
if os.path.exists(CACHE_FILE):
    monthly_pd = pd.read_parquet(CACHE_FILE)
    monthly_pd['ym'] = pd.to_datetime(monthly_pd['ym'])
    print(f"  Loaded from cache: {len(monthly_pd):,} firm-months")
else:
    def compute_delay(ri, rm, rm_lags):
        """Hou-Moskowitz PriceDelay from daily data within a window."""
        T = len(ri)
        if T < MIN_OBS:
            return np.nan

        ones = np.ones(T)
        ss_tot = np.sum((ri - ri.mean()) ** 2)
        if ss_tot < 1e-12:
            return np.nan

        # Restricted: ri ~ 1 + rm
        X_r = np.column_stack([ones, rm])
        beta_r = np.linalg.lstsq(X_r, ri, rcond=None)[0]
        ss_res_r = np.sum((ri - X_r @ beta_r) ** 2)
        r2_r = 1.0 - ss_res_r / ss_tot

        # Unrestricted: ri ~ 1 + rm + rm_lags
        X_u = np.column_stack([ones, rm, rm_lags])
        beta_u = np.linalg.lstsq(X_u, ri, rcond=None)[0]
        ss_res_u = np.sum((ri - X_u @ beta_u) ** 2)
        r2_u = 1.0 - ss_res_u / ss_tot

        if r2_u <= 0.005:
            return np.nan
        return np.clip(1.0 - r2_r / r2_u, 0.0, 1.0)

    results_list = []
    grouped = daily.groupby('Stkcd')
    n_firms = len(grouped)
    processed = 0

    for stkcd, grp in grouped:
        processed += 1
        if processed % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {processed}/{n_firms} firms ({elapsed:.0f}s)")

        grp = grp.sort_values('Trddt').reset_index(drop=True)
        if len(grp) < MIN_OBS:
            continue

        # Identify last trading day of each month
        ym_period = grp['Trddt'].dt.to_period('M')
        month_end_mask = ym_period != ym_period.shift(-1)
        month_end_mask.iloc[-1] = True
        me_positions = grp.index[month_end_mask].tolist()

        # Pre-extract numpy arrays
        ret_arr = grp['Dretwd'].values
        rm_arr = grp['Rm'].values
        lags_arr = grp[lag_cols].values
        dates = grp['Trddt'].values

        for me_pos in me_positions:
            start = max(0, me_pos - WINDOW + 1)
            n_obs = me_pos - start + 1
            if n_obs < MIN_OBS:
                continue

            sl = slice(start, me_pos + 1)
            delay = compute_delay(ret_arr[sl], rm_arr[sl], lags_arr[sl])

            if not np.isnan(delay):
                dt = pd.Timestamp(dates[me_pos])
                results_list.append((stkcd, dt.strftime('%Y-%m-01'), delay, n_obs))

    monthly_pd = pd.DataFrame(results_list,
                               columns=['Stkcd', 'ym', 'PriceDelay_m', 'n_days'])
    monthly_pd['ym'] = pd.to_datetime(monthly_pd['ym'])

    # Save cache
    monthly_pd.to_parquet(CACHE_FILE, index=False)
    print(f"  Saved cache: {CACHE_FILE}")

elapsed = time.time() - t0
print(f"\n  Result: {len(monthly_pd):,} firm-months, {monthly_pd.Stkcd.nunique():,} firms")
print(f"  PriceDelay_m: mean={monthly_pd.PriceDelay_m.mean():.4f}, "
      f"median={monthly_pd.PriceDelay_m.median():.4f}, "
      f"std={monthly_pd.PriceDelay_m.std():.4f}")
print(f"  Date range: {monthly_pd.ym.min().strftime('%Y-%m')} to "
      f"{monthly_pd.ym.max().strftime('%Y-%m')}")
print(f"  Time: {elapsed:.1f}s")

# === Validation: compare with annual PriceDelay ===
print("\n  --- Validation vs annual PriceDelay ---")
try:
    annual_pd = pd.read_parquet(f"{DATA_DIR}/price_delay.parquet")
    monthly_pd['year'] = monthly_pd['ym'].dt.year
    annual_agg = monthly_pd.groupby(['Stkcd', 'year'])['PriceDelay_m'].mean().reset_index()
    annual_agg.columns = ['Stkcd', 'year', 'PD_rolling_avg']
    val = annual_agg.merge(annual_pd[['Stkcd', 'year', 'PriceDelay']], on=['Stkcd', 'year'])
    corr = val['PD_rolling_avg'].corr(val['PriceDelay'])
    print(f"  Correlation(rolling_avg, annual): {corr:.4f} ({len(val):,} obs)")
    if corr > 0.5:
        print(f"  ✓ Good consistency with annual measure")
    else:
        print(f"  ! Low correlation - rolling measure behaves differently")
except Exception as e:
    print(f"  Skipped validation: {e}")

# ================================================================
# 3. Build monthly event study panel
# ================================================================
print("\n3. Building event study panel...")
t0 = time.time()

# --- Treatment group (same as annual) ---
panel = pd.read_parquet(f"{DATA_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{DATA_DIR}/annual_report_features.parquet")
panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_per10k']],
    on=['Stkcd', 'year'], how='left'
)

# DU_kw = kw_per10k, lagged by 1 year
panel = panel.sort_values(['Stkcd', 'year'])
panel['DU_kw'] = panel.groupby('Stkcd')['kw_per10k'].shift(1)

# Treatment: 2017-2021 DU_kw mean >= median
pre_firms = panel[(panel.year >= 2017) & (panel.year <= 2021)]
firm_kw_mean = pre_firms.groupby('Stkcd')['DU_kw'].mean()
treat_th = firm_kw_mean.median()
treat_set = set(firm_kw_mean[firm_kw_mean >= treat_th].index)
firm_kw_dict = firm_kw_mean.to_dict()
print(f"  Treatment threshold: {treat_th:.4f}, {len(treat_set)} treated firms")

# --- Industry info ---
fi = pd.read_parquet(f"{DATA_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')

# Get latest industry code per firm
firm_ind = fi.sort_values(['Stkcd', 'year']).drop_duplicates('Stkcd', keep='last')
firm_ind = firm_ind[['Stkcd', 'IndustryCodeC']].copy()
firm_ind['Ind2'] = firm_ind['IndustryCodeC'].str[:3]

# Exclude financial firms and ST
exclude_fin = set(firm_ind[firm_ind.IndustryCodeC.str.startswith('J', na=False)].Stkcd)
fi_latest = fi.sort_values(['Stkcd', 'year']).drop_duplicates('Stkcd', keep='last')
exclude_st = set(fi_latest[fi_latest.LISTINGSTATE.isin(['ST', '*ST'])].Stkcd)

# --- Annual controls ---
controls_list = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth',
                 'BoardSize', 'IndepRatio', 'Dual', 'Top1Share',
                 'SOE', 'InstHold', 'Amihud', 'Analyst', 'AuditType']
ctrl_available = [c for c in controls_list if c in panel.columns]
annual_ctrl = panel[['Stkcd', 'year'] + ctrl_available].copy()

# Winsorize controls
def winsorize(s, lower=0.01, upper=0.99):
    q_lo, q_hi = s.quantile([lower, upper])
    return s.clip(q_lo, q_hi)

cont_ctrls = [c for c in ctrl_available if c not in ['Dual', 'SOE', 'AuditType']]
for v in cont_ctrls:
    if annual_ctrl[v].notna().any():
        annual_ctrl[v] = winsorize(annual_ctrl[v])

# --- Assemble monthly panel ---
es = monthly_pd.copy()
es['year'] = es['ym'].dt.year

# Filter
es = es[~es.Stkcd.isin(exclude_fin | exclude_st)]

# Treatment
es['Treat'] = es['Stkcd'].isin(treat_set).astype(float)

# Industry
es = es.merge(firm_ind[['Stkcd', 'Ind2']], on='Stkcd', how='left')

# Annual controls (merge on year, forward-fill for 2025+)
es = es.merge(annual_ctrl, on=['Stkcd', 'year'], how='left')
es = es.sort_values(['Stkcd', 'ym'])
for c in ctrl_available:
    es[c] = es.groupby('Stkcd')[c].ffill()

# Relative month: 2024-01 = 0
es['rel_month'] = (es.ym.dt.year - 2024) * 12 + (es.ym.dt.month - 1)

# Filter to event window
es = es[(es.rel_month >= REL_LO) & (es.rel_month <= REL_HI)]

# Winsorize PriceDelay_m
q_lo, q_hi = es.PriceDelay_m.quantile([0.01, 0.99])
es['PriceDelay_m'] = es.PriceDelay_m.clip(q_lo, q_hi)

# FE variables
es['Stkcd_fe'] = es['Stkcd'].astype(str)
es['ym_fe'] = es['ym'].dt.strftime('%Y-%m')
es['IndMonth'] = es['Ind2'].astype(str) + '_' + es['ym_fe']

# Drop NA
es = es.dropna(subset=['PriceDelay_m', 'Treat'] + ctrl_available)
ctrl_str = ' + '.join(ctrl_available)

print(f"  Event study panel: {len(es):,} firm-months, {es.Stkcd.nunique():,} firms")
print(f"  Months: {es.ym_fe.min()} to {es.ym_fe.max()}")
print(f"  Treated: {es[es.Treat==1].Stkcd.nunique():,}, "
      f"Control: {es[es.Treat==0].Stkcd.nunique():,}")
print(f"  Relative months: {es.rel_month.min()} to {es.rel_month.max()}")
print(f"  Built in {time.time()-t0:.1f}s")

# ================================================================
# 4. Run monthly event study
# ================================================================
print("\n4. Running monthly event study...")

def run_monthly_es(df, treat_type, fe_type='firm_month'):
    """
    Monthly event study regression.
    treat_type: 'binary' or 'continuous'
    fe_type: 'firm_month' or 'firm_indmonth'
    """
    es_data = df.copy()

    if treat_type == 'binary':
        es_data['T'] = es_data['Treat']
    else:
        es_data['T'] = es_data['Stkcd'].map(firm_kw_dict).fillna(0)
        mu, sigma = es_data['T'].mean(), es_data['T'].std()
        es_data['T'] = (es_data['T'] - mu) / sigma

    # Create relative month dummies × Treatment
    rel_months = sorted([m for m in es_data.rel_month.unique() if m != BASE_REL])
    for m in rel_months:
        col = f'Dp{m}' if m >= 0 else f'Dn{abs(m)}'
        es_data[col] = es_data['T'] * (es_data['rel_month'] == m).astype(float)

    d_vars = ' + '.join([f'Dp{m}' if m >= 0 else f'Dn{abs(m)}' for m in rel_months])

    if fe_type == 'firm_indmonth':
        fe_formula = 'Stkcd_fe + IndMonth'
    else:
        fe_formula = 'Stkcd_fe + ym_fe'

    m = pf.feols(f"PriceDelay_m ~ {d_vars} + {ctrl_str} | {fe_formula}",
                 data=es_data, vcov={"CRV1": "Stkcd_fe"})

    rows = []
    for rm in rel_months:
        k = f'Dp{rm}' if rm >= 0 else f'Dn{abs(rm)}'
        c = m.coef()[k]
        t_val = m.tstat()[k]
        p_val = m.pvalue()[k]
        se = abs(c / t_val) if abs(t_val) > 1e-6 else np.nan
        rows.append({
            'rel_month': rm, 'Coef': c, 'SE': se,
            'CI_lo': c - 1.96 * se, 'CI_hi': c + 1.96 * se,
            't': t_val, 'p': p_val, 'N': m._N
        })
    # Base period
    rows.append({
        'rel_month': BASE_REL, 'Coef': 0, 'SE': 0,
        'CI_lo': 0, 'CI_hi': 0,
        't': np.nan, 'p': np.nan, 'N': m._N
    })
    return pd.DataFrame(rows).sort_values('rel_month').reset_index(drop=True)


# Run 4 specifications
specs = {
    'binary_fm': ('binary', 'firm_month'),
    'binary_fim': ('binary', 'firm_indmonth'),
    'cont_fm': ('continuous', 'firm_month'),
    'cont_fim': ('continuous', 'firm_indmonth'),
}

results = {}
for key, (ttype, fetype) in specs.items():
    print(f"  {key}...")
    t1 = time.time()
    results[key] = run_monthly_es(es, ttype, fetype)
    print(f"    done ({time.time()-t1:.1f}s)")

# Print key results
for key in ['binary_fim', 'cont_fim']:
    df = results[key]
    print(f"\n  === {key} ===")
    pre = df[(df.rel_month < 0) & (df.rel_month != BASE_REL)]
    post = df[df.rel_month >= 0]
    pre_sig = pre[pre.p < 0.1]
    post_sig = post[post.p < 0.1]
    print(f"  Pre-treatment: {len(pre_sig)}/{len(pre)} significant at 10%")
    print(f"  Post-treatment: {len(post_sig)}/{len(post)} significant at 10%")

    # Show post-treatment months
    print(f"  Post-treatment coefficients:")
    for _, r in post.iterrows():
        sig = '***' if r.p < 0.01 else '**' if r.p < 0.05 else '*' if r.p < 0.1 else ''
        ym_label = pd.Timestamp('2024-01-01') + pd.DateOffset(months=int(r.rel_month))
        print(f"    t={int(r.rel_month):+3d} ({ym_label.strftime('%Y-%m')}): "
              f"coef={r.Coef:+.6f} t={r.t:+.3f} {sig}")

# ================================================================
# 5. Also run quarterly binned version
# ================================================================
print("\n5. Running quarterly binned event study...")

def to_quarter(m):
    """Convert relative month to relative quarter."""
    if m >= 0:
        return m // 3
    return -((abs(m) - 1) // 3 + 1)

BASE_Q = to_quarter(BASE_REL)  # = -1

def run_quarterly_es(df, treat_type, fe_type='firm_month'):
    """Quarterly binned event study."""
    es_data = df.copy()
    es_data['rel_q'] = es_data['rel_month'].apply(to_quarter)

    if treat_type == 'binary':
        es_data['T'] = es_data['Treat']
    else:
        es_data['T'] = es_data['Stkcd'].map(firm_kw_dict).fillna(0)
        mu, sigma = es_data['T'].mean(), es_data['T'].std()
        es_data['T'] = (es_data['T'] - mu) / sigma

    rel_qs = sorted([q for q in es_data.rel_q.unique() if q != BASE_Q])
    for q in rel_qs:
        col = f'Qp{q}' if q >= 0 else f'Qn{abs(q)}'
        es_data[col] = es_data['T'] * (es_data['rel_q'] == q).astype(float)

    d_vars = ' + '.join([f'Qp{q}' if q >= 0 else f'Qn{abs(q)}' for q in rel_qs])

    if fe_type == 'firm_indmonth':
        fe_formula = 'Stkcd_fe + IndMonth'
    else:
        fe_formula = 'Stkcd_fe + ym_fe'

    m = pf.feols(f"PriceDelay_m ~ {d_vars} + {ctrl_str} | {fe_formula}",
                 data=es_data, vcov={"CRV1": "Stkcd_fe"})

    rows = []
    for q in rel_qs:
        k = f'Qp{q}' if q >= 0 else f'Qn{abs(q)}'
        c = m.coef()[k]
        t_val = m.tstat()[k]
        p_val = m.pvalue()[k]
        se = abs(c / t_val) if abs(t_val) > 1e-6 else np.nan
        rows.append({
            'rel_quarter': q, 'Coef': c, 'SE': se,
            'CI_lo': c - 1.96 * se, 'CI_hi': c + 1.96 * se,
            't': t_val, 'p': p_val, 'N': m._N
        })
    rows.append({
        'rel_quarter': BASE_Q, 'Coef': 0, 'SE': 0,
        'CI_lo': 0, 'CI_hi': 0,
        't': np.nan, 'p': np.nan, 'N': m._N
    })
    return pd.DataFrame(rows).sort_values('rel_quarter').reset_index(drop=True)

q_results = {}
for key in ['binary_fim', 'cont_fim']:
    ttype = key.split('_')[0]
    qkey = f'{key}_quarterly'
    print(f"  {qkey}...")
    q_results[qkey] = run_quarterly_es(es, ttype, 'firm_indmonth')

# ================================================================
# 6. Plotting
# ================================================================
print("\n6. Plotting...")

def plot_monthly_es(ax, df, title):
    months = df['rel_month'].values
    coefs = df['Coef'].values
    ci_lo = df['CI_lo'].values
    ci_hi = df['CI_hi'].values

    ax.fill_between(months, ci_lo, ci_hi, alpha=0.15, color='#2166AC')
    ax.plot(months, coefs, 'o-', color='#2166AC', markersize=3, linewidth=1.0, zorder=5)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Policy implementation (t=0, Jan 2024)
    ax.axvline(x=-0.5, color='#B2182B', linewidth=1.8, linestyle='--', alpha=0.8,
               label='2024-01 实施')
    # Announcement (t=-5, Aug 2023)
    ax.axvline(x=-5.5, color='#F4A582', linewidth=1.2, linestyle=':',
               alpha=0.7, label='2023-08 公告')

    # Significance markers for post-treatment
    for _, r in df.iterrows():
        if np.isnan(r.p) or r.rel_month < 0:
            continue
        if r.p < 0.01:
            marker = '***'
        elif r.p < 0.05:
            marker = '**'
        elif r.p < 0.1:
            marker = '*'
        else:
            continue
        offset_y = 6 if r.Coef >= 0 else -12
        ax.annotate(marker, (r.rel_month, r.Coef), textcoords="offset points",
                    xytext=(0, offset_y), ha='center', fontsize=7,
                    fontweight='bold', color='#B2182B')

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel('相对月 (0 = 2024年1月)', fontsize=10)
    ax.set_ylabel('系数估计值', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_quarterly_es(ax, df, title):
    qs = df['rel_quarter'].values
    coefs = df['Coef'].values
    ci_lo = df['CI_lo'].values
    ci_hi = df['CI_hi'].values

    ax.bar(qs, coefs, width=0.7, color='#2166AC', alpha=0.6, zorder=3)
    ax.errorbar(qs, coefs, yerr=[coefs - ci_lo, ci_hi - coefs],
                fmt='none', color='#2166AC', capsize=3, zorder=4)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvline(x=-0.5, color='#B2182B', linewidth=1.8, linestyle='--', alpha=0.8,
               label='2024Q1 实施')

    for _, r in df.iterrows():
        if np.isnan(r.p):
            continue
        if r.p < 0.01:
            marker = '***'
        elif r.p < 0.05:
            marker = '**'
        elif r.p < 0.1:
            marker = '*'
        else:
            continue
        offset_y = 6 if r.Coef >= 0 else -12
        ax.annotate(marker, (r.rel_quarter, r.Coef), textcoords="offset points",
                    xytext=(0, offset_y), ha='center', fontsize=9,
                    fontweight='bold', color='#B2182B')

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel('相对季度 (0 = 2024Q1)', fontsize=10)
    ax.set_ylabel('系数估计值', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# --- Monthly 2x2 plot ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
plot_monthly_es(axes[0, 0], results['binary_fm'],
                '(a) Binary, Firm+Month FE')
plot_monthly_es(axes[0, 1], results['binary_fim'],
                '(b) Binary, Firm+Ind×Month FE')
plot_monthly_es(axes[1, 0], results['cont_fm'],
                '(c) Continuous, Firm+Month FE')
plot_monthly_es(axes[1, 1], results['cont_fim'],
                '(d) Continuous, Firm+Ind×Month FE')

fig.suptitle(f'月度事件研究: 2024年入表新规对股价延迟的动态效应\n'
             f'(滚动{WINDOW}交易日PriceDelay, t={REL_LO} to t={REL_HI})',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f"{RES_DIR}/monthly_event_study.png", dpi=200, bbox_inches='tight')
print(f"  月度图: {RES_DIR}/monthly_event_study.png")

# --- Quarterly plot ---
fig_q, axes_q = plt.subplots(1, 2, figsize=(14, 5.5))
plot_quarterly_es(axes_q[0], q_results['binary_fim_quarterly'],
                  '(a) Binary, Firm+Ind×Month FE')
plot_quarterly_es(axes_q[1], q_results['cont_fim_quarterly'],
                  '(b) Continuous, Firm+Ind×Month FE')

fig_q.suptitle('季度归并事件研究: 2024年入表新规对股价延迟的动态效应',
               fontsize=14, fontweight='bold', y=1.02)
fig_q.tight_layout()
fig_q.savefig(f"{RES_DIR}/monthly_event_study_quarterly.png", dpi=200, bbox_inches='tight')
print(f"  季度图: {RES_DIR}/monthly_event_study_quarterly.png")

plt.close('all')

# ================================================================
# 7. Save results
# ================================================================
print("\n7. Saving results...")

# Monthly coefficients
all_dfs = []
for key, df in results.items():
    df_out = df.copy()
    df_out['Specification'] = key
    all_dfs.append(df_out)
df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv(f"{RES_DIR}/monthly_event_study_coefficients.csv", index=False)
print(f"  {RES_DIR}/monthly_event_study_coefficients.csv ({len(df_all)} rows)")

# Quarterly coefficients
all_q = []
for key, df in q_results.items():
    df_out = df.copy()
    df_out['Specification'] = key
    all_q.append(df_out)
df_q = pd.concat(all_q, ignore_index=True)
df_q.to_csv(f"{RES_DIR}/monthly_event_study_quarterly_coefficients.csv", index=False)
print(f"  {RES_DIR}/monthly_event_study_quarterly_coefficients.csv ({len(df_q)} rows)")

# ================================================================
# 8. Summary
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for label, key in [("Binary Firm+IndMonth", "binary_fim"),
                    ("Continuous Firm+IndMonth", "cont_fim")]:
    df = results[key]
    pre = df[(df.rel_month < -5) & df.p.notna()]  # before announcement
    announce = df[(df.rel_month >= -5) & (df.rel_month < 0) & df.p.notna()]
    post = df[df.rel_month >= 0]

    pre_sig = pre[pre.p < 0.1]
    ann_sig = announce[announce.p < 0.1]
    post_sig = post[post.p < 0.1]

    print(f"\n  {label}:")
    print(f"    Pre-announcement (t<-5): {len(pre_sig)}/{len(pre)} sig at 10%")
    print(f"    Announcement window (-5 to -1): {len(ann_sig)}/{len(announce)} sig at 10%")
    print(f"    Post-treatment (t>=0): {len(post_sig)}/{len(post)} sig at 10%")

    # Average post-treatment coefficient
    if len(post) > 0:
        avg_coef = post['Coef'].mean()
        avg_t = post['t'].mean()
        print(f"    Average post coef: {avg_coef:+.6f}, avg t: {avg_t:+.3f}")

print("\n完成！")
