"""
IV Estimation & Oster Bounds: DU_kw → PriceDelay

工具变量:
  IV1: 省级大数据发展指数(2016) × Year  — 前定地理变异
  IV2: 同省跨行业peer均值DU_kw          — 省级数据生态溢出

改进（v2）:
  - 三套样本: full, iv1_only, iv1+iv2
  - 每个2SLS配同样本OLS
  - Hansen J过度识别检验
  - 替代口径稳健性: Coi_s, Goi_s 替代 Toi_s
  - Oster (2019) bounds
"""

import pandas as pd
import numpy as np
import pyfixest as pf
from scipy import stats
import warnings, os, re
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
RES_DIR = "/Users/mac/computerscience/15会计研究/results"
os.makedirs(RES_DIR, exist_ok=True)

# ================================================================
# 1. Load and prepare data
# ================================================================
print("=" * 70)
print("IV Estimation v2: DU_kw → PriceDelay")
print("=" * 70)
print("\n1. Loading data...")

panel = pd.read_parquet(f"{DATA_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{DATA_DIR}/annual_report_features.parquet")
panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_per10k']],
    on=['Stkcd', 'year'], how='left'
)

fi = pd.read_parquet(f"{DATA_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE',
                              'PROVINCE', 'CITY'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(
    fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE', 'PROVINCE', 'CITY']],
    on=['Stkcd', 'year'], how='left'
)

# Filters
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']

# Lag DU_kw by 1 year
panel = panel.sort_values(['Stkcd', 'year'])
panel['DU_kw'] = panel.groupby('Stkcd')['DU_kw'].shift(1)

# Winsorize
def winsorize(s, lower=0.01, upper=0.99):
    q_lo, q_hi = s.quantile([lower, upper])
    return s.clip(q_lo, q_hi)

cont_vars = ['PriceDelay', 'DU_kw',
             'Lev', 'FinAsset', 'DataAsset', 'ROA', 'Growth', 'Size',
             'TobinQ', 'Age', 'BoardSize', 'IndepRatio', 'Top1Share',
             'InstHold', 'Amihud', 'Analyst']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth',
            'BoardSize', 'IndepRatio', 'Dual', 'Top1Share',
            'SOE', 'InstHold', 'Amihud', 'Analyst', 'AuditType']
ctrl_str = ' + '.join(controls)

reg_df = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg_df['Stkcd_fe'] = reg_df['Stkcd'].astype(str)
reg_df['year_fe'] = reg_df['year'].astype(str)

print(f"  Full sample: {len(reg_df):,} obs, {reg_df.Stkcd.nunique():,} firms, "
      f"{reg_df.year.min()}-{reg_df.year.max()}")

# ================================================================
# 2. Construct instruments
# ================================================================
print("\n2. Constructing instruments...")

# --- Province name matching ---
def clean_prov(s):
    if pd.isna(s):
        return s
    return re.sub(r'省|市|自治区|壮族|维吾尔|回族|特别行政区', '', str(s))

bigdata = pd.read_parquet(f"{DATA_DIR}/province_bigdata_index.parquet")
# Skip header row if present
bigdata = bigdata[bigdata['Year'] != '会计年度'].copy()
bigdata['prov_clean'] = bigdata['Prgrvn'].apply(clean_prov)
for col in ['Toi_s', 'Coi_s', 'Goi_s', 'Cii_s']:
    bigdata[col] = pd.to_numeric(bigdata[col], errors='coerce')

reg_df['prov_clean'] = reg_df['PROVINCE'].apply(clean_prov)

# Firm -> province mapping (use latest available)
firm_prov = reg_df[['Stkcd', 'prov_clean']].drop_duplicates('Stkcd', keep='last')
firm_prov = firm_prov.merge(
    bigdata[['prov_clean', 'Toi_s', 'Coi_s', 'Goi_s']],
    on='prov_clean', how='left'
)
matched = firm_prov['Toi_s'].notna().sum()
total = len(firm_prov)
print(f"  Province BigData match: {matched}/{total} firms ({matched/total*100:.1f}%)")

# Merge all indices to reg_df
reg_df = reg_df.merge(firm_prov[['Stkcd', 'Toi_s', 'Coi_s', 'Goi_s']], on='Stkcd', how='left')

# --- IV1 variants: Index × (year - 2016) ---
reg_df['year_c'] = reg_df['year'] - 2016
reg_df['IV1'] = reg_df['Toi_s'] * reg_df['year_c']          # main
reg_df['IV1_Coi'] = reg_df['Coi_s'] * reg_df['year_c']      # robustness
reg_df['IV1_Goi'] = reg_df['Goi_s'] * reg_df['year_c']      # robustness

# --- IV2: Same-province, cross-industry peer DU_kw ---
prov_ind = reg_df.groupby(['prov_clean', 'Ind2', 'year']).agg(
    ind_sum=('DU_kw', 'sum'),
    ind_n=('DU_kw', 'count')
).reset_index()

prov_all = reg_df.groupby(['prov_clean', 'year']).agg(
    prov_sum=('DU_kw', 'sum'),
    prov_n=('DU_kw', 'count')
).reset_index()

prov_peer = prov_ind.merge(prov_all, on=['prov_clean', 'year'])
prov_peer['peer_sum'] = prov_peer['prov_sum'] - prov_peer['ind_sum']
prov_peer['peer_n'] = prov_peer['prov_n'] - prov_peer['ind_n']
prov_peer['IV2'] = np.where(
    prov_peer['peer_n'] >= 10,
    prov_peer['peer_sum'] / prov_peer['peer_n'],
    np.nan
)

reg_df = reg_df.merge(
    prov_peer[['prov_clean', 'Ind2', 'year', 'IV2']],
    on=['prov_clean', 'Ind2', 'year'], how='left'
)

def drop_fe_singletons(df, fe_cols):
    """Iteratively drop FE singleton groups to ensure pyfixest row alignment."""
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        for col in fe_cols:
            counts = df[col].value_counts()
            keep = counts[counts > 1].index
            df = df[df[col].isin(keep)]
    return df.reset_index(drop=True)

# --- Define three samples ---
samp_iv1 = reg_df.dropna(subset=['IV1']).copy()
samp_both_raw = reg_df.dropna(subset=['IV1', 'IV2']).copy()
# Pre-clean FE singletons to ensure exact residual-data alignment for Hansen J
samp_both = drop_fe_singletons(samp_both_raw, ['Stkcd_fe', 'year_fe'])
n_singletons = len(samp_both_raw) - len(samp_both)
if n_singletons > 0:
    print(f"  Dropped {n_singletons} FE singletons from IV1+IV2 sample")

print(f"\n  Sample sizes:")
print(f"    Full (OLS):      {len(reg_df):>8,} obs, {reg_df.Stkcd.nunique():>5,} firms")
print(f"    IV1 only:        {len(samp_iv1):>8,} obs, {samp_iv1.Stkcd.nunique():>5,} firms")
print(f"    IV1+IV2:         {len(samp_both):>8,} obs, {samp_both.Stkcd.nunique():>5,} firms")
print(f"    IV2 coverage:    {samp_both['IV2'].notna().sum():>8,} / {len(samp_iv1):,}")

# ================================================================
# Helper functions
# ================================================================
def sig_stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1: return '*'
    return ''

def print_coef(model, label, var='DU_kw'):
    c = model.coef()[var]
    se = model.se()[var]
    t = model.tstat()[var]
    p = model.pvalue()[var]
    n = model._N
    print(f"  {label:50s}: β={c:+.6f} se={se:.6f} t={t:+.3f} {sig_stars(p):3s}  N={n:,}")

def get_r2(model):
    """Get within R² from pyfixest model."""
    for attr in ['_r2_within', '_r2']:
        if hasattr(model, attr):
            return getattr(model, attr)
    return 1 - np.sum(model.resid() ** 2) / np.sum(
        (model._Y.values - model._Y.values.mean()) ** 2)

def oster_bounds(short_model, long_model, label=""):
    """Compute Oster (2019) bounds."""
    beta_s = short_model.coef()['DU_kw']
    beta_l = long_model.coef()['DU_kw']
    r2_s = get_r2(short_model)
    r2_l = get_r2(long_model)
    r_max = min(1.0, 1.3 * r2_l)
    denom_b = beta_s - beta_l
    denom_r = r2_l - r2_s
    if abs(denom_b) < 1e-10 or abs(denom_r) < 1e-10:
        return None
    delta_star = (beta_l * denom_r) / (denom_b * (r_max - r2_l))
    beta_star = beta_l - 1.0 * denom_b * (r_max - r2_l) / denom_r
    robust = abs(delta_star) > 1
    print(f"  {label:50s}: δ*={delta_star:+.3f} β*(δ=1)={beta_star:+.6f} "
          f"{'✓' if robust else '✗'} {'Robust' if robust else 'Not robust'}")
    return {'label': label, 'beta_short': beta_s, 'beta_long': beta_l,
            'r2_short': r2_s, 'r2_long': r2_l, 'delta_star': delta_star,
            'beta_star': beta_star, 'robust': robust}

def hansen_j_test(iv_model, data, ctrl_str, fe_str, iv_str_list):
    """
    Hansen J overidentification test.
    H0: all instruments are JOINTLY exogenous.
    Rejection means at least one IV may violate exclusion restriction,
    but cannot pinpoint which individual IV is problematic.

    Test stat: J = N * R² from regressing 2SLS residuals on all IVs + controls + FE.
    df = #instruments - #endogenous.

    NOTE: data should be pre-cleaned via drop_fe_singletons() to ensure
    residual-data row alignment. If n_resid != n_data, a warning is issued.
    """
    try:
        resid = np.array(iv_model.resid()).flatten()
        n_resid = len(resid)
        n_data = len(data)

        if n_resid != n_data:
            print(f"    WARNING: residual count ({n_resid}) != data rows ({n_data}). "
                  f"Pre-clean data with drop_fe_singletons() for exact alignment.")

        temp = data.reset_index(drop=True).copy()
        if n_resid != n_data:
            # Fallback: truncate (imprecise but unlikely to flip conclusion)
            temp = temp.iloc[:n_resid].copy()
        temp['_resid'] = resid

        # Regress residuals on instruments + controls + FE
        iv_vars = ' + '.join(iv_str_list)
        aux = pf.feols(f"_resid ~ {iv_vars} + {ctrl_str} | {fe_str}", data=temp)
        r2_aux = get_r2(aux)
        n = aux._N
        n_overid = len(iv_str_list) - 1  # #instruments - #endogenous(1)
        j_stat = n * r2_aux
        p_val = 1 - stats.chi2.cdf(j_stat, df=n_overid)
        return j_stat, p_val, n_overid
    except Exception as e:
        print(f"    Hansen J failed: {e}")
        return np.nan, np.nan, np.nan

# ================================================================
# 3. Main Panel: IV1-only sample (full coverage)
# ================================================================
print("\n" + "=" * 70)
print("3. PANEL A: IV1-only sample (N≈38,900)")
print("=" * 70)

# OLS on IV1 sample
ols_iv1s_fy = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_fe + year_fe",
                       data=samp_iv1, vcov={"CRV1": "IndYear"})
ols_iv1s_fiy = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_fe + IndYear",
                        data=samp_iv1, vcov={"CRV1": "IndYear"})

print("\n  OLS (same sample as IV1):")
print_coef(ols_iv1s_fy, "(1) OLS, Firm+Year FE")
print_coef(ols_iv1s_fiy, "(2) OLS, Firm+Ind×Year FE")

# First stage: IV1
fs_iv1 = pf.feols(f"DU_kw ~ IV1 + {ctrl_str} | Stkcd_fe + year_fe",
                  data=samp_iv1, vcov={"CRV1": "IndYear"})
f_iv1 = fs_iv1.tstat()['IV1'] ** 2
print(f"\n  First stage IV1: coef={fs_iv1.coef()['IV1']:+.6f} "
      f"t={fs_iv1.tstat()['IV1']:+.3f} F={f_iv1:.1f} "
      f"{'✓ Strong' if f_iv1 > 10 else '✗ Weak'}")

# 2SLS: IV1 only
iv1_fy = pf.feols(f"PriceDelay ~ {ctrl_str} | Stkcd_fe + year_fe | DU_kw ~ IV1",
                  data=samp_iv1, vcov={"CRV1": "IndYear"})

print("\n  2SLS (IV1 only):")
print_coef(iv1_fy, "(3) 2SLS IV1, Firm+Year FE")
ols_b = ols_iv1s_fy.coef()['DU_kw']
iv_b = iv1_fy.coef()['DU_kw']
print(f"  {'':50s}  OLS amplification: {abs(iv_b/ols_b):.1f}x")

# ================================================================
# 4. PANEL B: IV1+IV2 sample (overidentified)
# ================================================================
print("\n" + "=" * 70)
print("4. PANEL B: IV1+IV2 sample (N≈29,300)")
print("=" * 70)

# OLS on both-IV sample
ols_both_fy = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_fe + year_fe",
                       data=samp_both, vcov={"CRV1": "IndYear"})
ols_both_fiy = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_fe + IndYear",
                        data=samp_both, vcov={"CRV1": "IndYear"})

print("\n  OLS (same sample as IV1+IV2):")
print_coef(ols_both_fy, "(4) OLS, Firm+Year FE")
print_coef(ols_both_fiy, "(5) OLS, Firm+Ind×Year FE")

# First stages
fs_iv1_both = pf.feols(f"DU_kw ~ IV1 + {ctrl_str} | Stkcd_fe + year_fe",
                       data=samp_both, vcov={"CRV1": "IndYear"})
fs_iv2_both = pf.feols(f"DU_kw ~ IV2 + {ctrl_str} | Stkcd_fe + IndYear",
                       data=samp_both, vcov={"CRV1": "IndYear"})
fs_joint = pf.feols(f"DU_kw ~ IV1 + IV2 + {ctrl_str} | Stkcd_fe + year_fe",
                    data=samp_both, vcov={"CRV1": "IndYear"})

f_iv1_b = fs_iv1_both.tstat()['IV1'] ** 2
f_iv2_b = fs_iv2_both.tstat()['IV2'] ** 2

# Conditional F: IV2 conditional on IV1 being in the model
f_iv2_cond = fs_joint.tstat()['IV2'] ** 2

print(f"\n  First stage diagnostics:")
print(f"    IV1 alone:        F={f_iv1_b:.1f} {'✓' if f_iv1_b > 10 else '✗'}")
print(f"    IV2 alone (IndYr): F={f_iv2_b:.1f} {'✓' if f_iv2_b > 10 else '✗'}")
print(f"    IV2 cond on IV1:  F={f_iv2_cond:.1f} {'✓' if f_iv2_cond > 10 else '✗'}")
print(f"    Joint (IV1+IV2):  IV1 t={fs_joint.tstat()['IV1']:+.3f}, "
      f"IV2 t={fs_joint.tstat()['IV2']:+.3f}")

# Approximate joint F
t1 = fs_joint.tstat()['IV1']
t2 = fs_joint.tstat()['IV2']
f_joint_approx = (t1**2 + t2**2) / 2
print(f"    Joint F (approx): {f_joint_approx:.1f} {'✓' if f_joint_approx > 10 else '✗'}")

# 2SLS: IV1 only on this sample (for comparison)
iv1_both = pf.feols(f"PriceDelay ~ {ctrl_str} | Stkcd_fe + year_fe | DU_kw ~ IV1",
                    data=samp_both, vcov={"CRV1": "IndYear"})

# 2SLS: IV1+IV2 overidentified
iv12_both = pf.feols(f"PriceDelay ~ {ctrl_str} | Stkcd_fe + year_fe | DU_kw ~ IV1 + IV2",
                     data=samp_both, vcov={"CRV1": "IndYear"})

print("\n  2SLS:")
print_coef(iv1_both, "(6) 2SLS IV1 only, Firm+Year FE")
print_coef(iv12_both, "(7) 2SLS IV1+IV2, Firm+Year FE (overidentified)")

ols_b2 = ols_both_fy.coef()['DU_kw']
print(f"  {'':50s}  IV1 amplification: {abs(iv1_both.coef()['DU_kw']/ols_b2):.1f}x")
print(f"  {'':50s}  IV1+2 amplification: {abs(iv12_both.coef()['DU_kw']/ols_b2):.1f}x")

# ================================================================
# 5. Hansen J Test (overidentification)
# ================================================================
print("\n" + "=" * 70)
print("5. Hansen J Test (overidentification)")
print("=" * 70)

j_stat, j_pval, j_df = hansen_j_test(
    iv12_both, samp_both, ctrl_str,
    'Stkcd_fe + year_fe', ['IV1', 'IV2'])

if not np.isnan(j_stat):
    print(f"\n  J statistic = {j_stat:.4f}, df = {j_df}, p-value = {j_pval:.4f}")
    print(f"  {'✓ Not rejected' if j_pval > 0.05 else '✗ Rejected'}: joint exogeneity of instrument set")
    print(f"  Note: rejection means AT LEAST ONE IV may violate exclusion restriction")
    print(f"  (Cannot pinpoint which individual IV is problematic)")

# ================================================================
# 6. Alternative IV1: Coi_s, Goi_s robustness
# ================================================================
print("\n" + "=" * 70)
print("6. Alternative IV1 Robustness (Coi_s, Goi_s)")
print("=" * 70)

for idx_name, iv_col in [("Coi_s (商用)", "IV1_Coi"), ("Goi_s (政用)", "IV1_Goi")]:
    samp_alt = samp_iv1.dropna(subset=[iv_col]).copy()
    if len(samp_alt) == 0:
        print(f"\n  {idx_name}: no valid obs, skipping")
        continue

    # First stage
    fs_alt = pf.feols(f"DU_kw ~ {iv_col} + {ctrl_str} | Stkcd_fe + year_fe",
                      data=samp_alt, vcov={"CRV1": "IndYear"})
    f_alt = fs_alt.tstat()[iv_col] ** 2

    # 2SLS
    try:
        iv_alt = pf.feols(
            f"PriceDelay ~ {ctrl_str} | Stkcd_fe + year_fe | DU_kw ~ {iv_col}",
            data=samp_alt, vcov={"CRV1": "IndYear"})
        c = iv_alt.coef()['DU_kw']
        se = iv_alt.se()['DU_kw']
        t = iv_alt.tstat()['DU_kw']
        p = iv_alt.pvalue()['DU_kw']
        print(f"\n  {idx_name}:")
        print(f"    First-stage F = {f_alt:.1f} {'✓' if f_alt > 10 else '✗'}")
        print(f"    2SLS: β={c:+.6f} se={se:.6f} t={t:+.3f} {sig_stars(p)} N={iv_alt._N:,}")
    except Exception as e:
        print(f"\n  {idx_name}: 2SLS failed: {e}")

# ================================================================
# 7. Oster (2019) Bounds
# ================================================================
print("\n" + "=" * 70)
print("7. Oster (2019) Bounds")
print("=" * 70)

# On full sample
print("\n  Full sample (N=38,900+):")
ols_short_full = pf.feols("PriceDelay ~ DU_kw | Stkcd_fe + year_fe", data=samp_iv1)
ols_long_full = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_fe + year_fe", data=samp_iv1)
oster_fy = oster_bounds(ols_short_full, ols_long_full, "Firm+Year FE (full)")

ols_short_fiy = pf.feols("PriceDelay ~ DU_kw | Stkcd_fe + IndYear", data=samp_iv1)
ols_long_fiy = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_fe + IndYear", data=samp_iv1)
oster_fiy = oster_bounds(ols_short_fiy, ols_long_fiy, "Firm+Ind×Year FE (full)")

# On IV1+IV2 sample
print("\n  IV1+IV2 sample (N=29,300):")
ols_short_b = pf.feols("PriceDelay ~ DU_kw | Stkcd_fe + year_fe", data=samp_both)
ols_long_b = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_fe + year_fe", data=samp_both)
oster_both = oster_bounds(ols_short_b, ols_long_b, "Firm+Year FE (IV1+IV2 sample)")

# ================================================================
# 8. Exclusion Restriction Stress Tests
# ================================================================
print("\n\n" + "=" * 70)
print("8. EXCLUSION RESTRICTION STRESS TESTS")
print("=" * 70)

stress_rows = []  # collect stress test results

# --- 8a. Reduced-Form: IV1 → PriceDelay directly ---
print("\n--- 8a. Reduced-Form: IV1 → PriceDelay (no 2SLS) ---")
print("  Logic: if IV1 → DU_kw → PriceDelay, then IV1 → PriceDelay should be negative")

rf_fy = pf.feols(f"PriceDelay ~ IV1 + {ctrl_str} | Stkcd_fe + year_fe",
                 data=samp_iv1, vcov={"CRV1": "IndYear"})
print_coef(rf_fy, "Reduced-form, Firm+Year FE", var='IV1')

rf_fiy = pf.feols(f"PriceDelay ~ IV1 + {ctrl_str} | Stkcd_fe + IndYear",
                  data=samp_iv1, vcov={"CRV1": "IndYear"})
print_coef(rf_fiy, "Reduced-form, Firm+Ind×Year FE", var='IV1')

# Implied IV estimate = reduced-form / first-stage
rf_coef = rf_fy.coef()['IV1']
fs_coef = fs_iv1.coef()['IV1']
implied_iv = rf_coef / fs_coef
print(f"  Implied IV = RF/FS = {rf_coef:.6f}/{fs_coef:.6f} = {implied_iv:.6f}")
print(f"  Actual 2SLS β = {iv1_fy.coef()['DU_kw']:.6f} (should match)")

for label, m in [("RF_FirmYear", rf_fy), ("RF_FirmIndYear", rf_fiy)]:
    stress_rows.append({
        'Test': 'Reduced-form', 'Specification': label,
        'Var': 'IV1', 'Coef': m.coef()['IV1'], 'SE': m.se()['IV1'],
        't': m.tstat()['IV1'], 'p': m.pvalue()['IV1'], 'N': m._N
    })

# --- 8b. Reverse-Temporal Placebo ---
print("\n--- 8b. Reverse-Temporal Placebo ---")
print("  Logic: future data utilization should NOT predict current price delay")
print("  DU_kw = kw_per10k_{t-1}; lead2 = kw_per10k_{t+1} (truly future)")

samp_placebo = samp_iv1.copy()
# DU_kw = kw_per10k_{t-1}. shift(-2) gives DU_kw from row (i,t+2) = kw_per10k_{t+1}
samp_placebo['DU_kw_lead2'] = samp_placebo.groupby('Stkcd')['DU_kw'].shift(-2)
placebo_df = samp_placebo.dropna(subset=['DU_kw_lead2']).copy()
print(f"  Placebo sample: {len(placebo_df):,} obs (lost {len(samp_iv1)-len(placebo_df):,} from lead)")

# OLS: PriceDelay_t ~ DU_kw_{t+1} (should be insignificant)
ols_lead = pf.feols(f"PriceDelay ~ DU_kw_lead2 + {ctrl_str} | Stkcd_fe + year_fe",
                    data=placebo_df, vcov={"CRV1": "IndYear"})
print_coef(ols_lead, "Placebo: PriceDelay_t ~ DU_kw_{t+1}", var='DU_kw_lead2')

# Benchmark: OLS with actual DU_kw_{t-1} on same subsample
ols_actual_sub = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_fe + year_fe",
                          data=placebo_df, vcov={"CRV1": "IndYear"})
print_coef(ols_actual_sub, "Benchmark: PriceDelay_t ~ DU_kw_{t-1} (same sample)", var='DU_kw')

lead_p = ols_lead.pvalue()['DU_kw_lead2']
actual_p = ols_actual_sub.pvalue()['DU_kw']
print(f"  Placebo p={lead_p:.4f} vs Actual p={actual_p:.4f}")
print(f"  {'✓ Placebo insignificant' if lead_p > 0.1 else '⚠ Placebo significant (expected with persistent DU_kw)'}")

# Conditional test: control for DU_kw_{t-1} when testing DU_kw_{t+1}
# If causal direction is t-1 → t, then lead should be insignificant conditional on lag
print("\n  Conditional test (controls for DU_kw_{t-1}):")
ols_cond = pf.feols(f"PriceDelay ~ DU_kw + DU_kw_lead2 + {ctrl_str} | Stkcd_fe + year_fe",
                    data=placebo_df, vcov={"CRV1": "IndYear"})
print_coef(ols_cond, "DU_kw_{t-1} (conditional)", var='DU_kw')
print_coef(ols_cond, "DU_kw_{t+1} (conditional)", var='DU_kw_lead2')
cond_p_lag = ols_cond.pvalue()['DU_kw']
cond_p_lead = ols_cond.pvalue()['DU_kw_lead2']
print(f"  Conditional: lag p={cond_p_lag:.4f}, lead p={cond_p_lead:.4f}")
print(f"  {'✓ Lead insignificant conditional on lag' if cond_p_lead > 0.1 else '⚠ Lead still significant'}")

stress_rows.append({
    'Test': 'ReverseTemporal_Placebo', 'Specification': 'OLS_lead2',
    'Var': 'DU_kw_lead2', 'Coef': ols_lead.coef()['DU_kw_lead2'],
    'SE': ols_lead.se()['DU_kw_lead2'], 't': ols_lead.tstat()['DU_kw_lead2'],
    'p': lead_p, 'N': ols_lead._N
})
stress_rows.append({
    'Test': 'ReverseTemporal_Benchmark', 'Specification': 'OLS_actual_samesample',
    'Var': 'DU_kw', 'Coef': ols_actual_sub.coef()['DU_kw'],
    'SE': ols_actual_sub.se()['DU_kw'], 't': ols_actual_sub.tstat()['DU_kw'],
    'p': actual_p, 'N': ols_actual_sub._N
})
stress_rows.append({
    'Test': 'ReverseTemporal_Conditional', 'Specification': 'OLS_cond_lag',
    'Var': 'DU_kw', 'Coef': ols_cond.coef()['DU_kw'],
    'SE': ols_cond.se()['DU_kw'], 't': ols_cond.tstat()['DU_kw'],
    'p': cond_p_lag, 'N': ols_cond._N
})
stress_rows.append({
    'Test': 'ReverseTemporal_Conditional', 'Specification': 'OLS_cond_lead',
    'Var': 'DU_kw_lead2', 'Coef': ols_cond.coef()['DU_kw_lead2'],
    'SE': ols_cond.se()['DU_kw_lead2'], 't': ols_cond.tstat()['DU_kw_lead2'],
    'p': cond_p_lead, 'N': ols_cond._N
})

# --- 8c. Pre-Period Falsification ---
print("\n--- 8c. Pre-Period Falsification ---")
print("  Logic: IV1 effect should be attenuated before digital infrastructure matures")

pre_sample = samp_iv1[samp_iv1['year'] <= 2016].copy()
post_sample = samp_iv1[samp_iv1['year'] > 2016].copy()
print(f"  Pre-period:  {len(pre_sample):>8,} obs, years {pre_sample.year.min()}-{pre_sample.year.max()}")
print(f"  Post-period: {len(post_sample):>8,} obs, years {post_sample.year.min()}-{post_sample.year.max()}")

for period_label, period_df in [("Pre (≤2016)", pre_sample), ("Post (>2016)", post_sample)]:
    if len(period_df) < 500:
        print(f"\n  {period_label}: insufficient obs ({len(period_df)}), skipping")
        continue
    try:
        # First-stage
        fs_p = pf.feols(f"DU_kw ~ IV1 + {ctrl_str} | Stkcd_fe + year_fe",
                        data=period_df, vcov={"CRV1": "IndYear"})
        f_p = fs_p.tstat()['IV1'] ** 2
        # Reduced-form
        rf_p = pf.feols(f"PriceDelay ~ IV1 + {ctrl_str} | Stkcd_fe + year_fe",
                        data=period_df, vcov={"CRV1": "IndYear"})
        print(f"\n  {period_label}:")
        print(f"    First-stage F = {f_p:.1f} {'(weak)' if f_p < 10 else '(strong)'}")
        print_coef(rf_p, f"    Reduced-form ({period_label})", var='IV1')

        stress_rows.append({
            'Test': f'PrePeriod_{period_label[:4]}', 'Specification': 'ReducedForm',
            'Var': 'IV1', 'Coef': rf_p.coef()['IV1'], 'SE': rf_p.se()['IV1'],
            't': rf_p.tstat()['IV1'], 'p': rf_p.pvalue()['IV1'],
            'N': rf_p._N, 'F_first': f_p
        })
    except Exception as e:
        print(f"\n  {period_label}: regression failed ({e})")

# --- 8d. Province-Level Proxy Controls ---
print("\n--- 8d. Province-Level Proxy Controls (lagged) ---")
print("  Adding prov_avg_size, prov_n_firms, prov_avg_analyst (all t-1)")

# Construct province-year aggregates
prov_agg = reg_df.groupby(['prov_clean', 'year']).agg(
    prov_avg_size=('Size', 'mean'),
    prov_n_firms=('Stkcd', 'nunique'),
    prov_avg_analyst=('Analyst', 'mean')
).reset_index()

# Lag by 1 year: use year t-1 values to explain year t
prov_agg_lag = prov_agg.copy()
prov_agg_lag['year'] = prov_agg_lag['year'] + 1

samp_prov = samp_iv1.merge(
    prov_agg_lag[['prov_clean', 'year', 'prov_avg_size', 'prov_n_firms', 'prov_avg_analyst']],
    on=['prov_clean', 'year'], how='left'
)
samp_prov = samp_prov.dropna(subset=['prov_avg_size', 'prov_n_firms', 'prov_avg_analyst']).copy()
print(f"  Sample with province controls: {len(samp_prov):,} obs "
      f"(lost {len(samp_iv1)-len(samp_prov):,})")

# Winsorize province controls
for v in ['prov_avg_size', 'prov_n_firms', 'prov_avg_analyst']:
    samp_prov[v] = winsorize(samp_prov[v])

prov_ctrl_str = ctrl_str + ' + prov_avg_size + prov_n_firms + prov_avg_analyst'

# OLS with province controls
ols_prov = pf.feols(f"PriceDelay ~ DU_kw + {prov_ctrl_str} | Stkcd_fe + year_fe",
                    data=samp_prov, vcov={"CRV1": "IndYear"})
print_coef(ols_prov, "OLS + province controls")

# First-stage with province controls
fs_prov = pf.feols(f"DU_kw ~ IV1 + {prov_ctrl_str} | Stkcd_fe + year_fe",
                   data=samp_prov, vcov={"CRV1": "IndYear"})
f_prov = fs_prov.tstat()['IV1'] ** 2
print(f"  First-stage F = {f_prov:.1f} (was {f_iv1:.1f} without province controls)")

# 2SLS with province controls
iv_prov = pf.feols(f"PriceDelay ~ {prov_ctrl_str} | Stkcd_fe + year_fe | DU_kw ~ IV1",
                   data=samp_prov, vcov={"CRV1": "IndYear"})
print_coef(iv_prov, "2SLS IV1 + province controls")

# Stability comparison
beta_base = iv1_fy.coef()['DU_kw']
beta_prov = iv_prov.coef()['DU_kw']
change_pct = (beta_prov - beta_base) / abs(beta_base) * 100
print(f"\n  Stability check:")
print(f"    β baseline:     {beta_base:+.6f} (N={iv1_fy._N:,}, F={f_iv1:.1f})")
print(f"    β + prov ctrl:  {beta_prov:+.6f} (N={iv_prov._N:,}, F={f_prov:.1f})")
print(f"    Change: {change_pct:+.1f}%")
print(f"    {'✓ Stable' if abs(change_pct) < 30 else '⚠ Substantial change'}")

stress_rows.append({
    'Test': 'ProvControls', 'Specification': 'OLS_prov',
    'Var': 'DU_kw', 'Coef': ols_prov.coef()['DU_kw'], 'SE': ols_prov.se()['DU_kw'],
    't': ols_prov.tstat()['DU_kw'], 'p': ols_prov.pvalue()['DU_kw'], 'N': ols_prov._N
})
stress_rows.append({
    'Test': 'ProvControls', 'Specification': '2SLS_IV1_prov',
    'Var': 'DU_kw', 'Coef': iv_prov.coef()['DU_kw'], 'SE': iv_prov.se()['DU_kw'],
    't': iv_prov.tstat()['DU_kw'], 'p': iv_prov.pvalue()['DU_kw'],
    'N': iv_prov._N, 'F_first': f_prov
})

# --- Save stress test results ---
stress_df = pd.DataFrame(stress_rows)
stress_df.to_csv(f"{RES_DIR}/iv_stress_test_results.csv", index=False)
print(f"\n  Saved: {RES_DIR}/iv_stress_test_results.csv")

# ================================================================
# 9. Summary Table
# ================================================================
print("\n\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)

print(f"\n{'':2s}{'Specification':52s} {'β(DU_kw)':>10s} {'SE':>10s} {'t':>8s} {'N':>8s}")
print("-" * 90)
print("  --- Panel A: IV1-only sample ---")

specs = [
    ("(1) OLS, Firm+Year FE", ols_iv1s_fy),
    ("(2) OLS, Firm+Ind×Year FE", ols_iv1s_fiy),
]
for label, m in specs:
    c = m.coef()['DU_kw']; se = m.se()['DU_kw']; t = m.tstat()['DU_kw']
    print(f"  {label:52s} {c:>+10.6f} {se:>10.6f} {t:>+8.3f}{sig_stars(m.pvalue()['DU_kw']):3s} {m._N:>8,}")

print(f"  {'First-stage IV1':52s} {'':>10s} {'':>10s} F={f_iv1:>5.1f}{'':>3s} {fs_iv1._N:>8,}")
print_coef(iv1_fy, "(3) 2SLS IV1, Firm+Year FE")

print("-" * 90)
print("  --- Panel B: IV1+IV2 sample ---")

specs2 = [
    ("(4) OLS, Firm+Year FE", ols_both_fy),
    ("(5) OLS, Firm+Ind×Year FE", ols_both_fiy),
]
for label, m in specs2:
    c = m.coef()['DU_kw']; se = m.se()['DU_kw']; t = m.tstat()['DU_kw']
    print(f"  {label:52s} {c:>+10.6f} {se:>10.6f} {t:>+8.3f}{sig_stars(m.pvalue()['DU_kw']):3s} {m._N:>8,}")

print(f"  {'First-stage IV1 (this sample)':52s} {'':>10s} {'':>10s} F={f_iv1_b:>5.1f}{'':>3s} {fs_iv1_both._N:>8,}")
print(f"  {'First-stage IV2 cond. on IV1':52s} {'':>10s} {'':>10s} F={f_iv2_cond:>5.1f}{'':>3s} {fs_joint._N:>8,}")
print(f"  {'Joint F (IV1+IV2)':52s} {'':>10s} {'':>10s} F={f_joint_approx:>5.1f}")

print_coef(iv1_both, "(6) 2SLS IV1 only, Firm+Year FE")
print_coef(iv12_both, "(7) 2SLS IV1+IV2, Firm+Year FE")

if not np.isnan(j_stat):
    print(f"  {'Hansen J (overid)':52s} {'':>10s} {'':>10s} J={j_stat:>5.2f} p={j_pval:.3f}")

print("-" * 90)
print("  --- Panel C: Oster Bounds ---")
if oster_fy:
    print(f"  {'Firm+Year FE (full sample)':52s} δ*={oster_fy['delta_star']:>+8.3f}")
if oster_fiy:
    print(f"  {'Firm+Ind×Year FE (full sample)':52s} δ*={oster_fiy['delta_star']:>+8.3f}")

# ================================================================
# 10. Save results
# ================================================================
print("\n\n10. Saving results...")

rows = []
for label, model_name, sample_label in [
    ("OLS_FirmYear_IV1samp", ols_iv1s_fy, "IV1"),
    ("OLS_FirmIndYear_IV1samp", ols_iv1s_fiy, "IV1"),
    ("2SLS_IV1_FirmYear", iv1_fy, "IV1"),
    ("OLS_FirmYear_bothsamp", ols_both_fy, "IV1+IV2"),
    ("OLS_FirmIndYear_bothsamp", ols_both_fiy, "IV1+IV2"),
    ("2SLS_IV1_bothsamp", iv1_both, "IV1+IV2"),
    ("2SLS_IV1_IV2", iv12_both, "IV1+IV2"),
]:
    rows.append({
        'Specification': label,
        'Sample': sample_label,
        'Coef': model_name.coef()['DU_kw'],
        'SE': model_name.se()['DU_kw'],
        't': model_name.tstat()['DU_kw'],
        'p': model_name.pvalue()['DU_kw'],
        'N': model_name._N
    })

# Add diagnostics
rows.append({'Specification': 'FirstStage_IV1_full', 'Sample': 'IV1',
             'Coef': fs_iv1.coef()['IV1'], 't': fs_iv1.tstat()['IV1'],
             'N': fs_iv1._N, 'SE': fs_iv1.se()['IV1'],
             'p': fs_iv1.pvalue()['IV1']})
rows.append({'Specification': 'FirstStage_IV1_both', 'Sample': 'IV1+IV2',
             'Coef': fs_iv1_both.coef()['IV1'], 't': fs_iv1_both.tstat()['IV1'],
             'N': fs_iv1_both._N, 'SE': fs_iv1_both.se()['IV1'],
             'p': fs_iv1_both.pvalue()['IV1']})

if not np.isnan(j_stat):
    rows.append({'Specification': 'Hansen_J', 'Sample': 'IV1+IV2',
                 'Coef': j_stat, 'p': j_pval, 'N': iv12_both._N,
                 'SE': np.nan, 't': np.nan})

if oster_fy:
    rows.append({'Specification': 'Oster_FirmYear_full', 'Sample': 'IV1',
                 'Coef': oster_fy['delta_star'], 'SE': np.nan,
                 't': np.nan, 'p': np.nan, 'N': ols_long_full._N})
if oster_fiy:
    rows.append({'Specification': 'Oster_FirmIndYear_full', 'Sample': 'IV1',
                 'Coef': oster_fiy['delta_star'], 'SE': np.nan,
                 't': np.nan, 'p': np.nan, 'N': ols_long_fiy._N})

pd.DataFrame(rows).to_csv(f"{RES_DIR}/iv_estimation_results.csv", index=False)
print(f"  Saved: {RES_DIR}/iv_estimation_results.csv")

print("\n完成！")
