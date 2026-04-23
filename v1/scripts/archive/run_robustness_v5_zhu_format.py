"""
v5 稳健性检验 + 基准回归重构
完全对标朱康(2025)会计研究表2/表3格式

表2(基准回归): 6列，全部带控制变量
  (1) Firm+Year FE, DU_kw
  (2) Firm+Year FE, DU_kw + 全部控制
  (3) Firm+Year FE, DU_kw_ln + 全部控制
  (4) Firm+Year FE, DU_sub_ln + 全部控制
  (5) Ind+Year FE, DU_kw + 全部控制
  (6) Firm+Year FE, DU_kw + 全部控制 + FinAsset中介

表3(稳健性): 9列
  (1) 替换被解释变量 SYNCH
  (2) 替换解释变量 DU_kw_ln
  (3) 控制时变行业 Ind×Year FE
  (4) 控制时变地区 Province×Year FE
  (5) 剔除2024年(政策冲击年)
  (6) 剔除信息技术业
  (7) PSM匹配
  (8) 同行业均值IV
  (9) Heckman两阶段
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import warnings, os, json
from scipy import stats

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data_parquet")
RES_DIR = os.path.join(BASE_DIR, "results/v5_zhu_format")
os.makedirs(RES_DIR, exist_ok=True)

# ================================================================
# 1. 加载数据（同v3）
# ================================================================
print("=" * 70)
print("加载数据")
print("=" * 70)

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")
synch = pd.read_parquet(f"{OUT_DIR}/price_synchronicity.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'mda_kw_per10k',
             'substantive_ratio', 'substantive_count',
             'kw_data_stock', 'kw_data_dev', 'kw_data_app', 'kw_data_value',
             'kw_data_gov', 'has_mda']],
    on=['Stkcd', 'year'], how='left'
)
panel = panel.merge(synch[['Stkcd', 'year', 'SYNCH']], on=['Stkcd', 'year'], how='left')

fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'ShortName', 'IndustryCodeC',
                              'LISTINGSTATE', 'PROVINCECODE', 'PROVINCE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE', 'PROVINCECODE']],
                    on=['Stkcd', 'year'], how='left')

# 样本筛选
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# 构造变量
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['ProvYear'] = panel['PROVINCECODE'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])

# Winsorize
def winsorize(s, lower=0.01, upper=0.99):
    q_lo, q_hi = s.quantile([lower, upper])
    return s.clip(q_lo, q_hi)

cont_vars = ['PriceDelay', 'SYNCH', 'DU_kw', 'DU_kw_ln', 'DU_sub_ln',
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
reg_df['Stkcd_str'] = reg_df['Stkcd'].astype(str)
reg_df['year_str'] = reg_df['year'].astype(str)

# 用于pyfixest的FE变量
reg_fe = reg_df.copy()
reg_fe['Stkcd'] = reg_fe['Stkcd_str']
reg_fe['year'] = reg_fe['year_str']

print(f"  回归样本: {len(reg_fe):,} obs, {reg_fe.Stkcd.nunique():,} firms")


# ================================================================
# 辅助函数
# ================================================================
def extract_result(m, var='DU_kw'):
    keys = [v for v in m.coef().keys() if var in v] if var == 'DU_kw' else [var]
    kv = keys[0] if keys else var
    c = m.coef().get(kv, np.nan)
    se = m.se().get(kv, np.nan)
    t = m.tstat().get(kv, np.nan)
    p = m.pvalue().get(kv, np.nan)
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    return {'Coef': c, 'SE': se, 't': t, 'p': p, 'Sig': sig, 'N': m._N, 'R2': m._r2}


# ================================================================
# 2. 表2：基准回归（6列，全部带控制）
# ================================================================
print("\n" + "=" * 70)
print("表2: 基准回归 (对标朱康表2)")
print("=" * 70)

baseline = {}

# (1) Firm+Year, DU_kw + Controls
baseline['(1)'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year",
                           data=reg_fe, vcov={"CRV1": "IndYear"})
# (2) Firm+Year, DU_kw_ln + Controls
baseline['(2)'] = pf.feols(f"PriceDelay ~ DU_kw_ln + {ctrl_str} | Stkcd + year",
                           data=reg_fe, vcov={"CRV1": "IndYear"})
# (3) Firm+Year, DU_sub_ln + Controls
baseline['(3)'] = pf.feols(f"PriceDelay ~ DU_sub_ln + {ctrl_str} | Stkcd + year",
                           data=reg_fe, vcov={"CRV1": "IndYear"})
# (4) Ind+Year, DU_kw + Controls
baseline['(4)'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Ind2 + year",
                           data=reg_fe, vcov={"CRV1": "IndYear"})
# (5) SYNCH as DV, Firm+Year
synch_df = reg_fe.dropna(subset=['SYNCH'])
baseline['(5)'] = pf.feols(f"SYNCH ~ DU_kw + {ctrl_str} | Stkcd + year",
                           data=synch_df, vcov={"CRV1": "IndYear"})
# (6) 加入中介变量FinAsset
baseline['(6)'] = pf.feols(f"PriceDelay ~ DU_kw + FinAsset + {ctrl_str} | Stkcd + year",
                           data=reg_fe, vcov={"CRV1": "IndYear"})

rows_b = []
for name, m in baseline.items():
    kv_list = [v for v in m.coef().keys() if v.startswith('DU_')]
    kv = kv_list[0] if kv_list else 'DU_kw'
    r = extract_result(m, kv)
    r['Model'] = name
    r['Var'] = kv

    # 如果是(6)，也记录FinAsset系数
    if name == '(6)' and 'FinAsset' in m.coef():
        r['FinAsset_coef'] = m.coef()['FinAsset']
        r['FinAsset_se'] = m.se()['FinAsset']
        r['FinAsset_t'] = m.tstat()['FinAsset']
        r['FinAsset_p'] = m.pvalue()['FinAsset']

    rows_b.append(r)
    print(f"  {name} {r['Var']:<12}: coef={r['Coef']:.6f} (SE={r['SE']:.6f}) t={r['t']:.3f} p={r['p']:.4f} {r['Sig']} N={r['N']:,} R2={r['R2']:.4f}")

pd.DataFrame(rows_b).to_csv(f"{RES_DIR}/baseline_v5.csv", index=False)

# 打印完整模型(1)用于论文
print("\n--- Model (1) 完整结果 ---")
print(baseline['(1)'].summary())


# ================================================================
# 3. 表3：稳健性检验（9列，对标朱康表3）
# ================================================================
print("\n" + "=" * 70)
print("表3: 稳健性检验 (对标朱康表3, 9列)")
print("=" * 70)

robust = {}

# --- (1) 替换被解释变量: SYNCH ---
print("\n  (1) 替换DV: SYNCH")
robust['(1)_替换DV'] = pf.feols(f"SYNCH ~ DU_kw + {ctrl_str} | Stkcd + year",
                                data=synch_df, vcov={"CRV1": "IndYear"})

# --- (2) 替换解释变量: DU_kw_ln ---
print("  (2) 替换IV: DU_kw_ln")
robust['(2)_替换IV'] = pf.feols(f"PriceDelay ~ DU_kw_ln + {ctrl_str} | Stkcd + year",
                                data=reg_fe, vcov={"CRV1": "IndYear"})

# --- (3) 控制时变行业: Ind×Year FE ---
print("  (3) Ind×Year FE")
robust['(3)_IndYear'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + IndYear",
                                 data=reg_fe, vcov={"CRV1": "IndYear"})

# --- (4) 控制时变地区: Province×Year FE ---
print("  (4) Province×Year FE")
prov_df = reg_fe.dropna(subset=['ProvYear'])
robust['(4)_ProvYear'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + ProvYear",
                                  data=prov_df, vcov={"CRV1": "IndYear"})

# --- (5) 剔除2024年(政策冲击年) ---
print("  (5) 剔除2024年")
no2024 = reg_fe[reg_fe['year'] != '2024']
robust['(5)_No2024'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year",
                                data=no2024, vcov={"CRV1": "IndYear"})

# --- (6) 剔除信息技术业 ---
print("  (6) 剔除信息传输/软件/信息技术服务业(I类)")
no_it = reg_fe[~reg_fe['Ind2'].str.startswith('I', na=False)]
robust['(6)_NoIT'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year",
                              data=no_it, vcov={"CRV1": "IndYear"})

# --- (7) PSM匹配 ---
print("  (7) PSM匹配")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

psm_df = reg_fe.dropna(subset=controls + ['DU_kw']).copy()
psm_df['DU_treat'] = (psm_df['DU_kw'] >= psm_df['DU_kw'].median()).astype(int)
X_psm = psm_df[controls].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_psm)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, psm_df['DU_treat'].values)
psm_df['pscore'] = lr.predict_proba(X_scaled)[:, 1]

treat_idx = psm_df[psm_df['DU_treat'] == 1].index
ctrl_pool = psm_df[psm_df['DU_treat'] == 0]
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(ctrl_pool[['pscore']].values)
distances, indices = nn.kneighbors(psm_df.loc[treat_idx, ['pscore']].values)

# 加caliper: 0.25 × SD(pscore)，标准做法
caliper = 0.25 * psm_df['pscore'].std()
within_caliper = distances.flatten() <= caliper
treat_matched = treat_idx[within_caliper]
ctrl_matched = ctrl_pool.index[indices.flatten()[within_caliper]]
psm_matched = pd.concat([psm_df.loc[treat_matched], psm_df.loc[ctrl_matched]])
# 去重（同一control可能被匹配多次）
psm_matched = psm_matched.drop_duplicates()
robust['(7)_PSM'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year",
                             data=psm_matched, vcov={"CRV1": "IndYear"})
print(f"    Caliper: {caliper:.4f}")
print(f"    Within caliper: {within_caliper.sum():,} / {len(treat_idx):,} treated")
print(f"    PSM matched sample (after dedup): {len(psm_matched):,}")

# --- (8) 工具变量: 同行业均值IV ---
print("  (8) 同行业均值IV")
iv_df = reg_fe.copy()
iv_df['DU_kw_ind_mean'] = iv_df.groupby(['Ind2', 'year_str'])['DU_kw'].transform(
    lambda x: (x.sum() - x) / (len(x) - 1) if len(x) > 1 else np.nan
)
iv_df = iv_df.dropna(subset=['DU_kw_ind_mean'])
robust['(8)_IV'] = pf.feols(f"PriceDelay ~ {ctrl_str} | Stkcd + year | DU_kw ~ DU_kw_ind_mean",
                            data=iv_df, vcov={"CRV1": "IndYear"})

# --- (9) Heckman两阶段 ---
print("  (9) Heckman两阶段")
# 第一阶段: Probit DU_kw > median
heck_df = reg_fe.copy()
heck_df['DU_high'] = (heck_df['DU_kw'] >= heck_df['DU_kw'].median()).astype(int)

# Probit用statsmodels
import statsmodels.api as sm
from scipy.stats import norm

probit_vars = controls.copy()
X_probit = sm.add_constant(heck_df[probit_vars].astype(float))
probit_model = sm.Probit(heck_df['DU_high'].values, X_probit.values)
probit_result = probit_model.fit(disp=0)

# 计算逆米尔斯比率
xb = probit_result.predict(X_probit.values)
heck_df['IMR'] = norm.pdf(norm.ppf(xb)) / xb  # λ = φ(xβ)/Φ(xβ)
heck_df['IMR'] = heck_df['IMR'].replace([np.inf, -np.inf], np.nan)
heck_df = heck_df.dropna(subset=['IMR'])

robust['(9)_Heckman'] = pf.feols(f"PriceDelay ~ DU_kw + IMR + {ctrl_str} | Stkcd + year",
                                 data=heck_df, vcov={"CRV1": "IndYear"})

# 汇总结果
print("\n" + "-" * 90)
print(f"{'Column':<20} {'DV':<12} {'Var':<12} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8} {'Sig':<4} {'N':>8} {'R2':>8}")
print("-" * 90)

rows_r = []
for name, m in robust.items():
    dv = 'SYNCH' if '替换DV' in name else 'PriceDelay'
    kv_candidates = [v for v in m.coef().keys() if v.startswith('DU_')]
    kv = kv_candidates[0] if kv_candidates else 'DU_kw'
    r = extract_result(m, kv)
    r['Column'] = name
    r['DV'] = dv
    r['Var'] = kv
    rows_r.append(r)
    print(f"  {name:<20} {dv:<12} {kv:<12} {r['Coef']:>10.6f} {r['SE']:>10.6f} {r['t']:>8.3f} {r['p']:>8.4f} {r['Sig']:<4} {r['N']:>8} {r['R2']:>8.4f}")

pd.DataFrame(rows_r).to_csv(f"{RES_DIR}/robustness_v5.csv", index=False)

# IV诊断统计量
print("\n--- IV诊断 ---")
iv_m = robust['(8)_IV']
print(iv_m.summary())
# 提取诊断统计量
try:
    iv_diag = {}
    if hasattr(iv_m, '_iv_diag') and iv_m._iv_diag is not None:
        iv_diag = iv_m._iv_diag
    elif hasattr(iv_m, 'diagn') and iv_m.diagn is not None:
        iv_diag = iv_m.diagn
    # pyfixest 0.25+ stores diagnostics differently
    for attr in ['_f_kp', '_f_ar', '_p_ar', '_p_kp']:
        if hasattr(iv_m, attr):
            iv_diag[attr] = getattr(iv_m, attr)
    print(f"  IV diagnostics dict: {iv_diag}")
except Exception as e:
    print(f"  IV diagnostics extraction error: {e}")

# 也尝试直接从model attributes获取
print("  IV model attributes containing 'diag' or 'f_' or 'kp':")
for attr in dir(iv_m):
    if any(kw in attr.lower() for kw in ['diag', 'f_kp', 'kp', '_ar', 'wald', 'first']):
        try:
            val = getattr(iv_m, attr)
            if not callable(val):
                print(f"    {attr} = {val}")
        except:
            pass

# ================================================================
# 4. 保存全部结果为JSON
# ================================================================
all_results = {
    'baseline': rows_b,
    'robustness': rows_r,
}
with open(f"{RES_DIR}/all_results_v5.json", 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

print("\n" + "=" * 70)
print(f"全部完成，结果保存在 {RES_DIR}/")
print("=" * 70)
