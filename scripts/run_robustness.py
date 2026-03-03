"""
稳健性检验汇总脚本

1. 替换因变量: SYNCH (股价同步性)
2. PSM匹配: 倾向得分匹配后重跑基准回归
3. 安慰剂检验: 虚假政策年DID
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import warnings, os
from scipy.special import expit  # logistic function
warnings.filterwarnings('ignore')

OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
RES_DIR = "/Users/mac/computerscience/15会计研究/results"
os.makedirs(RES_DIR, exist_ok=True)

# ================================================================
# 0. 加载基础数据
# ================================================================
print("=" * 60)
print("稳健性检验")
print("=" * 60)

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")
synch = pd.read_parquet(f"{OUT_DIR}/price_synchronicity.parquet")

# 合并年报特征
panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'substantive_count']],
    on=['Stkcd', 'year'], how='left'
)

# 合并同步性
panel = panel.merge(synch[['Stkcd', 'year', 'SYNCH', 'R2_synch']],
                    on=['Stkcd', 'year'], how='left')

# 合并行业
fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
                    on=['Stkcd', 'year'], how='left')

# 样本筛选
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# 构造变量
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])

# 时点对齐: 用t-1年年报指标解释t年结果变量
panel = panel.sort_values(['Stkcd', 'year'])
panel['DU_kw'] = panel.groupby('Stkcd')['DU_kw'].shift(1)
panel['DU_kw_ln'] = panel.groupby('Stkcd')['DU_kw_ln'].shift(1)

# Winsorize
def winsorize(s, lower=0.01, upper=0.99):
    q_lo, q_hi = s.quantile([lower, upper])
    return s.clip(q_lo, q_hi)

cont_vars = ['PriceDelay', 'SYNCH', 'DU_kw', 'DU_kw_ln',
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

print(f"  面板: {len(panel):,} obs, {panel.Stkcd.nunique():,} firms")
print(f"  PriceDelay非空: {panel.PriceDelay.notna().sum():,}")
print(f"  SYNCH非空: {panel.SYNCH.notna().sum():,}")

all_results = []

# ================================================================
# 1. 替换因变量: SYNCH
# ================================================================
print("\n" + "=" * 60)
print("1. 替换因变量: SYNCH (股价同步性)")
print("=" * 60)

reg_synch = panel.dropna(subset=['SYNCH', 'DU_kw'] + controls).copy()
reg_synch['Stkcd'] = reg_synch['Stkcd'].astype(str)
reg_synch['year'] = reg_synch['year'].astype(str)
print(f"  回归样本: {len(reg_synch):,} obs, {reg_synch.Stkcd.nunique():,} firms")

# M1: Firm+Year FE
m1 = pf.feols(f"SYNCH ~ DU_kw + {ctrl_str} | Stkcd + year",
              data=reg_synch, vcov={"CRV1": "IndYear"})
c1, t1, p1 = m1.coef()['DU_kw'], m1.tstat()['DU_kw'], m1.pvalue()['DU_kw']
sig1 = '***' if p1 < 0.01 else '**' if p1 < 0.05 else '*' if p1 < 0.1 else ''
print(f"  Firm+Year FE: DU_kw coef={c1:.6f}, t={t1:.3f}, p={p1:.4f} {sig1}")
all_results.append({'Test': 'Alt_DV_SYNCH', 'FE': 'Firm+Year', 'Var': 'DU_kw',
                    'Coef': c1, 't': t1, 'p': p1, 'N': m1._N})

# M2: ln指标
m2 = pf.feols(f"SYNCH ~ DU_kw_ln + {ctrl_str} | Stkcd + year",
              data=reg_synch, vcov={"CRV1": "IndYear"})
c2, t2, p2 = m2.coef()['DU_kw_ln'], m2.tstat()['DU_kw_ln'], m2.pvalue()['DU_kw_ln']
sig2 = '***' if p2 < 0.01 else '**' if p2 < 0.05 else '*' if p2 < 0.1 else ''
print(f"  Firm+Year FE: DU_kw_ln coef={c2:.6f}, t={t2:.3f}, p={p2:.4f} {sig2}")
all_results.append({'Test': 'Alt_DV_SYNCH_ln', 'FE': 'Firm+Year', 'Var': 'DU_kw_ln',
                    'Coef': c2, 't': t2, 'p': p2, 'N': m2._N})

# M3: Industry+Year FE
m3 = pf.feols(f"SYNCH ~ DU_kw + {ctrl_str} | Ind2 + year",
              data=reg_synch, vcov={"CRV1": "IndYear"})
c3, t3, p3 = m3.coef()['DU_kw'], m3.tstat()['DU_kw'], m3.pvalue()['DU_kw']
sig3 = '***' if p3 < 0.01 else '**' if p3 < 0.05 else '*' if p3 < 0.1 else ''
print(f"  Ind+Year FE:  DU_kw coef={c3:.6f}, t={t3:.3f}, p={p3:.4f} {sig3}")
all_results.append({'Test': 'Alt_DV_SYNCH', 'FE': 'Ind+Year', 'Var': 'DU_kw',
                    'Coef': c3, 't': t3, 'p': p3, 'N': m3._N})

print("\n--- SYNCH Full Model (Firm+Year FE) ---")
print(m1.summary())

# ================================================================
# 2. PSM匹配
# ================================================================
print("\n" + "=" * 60)
print("2. PSM匹配稳健性")
print("=" * 60)

# 处理组: DU_kw高于中位数
reg_psm = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg_psm['year_int'] = reg_psm['year']
du_median = reg_psm['DU_kw'].median()
reg_psm['Treated'] = (reg_psm['DU_kw'] >= du_median).astype(int)
print(f"  处理组阈值: DU_kw >= {du_median:.4f}")
print(f"  处理组: {reg_psm.Treated.sum():,}, 对照组: {(1-reg_psm.Treated).sum():,}")

# Logit倾向得分
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

psm_vars = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth',
            'BoardSize', 'IndepRatio', 'Dual', 'Top1Share',
            'SOE', 'Amihud', 'Analyst', 'AuditType']
psm_data = reg_psm.dropna(subset=psm_vars).copy()

scaler = StandardScaler()
X_psm = scaler.fit_transform(psm_data[psm_vars])
y_psm = psm_data['Treated'].values

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_psm, y_psm)
psm_data['pscore'] = lr.predict_proba(X_psm)[:, 1]

# 最近邻匹配 (1:1, 无放回, caliper=0.05)
print("  进行1:1最近邻匹配 (caliper=0.05)...")
treat_idx = psm_data[psm_data.Treated == 1].index.tolist()
ctrl_pool = psm_data[psm_data.Treated == 0].copy()
matched_pairs = []
used_ctrl = set()

for tidx in treat_idx:
    t_ps = psm_data.loc[tidx, 'pscore']
    # 在对照组中找最近的
    available = ctrl_pool[~ctrl_pool.index.isin(used_ctrl)]
    if len(available) == 0:
        break
    dists = (available['pscore'] - t_ps).abs()
    best_idx = dists.idxmin()
    best_dist = dists[best_idx]
    if best_dist <= 0.05:
        matched_pairs.append((tidx, best_idx))
        used_ctrl.add(best_idx)

matched_idx = set()
for t, c in matched_pairs:
    matched_idx.add(t)
    matched_idx.add(c)

psm_matched = psm_data.loc[list(matched_idx)].copy()
print(f"  匹配成功: {len(matched_pairs):,} 对, 总样本 {len(psm_matched):,}")

# 平衡性检验
print("\n  匹配后平衡性 (标准化偏差):")
for v in psm_vars:
    treat_mean = psm_matched[psm_matched.Treated == 1][v].mean()
    ctrl_mean = psm_matched[psm_matched.Treated == 0][v].mean()
    pooled_std = psm_matched[v].std()
    if pooled_std > 0:
        smd = abs(treat_mean - ctrl_mean) / pooled_std * 100
    else:
        smd = 0
    ok = '✓' if smd < 10 else '✗'
    print(f"    {v:15s}: SMD={smd:.2f}% {ok}")

# PSM样本回归
psm_reg = psm_matched.copy()
psm_reg['Stkcd'] = psm_reg['Stkcd'].astype(str)
psm_reg['year'] = psm_reg['year'].astype(str)

m_psm = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year",
                 data=psm_reg, vcov={"CRV1": "IndYear"})
c_psm = m_psm.coef()['DU_kw']
t_psm = m_psm.tstat()['DU_kw']
p_psm = m_psm.pvalue()['DU_kw']
sig_psm = '***' if p_psm < 0.01 else '**' if p_psm < 0.05 else '*' if p_psm < 0.1 else ''
print(f"\n  PSM回归: DU_kw coef={c_psm:.6f}, t={t_psm:.3f}, p={p_psm:.4f} {sig_psm} (N={m_psm._N})")
all_results.append({'Test': 'PSM', 'FE': 'Firm+Year', 'Var': 'DU_kw',
                    'Coef': c_psm, 't': t_psm, 'p': p_psm, 'N': m_psm._N})

print(m_psm.summary())

# ================================================================
# 3. 安慰剂检验: 虚假政策年
# ================================================================
print("\n" + "=" * 60)
print("3. 安慰剂检验 (虚假政策年)")
print("=" * 60)

reg_did = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg_did['year_int'] = reg_did['year']

# 用2019-2023均值定义处理组
pre = reg_did[(reg_did.year_int >= 2019) & (reg_did.year_int <= 2023)]
firm_kw = pre.groupby('Stkcd')['DU_kw'].mean()
treat_th = firm_kw.median()
treat_firms = set(firm_kw[firm_kw >= treat_th].index)

# 虚假政策年: 2020, 2021, 2022 (真实政策年: 2024)
print("  处理组定义同基准 (2019-2023均值 >= 中位数)")
print(f"  处理组: {len(treat_firms):,} firms")

for fake_year in [2020, 2021, 2022]:
    did_df = reg_did.copy()
    did_df['Treat'] = did_df['Stkcd'].isin(treat_firms).astype(int)
    did_df['Post'] = (did_df['year_int'] >= fake_year).astype(int)
    did_df['TreatPost'] = did_df['Treat'] * did_df['Post']
    did_df['Stkcd'] = did_df['Stkcd'].astype(str)
    did_df['year'] = did_df['year'].astype(str)

    # 窗口: fake_year前3年 ~ fake_year
    window_start = fake_year - 3
    did_sub = did_df[(did_df.year_int >= window_start) & (did_df.year_int <= fake_year)]

    try:
        m_fake = pf.feols(f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year",
                          data=did_sub, vcov={"CRV1": "IndYear"})
        c_f = m_fake.coef()['TreatPost']
        t_f = m_fake.tstat()['TreatPost']
        p_f = m_fake.pvalue()['TreatPost']
        sig_f = '***' if p_f < 0.01 else '**' if p_f < 0.05 else '*' if p_f < 0.1 else ''
        print(f"  虚假政策年={fake_year}: TreatPost coef={c_f:.6f}, t={t_f:.3f}, p={p_f:.4f} {sig_f} (N={m_fake._N})")
        all_results.append({'Test': f'Placebo_{fake_year}', 'FE': 'Firm+Year',
                            'Var': 'TreatPost', 'Coef': c_f, 't': t_f, 'p': p_f, 'N': m_fake._N})
    except Exception as e:
        print(f"  虚假政策年={fake_year}: ERROR {e}")

# 真实政策年 (对比)
did_real = reg_did.copy()
did_real['Treat'] = did_real['Stkcd'].isin(treat_firms).astype(int)
did_real['Post'] = (did_real['year_int'] >= 2024).astype(int)
did_real['TreatPost'] = did_real['Treat'] * did_real['Post']
did_real['Stkcd'] = did_real['Stkcd'].astype(str)
did_real['year'] = did_real['year'].astype(str)
did_real_sub = did_real[(did_real.year_int >= 2021) & (did_real.year_int <= 2024)]
m_real = pf.feols(f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year",
                  data=did_real_sub, vcov={"CRV1": "IndYear"})
c_r = m_real.coef()['TreatPost']
t_r = m_real.tstat()['TreatPost']
p_r = m_real.pvalue()['TreatPost']
sig_r = '***' if p_r < 0.01 else '**' if p_r < 0.05 else '*' if p_r < 0.1 else ''
print(f"  真实政策年=2024: TreatPost coef={c_r:.6f}, t={t_r:.3f}, p={p_r:.4f} {sig_r} (N={m_real._N})")
all_results.append({'Test': 'DID_Real_2024', 'FE': 'Firm+Year',
                    'Var': 'TreatPost', 'Coef': c_r, 't': t_r, 'p': p_r, 'N': m_real._N})

# ================================================================
# 4. 替换自变量: DU_kw_ln (对数变换)
# ================================================================
print("\n" + "=" * 60)
print("4. 替换自变量: ln(1+关键词总数)")
print("=" * 60)

reg_ln = panel.dropna(subset=['PriceDelay', 'DU_kw_ln'] + controls).copy()
reg_ln['Stkcd'] = reg_ln['Stkcd'].astype(str)
reg_ln['year'] = reg_ln['year'].astype(str)

m_ln = pf.feols(f"PriceDelay ~ DU_kw_ln + {ctrl_str} | Stkcd + year",
                data=reg_ln, vcov={"CRV1": "IndYear"})
c_ln = m_ln.coef()['DU_kw_ln']
t_ln = m_ln.tstat()['DU_kw_ln']
p_ln = m_ln.pvalue()['DU_kw_ln']
sig_ln = '***' if p_ln < 0.01 else '**' if p_ln < 0.05 else '*' if p_ln < 0.1 else ''
print(f"  Firm+Year FE: DU_kw_ln coef={c_ln:.6f}, t={t_ln:.3f}, p={p_ln:.4f} {sig_ln} (N={m_ln._N})")
all_results.append({'Test': 'Alt_IV_ln', 'FE': 'Firm+Year', 'Var': 'DU_kw_ln',
                    'Coef': c_ln, 't': t_ln, 'p': p_ln, 'N': m_ln._N})

# ================================================================
# 5. 子样本: 剔除2024 (排除政策冲击)
# ================================================================
print("\n" + "=" * 60)
print("5. 子样本: 剔除2024年")
print("=" * 60)

reg_no24 = panel[panel['year'] != 2024].dropna(
    subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg_no24['Stkcd'] = reg_no24['Stkcd'].astype(str)
reg_no24['year'] = reg_no24['year'].astype(str)

m_no24 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year",
                  data=reg_no24, vcov={"CRV1": "IndYear"})
c_no24 = m_no24.coef()['DU_kw']
t_no24 = m_no24.tstat()['DU_kw']
p_no24 = m_no24.pvalue()['DU_kw']
sig_no24 = '***' if p_no24 < 0.01 else '**' if p_no24 < 0.05 else '*' if p_no24 < 0.1 else ''
print(f"  剔除2024: DU_kw coef={c_no24:.6f}, t={t_no24:.3f}, p={p_no24:.4f} {sig_no24} (N={m_no24._N})")
all_results.append({'Test': 'Excl_2024', 'FE': 'Firm+Year', 'Var': 'DU_kw',
                    'Coef': c_no24, 't': t_no24, 'p': p_no24, 'N': m_no24._N})

# ================================================================
# 6. 排除创业板 / 科创板
# ================================================================
print("\n" + "=" * 60)
print("6. 子样本: 仅主板")
print("=" * 60)

reg_main = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
# 主板: 排除创业板(300/301)与科创板(688)
stk = reg_main['Stkcd'].astype(int).astype(str).str.zfill(6)
is_chinext = stk.str.startswith(('300', '301'))
is_star = stk.str.startswith('688')
reg_main = reg_main[~(is_chinext | is_star)]
reg_main['Stkcd'] = reg_main['Stkcd'].astype(str)
reg_main['year'] = reg_main['year'].astype(str)

m_main = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year",
                  data=reg_main, vcov={"CRV1": "IndYear"})
c_main = m_main.coef()['DU_kw']
t_main = m_main.tstat()['DU_kw']
p_main = m_main.pvalue()['DU_kw']
sig_main = '***' if p_main < 0.01 else '**' if p_main < 0.05 else '*' if p_main < 0.1 else ''
print(f"  仅主板: DU_kw coef={c_main:.6f}, t={t_main:.3f}, p={p_main:.4f} {sig_main} (N={m_main._N})")
all_results.append({'Test': 'MainBoard', 'FE': 'Firm+Year', 'Var': 'DU_kw',
                    'Coef': c_main, 't': t_main, 'p': p_main, 'N': m_main._N})

# ================================================================
# 汇总保存
# ================================================================
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)

res_df = pd.DataFrame(all_results)
print(res_df.to_string(index=False))
res_df.to_csv(f"{RES_DIR}/robustness_summary.csv", index=False)
print(f"\n已保存: {RES_DIR}/robustness_summary.csv")
print("完成！")
