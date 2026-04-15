#!/usr/bin/env python3
"""
内生性处理全套检验
1. 同行业均值IV (Industry Peer Mean)
2. Bartik/Shift-Share IV
3. Lewbel (2012) 异方差IV
4. Heckman两阶段
5. 强化现有IV排他性（加入地区控制）
"""

import os, sys
import pandas as pd
import numpy as np
import pyfixest as pf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_parquet")
RES_DIR = os.path.join(BASE_DIR, "results/endogeneity")
os.makedirs(RES_DIR, exist_ok=True)

# ── 加载数据 ──
panel = pd.read_parquet(os.path.join(DATA_DIR, "panel.parquet"))
af = pd.read_parquet(os.path.join(DATA_DIR, "annual_report_features.parquet"))
fi = pd.read_parquet(os.path.join(DATA_DIR, "firm_info.parquet"))
ps = pd.read_parquet(os.path.join(DATA_DIR, "per_share.parquet"))

# 合并DU_kw
af_merge = af[['Stkcd','year','kw_per10k','substantive_count']].copy()
af_merge.columns = ['Stkcd','year','DU_kw','DU_sub']

df = panel.merge(af_merge, on=['Stkcd','year'], how='left')

# 获取行业代码（证监会C级 = 二级行业）
# fi有多行per firm，取最新的
fi_latest = fi.sort_values('EndDate').groupby('Symbol').last().reset_index()
fi_latest = fi_latest[['Symbol','IndustryCodeC','PROVINCE','CITY']].rename(
    columns={'Symbol':'Stkcd','IndustryCodeC':'Indcd'}
)

# 也从per_share获取（可能更panel-level）
ps_ind = ps[['Stkcd','Indcd']].drop_duplicates()

# 用fi_latest做主匹配
df = df.merge(fi_latest[['Stkcd','Indcd','PROVINCE','CITY']], on='Stkcd', how='left')

# 填补：如果fi里没有，从per_share取
mask = df['Indcd'].isna()
if mask.sum() > 0:
    ps_map = ps_ind.drop_duplicates('Stkcd').set_index('Stkcd')['Indcd']
    df.loc[mask, 'Indcd'] = df.loc[mask, 'Stkcd'].map(ps_map).values

# 剔除金融J类（已在panel筛选过，但double check）
df = df[~df['Indcd'].str.startswith('J', na=False) | (df['Indcd'] == 'J66')].copy()
# Actually, J类应该已经被剔除了，J66是金融...
# panel.parquet已经剔除过金融了，所以这里不再过滤

# 筛选有效样本
controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize','IndepRatio',
            'Dual','Top1Share','SOE','InstHold','Amihud','Analyst','AuditType']

df = df.dropna(subset=['PriceDelay','DU_kw'] + controls)

# Winsorize
def winsorize(s, lo=0.01, hi=0.99):
    q = s.quantile([lo, hi])
    return s.clip(q.iloc[0], q.iloc[1])

cont_vars = ['PriceDelay','DU_kw','DU_sub'] + controls
for v in cont_vars:
    if v in df.columns:
        df[v] = winsorize(df[v])

# 创建固定效应ID
df['Stkcd_str'] = df['Stkcd'].astype(str)
df['year_str'] = df['year'].astype(str)
if df['Indcd'].notna().sum() > 0:
    df['IndYear'] = df['Indcd'].astype(str) + '_' + df['year'].astype(str)

print(f"样本: N={len(df)}, Firms={df['Stkcd'].nunique()}")
print(f"行业: {df['Indcd'].nunique()} unique")
print(f"省份: {df['PROVINCE'].nunique()} unique")
print()

results = {}

# ============================================================
#  1. 同行业均值IV (Industry Peer Mean)
# ============================================================
print("="*60)
print("1. 同行业均值IV")
print("="*60)

# 构造：同年同行业排除自身的DU_kw均值
ind_year_sum = df.groupby(['Indcd','year'])['DU_kw'].agg(['sum','count'])
ind_year_sum.columns = ['iy_sum','iy_count']

df = df.merge(ind_year_sum, on=['Indcd','year'], how='left')
df['DU_kw_peer'] = (df['iy_sum'] - df['DU_kw']) / (df['iy_count'] - 1)

# 检查IV有效性
corr_peer = df[['DU_kw','DU_kw_peer']].corr().iloc[0,1]
print(f"  Peer IV与DU_kw相关系数: {corr_peer:.4f}")

# 一阶段
ctrl_str = ' + '.join(controls)
try:
    first_stage = pf.feols(f"DU_kw ~ DU_kw_peer + {ctrl_str} | Stkcd_str + year_str",
                           data=df, vcov={'CRV1': 'IndYear'})
    fs_coef = first_stage.coef()['DU_kw_peer']
    fs_se = first_stage.se()['DU_kw_peer']
    fs_t = fs_coef / fs_se
    print(f"  一阶段: coef={fs_coef:.4f}, t={fs_t:.2f}")

    # 2SLS
    second_stage = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw ~ DU_kw_peer",
                            data=df, vcov={'CRV1': 'IndYear'})
    iv_coef = second_stage.coef()['DU_kw']
    iv_se = second_stage.se()['DU_kw']
    iv_t = iv_coef / iv_se
    iv_p = second_stage.pvalue()['DU_kw']
    print(f"  2SLS: DU_kw coef={iv_coef:.6f}, t={iv_t:.2f}, p={iv_p:.4f}")
    results['peer_iv'] = {'coef': iv_coef, 'se': iv_se, 't': iv_t, 'p': iv_p}
except Exception as e:
    print(f"  Error: {e}")

print()

# ============================================================
#  2. Bartik/Shift-Share IV
# ============================================================
print("="*60)
print("2. Bartik/Shift-Share IV")
print("="*60)

# 构造: 企业所在行业初期(2011)DU_kw水平 × 全国行业年度DU_kw均值增长率
# Step1: 2011年行业DU_kw均值（作为初始份额/暴露度）
base_year = 2011
ind_base = df[df['year']==base_year].groupby('Indcd')['DU_kw'].mean().reset_index()
ind_base.columns = ['Indcd','DU_kw_base']

# Step2: 全国行业年度DU_kw均值
ind_year_mean = df.groupby(['Indcd','year'])['DU_kw'].mean().reset_index()
ind_year_mean.columns = ['Indcd','year','DU_kw_ind_mean']

# Step3: 行业年度增长率 (相对于基期)
ind_year_mean = ind_year_mean.merge(
    ind_year_mean[ind_year_mean['year']==base_year][['Indcd','DU_kw_ind_mean']].rename(
        columns={'DU_kw_ind_mean':'DU_kw_ind_base'}
    ), on='Indcd', how='left'
)
ind_year_mean['ind_growth'] = ind_year_mean['DU_kw_ind_mean'] / ind_year_mean['DU_kw_ind_base'].clip(lower=0.001)

# Step4: Bartik IV = base exposure × growth
df = df.merge(ind_base, on='Indcd', how='left')
df = df.merge(ind_year_mean[['Indcd','year','ind_growth']], on=['Indcd','year'], how='left')
df['bartik_iv'] = df['DU_kw_base'] * df['ind_growth']

# 处理NaN
df['bartik_iv'] = df['bartik_iv'].fillna(0)

corr_bartik = df[['DU_kw','bartik_iv']].corr().iloc[0,1]
print(f"  Bartik IV与DU_kw相关系数: {corr_bartik:.4f}")

try:
    # 一阶段
    first_bartik = pf.feols(f"DU_kw ~ bartik_iv + {ctrl_str} | Stkcd_str + year_str",
                            data=df, vcov={'CRV1': 'IndYear'})
    fb_coef = first_bartik.coef()['bartik_iv']
    fb_t = fb_coef / first_bartik.se()['bartik_iv']
    print(f"  一阶段: coef={fb_coef:.4f}, t={fb_t:.2f}")

    # 2SLS
    second_bartik = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw ~ bartik_iv",
                             data=df, vcov={'CRV1': 'IndYear'})
    bk_coef = second_bartik.coef()['DU_kw']
    bk_se = second_bartik.se()['DU_kw']
    bk_t = bk_coef / bk_se
    bk_p = second_bartik.pvalue()['DU_kw']
    print(f"  2SLS: DU_kw coef={bk_coef:.6f}, t={bk_t:.2f}, p={bk_p:.4f}")
    results['bartik_iv'] = {'coef': bk_coef, 'se': bk_se, 't': bk_t, 'p': bk_p}
except Exception as e:
    print(f"  Error: {e}")

print()

# ============================================================
#  3. Lewbel (2012) 异方差IV
# ============================================================
print("="*60)
print("3. Lewbel (2012) 异方差IV")
print("="*60)

# Lewbel方法: 用(Z - mean(Z)) * residual 作为内部IV
# Step1: 回归DU_kw on controls，得到残差
# Step2: 用残差与中心化的外生变量的乘积作为IV

# 先去固定效应
df['PD_dm'] = df.groupby('Stkcd_str')['PriceDelay'].transform(lambda x: x - x.mean())
df['DU_dm'] = df.groupby('Stkcd_str')['DU_kw'].transform(lambda x: x - x.mean())

for v in controls:
    df[f'{v}_dm'] = df.groupby('Stkcd_str')[v].transform(lambda x: x - x.mean())

# Year demeaning
df['PD_dm'] = df.groupby('year')['PD_dm'].transform(lambda x: x - x.mean())
df['DU_dm'] = df.groupby('year')['DU_dm'].transform(lambda x: x - x.mean())
for v in controls:
    df[f'{v}_dm'] = df.groupby('year')[f'{v}_dm'].transform(lambda x: x - x.mean())

# 第一阶段: DU_kw on controls (demeaned)
from sklearn.linear_model import LinearRegression

ctrl_dm = [f'{v}_dm' for v in controls]
X_ctrl = df[ctrl_dm].values
y_du = df['DU_dm'].values

reg_first = LinearRegression().fit(X_ctrl, y_du)
resid_du = y_du - reg_first.predict(X_ctrl)

# 构造Lewbel IV: 选3个外生变量中心化后乘残差
lewbel_vars = ['Size','Age','Analyst']  # 选相对"最外生"的
for lv in lewbel_vars:
    z_centered = df[f'{lv}_dm'].values - df[f'{lv}_dm'].mean()
    df[f'lewbel_{lv}'] = z_centered * resid_du

# 用ivreg2-style: 2SLS with Lewbel instruments
# 简化实现：手动一阶段+二阶段
lewbel_cols = [f'lewbel_{lv}' for lv in lewbel_vars]
X_first_lewbel = df[ctrl_dm + lewbel_cols].values
reg_lewbel_first = LinearRegression().fit(X_first_lewbel, y_du)
du_hat_lewbel = reg_lewbel_first.predict(X_first_lewbel)

# F-stat for excluded instruments
from sklearn.metrics import r2_score
r2_restricted = r2_score(y_du, reg_first.predict(X_ctrl))
r2_unrestricted = r2_score(y_du, reg_lewbel_first.predict(X_first_lewbel))
n = len(y_du)
k_excl = len(lewbel_cols)
k_full = X_first_lewbel.shape[1]
f_lewbel = ((r2_unrestricted - r2_restricted) / k_excl) / ((1 - r2_unrestricted) / (n - k_full))
print(f"  Lewbel一阶段F统计量: {f_lewbel:.2f}")

# 二阶段
y_pd = df['PD_dm'].values
X_second = np.column_stack([du_hat_lewbel] + [df[c].values for c in ctrl_dm])
reg_second_lewbel = LinearRegression().fit(X_second, y_pd)
coef_lewbel = reg_second_lewbel.coef_[0]

# 手算SE (二阶段异方差稳健SE)
resid_second = y_pd - reg_second_lewbel.predict(X_second)
# 用原始DU_dm而非fitted做residual
X_original = np.column_stack([df['DU_dm'].values] + [df[c].values for c in ctrl_dm])
resid_true = y_pd - LinearRegression().fit(X_original, y_pd).predict(X_original)

# Robust SE for 2SLS (memory-efficient, no diag matrix)
X2 = X_second
bread = np.linalg.inv(X2.T @ X2)
# meat = X2.T @ diag(e^2) @ X2 = sum_i (e_i^2 * x_i x_i')
X2_weighted = X2 * resid_second[:, np.newaxis]  # element-wise multiply rows by residuals
meat = X2_weighted.T @ X2_weighted
vcov_robust = bread @ meat @ bread
se_lewbel = np.sqrt(vcov_robust[0, 0])
t_lewbel = coef_lewbel / se_lewbel

print(f"  2SLS: DU_kw coef={coef_lewbel:.6f}, se={se_lewbel:.6f}, t={t_lewbel:.2f}")
results['lewbel_iv'] = {'coef': coef_lewbel, 'se': se_lewbel, 't': t_lewbel, 'p': 2*(1-stats.norm.cdf(abs(t_lewbel)))}

print()

# ============================================================
#  4. Heckman两阶段
# ============================================================
print("="*60)
print("4. Heckman两阶段")
print("="*60)

from statsmodels.discrete.discrete_model import Probit as smProbit

# 第一阶段: Probit预测"高数据利用"概率
df['high_du'] = (df['DU_kw'] >= df.groupby('year')['DU_kw'].transform('median')).astype(int)

# 选择排他性变量（影响是否高数据利用但不直接影响PriceDelay）
# 用行业DU_kw均值和企业Age作为exclusion variables
df['ind_du_mean'] = df.groupby(['Indcd','year'])['DU_kw'].transform('mean')

probit_vars = controls + ['ind_du_mean']
X_probit = df[probit_vars].values
y_probit = df['high_du'].values

try:
    import statsmodels.api as sm
    X_p = sm.add_constant(X_probit)
    probit_model = sm.Probit(y_probit, X_p).fit(disp=0)

    # 计算逆米尔斯比率
    xb = probit_model.predict(X_p, linear=True)
    pdf_val = stats.norm.pdf(xb)
    cdf_val = stats.norm.cdf(xb)

    # IMR = phi(xb)/Phi(xb) for high_du=1, -phi(xb)/(1-Phi(xb)) for high_du=0
    df['IMR'] = np.where(
        df['high_du'] == 1,
        pdf_val / np.clip(cdf_val, 1e-8, None),
        -pdf_val / np.clip(1 - cdf_val, 1e-8, None)
    )

    # 第二阶段: 加入IMR的OLS
    heckman_reg = pf.feols(f"PriceDelay ~ DU_kw + IMR + {ctrl_str} | Stkcd_str + year_str",
                           data=df, vcov={'CRV1': 'IndYear'})

    hk_du_coef = heckman_reg.coef()['DU_kw']
    hk_du_se = heckman_reg.se()['DU_kw']
    hk_du_t = hk_du_coef / hk_du_se
    hk_du_p = heckman_reg.pvalue()['DU_kw']
    hk_imr_coef = heckman_reg.coef()['IMR']
    hk_imr_t = hk_imr_coef / heckman_reg.se()['IMR']

    print(f"  DU_kw: coef={hk_du_coef:.6f}, t={hk_du_t:.2f}, p={hk_du_p:.4f}")
    print(f"  IMR:   coef={hk_imr_coef:.6f}, t={hk_imr_t:.2f}")
    print(f"  IMR显著{'√ 存在选择偏差' if abs(hk_imr_t) > 1.96 else '× 无显著选择偏差'}")
    results['heckman'] = {'coef': hk_du_coef, 'se': hk_du_se, 't': hk_du_t, 'p': hk_du_p,
                          'imr_coef': hk_imr_coef, 'imr_t': hk_imr_t}
except Exception as e:
    print(f"  Error: {e}")

print()

# ============================================================
#  5. OLS基准对照 (for comparison)
# ============================================================
print("="*60)
print("5. OLS基准对照")
print("="*60)

ols_reg = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
                   data=df, vcov={'CRV1': 'IndYear'})
ols_coef = ols_reg.coef()['DU_kw']
ols_se = ols_reg.se()['DU_kw']
ols_t = ols_coef / ols_se
ols_p = ols_reg.pvalue()['DU_kw']
print(f"  OLS: DU_kw coef={ols_coef:.6f}, t={ols_t:.2f}, p={ols_p:.4f}")
results['ols'] = {'coef': ols_coef, 'se': ols_se, 't': ols_t, 'p': ols_p}

print()

# ============================================================
#  汇总
# ============================================================
print("="*60)
print("汇总比较")
print("="*60)

summary_data = []
method_names = {
    'ols': 'OLS基准',
    'peer_iv': '同行业均值IV',
    'bartik_iv': 'Bartik IV',
    'lewbel_iv': 'Lewbel异方差IV',
    'heckman': 'Heckman二阶段',
}

for key, name in method_names.items():
    if key in results:
        r = results[key]
        sig = '***' if r['p'] < 0.01 else '**' if r['p'] < 0.05 else '*' if r['p'] < 0.1 else ''
        summary_data.append({
            '方法': name,
            '系数': f"{r['coef']:.6f}",
            '标准误': f"{r['se']:.6f}",
            't值': f"{r['t']:.2f}",
            'p值': f"{r['p']:.4f}",
            '显著性': sig
        })
        print(f"  {name:20s}: coef={r['coef']:.6f}, t={r['t']:.2f} {sig}")

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(RES_DIR, "endogeneity_summary.csv"), index=False, encoding='utf-8-sig')

# 也保存完整results dict
import json
with open(os.path.join(RES_DIR, "endogeneity_results.json"), 'w') as f:
    json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                   for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

print(f"\n结果已保存至 {RES_DIR}/")
