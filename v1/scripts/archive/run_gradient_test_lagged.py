"""
梯度检验：用 t-1 期 Analyst 分组，避免内生性
逻辑：DU_kw(t) → Analyst(t) 是机制声称，所以分组不能用当期 Analyst
用 Analyst(t-1) 作为"预定的信息环境"来分组

回归：PriceDelay(t) ~ DU_kw(t) + controls(t) | Firm + Year FE
分组：Analyst(t-1) 中位数
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import warnings, os, json
from scipy import stats
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"
OUT_DIR = f"{BASE}/data_parquet"

# 复用 generate_tables_v10.py 的数据加载流程
panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")
panel = panel.merge(
    ar_feat[['Stkcd','year','kw_total','kw_per10k','substantive_count']],
    on=['Stkcd','year'], how='left'
)

fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC','LISTINGSTATE'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(subset=['Stkcd','year'], keep='last')

# 公司众数填补IndustryCodeC
ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
panel = panel.merge(ind_mode.rename('IndCode_mode').reset_index(), on='Stkcd', how='left')
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE']], on=['Stkcd','year'], how='left')
panel['IndustryCodeC'] = panel['IndustryCodeC'].fillna(panel['IndCode_mode'])

mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST','*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']

def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

cont_vars = ['PriceDelay','DU_kw','Lev','ROA','Growth','Size','TobinQ','Age',
             'BoardSize','IndepRatio','Top1Share','InstHold','Amihud','Analyst']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

# 构造 Analyst(t-1)：按公司分组，Analyst 向后移一期
panel = panel.sort_values(['Stkcd', 'year'])
panel['Analyst_lag1'] = panel.groupby('Stkcd')['Analyst'].shift(1)

controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize','IndepRatio',
            'Dual','Top1Share','SOE','InstHold','Amihud','Analyst','AuditType']

reg = panel.dropna(subset=['PriceDelay','DU_kw','Analyst_lag1'] + controls).copy()
reg['Stkcd_str'] = reg['Stkcd'].astype(str)
reg['year_str'] = reg['year'].astype(str)

print(f"回归样本（含Analyst_lag1）: N={len(reg):,}")

# 按 Analyst(t-1) 中位数分组
analyst_lag_median = reg['Analyst_lag1'].median()
print(f"Analyst(t-1) 中位数: {analyst_lag_median:.4f}")

high_analyst = reg[reg['Analyst_lag1'] >= analyst_lag_median]
low_analyst = reg[reg['Analyst_lag1'] < analyst_lag_median]
print(f"高覆盖组 (Analyst_lag1>=median): N={len(high_analyst):,}")
print(f"低覆盖组 (Analyst_lag1<median):  N={len(low_analyst):,}")

ctrl_str = " + ".join(controls)
base_fml = f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str"

# 高覆盖组
m_high = pf.feols(base_fml, data=high_analyst, vcov={"CRV1": "IndYear"})
c_high = float(m_high.coef()['DU_kw'])
se_high = float(m_high.se()['DU_kw'])
p_high = float(m_high.pvalue()['DU_kw'])
sig_high = '***' if p_high < 0.01 else '**' if p_high < 0.05 else '*' if p_high < 0.1 else ''

# 低覆盖组
m_low = pf.feols(base_fml, data=low_analyst, vcov={"CRV1": "IndYear"})
c_low = float(m_low.coef()['DU_kw'])
se_low = float(m_low.se()['DU_kw'])
p_low = float(m_low.pvalue()['DU_kw'])
sig_low = '***' if p_low < 0.01 else '**' if p_low < 0.05 else '*' if p_low < 0.1 else ''

# Fisher检验
z = (c_high - c_low) / np.sqrt(se_high**2 + se_low**2)
p_fisher = 2 * (1 - stats.norm.cdf(abs(z)))
sig_fisher = '***' if p_fisher < 0.01 else '**' if p_fisher < 0.05 else '*' if p_fisher < 0.1 else ''

print(f"\n{'='*60}")
print(f"梯度检验结果（Analyst(t-1) 分组）")
print(f"{'='*60}")
print(f"高覆盖组: coef={c_high:.5f}, se={se_high:.5f}, t={c_high/se_high:.3f}, p={p_high:.4f} {sig_high}")
print(f"  N={m_high._N:,}, R2={m_high._r2:.3f}")
print(f"低覆盖组: coef={c_low:.5f}, se={se_low:.5f}, t={c_low/se_low:.3f}, p={p_low:.4f} {sig_low}")
print(f"  N={m_low._N:,}, R2={m_low._r2:.3f}")
print(f"Fisher z={z:.3f}, p={p_fisher:.4f} {sig_fisher}")

# 保存结果
result = {
    "高覆盖_Analyst_lag1": {
        "coef": round(c_high, 5), "se": round(se_high, 5),
        "sig": sig_high, "N": int(m_high._N), "R2": round(m_high._r2, 3),
        "p": round(p_high, 4)
    },
    "低覆盖_Analyst_lag1": {
        "coef": round(c_low, 5), "se": round(se_low, 5),
        "sig": sig_low, "N": int(m_low._N), "R2": round(m_low._r2, 3),
        "p": round(p_low, 4)
    },
    "fisher": {"z": round(z, 3), "p": round(p_fisher, 4), "sig": sig_fisher},
    "grouping": "Analyst(t-1) median split",
    "note": "分组变量为t-1期分析师覆盖，避免DU_kw(t)→Analyst(t)的内生性"
}

os.makedirs(f"{BASE}/results/v11_expanded", exist_ok=True)
with open(f"{BASE}/results/v11_expanded/gradient_test_lagged.json", 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: results/v11_expanded/gradient_test_lagged.json")
