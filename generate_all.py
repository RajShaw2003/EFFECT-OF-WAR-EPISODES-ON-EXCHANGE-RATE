"""
=============================================================================
  generate_all.py
  Complete Python script — generates every figure and table in the paper
  "War Shocks, Exchange Rate Dynamics, and Trade Structure"

  USAGE:
      python generate_all.py

  REQUIREMENTS:
      pip install pandas numpy matplotlib scipy statsmodels arch openpyxl

  INPUT:
      INR_PROJECT_1.xls  (your original dataset, Sheet3)
      All GDP/Trade xlsx files in the same folder (optional — for ITS section)

  OUTPUT (all saved to ./output/):
      Figures  : fig1_overview.png  fig2_irf.png  fig3_garch.png
                 fig4_fevd.png  fig5_ardl_fit.png  fig6_ardl_coefs.png
                 fig7_ardl_lr.png  fig8_mediation.png
                 fig9_comp_fit.png  fig10_comp_wardummies.png
                 fig11_comp_longrun.png  fig12_comp_heatmap.png
                 fig13_comp_structure.png
                 fig14_its_gdp.png  fig15_its_exports.png
                 fig16_its_imports.png  fig17_its_heatmap.png
      Tables   : tables.xlsx  (all paper tables in separate sheets)
=============================================================================
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from scipy.linalg import cholesky
from arch import arch_model
warnings.filterwarnings('ignore')

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs('output', exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = dict(BLUE='#2563EB', RED='#DC2626', GREEN='#16A34A', AMBER='#D97706',
         GRAY='#6B7280', DARK='#1E293B', PURPLE='#7C3AED', TEAL='#0D9488')
GROUP_COLOR = {'Developed': '#1D4ED8', 'Developing': '#DC2626'}
COUNTRY_COLOR = {'India':'#DC2626','Euro Area':'#1D4ED8','UK':'#7C3AED',
                 'Japan':'#0D9488','China':'#EA580C','Brazil':'#DB2777',
                 'Russia':'#92400E','Ukraine':'#065F46'}

print("=" * 60)
print("  PAPER FIGURE & TABLE GENERATOR")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[0] Loading and preparing data...")

import os
# Try current directory first, then uploads folder
_xls_path = 'INR_PROJECT_1.xls'
if not os.path.exists(_xls_path):
    _xls_path = '/mnt/user-data/uploads/INR_PROJECT_1.xls'
df = pd.read_excel(_xls_path, sheet_name='Sheet3', engine='xlrd')
df['MONTH'] = pd.to_datetime(df['MONTH'])
df = df.sort_values('MONTH').reset_index(drop=True)

# Log transforms
for col in ['INR_ER','EURO_ER','UK_ER','JAPAN_ER','CHINA_ER','BRAZIL_ER',
            'RUSSINA_ER','UKRAINE_ER','GRP','CRUDE_OIL','GOLD_PRICES']:
    df[f'ln_{col}'] = np.log(df[col].replace(0, np.nan))
df['ln_VIX']   = np.log(df['Monthly Avg VIX'])
df['ln_GOLD']  = df['ln_GOLD_PRICES']
df['ln_CRUDE'] = df['ln_CRUDE_OIL']
df['t']        = np.arange(1, len(df)+1)
df['INR_ER_logreturn'] = df['ln_INR_ER'].diff() * 100

# Disaggregated war dummies
m = df['MONTH']
df['War_Russia_Chechnya1']  = ((m>='1994-12-01')&(m<='1996-08-01')).astype(int)
df['War_Russia_Chechnya2']  = ((m>='1999-08-01')&(m<='2009-04-01')).astype(int)
df['War_Russia_Crimea']     = ((m>='2014-02-01')&(m<='2022-01-01')).astype(int)
df['War_Russia_Ukraine']    = ((m>='2022-02-01')).astype(int)
df['War_Ukraine_Donbas']    = ((m>='2014-04-01')&(m<='2022-01-01')).astype(int)
df['War_Ukraine_Fullscale'] = ((m>='2022-02-01')).astype(int)
df['War_UK_Gulf']           = ((m>='1990-08-01')&(m<='1991-02-01')).astype(int)
df['War_UK_Afghan']         = ((m>='2001-10-01')&(m<='2014-12-01')).astype(int)

WAR_D = ['War_Russia_Chechnya1','War_Russia_Chechnya2','War_Russia_Crimea',
         'War_Russia_Ukraine','War_Ukraine_Donbas','War_UK_Gulf','War_UK_Afghan']

print(f"  Data loaded: {len(df)} observations, {df['MONTH'].min().date()} to {df['MONTH'].max().date()}")

# ── Helper: war shading ───────────────────────────────────────────────────────
def shade_wars(ax, df_plot):
    for col, color, alpha in [
        ('War_Russia_Chechnya1','#FCA5A5',0.30),
        ('War_Russia_Chechnya2','#F87171',0.18),
        ('War_Russia_Crimea',   '#FEE2E2',0.30),
        ('War_Russia_Ukraine',  '#DC2626',0.13)]:
        if col not in df_plot.columns: continue
        mask = df_plot[col].values == 1
        starts = df_plot['MONTH'][pd.Series(mask) & ~pd.Series(mask).shift(1,fill_value=False)]
        ends   = df_plot['MONTH'][pd.Series(mask) & ~pd.Series(mask).shift(-1,fill_value=False)]
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=alpha, color=color, lw=0)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — UNIT ROOT TESTS  (Table 1 in paper)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Running unit root tests (Table 1)...")

ur_series = {
    'ln_INR_ER':'ln(INR/USD)', 'ln_GOLD':'ln(Gold)', 'ln_CRUDE':'ln(Crude Oil)',
    'ln_VIX':'ln(VIX)', 'ln_GRP':'ln(GPR Index)',
}
ur_rows = []
for col, label in ur_series.items():
    s = df[col].dropna()
    adf_l = adfuller(s, autolag='AIC')
    adf_d = adfuller(s.diff().dropna(), autolag='AIC')
    kp    = kpss(s, regression='c', nlags='auto')
    order = 'I(0)' if (adf_l[1]<0.05 and kp[1]>0.05) else 'I(1)'
    ur_rows.append({'Variable':label,
                    'ADF Stat (level)':round(adf_l[0],4),
                    'ADF p (level)':round(adf_l[1],4),
                    'ADF p (diff)':round(adf_d[1],4),
                    'KPSS Stat':round(kp[0],4),
                    'KPSS p':round(kp[1],4),
                    'Order':order})
    print(f"  {label:<20} ADF_p={adf_l[1]:.4f}  diff_p={adf_d[1]:.4f}  KPSS_p={kp[1]:.4f}  → {order}")
TABLE_UR = pd.DataFrame(ur_rows)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — STRUCTURAL BREAK TESTS  (Table 5 in paper)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Running structural break tests (Table 5)...")

all_sb_series = {
    'ln_INR_ER':'INR/USD',    'ln_EURO_ER':'EUR/USD',
    'ln_UK_ER':'GBP/USD',     'ln_JAPAN_ER':'JPY/USD',
    'ln_CHINA_ER':'CNY/USD',  'ln_BRAZIL_ER':'BRL/USD',
    'ln_RUSSINA_ER':'RUB/USD','ln_UKRAINE_ER':'UAH/USD',
    'ln_GRP':'GPR Index',     'ln_VIX':'VIX',
    'ln_CRUDE':'Crude Oil',   'ln_GOLD':'Gold',
    'MARKET_SENTIMENT_INDEX':'Sentiment',
}

# ── Zivot-Andrews (Model C) ───────────────────────────────────────────────────
def zivot_andrews(series, maxlag=1):
    y = series.dropna().values.astype(float)
    n = len(y)
    lo = int(0.15*n); hi = int(0.85*n)
    best_t = np.inf; best_k = None
    dy = np.diff(y)          # length n-1
    for k in range(lo, hi):
        DU = (np.arange(n) >= k).astype(float)
        DT = np.where(np.arange(n) >= k, np.arange(n)-k, 0).astype(float)
        t  = np.arange(1, n+1, dtype=float)
        # y[:-1] and dy are both length n-1; DU[1:], DT[1:], t[1:] also n-1
        lag = min(maxlag, 1)
        if lag == 0 or n-1 <= 5:
            X = np.column_stack([np.ones(n-1), t[1:], DU[1:], DT[1:], y[:-1]])
            y_dep = dy
        else:
            # drop first `lag` obs so all arrays align
            X = np.column_stack([np.ones(n-1-lag), t[1+lag:], DU[1+lag:],
                                  DT[1+lag:], y[lag:-1], dy[lag-1:-1]])
            y_dep = dy[lag:]
        try:
            res = OLS(y_dep, X).fit()
            t_stat = res.tvalues[4]   # coef on y[t-1]
            if t_stat < best_t:
                best_t = t_stat; best_k = k
        except: pass
    return best_t, best_k

# ── CMR Double-break (AO) ─────────────────────────────────────────────────────
def cmr_ao(series):
    y = series.dropna().values.astype(float)
    n = len(y)
    lo = int(0.10*n); hi = int(0.90*n)
    best_t = np.inf; best_tb1 = best_tb2 = None
    for i in range(lo, hi, 5):
        for j in range(i+8, hi, 5):
            DU1 = (np.arange(n)>=i).astype(float)
            DU2 = (np.arange(n)>=j).astype(float)
            try:
                y_tilde = OLS(y, np.column_stack([np.ones(n),DU1,DU2])).fit().resid
            except: continue
            dy = np.diff(y_tilde); m = len(dy)
            X2 = np.column_stack([DU1[1:m+1], DU2[1:m+1], y_tilde[:m]])
            try:
                res2 = OLS(dy, X2).fit()
                t_stat = res2.tvalues[2]
                if t_stat < best_t:
                    best_t = t_stat; best_tb1 = i; best_tb2 = j
            except: pass
    return best_t, best_tb1, best_tb2

# ── Chow test ─────────────────────────────────────────────────────────────────
def chow_test(series, t_vec, break_date_str):
    mask  = series.notna()
    y     = series[mask].values
    t     = t_vec[mask].values
    dates = df.loc[mask, 'MONTH'].values
    POST  = (dates >= np.datetime64(break_date_str)).astype(float)
    X     = add_constant(np.column_stack([t, POST, POST*t]))
    try:
        res = OLS(y, X).fit()
        R   = np.zeros((2,4)); R[0,2]=1; R[1,3]=1
        ft  = res.f_test(R)
        return float(ft.fvalue), float(ft.pvalue)
    except: return np.nan, np.nan

war_chow_dates = {
    'Gulf_War':'1990-08-01', 'Chechnya1':'1994-12-01',
    '9/11_Afghan':'2001-10-01', 'Iraq_War':'2003-03-01',
    'GFC':'2008-09-01', 'Crimea':'2014-02-01', 'Ukraine_Inv':'2022-02-01'
}

sb_rows = []
za_store = {}; cmr_store = {}; chow_store = {}

for col, label in all_sb_series.items():
    s     = df[col].dropna()
    dates = df.loc[s.index, 'MONTH'].reset_index(drop=True)

    # ZA
    za_stat, za_brk = zivot_andrews(s)
    za_date = dates.iloc[za_brk].strftime('%Y-%m') if za_brk else 'NA'
    za_sig  = '***' if za_stat<-5.08 else ('**' if za_stat<-4.80 else '')
    za_store[col] = {'stat':round(za_stat,3), 'date':za_date, 'sig':za_sig}

    # CMR
    cmr_stat, tb1, tb2 = cmr_ao(s)
    cmr_b1 = dates.iloc[tb1].strftime('%Y-%m') if tb1 else 'NA'
    cmr_b2 = dates.iloc[tb2].strftime('%Y-%m') if tb2 else 'NA'
    cmr_sig = '***' if cmr_stat<-5.49 else ('**' if cmr_stat<-5.20 else '')
    cmr_store[col] = {'stat':round(cmr_stat,3),'b1':cmr_b1,'b2':cmr_b2,'sig':cmr_sig}

    # Chow at Crimea and invasion
    _, p_cr  = chow_test(df[col], df['t'], '2014-02-01')
    _, p_ukr = chow_test(df[col], df['t'], '2022-02-01')
    chow_store[col] = {'Crimea':round(p_cr,4) if not np.isnan(p_cr) else np.nan,
                       'Ukraine_Inv':round(p_ukr,4) if not np.isnan(p_ukr) else np.nan}

    def fmt_p(p):
        if np.isnan(p): return 'NA'
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        return f'{p:.3f}{sig}'

    sb_rows.append({
        'Series': label,
        'ZA Stat': f'{za_stat:.3f}{za_sig}',
        'ZA Break': za_date,
        'CMR Stat': f'{cmr_stat:.3f}{cmr_sig}',
        'CMR Break1': cmr_b1,
        'CMR Break2': cmr_b2,
        'Chow Crimea p': fmt_p(p_cr),
        'Chow Invasion p': fmt_p(p_ukr),
    })
    print(f"  {label:<14} ZA={za_stat:6.3f}{za_sig:3s}  CMR={cmr_stat:6.3f}{cmr_sig:3s}  "
          f"Chow_Crimea={fmt_p(p_cr)}  Chow_Inv={fmt_p(p_ukr)}")

TABLE_SB = pd.DataFrame(sb_rows)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ARDL (India) + Long-run + Bounds Test  (Tables 2,3,4)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Running India ARDL(2,1,1,1,0)...")

dep = 'ln_INR_ER'
specs_ardl = {'ln_GOLD':1,'ln_CRUDE':1,'ln_VIX':1,'ln_GRP':0}
controls   = ['ln_GOLD','ln_CRUDE','ln_VIX','ln_GRP']

def build_ardl(df, dep, controls, specs, war_d, p=2, extra_dummies=None):
    data = {dep: df[dep]}
    for k in range(1, p+1): data[f'{dep}_L{k}'] = df[dep].shift(k)
    for x in controls:
        data[x] = df[x]
        for k in range(1, specs.get(x,0)+1): data[f'{x}_L{k}'] = df[x].shift(k)
    for w in war_d: data[w] = df[w]
    if extra_dummies:
        for col in extra_dummies: data[col] = df[col]
    tmp  = pd.DataFrame(data).dropna()
    y    = tmp[dep]
    xcols= [c for c in tmp.columns if c != dep]
    X    = add_constant(tmp[xcols])
    return y, X

y_ardl, X_ardl = build_ardl(df, dep, controls, specs_ardl, WAR_D)
res_ardl = OLS(y_ardl, X_ardl).fit(cov_type='HAC', cov_kwds={'maxlags':12})

# Long-run coefficients
theta = 1 - res_ardl.params.get('ln_INR_ER_L1',0) - res_ardl.params.get('ln_INR_ER_L2',0)
lr = {}
for x in controls:
    num = sum(res_ardl.params.get(f'{x}_L{k}' if k>0 else x, 0)
              for k in range(0, specs_ardl.get(x,0)+1))
    lr[x] = round(num/theta, 4) if theta>0 else np.nan
hl = round(np.log(0.5)/np.log(1-theta), 1) if 0<theta<1 else np.nan

# Bounds test
level_vars = [f'{dep}_L1','ln_GOLD','ln_CRUDE','ln_VIX','ln_GRP']
R_bounds = np.zeros((5, X_ardl.shape[1]))
for i,v in enumerate(level_vars):
    if v in list(X_ardl.columns):
        R_bounds[i, list(X_ardl.columns).index(v)] = 1
ft_bounds = res_ardl.f_test(R_bounds)
F_bounds  = float(ft_bounds.fvalue)
p_bounds  = float(ft_bounds.pvalue)

# Diagnostics
resid_ardl = res_ardl.resid
dw_ardl    = durbin_watson(resid_ardl)
bg_ardl    = acorr_breusch_godfrey(res_ardl, nlags=12)[1]
bp_ardl    = het_breuschpagan(resid_ardl, res_ardl.model.exog)[1]
adf_ardl   = adfuller(resid_ardl, autolag='AIC')[1]

print(f"  ARDL: R²={res_ardl.rsquared:.4f}  theta={theta:.4f}  HL={hl}mo")
print(f"  Bounds F={F_bounds:.3f}  DW={dw_ardl:.3f}  BG_p={bg_ardl:.3f}")
print(f"  LR: Gold={lr['ln_GOLD']:.3f}  Crude={lr['ln_CRUDE']:.3f}  "
      f"VIX={lr['ln_VIX']:.3f}  GRP={lr['ln_GRP']:.3f}")

# Build results dataframe for Table 2
ardl_rows = []
for var in res_ardl.params.index:
    ardl_rows.append({'Variable':var,
                      'Coefficient':round(res_ardl.params[var],5),
                      'Std Error':round(res_ardl.bse[var],5),
                      't-stat':round(res_ardl.tvalues[var],3),
                      'p-value':round(res_ardl.pvalues[var],4),
                      'CI Lower':round(res_ardl.conf_int().loc[var,0],5),
                      'CI Upper':round(res_ardl.conf_int().loc[var,1],5)})
TABLE_ARDL = pd.DataFrame(ardl_rows)

# ARDL with structural breaks (Table 4)
df['SB_GFC']      = (df['MONTH']>='2008-09-01').astype(int)
df['SB_Crimea']   = (df['MONTH']>='2014-02-01').astype(int)
df['SB_Invasion'] = (df['MONTH']>='2022-02-01').astype(int)
y_sb, X_sb = build_ardl(df, dep, controls, specs_ardl, WAR_D,
                         extra_dummies=['SB_GFC','SB_Crimea','SB_Invasion'])
res_sb   = OLS(y_sb, X_sb).fit(cov_type='HAC', cov_kwds={'maxlags':12})
ft_sb    = res_sb.f_test(R_bounds[:, :X_sb.shape[1]] if R_bounds.shape[1]>X_sb.shape[1]
                         else np.zeros((5, X_sb.shape[1])))
# Recompute bounds for sb model
R_sb = np.zeros((5, X_sb.shape[1]))
for i,v in enumerate(level_vars):
    if v in list(X_sb.columns):
        R_sb[i, list(X_sb.columns).index(v)] = 1
ft_sb2   = res_sb.f_test(R_sb)
F_sb     = float(ft_sb2.fvalue)

print(f"  ARDL with breaks: R²={res_sb.rsquared:.4f}  Bounds F={F_sb:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — GARCH(1,1)-t  (Table in paper)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Running GARCH models...")

returns = df['INR_ER_logreturn'].dropna()
ret_idx = returns.index

m1 = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal', mean='Constant')
r1 = m1.fit(disp='off')
m2 = arch_model(returns, vol='GARCH', p=1, q=1, dist='t', mean='Constant')
r2 = m2.fit(disp='off')
war_mean_cols = ['War_Russia_Crimea','War_Russia_Ukraine','War_Ukraine_Donbas','War_UK_Afghan']
X_war = df[war_mean_cols].iloc[ret_idx].values
m3 = arch_model(returns, vol='GARCH', p=1, q=1, dist='t', mean='ARX', lags=0, x=X_war)
r3 = m3.fit(disp='off')

garch_rows = [
    {'Model':'Baseline GARCH(1,1)-N', 'omega':r1.params.get('omega',np.nan),
     'alpha':r1.params.get('alpha[1]',np.nan),'beta':r1.params.get('beta[1]',np.nan),
     'nu':np.nan,'AIC':round(r1.aic,2),'BIC':round(r1.bic,2),'LogL':round(r1.loglikelihood,2)},
    {'Model':'GARCH(1,1)-t', 'omega':r2.params.get('omega',np.nan),
     'alpha':r2.params.get('alpha[1]',np.nan),'beta':r2.params.get('beta[1]',np.nan),
     'nu':round(r2.params.get('nu',np.nan),3),'AIC':round(r2.aic,2),
     'BIC':round(r2.bic,2),'LogL':round(r2.loglikelihood,2)},
    {'Model':'GARCH(1,1)-t + War dummies (mean)', 'omega':r3.params.get('omega',np.nan),
     'alpha':r3.params.get('alpha[1]',np.nan),'beta':r3.params.get('beta[1]',np.nan),
     'nu':round(r3.params.get('nu',np.nan),3),'AIC':round(r3.aic,2),
     'BIC':round(r3.bic,2),'LogL':round(r3.loglikelihood,2)},
]
for row in garch_rows:
    if not np.isnan(row['alpha']) and not np.isnan(row['beta']):
        row['alpha+beta'] = round(row['alpha']+row['beta'],4)
TABLE_GARCH = pd.DataFrame(garch_rows)

cond_vol = pd.DataFrame({
    'MONTH': df['MONTH'].iloc[ret_idx].values,
    'returns': returns.values,
    'cond_vol': r2.conditional_volatility.values,
    'GRP': df['GRP'].iloc[ret_idx].values,
})
print(f"  GARCH-t: alpha={r2.params.get('alpha[1]',np.nan):.4f}  "
      f"beta={r2.params.get('beta[1]',np.nan):.4f}  "
      f"nu={r2.params.get('nu',np.nan):.3f}  AIC={r2.aic:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — VAR + LP-IRF + FEVD
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Running VAR and LP-IRF...")

df['dln_INR']   = df['ln_INR_ER'].diff()
df['dln_CRUDE'] = df['ln_CRUDE'].diff()
df['dln_GOLD']  = df['ln_GOLD'].diff()

var_cols  = ['GRP','dln_INR','ln_VIX','dln_CRUDE','dln_GOLD']
var_data  = df[var_cols].dropna().reset_index(drop=True)
var_model = VAR(var_data)
var_res   = var_model.fit(1)

# Bootstrap IRF
np.random.seed(42)
A = var_res.coefs[0]; Sigma = var_res.sigma_u; k = 5
P = cholesky(Sigma, lower=True)

def compute_orth_irf(A, P, steps):
    Phi = np.eye(k); irfs = []
    for _ in range(steps+1):
        irfs.append(Phi @ P); Phi = Phi @ A
    return np.array(irfs)

base_irf = compute_orth_irf(A, P, 24)
resid_var = var_res.resid.values; intercept_var = var_res.intercept
B = 500; boot = np.zeros((B,25,k,k))
for b in range(B):
    idx = np.random.choice(len(resid_var), len(resid_var), replace=True)
    br  = resid_var[idx]
    Y   = np.zeros((len(var_data),k)); Y[0] = var_data.values[0]
    for t2 in range(1, len(var_data)):
        Y[t2] = intercept_var + A @ Y[t2-1] + br[t2-1]
    try:
        rb = VAR(pd.DataFrame(Y, columns=var_cols)).fit(1)
        boot[b] = compute_orth_irf(rb.coefs[0], cholesky(rb.sigma_u,lower=True), 24)
    except: boot[b] = base_irf

irf_vals = base_irf[:,1,0]
irf_lo   = np.percentile(boot,2.5,axis=0)[:,1,0]
irf_hi   = np.percentile(boot,97.5,axis=0)[:,1,0]

# FEVD
fevd     = var_res.fevd(24)
fevd_inr = pd.DataFrame(fevd.decomp[1]*100, columns=var_cols, index=range(1,25))

# LP-IRF
h_range = range(0,25)
lp_betas=[]; lp_lo=[]; lp_hi=[]
for h in h_range:
    y_lp = df['dln_INR'].shift(-h)
    X_lp = df[['GRP','ln_VIX','dln_CRUDE','dln_GOLD','dln_INR']].copy()
    X_lp = add_constant(X_lp)
    mask = y_lp.notna() & X_lp.notna().all(axis=1)
    res_lp = OLS(y_lp[mask], X_lp[mask]).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    lp_betas.append(res_lp.params['GRP'])
    lp_lo.append(res_lp.conf_int().loc['GRP',0])
    lp_hi.append(res_lp.conf_int().loc['GRP',1])
lp_betas=np.array(lp_betas); lp_lo=np.array(lp_lo); lp_hi=np.array(lp_hi)
print(f"  VAR(1): GRP→INR IRF h=0: {irf_vals[0]:.6f}  LP-IRF h=0: {lp_betas[0]:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MULTI-COUNTRY ARDL  (Tables 6,7)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Running multi-country ARDL...")

countries = {
    'INR':    {'col':'ln_INR_ER',    'name':'India',    'group':'Developing','start':'1990-01-01'},
    'EURO':   {'col':'ln_EURO_ER',   'name':'Euro Area','group':'Developed', 'start':'1990-01-01'},
    'UK':     {'col':'ln_UK_ER',     'name':'UK',       'group':'Developed', 'start':'1990-01-01'},
    'JAPAN':  {'col':'ln_JAPAN_ER',  'name':'Japan',    'group':'Developed', 'start':'1990-01-01'},
    'CHINA':  {'col':'ln_CHINA_ER',  'name':'China',    'group':'Developing','start':'1990-01-01'},
    'BRAZIL': {'col':'ln_BRAZIL_ER', 'name':'Brazil',   'group':'Developing','start':'1990-01-01'},
    'RUSSIA': {'col':'ln_RUSSINA_ER','name':'Russia',   'group':'Developing','start':'1992-06-01'},
    'UKRAINE':{'col':'ln_UKRAINE_ER','name':'Ukraine',  'group':'Developing','start':'1996-09-01'},
}
ORDER = ['EURO','UK','JAPAN','INR','CHINA','BRAZIL','RUSSIA','UKRAINE']

def select_lag(df_c, dep, controls, war_d, max_p=4, max_q=3):
    best_aic=np.inf; best=(1,0,0,0,0)
    for p in range(1,max_p+1):
        for q_g in range(0,max_q):
            for q_c in range(0,max_q):
                for q_v in range(0,2):
                    data={dep:df_c[dep]}
                    for k in range(1,p+1): data[f'{dep}_L{k}']=df_c[dep].shift(k)
                    for x,q in zip(controls,[q_g,q_c,q_v]):
                        data[x]=df_c[x]
                        for k in range(1,q+1): data[f'{x}_L{k}']=df_c[x].shift(k)
                    for w in war_d: data[w]=df_c[w]
                    try:
                        tmp=pd.DataFrame(data).dropna()
                        y_=tmp[dep]; X_=add_constant(tmp[[c for c in tmp.columns if c!=dep]])
                        r_=OLS(y_,X_).fit()
                        n_=len(y_); k_=X_.shape[1]
                        aic=n_*np.log(r_.ssr/n_)+2*k_
                        if aic<best_aic: best_aic=aic; best=(p,q_g,q_c,q_v,0)
                    except: pass
    return best

mc_ctrl = ['ln_GRP','ln_CRUDE','ln_GOLD','ln_VIX']
mc_results = {}

for key in ORDER:
    meta = countries[key]; dep_c = meta['col']
    df_c = df[df['MONTH']>=meta['start']].copy().reset_index(drop=True)
    df_c = df_c.dropna(subset=[dep_c])
    spec = select_lag(df_c, dep_c, mc_ctrl, WAR_D)
    p,q_g,q_c,q_v,_ = spec
    sp = {'ln_GRP':q_g,'ln_CRUDE':q_c,'ln_GOLD':q_c,'ln_VIX':q_v}

    data={dep_c:df_c[dep_c]}
    for k in range(1,p+1): data[f'{dep_c}_L{k}']=df_c[dep_c].shift(k)
    for x in mc_ctrl:
        data[x]=df_c[x]
        for k in range(1,sp.get(x,0)+1): data[f'{x}_L{k}']=df_c[x].shift(k)
    for w in WAR_D: data[w]=df_c[w]
    tmp=pd.DataFrame(data).dropna(); y_=tmp[dep_c]
    X_=add_constant(tmp[[c for c in tmp.columns if c!=dep_c]])
    res_=OLS(y_,X_).fit(cov_type='HAC',cov_kwds={'maxlags':12})

    denom = 1-sum(res_.params.get(f'{dep_c}_L{k}',0) for k in range(1,p+1))
    lr_c  = {}
    for x in mc_ctrl:
        num=sum(res_.params.get(f'{x}_L{k}' if k>0 else x,0)
                for k in range(0,sp.get(x,0)+1))
        lr_c[x] = round(num/denom,4) if denom!=0 else np.nan
    hl_c = round(np.log(0.5)/np.log(1-denom),1) if 0<denom<1 else np.nan

    mc_results[key] = {'meta':meta,'spec':spec,'res':res_,'lr':lr_c,
                       'theta':denom,'hl':hl_c,'y':y_,'dep':dep_c,'df':df_c}
    print(f"  {meta['name']:<10} ARDL{spec}  R²={res_.rsquared:.4f}  HL={hl_c}mo")

# Build Tables 6 & 7
def build_mc_table(keys):
    rows=[]
    for key in keys:
        r=mc_results[key]; res=r['res']; meta=r['meta']
        row={'Country':meta['name'],'Group':meta['group'],
             'Spec':str(r['spec']),'n':len(r['y']),
             'R2':round(res.rsquared,4),'AdjR2':round(res.rsquared_adj,4),
             'AIC':round(res.aic,1),'theta':round(r['theta'],4),'HL':r['hl'],
             'LR_GRP':r['lr'].get('ln_GRP',np.nan),
             'LR_Crude':r['lr'].get('ln_CRUDE',np.nan),
             'LR_Gold':r['lr'].get('ln_GOLD',np.nan),
             'LR_VIX':r['lr'].get('ln_VIX',np.nan)}
        for w in WAR_D:
            if w in res.params:
                row[f'{w}_coef'] = round(res.params[w]*100,4)
                row[f'{w}_p']    = round(res.pvalues[w],4)
        rows.append(row)
    return pd.DataFrame(rows)

dev_keys  = ['EURO','UK','JAPAN']
devg_keys = ['INR','CHINA','BRAZIL','RUSSIA','UKRAINE']
TABLE_DEV  = build_mc_table(dev_keys)
TABLE_DEVG = build_mc_table(devg_keys)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — SAVE ALL TABLES TO EXCEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Saving all tables to output/tables.xlsx...")

with pd.ExcelWriter('output/tables.xlsx', engine='openpyxl') as writer:
    TABLE_UR.to_excel(writer,   sheet_name='Table1_UnitRoot',     index=False)
    TABLE_ARDL.to_excel(writer, sheet_name='Table2_ARDL_India',   index=False)
    pd.DataFrame([{'Variable':k,'LR_Elasticity':v} for k,v in lr.items()
                  ]+[{'Variable':'theta','LR_Elasticity':round(theta,4)},
                     {'Variable':'Half-life (months)','LR_Elasticity':hl},
                     {'Variable':'Bounds F-stat','LR_Elasticity':round(F_bounds,3)},
                     {'Variable':'Bounds p','LR_Elasticity':round(p_bounds,4)}]
                 ).to_excel(writer, sheet_name='Table3_LongRun',  index=False)
    pd.DataFrame([{'Model':'Without breaks','R2':round(res_ardl.rsquared,4),
                   'AIC':round(res_ardl.aic,1),'Bounds_F':round(F_bounds,3)},
                  {'Model':'With SB dummies','R2':round(res_sb.rsquared,4),
                   'AIC':round(res_sb.aic,1),'Bounds_F':round(F_sb,3)}]
                ).to_excel(writer, sheet_name='Table4_ARDL_Breaks',index=False)
    TABLE_SB.to_excel(writer,   sheet_name='Table5_StructBreaks',index=False)
    TABLE_DEV.to_excel(writer,  sheet_name='Table6_Developed',   index=False)
    TABLE_DEVG.to_excel(writer, sheet_name='Table7_Developing',  index=False)
    TABLE_GARCH.to_excel(writer,sheet_name='Table8_GARCH',       index=False)
    fevd_inr.to_excel(writer,   sheet_name='Table9_FEVD')
    # LP-IRF table
    pd.DataFrame({'h':list(h_range),'LP_beta':lp_betas,
                  'CI_lo':lp_lo,'CI_hi':lp_hi}).to_excel(
        writer, sheet_name='Table10_LP_IRF', index=False)

print("  tables.xlsx saved with 10 sheets.")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — GENERATE ALL FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Generating all figures...")

# ── FIG 1: INR overview ───────────────────────────────────────────────────────
fig, axes = plt.subplots(3,1,figsize=(14,10),facecolor='white')
fig.suptitle('INR/USD Exchange Rate and Geopolitical Risk Index (1990–2026)',
             fontsize=13,fontweight='bold',color=C['DARK'])

ax=axes[0]; shade_wars(ax,df)
ax.plot(df['MONTH'],df['INR_ER'],color=C['BLUE'],lw=1.5)
ax.set_title('A.  INR/USD Exchange Rate (Levels)',fontsize=9,fontweight='bold',loc='left')
ax.set_ylabel('INR per USD'); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax=axes[1]; shade_wars(ax,df)
ax.plot(df['MONTH'],df['GRP'],color=C['RED'],lw=1.2,alpha=0.9)
ax.axhline(df['GRP'].mean(),color=C['GRAY'],lw=0.8,ls='--',label=f"Mean={df['GRP'].mean():.0f}")
ax.set_title('B.  Geopolitical Risk Index (Caldara & Iacoviello 2022)',fontsize=9,fontweight='bold',loc='left')
ax.set_ylabel('GPR Index'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

roll_corr = df[['INR_ER','GRP']].set_index(df['MONTH']).rolling(24).corr().unstack()['INR_ER']['GRP']
ax=axes[2]
ax.plot(df['MONTH'],roll_corr.values,color=C['GREEN'],lw=1.3)
ax.axhline(0,color=C['DARK'],lw=0.7)
ax.fill_between(df['MONTH'],roll_corr.values,0,where=roll_corr.values>0,alpha=0.2,color=C['RED'])
ax.fill_between(df['MONTH'],roll_corr.values,0,where=roll_corr.values<0,alpha=0.2,color=C['BLUE'])
ax.set_title('C.  Rolling 24-Month Correlation: INR vs GPR',fontsize=9,fontweight='bold',loc='left')
ax.set_ylabel('Correlation'); ax.set_ylim(-1,1); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

for ax in axes: ax.set_xlim(df['MONTH'].min(),df['MONTH'].max()); ax.tick_params(labelsize=8)
plt.tight_layout(); fig.savefig('output/fig1_overview.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig1_overview.png")

# ── FIG 2: LP-IRF ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(14,5),facecolor='white')
fig.suptitle('Impulse Response Functions: Geopolitical Risk Shock → INR/USD',
             fontsize=12,fontweight='bold',color=C['DARK'])
h = np.arange(25)
ax=axes[0]
ax.fill_between(h,lp_lo*100,lp_hi*100,alpha=0.2,color=C['BLUE'],label='95% CI (HAC)')
ax.plot(h,lp_betas*100,color=C['BLUE'],lw=2,marker='o',ms=4,label='LP-IRF')
ax.axhline(0,color=C['DARK'],lw=0.8,ls='--')
sig_m = (lp_lo>0)|(lp_hi<0)
ax.scatter(h[sig_m],lp_betas[sig_m]*100,color=C['RED'],zorder=5,s=50,label='Significant')
ax.set_xlabel('Months after shock'); ax.set_ylabel('% change in INR/USD')
ax.set_title('A.  Local Projection IRF (Jordà 2005)',fontsize=9,fontweight='bold',loc='left')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax=axes[1]
cum=np.cumsum(lp_betas)*100
ax.fill_between(h,np.cumsum(lp_lo)*100,np.cumsum(lp_hi)*100,alpha=0.2,color=C['GREEN'])
ax.plot(h,cum,color=C['GREEN'],lw=2,marker='s',ms=4)
ax.axhline(0,color=C['DARK'],lw=0.8,ls='--')
ax.set_xlabel('Months after shock'); ax.set_ylabel('Cumulative % change')
ax.set_title('B.  Cumulative LP-IRF',fontsize=9,fontweight='bold',loc='left')
ax.grid(alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig('output/fig2_irf.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig2_irf.png")

# ── FIG 3: GARCH ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2,2,figsize=(14,9),facecolor='white')
fig.suptitle('GARCH(1,1)-t: INR/USD Conditional Volatility and War Episodes',
             fontsize=12,fontweight='bold',color=C['DARK'])

ax=axes[0,0]
ax.bar(cond_vol['MONTH'],cond_vol['returns'],
       color=np.where(cond_vol['returns']>0,C['BLUE'],C['RED']),alpha=0.7,width=20)
ax.axhline(0,color=C['DARK'],lw=0.6)
ax.set_title('A.  INR/USD Monthly Log Returns (%)',fontsize=9,fontweight='bold',loc='left')
ax.set_ylabel('Return (%)'); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax=axes[0,1]; ax2b=ax.twinx()
ax.fill_between(cond_vol['MONTH'],cond_vol['cond_vol'],alpha=0.35,color=C['AMBER'])
ax.plot(cond_vol['MONTH'],cond_vol['cond_vol'],color=C['AMBER'],lw=1.2)
ax2b.plot(cond_vol['MONTH'],cond_vol['GRP'],color=C['RED'],lw=0.8,alpha=0.7)
ax.set_title('B.  Conditional Volatility vs GPR',fontsize=9,fontweight='bold',loc='left')
ax.set_ylabel('Cond. Std Dev (%)',color=C['AMBER']); ax2b.set_ylabel('GPR',color=C['RED'])
for sp in ['top']: ax.spines[sp].set_visible(False)

ax=axes[1,0]; shade_wars(ax,df)
ax.plot(cond_vol['MONTH'],cond_vol['cond_vol'],color=C['AMBER'],lw=1.5,label='Cond. vol')
ax.set_title('C.  Conditional Volatility — Russia Conflict Shading',fontsize=9,fontweight='bold',loc='left')
ax.set_ylabel('Cond. Std Dev (%)'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax=axes[1,1]
periods = {
    'Pre-\nChechnya': ('1991-01-01','1994-11-01'),
    'Chechnya\n-1': ('1994-12-01','1996-08-01'),
    'Chechnya\n-2': ('1999-08-01','2009-04-01'),
    'Peace\n09-14':  ('2009-05-01','2014-01-01'),
    'Crimea/\nDonbas':('2014-02-01','2022-01-01'),
    'Ukraine\nWar':  ('2022-02-01','2026-01-01'),
}
cv_data=[]; labs_bp=[]
colors_bp=['#BFDBFE','#FCA5A5','#F87171','#D1FAE5','#FEE2E2','#DC2626']
for lbl,(s,e) in periods.items():
    mask=(cond_vol['MONTH']>=s)&(cond_vol['MONTH']<=e)
    cv_data.append(cond_vol.loc[mask,'cond_vol'].dropna().values)
    labs_bp.append(lbl)
bp=ax.boxplot(cv_data,patch_artist=True,widths=0.55,
              medianprops=dict(color=C['DARK'],lw=1.8))
for patch,col in zip(bp['boxes'],colors_bp):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.set_xticklabels(labs_bp,fontsize=8)
ax.set_title('D.  Volatility by Conflict Period',fontsize=9,fontweight='bold',loc='left')
ax.set_ylabel('Cond. Std Dev (%)'); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig('output/fig3_garch.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig3_garch.png")

# ── FIG 4: FEVD ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,5),facecolor='white')
fig.suptitle('Forecast Error Variance Decomposition of Δln(INR/USD) — VAR(1)',
             fontsize=11,fontweight='bold',color=C['DARK'])
h_fevd=[1,3,6,12,24]; fevd_p=fevd_inr.loc[h_fevd]
colors_f=[C['RED'],C['BLUE'],C['AMBER'],C['GREEN'],C['GRAY']]
labs_f=['GPR Index','Own (INR)','VIX','Crude Oil','Gold']
bottom=np.zeros(len(h_fevd))
for col_f,col_c,lbl_f in zip(var_cols,colors_f,labs_f):
    vals=fevd_p[col_f].values
    bars=ax.bar([str(h) for h in h_fevd],vals,bottom=bottom,color=col_c,alpha=0.82,label=lbl_f,width=0.5)
    for bar,v,b in zip(bars,vals,bottom):
        if v>1: ax.text(bar.get_x()+bar.get_width()/2,b+v/2,f'{v:.1f}%',
                        ha='center',va='center',fontsize=8,color='white',fontweight='bold')
    bottom+=vals
ax.set_xlabel('Forecast Horizon (months)'); ax.set_ylabel('% Variance Explained')
ax.set_ylim(0,100); ax.legend(fontsize=9,loc='upper right'); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig('output/fig4_fevd.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig4_fevd.png")

# ── FIG 5: ARDL fit ───────────────────────────────────────────────────────────
ardl_dates = df['MONTH'].iloc[y_ardl.index]
fig, axes = plt.subplots(3,1,figsize=(14,10),facecolor='white')
fig.suptitle('ARDL(2,1,1,1,0) — Model Fit and Residual Diagnostics',
             fontsize=12,fontweight='bold',color=C['DARK'])
ax=axes[0]; shade_wars(ax,df)
ax.plot(ardl_dates,np.exp(y_ardl.values),color=C['DARK'],lw=1.5,label='Actual')
ax.plot(ardl_dates,np.exp(res_ardl.fittedvalues.values),color=C['BLUE'],lw=1.2,ls='--',label='Fitted')
ax.set_title(f'A.  Actual vs Fitted  |  R²={res_ardl.rsquared:.4f}',fontsize=9,fontweight='bold',loc='left')
ax.set_ylabel('INR per USD'); ax.legend(fontsize=9); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax=axes[1]; resid_a=res_ardl.resid
ax.bar(ardl_dates,resid_a,color=np.where(resid_a>0,C['RED'],C['BLUE']),alpha=0.6,width=20)
ax.axhline(0,color=C['DARK'],lw=0.8)
ax.axhline(resid_a.std()*2,color=C['RED'],lw=0.8,ls='--',alpha=0.6,label='±2 SD')
ax.axhline(-resid_a.std()*2,color=C['RED'],lw=0.8,ls='--',alpha=0.6)
ax.set_title('B.  Residuals over Time',fontsize=9,fontweight='bold',loc='left')
ax.set_ylabel('Residuals'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax=axes[2]
ax.hist(resid_a,bins=60,color=C['BLUE'],alpha=0.7,density=True,edgecolor='white',lw=0.3)
xr=np.linspace(resid_a.min(),resid_a.max(),200)
ax.plot(xr,stats.norm.pdf(xr,resid_a.mean(),resid_a.std()),color=C['RED'],lw=2,label='Normal fit')
ax.set_title(f'C.  Residual Distribution  (skew={resid_a.skew():.2f}, kurt={resid_a.kurt():.2f})',
             fontsize=9,fontweight='bold',loc='left')
ax.set_xlabel('Residual'); ax.set_ylabel('Density'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig('output/fig5_ardl_fit.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig5_ardl_fit.png")

# ── FIG 6: ARDL coefficient forest ───────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(14,7),facecolor='white')
fig.suptitle('ARDL Coefficient Forest Plot — India INR/USD',
             fontsize=12,fontweight='bold',color=C['DARK'])

macro_vars_plot = ['ln_GRP','ln_VIX_L1','ln_VIX','ln_CRUDE_L1','ln_CRUDE',
                   'ln_GOLD_L1','ln_GOLD','ln_INR_ER_L2','ln_INR_ER_L1']
macro_labels_p  = ['ln(GRP)','ln(VIX) L1','ln(VIX)','ln(Crude) L1','ln(Crude)',
                   'ln(Gold) L1','ln(Gold)','ln(INR) L2','ln(INR) L1']

for ax_i, (var_list, lbl_list, title) in enumerate([
    (macro_vars_plot, macro_labels_p, 'A.  Macro Variables'),
    (WAR_D, [w.replace('War_','').replace('_',' ') for w in WAR_D], 'B.  War Dummies')]):
    ax = axes[ax_i]
    for i,(var,lbl) in enumerate(zip(var_list,lbl_list)):
        if var not in res_ardl.params.index: continue
        c=res_ardl.params[var]; p=res_ardl.pvalues[var]
        lo=res_ardl.conf_int().loc[var,0]; hi=res_ardl.conf_int().loc[var,1]
        color=C['RED'] if p<0.01 else C['AMBER'] if p<0.05 else C['GREEN'] if p<0.1 else C['GRAY']
        ax.plot([lo,hi],[i,i],color=color,lw=2.2,solid_capstyle='round',zorder=2)
        ax.scatter(c,i,color=color,s=55,zorder=3)
        stars='***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        if ax_i==1:
            ax.text(max(hi,lo)+abs(hi-lo)*0.05, i, f'{c*100:+.2f}%{stars}',
                    va='center',fontsize=8,color=color)
        else:
            ax.text(max(abs(hi),abs(lo))+0.002, i, f'{c:.4f}{stars}',
                    va='center',fontsize=8,color=color)
    ax.axvline(0,color=C['DARK'],lw=0.8,ls='--',alpha=0.7)
    ax.set_yticks(range(len(var_list))); ax.set_yticklabels(lbl_list,fontsize=9)
    ax.set_title(title,fontsize=10,fontweight='bold',loc='left')
    ax.grid(axis='x',alpha=0.3)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for col,lbl_leg in [(C['RED'],'p<0.01'),(C['AMBER'],'p<0.05'),
                        (C['GREEN'],'p<0.10'),(C['GRAY'],'n.s.')]:
        ax.scatter([],[],color=col,s=40,label=lbl_leg)
    ax.legend(fontsize=8,loc='lower right')
plt.tight_layout(); fig.savefig('output/fig6_ardl_coefs.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig6_ardl_coefs.png")

# ── FIG 7: Long-run elasticities ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5),facecolor='white')
fig.suptitle('Long-Run Elasticities from ARDL(2,1,1,1,0) — India INR/USD',
             fontsize=11,fontweight='bold',color=C['DARK'])
lr_vals_plot = [lr['ln_GOLD'],lr['ln_CRUDE'],lr['ln_VIX'],lr['ln_GRP']]
lr_labs_plot = ['ln(Gold)','ln(Crude Oil)','ln(VIX)','ln(GPR)']
colors_lr = [C['AMBER'],C['RED'],C['BLUE'],C['GRAY']]
bars=ax.barh(lr_labs_plot,lr_vals_plot,color=colors_lr,alpha=0.82,height=0.45)
ax.axvline(0,color=C['DARK'],lw=0.8)
for bar,v in zip(bars,lr_vals_plot):
    ha='left' if v>=0 else 'right'; off=0.1 if v>=0 else -0.1
    ax.text(v+off,bar.get_y()+bar.get_height()/2,f'{v:.3f}',
            va='center',ha=ha,fontsize=10,fontweight='bold',color=C['DARK'])
ax.set_xlabel('Long-run elasticity'); ax.grid(axis='x',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.text(0.98,0.05,f"Adj. speed θ={theta:.4f}\nHalf-life={hl} months",
        transform=ax.transAxes,ha='right',fontsize=9,
        bbox=dict(boxstyle='round',facecolor='#F8FAFC',alpha=0.8))
plt.tight_layout(); fig.savefig('output/fig7_ardl_lr.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig7_ardl_lr.png")

# ── FIG 8: Mediation ─────────────────────────────────────────────────────────
def partial_corr(x_col, y_col, control_cols):
    tmp = df[[x_col,y_col]+control_cols].dropna()
    rx = OLS(tmp[x_col], add_constant(tmp[control_cols])).fit().resid
    ry = OLS(tmp[y_col], add_constant(tmp[control_cols])).fit().resid
    return np.corrcoef(rx,ry)[0,1]

pc_labels = ['No controls','Partial out\nCrude','Partial out\nGold',
             'Partial out\nVIX','Partial out\nGPR','Crude+Gold','All macro']
pc_vals   = [
    df[['War_Russia_Ukraine','ln_INR_ER']].corr().iloc[0,1],
    partial_corr('War_Russia_Ukraine','ln_INR_ER',['ln_CRUDE']),
    partial_corr('War_Russia_Ukraine','ln_INR_ER',['ln_GOLD']),
    partial_corr('War_Russia_Ukraine','ln_INR_ER',['ln_VIX']),
    partial_corr('War_Russia_Ukraine','ln_INR_ER',['ln_GRP']),
    partial_corr('War_Russia_Ukraine','ln_INR_ER',['ln_CRUDE','ln_GOLD']),
    partial_corr('War_Russia_Ukraine','ln_INR_ER',['ln_CRUDE','ln_GOLD','ln_VIX','ln_GRP']),
]
fig, axes = plt.subplots(1,2,figsize=(14,5.5),facecolor='white')
fig.suptitle('Mediation Analysis: War_Russia_Ukraine → INR/USD',
             fontsize=12,fontweight='bold',color=C['DARK'])
ax=axes[0]
colors_pc=[C['RED'] if v>0.4 else C['AMBER'] if v>0.2 else C['GRAY'] for v in pc_vals]
ax.bar(range(len(pc_labels)),pc_vals,color=colors_pc,alpha=0.82,width=0.55)
ax.axhline(0,color=C['DARK'],lw=0.8)
for i,v in enumerate(pc_vals):
    ax.text(i,v+0.01,f'{v:.3f}',ha='center',va='bottom',fontsize=9,fontweight='bold',color=C['DARK'])
ax.set_xticks(range(len(pc_labels))); ax.set_xticklabels(pc_labels,fontsize=8.5)
ax.set_ylabel('Partial correlation with ln(INR/USD)'); ax.set_ylim(0,0.65)
ax.set_title('A.  Partial Correlation Analysis',fontsize=9,fontweight='bold',loc='left')
ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax=axes[1]; ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(ax,x,y,w,h,text,fc,fontsize=9):
    ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.15',
                  facecolor=fc,edgecolor='white',lw=1,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,
            fontweight='bold',color='white',zorder=3,multialignment='center')
box(ax,0.3,4.2,2.6,1.3,'Russia–Ukraine\nWar (Feb 2022)','#991B1B',8)
box(ax,3.8,7.2,2.8,1.1,'Crude Oil\nSpike',C['AMBER'],9)
box(ax,3.8,4.2,2.8,1.1,'Gold Price\nSurge','#B45309',9)
box(ax,3.8,1.2,2.8,1.1,'VIX / GPR\nSpike',C['PURPLE'],9)
box(ax,7.3,4.2,2.4,1.3,'INR/USD\nDepreciation',C['BLUE'],9)
for (x1,y1,x2,y2,col) in [(2.9,5.4,3.8,7.7,C['AMBER']),(2.9,4.85,3.8,4.75,'#B45309'),
                            (2.9,4.4,3.8,1.75,C['PURPLE'])]:
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=col,lw=1.8))
for (x1,y1,x2,y2) in [(6.6,7.75,7.3,5.3),(6.6,4.75,7.3,4.85),(6.6,1.75,7.3,4.45)]:
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=C['DARK'],lw=1.5))
ax.annotate('',xy=(7.3,4.85),xytext=(2.9,4.85),
            arrowprops=dict(arrowstyle='->',color=C['GRAY'],lw=1.2,linestyle='dashed'))
ax.text(5.1,4.3,'Direct path\n(β=+0.42%, p=0.518, n.s.)',ha='center',fontsize=7.5,color=C['GRAY'],style='italic')
ax.set_title('B.  Causal Mediation Pathway',fontsize=9,fontweight='bold',loc='left')
plt.tight_layout(); fig.savefig('output/fig8_mediation.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig8_mediation.png")

# ── FIG 9–13: Multi-country comparison ───────────────────────────────────────
# FIG 9: Actual vs Fitted all countries
fig, axes = plt.subplots(2,4,figsize=(20,9),facecolor='white')
fig.suptitle('ARDL Models: Actual vs Fitted — All Countries',
             fontsize=12,fontweight='bold',color=C['DARK'])
for ax,key in zip(axes.flat,ORDER):
    r=mc_results[key]; meta=r['meta']; res=r['res']
    df_c=r['df']; y_c=r['y']
    dates_c = df_c['MONTH'].iloc[y_c.index] if 'MONTH' in df_c.columns else pd.Series(range(len(y_c)))
    ax.plot(dates_c,np.exp(y_c.values),color=C['DARK'],lw=1.2,label='Actual',zorder=3)
    ax.plot(dates_c,np.exp(res.fittedvalues.values),color=COUNTRY_COLOR[meta['name']],
            lw=1.1,ls='--',alpha=0.9,label='Fitted',zorder=4)
    grp_c=GROUP_COLOR[meta['group']]
    ax.set_title(f"{meta['name']} ({meta['group']})",fontsize=8.5,fontweight='bold',color=grp_c,loc='left')
    ax.text(0.97,0.05,f"R²={res.rsquared:.3f}",transform=ax.transAxes,ha='right',fontsize=7.5,
            color=COUNTRY_COLOR[meta['name']])
    ax.grid(axis='y',alpha=0.25); ax.tick_params(labelsize=7)
    ax.set_ylabel('LCU/USD',fontsize=7.5)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig('output/fig9_comp_fit.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig9_comp_fit.png")

# FIG 10: War dummy forest all countries
war_short = {'War_Russia_Chechnya1':'Chech-1','War_Russia_Chechnya2':'Chech-2',
             'War_Russia_Crimea':'Crimea','War_Russia_Ukraine':'Rus-Ukr',
             'War_Ukraine_Donbas':'Donbas','War_UK_Gulf':'Gulf','War_UK_Afghan':'Afghan'}
fig, axes = plt.subplots(2,4,figsize=(20,10),facecolor='white')
fig.suptitle('ARDL War Dummy Coefficients — All Countries (% effect on LCU/USD)',
             fontsize=12,fontweight='bold',color=C['DARK'])
for ax,key in zip(axes.flat,ORDER):
    r=mc_results[key]; res=r['res']; meta=r['meta']
    for i,w in enumerate(WAR_D):
        if w not in res.params: continue
        c=res.params[w]*100; p=res.pvalues[w]
        lo=res.conf_int().loc[w,0]*100; hi=res.conf_int().loc[w,1]*100
        col=C['RED'] if p<0.01 else C['AMBER'] if p<0.05 else C['GREEN'] if p<0.1 else C['GRAY']
        ax.plot([lo,hi],[i,i],color=col,lw=2,solid_capstyle='round',zorder=2)
        ax.scatter(c,i,color=col,s=45,zorder=3)
        stars='***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        ax.text(max(hi,lo)+0.1,i,f'{c:+.2f}%{stars}',va='center',fontsize=7,color=col)
    ax.axvline(0,color=C['DARK'],lw=0.8,ls='--',alpha=0.7)
    ax.set_yticks(range(len(WAR_D)))
    ax.set_yticklabels([war_short[w] for w in WAR_D],fontsize=7.5)
    ax.set_title(f"{meta['name']} ({meta['group']})",fontsize=8.5,fontweight='bold',
                 color=GROUP_COLOR[meta['group']],loc='left')
    ax.grid(axis='x',alpha=0.25)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig('output/fig10_comp_wardummies.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig10_comp_wardummies.png")

# FIG 11: Long-run elasticities comparison
fig, axes = plt.subplots(1,4,figsize=(18,6),facecolor='white')
fig.suptitle('Long-Run Elasticities: Developed vs Developing Economies',
             fontsize=11,fontweight='bold',color=C['DARK'])
for ax_i,(lrv,lrlab) in enumerate(zip(
    ['ln_GRP','ln_CRUDE','ln_GOLD','ln_VIX'],['GPR Index','Crude Oil','Gold','VIX'])):
    ax=axes[ax_i]
    names_=[mc_results[k]['meta']['name'] for k in ORDER]
    vals_=[mc_results[k]['lr'].get(lrv,np.nan) for k in ORDER]
    cols_=[GROUP_COLOR[mc_results[k]['meta']['group']] for k in ORDER]
    bars_=ax.bar(range(len(ORDER)),vals_,color=cols_,alpha=0.82,width=0.6)
    ax.axhline(0,color=C['DARK'],lw=0.8)
    for i,(b,v) in enumerate(zip(bars_,vals_)):
        if not np.isnan(v):
            va_='bottom' if v>=0 else 'top'
            ax.text(i,v+(max(abs(v)*0.05,0.05) if v>=0 else -max(abs(v)*0.05,0.05)),
                    f'{v:.2f}',ha='center',va=va_,fontsize=7.5,fontweight='bold',color=C['DARK'])
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(names_,rotation=30,ha='right',fontsize=8)
    ax.set_title(lrlab,fontsize=10,fontweight='bold',loc='left')
    ax.set_ylabel('Long-run elasticity',fontsize=8.5); ax.grid(axis='y',alpha=0.3)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
dev_p=mpatches.Patch(color=GROUP_COLOR['Developed'],alpha=0.82,label='Developed')
devg_p=mpatches.Patch(color=GROUP_COLOR['Developing'],alpha=0.82,label='Developing')
fig.legend(handles=[dev_p,devg_p],loc='upper right',fontsize=9,bbox_to_anchor=(0.99,0.99))
plt.tight_layout(rect=[0,0,0.97,0.95])
fig.savefig('output/fig11_comp_longrun.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig11_comp_longrun.png")

# FIG 12: Heatmap
fig, ax = plt.subplots(figsize=(14,6),facecolor='white')
fig.suptitle('ARDL War Dummy Significance Matrix — All Countries',
             fontsize=11,fontweight='bold',color=C['DARK'])
mat=np.zeros((len(ORDER),len(WAR_D))); mat[:]=np.nan
for ci,key in enumerate(ORDER):
    res=mc_results[key]['res']
    for wi,w in enumerate(WAR_D):
        if w in res.params:
            c=res.params[w]; p=res.pvalues[w]
            if p<0.05: mat[ci,wi]=np.sign(c)
from matplotlib.colors import ListedColormap,BoundaryNorm
cmap_hm=ListedColormap(['#16A34A','#D1D5DB','#DC2626'])
norm_hm=BoundaryNorm([-1.5,-0.5,0.5,1.5],cmap_hm.N)
ax.imshow(mat,cmap=cmap_hm,norm=norm_hm,aspect='auto',alpha=0.85)
ax.set_xticks(range(len(WAR_D))); ax.set_xticklabels([war_short[w] for w in WAR_D],fontsize=10)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([f"{mc_results[k]['meta']['name']}\n({mc_results[k]['meta']['group']})"
                    for k in ORDER],fontsize=9)
for ci in range(len(ORDER)):
    for wi in range(len(WAR_D)):
        sym=('+' if mat[ci,wi]>0 else '−') if not np.isnan(mat[ci,wi]) else '·'
        col='white' if not np.isnan(mat[ci,wi]) else C['GRAY']
        ax.text(wi,ci,sym,ha='center',va='center',fontsize=13,color=col,fontweight='bold')
ax.axhline(2.5,color=C['DARK'],lw=1.5,ls='--',alpha=0.5)
for spine in ax.spines.values(): spine.set_visible(False)
ax.tick_params(length=0)
leg_hm=[mpatches.Patch(color='#DC2626',alpha=0.85,label='Sig. depreciation (p<0.05)'),
        mpatches.Patch(color='#16A34A',alpha=0.85,label='Sig. appreciation (p<0.05)'),
        mpatches.Patch(color='#D1D5DB',alpha=0.85,label='Not significant')]
ax.legend(handles=leg_hm,loc='upper right',fontsize=9,bbox_to_anchor=(1.01,1.1))
plt.tight_layout(); fig.savefig('output/fig12_comp_heatmap.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig12_comp_heatmap.png")

# FIG 13: Structural comparison
fig, axes = plt.subplots(1,3,figsize=(17,5.5),facecolor='white')
fig.suptitle('ARDL Structural Comparison: Developed vs Developing Economies',
             fontsize=11,fontweight='bold',color=C['DARK'])
names_all=[mc_results[k]['meta']['name'] for k in ORDER]
colors_all=[GROUP_COLOR[mc_results[k]['meta']['group']] for k in ORDER]

ax=axes[0]
hls_=[min(mc_results[k]['hl'],150) if mc_results[k]['hl'] and not np.isnan(mc_results[k]['hl']) else 0 for k in ORDER]
bars_=ax.bar(range(len(ORDER)),hls_,color=colors_all,alpha=0.82,width=0.6)
for i,(b,h_) in enumerate(zip(bars_,hls_)):
    raw=mc_results[ORDER[i]]['hl']
    lbl=f'{raw:.0f}m' if raw and not np.isnan(raw) and raw<150 else '>150m'
    ax.text(i,h_+1,lbl,ha='center',va='bottom',fontsize=8,color=C['DARK'])
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(names_all,rotation=30,ha='right',fontsize=8.5)
ax.set_ylabel('Half-life (months)'); ax.set_title('A.  Adjustment Speed',fontsize=9,fontweight='bold',loc='left')
ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax=axes[1]
r2s_=[mc_results[k]['res'].rsquared for k in ORDER]
ax.bar(range(len(ORDER)),r2s_,color=colors_all,alpha=0.82,width=0.6)
for i,v in enumerate(r2s_):
    ax.text(i,v+0.0002,f'{v:.4f}',ha='center',va='bottom',fontsize=7.5,color=C['DARK'])
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(names_all,rotation=30,ha='right',fontsize=8.5)
ax.set_ylabel('R²'); ax.set_ylim(0.95,1.002)
ax.set_title('B.  Model Fit (R²)',fontsize=9,fontweight='bold',loc='left')
ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

ax=axes[2]
x_=np.arange(len(ORDER)); w_=0.25
n01_=[sum(1 for w in WAR_D if w in mc_results[k]['res'].pvalues and mc_results[k]['res'].pvalues[w]<0.01) for k in ORDER]
n05_=[sum(1 for w in WAR_D if w in mc_results[k]['res'].pvalues and 0.01<=mc_results[k]['res'].pvalues[w]<0.05) for k in ORDER]
n10_=[sum(1 for w in WAR_D if w in mc_results[k]['res'].pvalues and 0.05<=mc_results[k]['res'].pvalues[w]<0.10) for k in ORDER]
ax.bar(x_-w_,n01_,w_,color=C['RED'],alpha=0.85,label='p<0.01')
ax.bar(x_,n05_,w_,color=C['AMBER'],alpha=0.85,label='p<0.05')
ax.bar(x_+w_,n10_,w_,color=C['GREEN'],alpha=0.85,label='p<0.10')
ax.set_xticks(x_); ax.set_xticklabels(names_all,rotation=30,ha='right',fontsize=8.5)
ax.set_ylabel('No. significant war dummies')
ax.set_title('C.  War Sensitivity',fontsize=9,fontweight='bold',loc='left')
ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.3)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
dev_p=mpatches.Patch(color=GROUP_COLOR['Developed'],alpha=0.82,label='Developed')
devg_p=mpatches.Patch(color=GROUP_COLOR['Developing'],alpha=0.82,label='Developing')
fig.legend(handles=[dev_p,devg_p],loc='upper right',fontsize=9,bbox_to_anchor=(0.99,0.99))
plt.tight_layout(rect=[0,0,0.97,0.95])
fig.savefig('output/fig13_comp_structure.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig13_comp_structure.png")

# ── FIG 14: Structural break heatmap ─────────────────────────────────────────
# Build significance matrix: rows=series, cols=chow dates
fig, ax = plt.subplots(figsize=(16,7),facecolor='white')
fig.suptitle('Structural Break Chow Test — p-values Matrix (All Series × All War Dates)',
             fontsize=11,fontweight='bold',color=C['DARK'])

sb_series_order = list(all_sb_series.keys())
sb_series_labels= list(all_sb_series.values())
war_date_names  = list(war_chow_dates.keys())
war_dates_list  = list(war_chow_dates.values())

pmat = np.zeros((len(sb_series_order), len(war_date_names)))
for si,col in enumerate(sb_series_order):
    for di,(wname,wdate) in enumerate(war_chow_dates.items()):
        _,p = chow_test(df[col],df['t'],wdate)
        pmat[si,di] = p if not np.isnan(p) else 1.0

from matplotlib.colors import LogNorm
im = ax.imshow(pmat, cmap='RdYlGn_r', vmin=0, vmax=0.10, aspect='auto')
plt.colorbar(im, ax=ax, label='Chow test p-value', shrink=0.8)
ax.set_xticks(range(len(war_date_names)))
ax.set_xticklabels([w.replace('_','\n') for w in war_date_names],fontsize=9)
ax.set_yticks(range(len(sb_series_labels)))
ax.set_yticklabels(sb_series_labels,fontsize=9)
for si in range(len(sb_series_order)):
    for di in range(len(war_date_names)):
        p=pmat[si,di]
        sym='***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        ax.text(di,si,f'{p:.2f}{sym}',ha='center',va='center',fontsize=7,
                color='white' if p<0.05 else C['DARK'])
ax.axhline(7.5,color='white',lw=2,ls='--')
ax.set_title('Green/red = significant structural break; white dashed = exchange rates / macro boundary',
             fontsize=8.5,loc='left')
for spine in ax.spines.values(): spine.set_visible(False)
plt.tight_layout(); fig.savefig('output/fig14_structbreaks_heatmap.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig14_structbreaks_heatmap.png")

# ── FIG 15: ZA + CMR bar chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(16,6),facecolor='white')
fig.suptitle('Structural Break Test Statistics — Zivot-Andrews and CMR',
             fontsize=11,fontweight='bold',color=C['DARK'])

za_stats_  = [za_store[c]['stat'] for c in sb_series_order]
cmr_stats_ = [cmr_store[c]['stat'] for c in sb_series_order]
cols_za    = [C['RED'] if v<-5.08 else C['AMBER'] if v<-4.80 else C['GRAY'] for v in za_stats_]
cols_cmr   = [C['RED'] if v<-5.49 else C['AMBER'] if v<-5.20 else C['GRAY'] for v in cmr_stats_]

for ax,(stats_,cols_,cv,title,cv_label) in zip(axes,[
    (za_stats_, cols_za, -5.08, 'A.  Zivot-Andrews Statistics (Model C)', '5% CV = −5.08'),
    (cmr_stats_,cols_cmr,-5.49,'B.  CMR Double-Break Statistics (AO)',   '5% CV = −5.49')]):
    bars_=ax.barh(range(len(sb_series_labels)),stats_,color=cols_,alpha=0.82,height=0.6)
    ax.axvline(cv,color=C['RED'],lw=1.5,ls='--',label=cv_label)
    ax.axvline(0,color=C['DARK'],lw=0.7)
    for i,(b,v) in enumerate(zip(bars_,stats_)):
        ax.text(v-0.1,i,f'{v:.3f}',va='center',ha='right',fontsize=7.5,color='white',fontweight='bold')
    ax.set_yticks(range(len(sb_series_labels))); ax.set_yticklabels(sb_series_labels,fontsize=9)
    ax.set_xlabel('Test statistic'); ax.set_title(title,fontsize=9,fontweight='bold',loc='left')
    ax.legend(fontsize=9); ax.grid(axis='x',alpha=0.3)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for col,lbl in [(C['RED'],'Sig. at 1%'),(C['AMBER'],'Sig. at 5%'),(C['GRAY'],'n.s.')]:
        ax.barh([],[], color=col, alpha=0.82, label=lbl)
    ax.legend(fontsize=8,loc='lower right')
plt.tight_layout(); fig.savefig('output/fig15_za_cmr.png',dpi=150,bbox_inches='tight')
plt.close(); print("  fig15_za_cmr.png")

# ══════════════════════════════════════════════════════════════════════════════
#  DONE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  ALL DONE — check the output/ folder")
print("  Figures: 15 PNG files")
print("  Tables:  tables.xlsx with 10 sheets")
print("="*60)
print("\nFile list:")
for f in sorted(os.listdir('output')):
    size = os.path.getsize(f'output/{f}')
    print(f"  {f:<45} {size/1024:>7.1f} KB")
