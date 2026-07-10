
#####################################################
#                                                   #
# Piece of code for fuzzy numbers                   #
# including portfolio optimization                  #
#                                                   #
# 10.07.2025                                        #
#                                                   #
# Jan Schneider, jan.schneider@pwr.edu.pl           #
# Kaja Bilińska, kaja.bilinska@pwr.edu.pl           # 
#                                                   #
#   	arXiv:2602.20183                            #
#                                                   #
#####################################################


import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

print("Loading required packages...")

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
main_dir = f"Fuzzy_Analysis_Quantile_{run_id}"
plot_dir = os.path.join(main_dir, "Fuzzy_Asset_Plots")
report_dir = os.path.join(main_dir, "Reports")
sub_report_dir = os.path.join(report_dir, "Individual_Combos")

os.makedirs(main_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)
os.makedirs(sub_report_dir, exist_ok=True)

tickers = ["AAPL", "MSFT", "AMZN", "NVDA", 
           "TSLA", "NFLX", "AMD", 
           "GOOGL", "META"]

start_date = "2012-01-01"
end_date = "2019-12-31"
num_assets = len(tickers)

yf_end_date = "2020-01-01"

print(f"Downloading historical data for {num_assets} assets...")
returns_list = []

for tkr in tickers:
    df = yf.download(tkr, start=start_date, end=yf_end_date, progress=False)
    if 'Adj Close' in df.columns:
        prices = df['Adj Close']
    else:
        prices = df['Close']
    
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    
    monthly_last = prices.resample('ME').last()
    monthly_ret = monthly_last.pct_change()
    
    first_daily_price = prices.iloc[0]
    first_month_last_price = monthly_last.iloc[0]
    monthly_ret.iloc[0] = (first_month_last_price - first_daily_price) / first_daily_price
        
    monthly_ret.name = tkr
    returns_list.append(monthly_ret)

all_returns = pd.concat(returns_list, axis=1).dropna()

n_months = len(all_returns)
train_idx = int(np.floor((2/3) * n_months))
train_data = all_returns.iloc[:train_idx, :]
test_data  = all_returns.iloc[train_idx:n_months, :]
test_mat = test_data.values

def fuzzify_asset_quantile(return_vector, alpha_step=0.05):
    alphas = np.arange(0, 1 + alpha_step, alpha_step)
    alphas = np.round(alphas, 6)
    xi_d = np.zeros(len(alphas))
    xi_u = np.zeros(len(alphas))
    
    for i, a in enumerate(alphas):
        xi_d[i] = np.nanquantile(return_vector, q=a / 2, method='linear')
        xi_u[i] = np.nanquantile(return_vector, q=1 - a / 2, method='linear')
        
    return pd.DataFrame({'alpha': alphas, 'xi_d': xi_d, 'xi_u': xi_u})

print("Fuzzifying assets (empirical quantiles)...")
fuzzy_assets = {tkr: fuzzify_asset_quantile(train_data[tkr].values) for tkr in tickers}

for tkr in tickers:
    df = fuzzy_assets[tkr]
    fig, ax = plt.subplots(figsize=(5, 3))
    x_coords = list(df['xi_d']) + list(df['xi_u'])[::-1]
    y_coords = list(df['alpha']) + list(df['alpha'])[::-1]
    verts = list(zip(x_coords, y_coords))
    
    poly = Polygon(verts, facecolor="lightblue", edgecolor="darkblue", alpha=0.6)
    ax.add_patch(poly)
    ax.autoscale()
    ax.set_title(f"Quantile-based fuzzy number: {tkr}")
    ax.set_xlabel("Return")
    ax.set_ylabel(r"$\alpha$")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{tkr}.png"))
    plt.close()

print("Generating Universe...")
np.random.seed(42)
S_universe = 50000

random_mat = np.random.exponential(scale=1.0, size=(S_universe, num_assets))
W_universe = random_mat / random_mat.sum(axis=1, keepdims=True)

alphas = fuzzy_assets[tickers[0]]['alpha'].values
delta_alpha = alphas[1] - alphas[0]
int_weights = np.full(len(alphas), delta_alpha)
int_weights[0] = delta_alpha / 2
int_weights[-1] = delta_alpha / 2

Xi_d = np.column_stack([fuzzy_assets[t]['xi_d'] for t in tickers])
Xi_u = np.column_stack([fuzzy_assets[t]['xi_u'] for t in tickers])

Port_d = W_universe @ Xi_d.T
Port_u = W_universe @ Xi_u.T

E_P_universe = 0.5 * (Port_d @ int_weights + Port_u @ int_weights)
Risk_omega_universe = Port_u @ int_weights - Port_d @ int_weights
Risk_omega_sq_universe = Risk_omega_universe**2

lgy_eff_w = alphas * int_weights
lgy_w_mat = np.tile(lgy_eff_w, (S_universe, 1))

E_LGY_universe = np.sum((Port_d + Port_u) * lgy_w_mat, axis=1)
Var_LGY_universe = np.sum(((Port_d - E_LGY_universe[:, None])**2 + 
                           (Port_u - E_LGY_universe[:, None])**2) * lgy_w_mat, axis=1)

eq_w = np.full(num_assets, 1 / num_assets)
eq_Pd = eq_w @ Xi_d.T
eq_Pu = eq_w @ Xi_u.T

eq_E_P = np.sum(0.5 * (eq_Pd * int_weights + eq_Pu * int_weights))
eq_Risk_sq = np.sum(eq_Pu * int_weights - eq_Pd * int_weights)**2
eq_E_LGY = np.sum((eq_Pd + eq_Pu) * lgy_eff_w)
eq_Var_LGY = np.sum(((eq_Pd - eq_E_LGY)**2 + (eq_Pu - eq_E_LGY)**2) * lgy_eff_w)

print("Computing asset-level fuzzy metrics from KDE fuzzy numbers...")

def compute_asset_fuzzy_metrics():
    a_vals = alphas
    dalpha = int_weights
    n_assets = Xi_d.shape[1]
    rows = []
    
    for j in range(n_assets):
        xi_d = Xi_d[:, j]
        xi_u = Xi_u[:, j]
        
        mean_VB = 0.5 * np.sum((xi_d + xi_u) * dalpha)
        omega = np.sum((xi_u - xi_d) * dalpha)
        var_VB = omega**2
        
        third_VB = 0.5 * np.sum((xi_d - mean_VB)**3 * dalpha) + \
                   0.5 * np.sum((xi_u - mean_VB)**3 * dalpha)
        skew_VB = third_VB / (omega**3 + 1e-9)
        
        mean_LGY = np.sum((xi_d + xi_u) * lgy_eff_w)
        var_LGY = np.sum(((xi_d - mean_LGY)**2 + (xi_u - mean_LGY)**2) * lgy_eff_w)
        skew_LGY = np.sum(((xi_d - mean_LGY)**3 + (xi_u - mean_LGY)**3) * lgy_eff_w)
        
        m_lower = xi_d[np.argmax(a_vals)]
        m_upper = xi_u[np.argmax(a_vals)]
        
        jkpt1_inner = {}
        jkpt1_outer = {}
        
        for p in [0.10, 0.25]:
            ia = np.argmin(np.abs(a_vals - p))
            i1a = np.argmin(np.abs(a_vals - (1 - p)))
            i05 = np.argmin(np.abs(a_vals - 0.50))
            
            S_GM1_d = (xi_d[i1a] + xi_d[ia] - 2 * xi_d[i05]) / (xi_d[i1a] - xi_d[ia] + 1e-9)
            S_GM1_u = (xi_u[ia] + xi_u[i1a] - 2 * xi_u[i05]) / (xi_u[ia] - xi_u[i1a] + 1e-9)
            
            S_in = 0.5 * (S_GM1_d + S_GM1_u)
            S_out = ((xi_d[ia] - m_lower) + (xi_u[ia] - m_upper)) / (xi_u[ia] - xi_d[ia] + 1e-9)
            
            jkpt1_inner[str(p)] = S_in
            jkpt1_outer[str(p)] = S_out
            
        half_idx = np.where(a_vals <= 0.5)[0]
        p_vals = a_vals[half_idx]
        
        one_m_idx = [np.argmin(np.abs(a_vals - (1 - pv))) for pv in p_vals]
        i05 = np.argmin(np.abs(a_vals - 0.50))
        
        num_d = xi_d[one_m_idx] + xi_d[half_idx] - 2 * xi_d[i05]
        den_d = xi_d[one_m_idx] - xi_d[half_idx]
        
        num_u = xi_u[half_idx] + xi_u[one_m_idx] - 2 * xi_u[i05]
        den_u = xi_u[half_idx] - xi_u[one_m_idx]
        
        hw = int_weights[half_idx]
        S_in_int = 0.5 * (
            np.sum(num_d * hw) / (np.sum(den_d * hw) + 1e-9) + 
            np.sum(num_u * hw) / (np.sum(den_u * hw) + 1e-9)
        )
        
        S_out_num = (xi_d[half_idx] - m_lower) + (xi_u[half_idx] - m_upper)
        S_out_den = xi_u[half_idx] - xi_d[half_idx]
        S_out_int = np.sum(S_out_num * hw) / (np.sum(S_out_den * hw) + 1e-9)
        
        rows.append({
            'Asset': tickers[j],
            'Mean_VB13': mean_VB,
            'Var_VB13': var_VB,
            'Mean_LGY15': mean_LGY,
            'Var_LGY15': var_LGY,
            'Skew_VB13': skew_VB,
            'Skew_LGY15': skew_LGY,
            'JKPT1_Inner_p0.10': jkpt1_inner["0.1"],
            'JKPT1_Outer_p0.10': jkpt1_outer["0.1"],
            'JKPT1_Inner_p0.25': jkpt1_inner["0.25"],
            'JKPT1_Outer_p0.25': jkpt1_outer["0.25"],
            'JKPT2_InnerInt': S_in_int,
            'JKPT2_OuterInt': S_out_int
        })
        
    return pd.DataFrame(rows)

asset_metrics_df = compute_asset_fuzzy_metrics()
asset_metrics_csv = os.path.join(report_dir, "Fuzzy_Asset_Metrics.csv")
asset_metrics_df.to_csv(asset_metrics_csv, index=False)

asset_metrics_tex = os.path.join(report_dir, "Fuzzy_Asset_Metrics.tex")
colnames_ltx = [c.replace("_", "\\_") for c in asset_metrics_df.columns]

tex_lines = [
    "\\documentclass{article}",
    "\\usepackage[landscape, margin=1in]{geometry}",
    "\\usepackage{booktabs}",
    "\\begin{document}",
    "\\section*{Fuzzy Asset Metrics (quantile-based)}",
    "\\begin{tabular}{l" + "r" * (len(asset_metrics_df.columns) - 1) + "}",
    "\\toprule",
    " & ".join(colnames_ltx) + " \\\\ \\midrule"
]

cols = list(asset_metrics_df.columns)
sk_lgy_idx = cols.index("Skew_LGY15")

for _, row in asset_metrics_df.iterrows():
    formatted = []
    for i, val in enumerate(row[1:]):
        if i == (sk_lgy_idx - 1):
            formatted.append(f"{val:.6f}")
        else:
            formatted.append(f"{val:.3f}")
    row_str = f"{row['Asset']} & " + " & ".join(formatted) + " \\\\"
    tex_lines.append(row_str)

tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{document}"])

with open(asset_metrics_tex, "w") as f:
    f.write("\n".join(tex_lines))

print(f"Asset-level fuzzy metrics written to:\n  {asset_metrics_csv}\n  {asset_metrics_tex}")

def run_grid(rho, beta):
    res_list = []
    
    feas_JKPT = np.where((E_P_universe >= eq_E_P * rho) & (Risk_omega_sq_universe <= eq_Risk_sq * beta))[0]
    feas_VB13 = np.where((E_P_universe >= eq_E_P * rho) & (Risk_omega_universe <= np.sqrt(eq_Risk_sq) * beta))[0]
    feas_LGY  = np.where((E_LGY_universe >= eq_E_LGY * rho) & (Var_LGY_universe <= eq_Var_LGY * beta))[0]
    
    def get_oos(w, method, feas_cnt):
        if len(feas_cnt) == 0 or w is None:
            df = {'Method': method, 'Feasible': 0}
            for i in range(1, num_assets + 1):
                df[f'W{i}'] = np.nan
            df.update({'Mean': np.nan, 'Skew': np.nan, 'Sharpe': np.nan})
            return pd.DataFrame([df])
        
        ret = test_mat @ w
        mu = np.mean(ret) * 100
        vol = np.std(ret, ddof=1) * 100
        sk = skew(ret)
        sh = mu / (vol + 1e-12)
        
        df = {'Method': method, 'Feasible': len(feas_cnt)}
        for i in range(1, num_assets + 1):
            df[f'W{i}'] = w[i-1]
        df.update({'Mean': mu, 'Skew': sk, 'Sharpe': sh})
        return pd.DataFrame([df])

    # VB13
    if len(feas_VB13) > 0:
        Pd_f = Port_d[feas_VB13, :]
        Pu_f = Port_u[feas_VB13, :]
        E_f  = E_P_universe[feas_VB13]
        R_f  = Risk_omega_universe[feas_VB13]
        
        mu3 = 0.5 * (np.sum((Pd_f - E_f[:, None])**3 * int_weights, axis=1) +
                     np.sum((Pu_f - E_f[:, None])**3 * int_weights, axis=1))
        best_idx = np.argmax(mu3 / (R_f**3 + 1e-9))
        w_star = W_universe[feas_VB13][best_idx]
        res_list.append(get_oos(w_star, "VB13", feas_VB13))
    else:
        res_list.append(get_oos(None, "VB13", []))
        
    # LGY15
    if len(feas_LGY) > 0:
        Pd_f = Port_d[feas_LGY, :]
        Pu_f = Port_u[feas_LGY, :]
        E_f  = E_LGY_universe[feas_LGY]
        
        S_LGY = np.sum((Pd_f - E_f[:, None])**3 * lgy_eff_w, axis=1) + \
                np.sum((Pu_f - E_f[:, None])**3 * lgy_eff_w, axis=1)
        best_idx = np.argmax(S_LGY)
        w_star = W_universe[feas_LGY][best_idx]
        res_list.append(get_oos(w_star, "LGY15", feas_LGY))
    else:
        res_list.append(get_oos(None, "LGY15", []))
        
    v_vals = np.linspace(0, 1, 11)
    
    if len(feas_JKPT) > 0:
        W_f = W_universe[feas_JKPT, :]
        Pd_f = Port_d[feas_JKPT, :]
        Pu_f = Port_u[feas_JKPT, :]
        
        idx_100 = np.argmin(np.abs(alphas - 1.00))
        m_l = Pd_f[:, idx_100]
        m_u = Pu_f[:, idx_100]
        
        # JKPT1
        for a_val in [0.10, 0.25]:
            ia  = np.argmin(np.abs(alphas - a_val))
            i1a = np.argmin(np.abs(alphas - (1 - a_val)))
            i05 = np.argmin(np.abs(alphas - 0.50))
            
            S_GM1_d = (Pd_f[:, i1a] + Pd_f[:, ia] - 2 * Pd_f[:, i05]) / (Pd_f[:, i1a] - Pd_f[:, ia] + 1e-9)
            S_GM1_u = (Pu_f[:, ia] + Pu_f[:, i1a] - 2 * Pu_f[:, i05]) / (Pu_f[:, ia] - Pu_f[:, i1a] + 1e-9)
            
            S_in = 0.5 * (S_GM1_d + S_GM1_u)
            S_out = ((Pd_f[:, ia] - m_l) + (Pu_f[:, ia] - m_u)) / (Pu_f[:, ia] - Pd_f[:, ia] + 1e-9)
            
            for v in v_vals:
                w_star = W_f[np.argmax(v * S_out + (1 - v) * S_in)]
                lab = f"JKPT1_{{{a_val:.2f}}} v={v:.1f}"
                res_list.append(get_oos(w_star, lab, feas_JKPT))
                
        # JKPT2
        half_idx = np.where(alphas <= 0.5)[0]
        p_vals = alphas[half_idx]
        one_m_idx = [np.argmin(np.abs(alphas - (1 - pv))) for pv in p_vals]
        
        i05_mat = np.tile(Pd_f[:, np.argmin(np.abs(alphas - 0.50))], (len(half_idx), 1)).T
        
        num_d = Pd_f[:, one_m_idx] + Pd_f[:, half_idx] - 2 * i05_mat
        den_d = Pd_f[:, one_m_idx] - Pd_f[:, half_idx]
        
        num_u = Pu_f[:, half_idx] + Pu_f[:, one_m_idx] - 2 * i05_mat
        den_u = Pu_f[:, half_idx] - Pu_f[:, one_m_idx]
        
        hw = np.tile(int_weights[half_idx], (Pd_f.shape[0], 1))
        
        S_in_int = 0.5 * (np.sum(num_d * hw, axis=1) / (np.sum(den_d * hw, axis=1) + 1e-9) +
                          np.sum(num_u * hw, axis=1) / (np.sum(den_u * hw, axis=1) + 1e-9))
        
        m_l_mat = np.tile(m_l, (len(half_idx), 1)).T
        m_u_mat = np.tile(m_u, (len(half_idx), 1)).T
        
        S_out_num = (Pd_f[:, half_idx] - m_l_mat) + (Pu_f[:, half_idx] - m_u_mat)
        S_out_den = Pu_f[:, half_idx] - Pd_f[:, half_idx]
        S_out_int = np.sum(S_out_num * hw, axis=1) / (np.sum(S_out_den * hw, axis=1) + 1e-9)
        
        for v in v_vals:
            w_star = W_f[np.argmax(v * S_out_int + (1 - v) * S_in_int)]
            lab = f"JKPT2 v={v:.1f}"
            res_list.append(get_oos(w_star, lab, feas_JKPT))
            
    else:
        for a_val in [0.10, 0.25]:
            for v in v_vals:
                lab = f"JKPT1_{{{a_val:.2f}}} v={v:.1f}"
                res_list.append(get_oos(None, lab, []))
        for v in v_vals:
            lab = f"JKPT2 v={v:.1f}"
            res_list.append(get_oos(None, lab, []))
            
    return pd.concat(res_list, ignore_index=True)

def generate_latex(df, rho, beta, filepath):
    comp_df = df[df['Method'].isin(['VB13', 'LGY15'])]
    max_mean = comp_df['Mean'].max()
    max_skew = comp_df['Skew'].max()
    max_sharpe = comp_df['Sharpe'].max()
    
    tex_df = df.copy()
    tex_df['Feasible'] = tex_df['Feasible'].astype(str)
    
    for i in range(1, num_assets + 1):
        tex_df[f'W{i}'] = df[f'W{i}'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "NA")
        
    tex_df['Method'] = tex_df['Method'].str.replace("_", "\\_")
    
    for idx, row in df.iterrows():
        if "JKPT" not in str(row['Method']) or pd.isnull(row['Mean']):
            tex_df.at[idx, 'Mean'] = f"{row['Mean']:.3f}" if pd.notnull(row['Mean']) else "NA"
            tex_df.at[idx, 'Skew'] = f"{row['Skew']:.3f}" if pd.notnull(row['Skew']) else "NA"
            tex_df.at[idx, 'Sharpe'] = f"{row['Sharpe']:.3f}" if pd.notnull(row['Sharpe']) else "NA"
            continue
            
        b_mean = row['Mean'] > max_mean
        b_skew = row['Skew'] > max_skew
        b_sharp = row['Sharpe'] > max_sharpe
        
        m_str = f"{row['Mean']:.3f}"
        s_str = f"{row['Skew']:.3f}"
        sh_str = f"{row['Sharpe']:.3f}"
        
        if b_mean and b_skew and b_sharp:
            m_str = f"\\cellcolor{{magenta!30}}{m_str}"
            s_str = f"\\cellcolor{{magenta!30}}{s_str}"
            sh_str = f"\\cellcolor{{magenta!30}}{sh_str}"
        else:
            if b_mean: m_str = f"\\cellcolor{{green!30}}{m_str}"
            if b_skew: s_str = f"\\cellcolor{{blue!20}}{s_str}"
            if b_sharp: sh_str = f"\\cellcolor{{yellow!40}}{sh_str}"
            
        tex_df.at[idx, 'Mean'] = m_str
        tex_df.at[idx, 'Skew'] = s_str
        tex_df.at[idx, 'Sharpe'] = sh_str

    col_str = "l " + "r " * (2 + num_assets + 3 - 1)
    header = " & ".join(["Method", "Feas."] + tickers + ["Mean", "Skew", "Sharpe"])
    
    lines = [
        "\\documentclass{article}",
        "\\usepackage[landscape, margin=0.5in]{geometry}",
        "\\usepackage{booktabs, longtable}",
        "\\usepackage[table]{xcolor}",
        "\\begin{document}",
        f"\\section*{{Results: $\\rho={rho:.2f}$, $\\beta={beta:.2f}$}}",
        "\\textbf{Legend:} \\colorbox{green!30}{Beats Comp. Mean} | \\colorbox{blue!20}{Beats Comp. Skew} | \\colorbox{yellow!40}{Beats Comp. Sharpe} | \\colorbox{magenta!30}{Beats ALL}",
        "\\vspace{1em}",
        f"\\begin{{longtable}}{{{col_str}}}",
        "\\toprule",
        header + " \\\\ \\midrule"
    ]
    
    for _, row in tex_df.iterrows():
        lines.append(" & ".join([str(x) for x in row.values]) + " \\\\")
        
    lines.extend(["\\bottomrule", "\\end{longtable}", "\\end{document}"])
    
    with open(filepath, "w") as f:
        f.write("\n".join(lines))

def plot_sharpe_profile(df, rho, beta, out_path):
    vb = df[df['Method'] == 'VB13']
    lgy = df[df['Method'] == 'LGY15']
    
    if vb.empty or lgy.empty or pd.isnull(vb['Sharpe'].iloc[0]) or pd.isnull(lgy['Sharpe'].iloc[0]):
        print("No VB13 or LGY15 valid row for this (rho,beta), skipping plot.")
        return
        
    sharpe_vb = vb['Sharpe'].iloc[0]
    sharpe_lgy = lgy['Sharpe'].iloc[0]
    
    jkpt_df = df[df['Method'].str.contains("JKPT")].copy()
    if jkpt_df.empty or jkpt_df['Sharpe'].isnull().all():
        print("No JKPT rows for this (rho,beta), skipping plot.")
        return
        
    jkpt_df['v_num'] = jkpt_df['Method'].str.extract(r'v=([0-9\.]+)').astype(float)
    
    def get_group(m):
        if "JKPT1_{0.10}" in m: return 1
        if "JKPT1_{0.25}" in m: return 2
        return 3
    
    jkpt_df['group'] = jkpt_df['Method'].apply(get_group)
    jkpt_df = jkpt_df.sort_values(['group', 'v_num'])
    
    labels = []
    for _, row in jkpt_df.iterrows():
        if row['group'] == 1: labels.append(f"α=0.10, v={row['v_num']:.1f}")
        elif row['group'] == 2: labels.append(f"α=0.25, v={row['v_num']:.1f}")
        else: labels.append(f"v={row['v_num']:.1f}")
        
    jkpt_df['Label'] = labels
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(jkpt_df['Label'], jkpt_df['Sharpe'], marker='o', color='steelblue', label=r'JKPT Variants ($\alpha$ and $v$)')
    ax.axhline(y=sharpe_vb, color='red', linestyle='dashed', linewidth=1.5, label=f'VB13 (Sharpe = {sharpe_vb:.3f})')
    ax.axhline(y=sharpe_lgy, color='darkgreen', linestyle='dotted', linewidth=1.5, label=f'LGY15 (Sharpe = {sharpe_lgy:.3f})')
    ax.legend(loc='best', fontsize=15)
    ax.set_title(rf"Sharpe profile, $\rho={rho:.2f}$, $\beta={beta:.2f}$, $r_f=0$", fontsize=20)
    ax.set_ylabel(r"Sharpe ($r_f=0$)", fontsize=20)
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

print("Running Grid Search, Generating LaTeX & Plots...")
rho_vals = [0.95, 1.00, 1.05]
beta_vals = [0.95, 1.00, 1.05]
all_combos = []

for r in rho_vals:
    for b in beta_vals:
        print(f" Processing Rho={r:.2f}, Beta={b:.2f}")
        df = run_grid(r, b)
        
        combo_dir = os.path.join(sub_report_dir, f"Rho{r:.2f}_Beta{b:.2f}")
        os.makedirs(combo_dir, exist_ok=True)
        
        tex_path = os.path.join(combo_dir, f"Table_Rho{r:.2f}_Beta{b:.2f}.tex")
        generate_latex(df, r, b, tex_path)
        
        plot_path = os.path.join(combo_dir, f"Sharpe_Profile_Rho{r:.2f}_Beta{b:.2f}.png")
        plot_sharpe_profile(df, r, b, plot_path)
        
        all_combos.append(df)

print(f"Finished! Check the folder: {main_dir}")
