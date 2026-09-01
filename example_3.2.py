import numpy as np
import pandas as pd
from scipy.stats import beta, norm, uniform
from scipy.integrate import quad
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# --- Class for scaled distributions ---
class DistInfo:
    def __init__(self, dist_name, params, lower_bound, upper_bound):
        self.min_val = lower_bound
        self.max_val = upper_bound
        
        if dist_name == "beta":
            self.q_fn = lambda p: lower_bound + (upper_bound - lower_bound) * \
                                  beta.ppf(p, params['shape1'], params['shape2'])
            self.p_fn = lambda x: beta.cdf((x - lower_bound)/(upper_bound - lower_bound),
                                          params['shape1'], params['shape2'])
            self.d_fn = lambda x: beta.pdf((x - lower_bound)/(upper_bound - lower_bound),
                                          params['shape1'], params['shape2']) / (upper_bound - lower_bound)
        
        elif dist_name == "uniform":
            self.q_fn = lambda p: lower_bound + (upper_bound - lower_bound) * uniform.ppf(p)
            self.p_fn = lambda x: uniform.cdf((x - lower_bound)/(upper_bound - lower_bound))
            self.d_fn = lambda x: uniform.pdf((x - lower_bound)/(upper_bound - lower_bound)) / (upper_bound - lower_bound)
        
        elif dist_name == "truncated_normal":
            mu = params['mean']
            sigma = params['sd']
            Phi_lower = norm.cdf(lower_bound, loc=mu, scale=sigma)
            Phi_upper = norm.cdf(upper_bound, loc=mu, scale=sigma)
            norm_const = Phi_upper - Phi_lower
            self.q_fn = lambda p: norm.ppf(p * norm_const + Phi_lower, loc=mu, scale=sigma)
            self.p_fn = lambda x: np.clip((norm.cdf(x, loc=mu, scale=sigma) - Phi_lower) / norm_const, 0, 1)
            self.d_fn = lambda x: np.where((x >= lower_bound) & (x <= upper_bound),
                                           norm.pdf(x, loc=mu, scale=sigma)/norm_const, 0)

# --- Skewness ---
def calc_GM_integral_individual(q_fn):
    num_integral, _ = quad(lambda a: q_fn(1-a)+q_fn(a)-2*q_fn(0.5), 0, 0.5)
    den_integral, _ = quad(lambda a: q_fn(1-a)-q_fn(a), 0, 0.5)
    return num_integral/den_integral if abs(den_integral)>1e-9 else 0

def calc_VB13_skew(q_XL_fn, q_XR_fn):
    E, _ = quad(lambda a: 0.5*(q_XL_fn(a)+q_XR_fn(1-a)), 0, 1)
    omega, _ = quad(lambda a: q_XR_fn(1-a)-q_XL_fn(a), 0, 1)
    mu3L, _ = quad(lambda a: (q_XL_fn(a)-E)**3, 0, 1)
    mu3R, _ = quad(lambda a: (q_XR_fn(1-a)-E)**3, 0, 1)
    mu3 = 0.5*(mu3L+mu3R)
    return mu3/(omega**3) if abs(omega)>1e-9 else 0


def analyze_skewness_pair(dist_name_L, params_L, l, u_m,
                          dist_name_R, params_R, o_m, r):

    dist_L = DistInfo(dist_name_L, params_L, l, u_m)
    dist_R = DistInfo(dist_name_R, params_R, o_m, r)

    x_min_plot = l - 0.05*(r-l)
    x_max_plot = r + 0.05*(r-l)
    x_seq = np.linspace(x_min_plot, x_max_plot, 800)

    pdf_L = np.array([dist_L.d_fn(x) for x in x_seq])
    pdf_R = np.array([dist_R.d_fn(x) for x in x_seq])
    cdf_L = np.array([dist_L.p_fn(x) for x in x_seq])
    cdf_R = np.array([dist_R.p_fn(x) for x in x_seq])

    xi_val = np.array([
        dist_L.p_fn(x) if l <= x < u_m else
        1 if u_m <= x <= o_m else
        1 - dist_R.p_fn(x) if o_m < x <= r else
        0
        for x in x_seq
    ])

    # # --- Selected quantile points ---
    # # median = dist_L.q_fn(0.5)
    # q75_R = dist_R.q_fn(0.75)
    # q90_R = dist_R.q_fn(0.90)

    # xticks = [l, m_value, q75_R, q90_R, r]
    # xlabels = [
    #     r"$l~(100)$",
    #     r"$m$",
    #     r"$Q_R~(0.75)$",
    #     r"$Q_R~(0.90)$",
    #     r"$r~(110)$"
    # ]

    q10_L = dist_L.q_fn(0.1)
    q25_L = dist_L.q_fn(0.25)
    q50_L = dist_L.q_fn(0.5)
    q75_L = dist_L.q_fn(0.75)
    q90_L = dist_L.q_fn(0.90)

    m_value = u_m

    q10_R = dist_R.q_fn(0.1)
    q25_R = dist_R.q_fn(0.25)
    q50_R = dist_R.q_fn(0.5)
    q75_R = dist_R.q_fn(0.75)
    q90_R = dist_R.q_fn(0.90)

    xticks = [
    l,
    q10_L, q25_L, q50_L, q75_L, q90_L,
    m_value,
    q10_R, q25_R, q50_R, q75_R, q90_R,
    r
]
    
    xlabels = [
        r"$l~(100)$",
        '',
        '',
        '',
        '',
        '',
        r"$m$",
        '',
        '',
        # r"$Q_R~(0.50)$",
        '',
        r"$Q_R~(0.75)$",
        r"$Q_R~(0.90)$",
        r"$r~(110)$"
    ]

    # --- Plot ---
    # fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig, axes = plt.subplots(3, 1, figsize=(8, 18))


    # PDF
    axes[0].plot(x_seq, pdf_L, color='blue', lw=2, label=r'$f_{X_L}(x)$')
    axes[0].plot(x_seq, pdf_R, color='red', lw=2, linestyle='--', label=r'$f_{X_R}(x)$')
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xlabels)
    axes[0].set_title("Probability Density Functions", fontsize=15)
    axes[0].legend(frameon=True, edgecolor="black")
    axes[0].set_ylabel("PDF")


    # CDF
    axes[1].plot(x_seq, cdf_L, color='blue', lw=2, label=r'$F_{X_L}(x)$')
    axes[1].plot(x_seq, cdf_R, color='red', lw=2, linestyle='--', label=r'$F_{X_R}(x)$')
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xlabels)
    axes[1].set_title("Cumulative Distribution Functions", fontsize=15)
    axes[1].legend(frameon=True, edgecolor="black")
    axes[1].set_ylabel("CDF")


    # Membership
    axes[2].plot(x_seq, xi_val, color='darkgreen', lw=2, label=r'$\xi(x)$')
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(xlabels)
    axes[2].set_title(r"Membership Function $\xi(x)$", fontsize=15)
    axes[2].legend(frameon=True, edgecolor="black")
    axes[2].set_ylabel(r"$\xi(x)$")


    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.15)
    plt.subplots_adjust(hspace=0.45)

    plt.savefig("Example_3_2_Selected_Quantiles.pdf")
    plt.show()
    plt.close()

    # --- Skewness ---
    GM_XL = calc_GM_integral_individual(dist_L.q_fn)
    GM_XR = calc_GM_integral_individual(dist_R.q_fn)
    VB13 = calc_VB13_skew(dist_L.q_fn, dist_R.q_fn)

    return pd.DataFrame({
        "Coefficient": ["GM_Skewness_XL", "GM_Skewness_XR", "VB13_Skewness"],
        "Value": [GM_XL, GM_XR, VB13]
    })


# --- Example 3.2 ---
result = analyze_skewness_pair(
    dist_name_L="beta",
    params_L={"shape1": 0.5, "shape2": 2},
    l=100,
    u_m=102,
    dist_name_R="beta",
    params_R={"shape1": 0.5, "shape2": 2},
    o_m=102,
    r=110
)

print(result)