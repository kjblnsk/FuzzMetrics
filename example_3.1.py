import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

plt.style.use("seaborn-v0_8-whitegrid")

XL_x = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 1.4, 1.6, 2.5, 2.8, 3.0])
XL_y = np.array([0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.7, 0.85, 0.9, 1.0])

XR_x = np.array([7.6, 7.2, 7.1, 6.3, 6.0, 5.1, 4.8, 3.7, 3.4, 3.0])
XR_y = np.array([0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0])

idx = np.argsort(XR_x)
XR_x = XR_x[idx]
XR_y = XR_y[idx]

F_XL = interp1d(XL_x, XL_y, kind="linear",
                bounds_error=False, fill_value=(0, 1))

F_XR_comp = interp1d(XR_x, XR_y, kind="linear",
                     bounds_error=False, fill_value=(0, 1))

def F_XR(x):
    return 1 - F_XR_comp(x)

# -------- funkcja kwantylowa dla F_XR --------
def q_fn_R(p):
    x_dense = np.linspace(XR_x.min(), XR_x.max(), 5000)
    F_vals = F_XR(x_dense)
    return np.interp(p, F_vals, x_dense)

# -------- funkcja kwantylowa dla F_XL --------
def q_fn_L(p):
    x_dense = np.linspace(XL_x.min(), XL_x.max(), 5000)
    F_vals = F_XL(x_dense)
    return np.interp(p, F_vals, x_dense)

def piecewise_pdf(x_vals, x_points, y_points, sign=1):
    result = np.zeros_like(x_vals)
    for i in range(len(x_points) - 1):
        x0, x1 = x_points[i], x_points[i+1]
        slope = (y_points[i+1] - y_points[i]) / (x1 - x0)
        mask = (x_vals >= x0) & (x_vals < x1)
        result[mask] = sign * slope
    return result

def f_XL(x):
    return piecewise_pdf(x, XL_x, XL_y, sign=1)

def f_XR(x):
    return piecewise_pdf(x, XR_x, XR_y, sign=-1)

l = XL_x.min()
u_m = XL_x.max()
o_m = XR_x.min()
r = XR_x.max()

def xi(x):
    x = np.array(x)
    result = np.zeros_like(x)

    left = (x >= l) & (x <= u_m)
    middle = (x > u_m) & (x < o_m)
    right = (x >= o_m) & (x <= r)

    result[left] = F_XL(x[left])
    result[middle] = 1
    result[right] = F_XR_comp(x[right])

    return result

x_plot_min = min(l, o_m) - 0.5
x_plot_max = max(u_m, r) + 0.5
x_seq = np.linspace(x_plot_min, x_plot_max, 2000)

display_x_points = np.array([0.1, 0.8, 1.6, 2.5, 3.0,
                             3.7, 4.8, 6.0, 7.1, 7.6])
display_labels = [f"{x:.1f}" for x in display_x_points]
custom_y_breaks = [0, 0.25, 0.5, 0.75, 1.0]

fig, axes = plt.subplots(3, 1, figsize=(8, 18))

# ------------------ MEMBERSHIP ------------------
axes[0].plot(x_seq, xi(x_seq),
             color="darkgreen", linewidth=2, label=r'$\xi(x)$')

axes[0].set_title(r"Membership Function $\xi(x)$", fontsize=15)
axes[0].set_xlabel("x")
axes[0].set_ylabel(r"$\xi(x)$")
axes[0].set_xlim(x_plot_min, x_plot_max)
axes[0].set_ylim(-0.05, 1.05)
axes[0].set_xticks(display_x_points)
axes[0].set_xticklabels(display_labels)
axes[0].set_yticks(custom_y_breaks)
axes[0].legend(frameon=True, edgecolor="black")

# ------------------ CDF ------------------
mask_blue_main = x_seq <= 3.0
axes[1].plot(x_seq[mask_blue_main],
             F_XL(x_seq[mask_blue_main]),
             color="blue", linewidth=2)

mask_blue_tail = x_seq >= 3.0
axes[1].plot(x_seq[mask_blue_tail],
             np.ones_like(x_seq[mask_blue_tail]),
             color="blue", linewidth=2,
             label=r"$F_{X_L}(x)$")

mask_red_left = x_seq <= 3.0
axes[1].plot(x_seq[mask_red_left],
             np.zeros_like(x_seq[mask_red_left]),
             color="red", linewidth=2, linestyle='--')

mask_red_main = (x_seq >= 3.0) & (x_seq <= 7.6)
axes[1].plot(x_seq[mask_red_main],
             F_XR(x_seq[mask_red_main]),
             color="red", linewidth=2, linestyle='--')

mask_red_right = x_seq >= 7.6
axes[1].plot(x_seq[mask_red_right],
             np.ones_like(x_seq[mask_red_right]),
             color="red", linewidth=2,
             label=r"$F_{X_R}(x)$", linestyle='--')

axes[1].set_title("Cumulative Distribution Functions", fontsize=15)
axes[1].set_xlabel("x")
axes[1].set_ylabel("CDF")
axes[1].set_xlim(x_plot_min, x_plot_max)
axes[1].set_ylim(-0.05, 1.05)
axes[1].set_xticks(display_x_points)
axes[1].set_xticklabels(display_labels)
axes[1].set_yticks(custom_y_breaks)
axes[1].legend(frameon=True, edgecolor="black")

# ------------------ PDF ------------------
pdf_XL_vals = f_XL(x_seq)
pdf_XR_vals = f_XR(x_seq)

axes[2].plot(x_seq[pdf_XL_vals > 0],
             pdf_XL_vals[pdf_XL_vals > 0],
             label=r"$f_{X_L}(x)$",
             color="blue", linewidth=2)

axes[2].plot(x_seq[pdf_XR_vals > 0],
             pdf_XR_vals[pdf_XR_vals > 0],
             label=r"$f_{X_R}(x)$",
             color="red", linewidth=2, linestyle='--')

# ---- pionowe linie ----
axes[2].vlines(l, 0,
               f_XL(np.array([l + 1e-4]))[0],
               color="blue", linewidth=2)

axes[2].vlines(u_m, 0,
               f_XL(np.array([u_m - 1e-4]))[0],
               color="blue", linewidth=2)

axes[2].vlines(r, 0,
               f_XR(np.array([r - 1e-4]))[0],
               color="red", linewidth=2, linestyle='--')

# ---- dodatkowa czerwona przerywana linia w x = m ----
axes[2].vlines(u_m, 0, 0.25, color="red", linewidth=2, linestyle='--')

# ---- krótkie poziome linie ----
axes[2].hlines(0, x_plot_min, l, color="blue", linewidth=2, linestyle='-')   # niebieska ciągła
axes[2].hlines(0, r, x_plot_max, color="red", linewidth=2, linestyle='--')      # czerwona przerywana

# # ---- kwantyle ----
# q50_R = q_fn_R(0.5)
# q10_R = q_fn_R(0.10)  # zmienione z 0.90 na 0.10

# # tick_positions = [l, u_m, q50_R, q10_R, r]
# tick_positions = [l, u_m, q50_R, r]
# tick_labels = [
#     r"$l$",
#     r"$m$",
#     r"$Q_R(0.50)$",
#     # r"$Q_R(0.10)$",
#     r"$r$"
# ]

q10_L = q_fn_L(0.1)
q25_L = q_fn_L(0.25)
q50_L = q_fn_L(0.5)
q75_L = q_fn_L(0.75)
q90_L = q_fn_L(0.90)

m_value = u_m

q10_R = q_fn_R(0.1)
q25_R = q_fn_R(0.25)
q50_R = q_fn_R(0.5)
q75_R = q_fn_R(0.75)
q90_R = q_fn_R(0.90)

tick_positions = [l, q_fn_L(0.1), q_fn_L(0.25), q_fn_L(0.5), q_fn_L(0.75), q_fn_L(0.9), m_value, q10_R, q25_R, q50_R, q75_R, q90_R, r]
tick_labels = [
    r"$l~(100)$",
    '',
    '',
    r"$Q_L~(0.50)$",
    '',
    '',
    r"$m$",
    '',
    r"$Q_R~(0.25)$",
    r"$Q_R~(0.50)$",
    r"$Q_R~(0.75)$",
    '',
    r"$r~(110)$"
]

# sortowanie rosnąco
sorted_ticks = sorted(zip(tick_positions, tick_labels), key=lambda x: x[0])
axes[2].set_xticks([t[0] for t in sorted_ticks])
axes[2].set_xticklabels([t[1] for t in sorted_ticks])

axes[2].set_title("Probability Density Functions", fontsize=15)
axes[2].set_xlabel("x")
axes[2].set_ylabel("PDF")
axes[2].set_xlim(x_plot_min, x_plot_max)
axes[2].set_ylim(-0.05, 1.05)
axes[2].set_yticks(custom_y_breaks)
axes[2].legend(frameon=True, edgecolor="black")

plt.tight_layout()
plt.subplots_adjust(hspace=0.45)

folder_name = "JKPT_Fuzzy_to_RV_Example"
os.makedirs(folder_name, exist_ok=True)

file_path = os.path.join(folder_name,
                         "combined_JKPT_final_python.png")

plt.savefig(file_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white")

print("Combined plot saved to:", file_path)
plt.show()