#!/usr/bin/env python3
"""
Master supplementary plotting script — corrected genus colour scheme
===================================================================

Outputs
-------
Figure_S1_frequency_sweep_replicates.png/.pdf
Figure_S2_dehydration_control.png/.pdf
Figure_S3_compliance_vs_stress.png/.pdf

Palette matches the main master figures:
  Phalaenopsis = purple
  Cattleya     = orange
  Cymbidium    = green
  Dendrobium   = blue
  Laelia       = pink
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, to_hex

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "supplemental_outputs_corrected"
OUT.mkdir(parents=True, exist_ok=True)

# =====================================================================
# GLOBAL STYLE + COLOUR MAP
# =====================================================================

COLORS = {
    "Phalaenopsis": "#7E57C2",  # purple
    "Cattleya":     "#E69F00",  # orange
    "Cymbidium":    "#009E73",  # green
    "Dendrobium":   "#0072B2",  # blue
    "Laelia":       "#CC79A7",  # pink
}

MARKERS = {
    "Phalaenopsis": "s",
    "Cattleya":     "^",
    "Cymbidium":    "D",
    "Dendrobium":   "v",
    "Laelia":       "o",
}

ORDER = ["Phalaenopsis", "Cattleya", "Cymbidium", "Dendrobium", "Laelia"]


def darken(hex_color: str, factor: float = 0.70) -> str:
    rgb = np.array(to_rgb(hex_color))
    return to_hex(np.clip(rgb * factor, 0, 1))


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# =====================================================================
# SUPPLEMENTARY FIGURE S1 — FREQUENCY SWEEP REPLICATES
# =====================================================================

CATTLEYA_S1 = {
    "omega": np.array([100, 55.8, 31.2, 17.4, 9.73, 5.43, 3.03, 1.69, 0.946,
                        0.528, 0.295, 0.165, 0.092, 0.0514, 0.0287, 0.016, 0.00895, 0.005]),
    "Gp": np.array([858000, 804000, 755000, 706000, 659000, 611000, 565000, 517000, 475000,
                     441000, 381000, 332000, 287000, 243000, 197000, 161000, 122000, 92900]),
    "Gpp": np.array([143000, 140000, 139000, 137000, 137000, 135000, 135000, 132000, 136000,
                      138000, 132000, 140000, 126000, 118000, 110000, 98170, 85630, 82030]),
    "gamma0": "0.05%",
}
CATTLEYA_S2 = {
    "omega": np.array([100, 56.2, 31.6, 17.8, 10, 5.62, 3.16, 1.78, 1.0]),
    "Gp": np.array([494850, 469730, 432280, 391000, 349030, 314820, 277740, 238030, 216690]),
    "Gpp": np.array([123730, 121960, 119270, 114970, 110520, 104350, 100680, 90645, 86715]),
    "gamma0": "0.015%",
}
CYMBIDIUM_S1 = {
    "omega": np.array([100, 55.8, 31.2, 17.4, 9.73, 5.43, 3.03, 1.69, 0.946,
                        0.528, 0.295, 0.165, 0.092, 0.0514, 0.0287, 0.016, 0.00895, 0.005]),
    "Gp": np.array([49246, 40074, 31671, 24413, 18359, 13484, 9687.9, 6841.7, 4763.6,
                     3310.2, 2304, 1616, 1147.9, 823.01, 609.02, 458.63, 336.56, 262.8]),
    "Gpp": np.array([32608, 27628, 23395, 19542, 16088, 13003, 10260, 7927.2, 5983.8,
                      4454.2, 3269.9, 2380.5, 1730.2, 1259.3, 923.65, 676.31, 506.89, 376.57]),
    "gamma0": "0.25%",
}
CYMBIDIUM_S2 = {
    "omega": np.array([100, 55.7, 31.1, 17.3, 9.64, 5.37, 3.0, 1.67,
                        0.93, 0.518, 0.289, 0.161, 0.0897, 0.05]),
    "Gp": np.array([39162, 35274, 29626, 24262, 19394, 15132, 11738, 8941.7,
                     6699, 5013.7, 3819.1, 2870.1, 2137.2, 1556.2]),
    "Gpp": np.array([27161, 22952, 19064, 16148, 13631, 11335, 9430.3, 7668.7,
                      5954.7, 4950.7, 3935.5, 3093.6, 2474.6, 1920.4]),
    "gamma0": "0.1%",
}
PHALAENOPSIS_S1 = {
    "omega": np.array([100, 55.8, 31.2, 17.4, 9.73, 5.43, 3.03, 1.69, 0.946,
                        0.528, 0.295, 0.165, 0.092, 0.0514, 0.0287, 0.016, 0.00895, 0.005]),
    "Gp": np.array([41768, 33004, 27165, 22578, 18588, 15231, 12429, 10138, 8217.1,
                     6604.4, 5314, 4304.6, 3482.8, 2798, 2213, 1758, 1385, 1162]),
    "Gpp": np.array([14961, 13509, 12139, 10818, 9666.1, 8547.7, 7486.7, 6503.6, 5619.9,
                      4711.4, 4050.6, 3384.1, 2828.3, 2358, 1966, 1599, 1368, 1120]),
    "gamma0": "0.25%",
}
PHALAENOPSIS_S2 = {
    "omega": np.array([100, 82.9, 68.7, 56.9, 47.1, 39.1, 32.4, 26.8, 22.2,
                        18.4, 15.3, 12.6, 10.5, 8.69, 7.2, 5.96, 4.94, 4.09,
                        3.39, 2.81, 2.33, 1.93, 1.6, 1.33, 1.1, 0.91, 0.754,
                        0.625, 0.518, 0.429, 0.356, 0.295, 0.244, 0.202, 0.168,
                        0.139, 0.115, 0.0954, 0.0791, 0.0655, 0.0543, 0.045,
                        0.0373, 0.0309, 0.0256, 0.0212, 0.0176, 0.0146, 0.0121]),
    "Gp": np.array([64512, 61995, 59608, 57020, 54595, 52094, 49611, 47174, 44764,
                     42397, 40085, 37813, 35592, 33449, 31364, 29342, 27437, 25569,
                     23786, 22073, 20486, 18967, 17489, 16103, 14902, 13592, 12431,
                     11476, 10447, 9327.9, 8655.5, 7754.2, 7057, 6278.5, 5711.7,
                     5175.4, 4673.1, 4228.1, 3728.5, 3456.7, 2990.2, 2731.9,
                     2440.5, 2198.1, 2016.9, 1808.5, 1576.6, 1448.9, 1309.3]),
    "Gpp": np.array([19720, 19971, 20297, 19842, 19781, 19817, 19579, 19344, 19092,
                      18776, 18424, 18040, 17601, 17139, 16650, 16130, 15569, 15023,
                      14450, 13852, 13289, 12686, 12097, 11498, 10938, 10357, 9756.3,
                      9296.7, 8679.2, 8200.4, 7636.3, 7198.3, 6822.9, 6280.3, 5841.1,
                      5456.7, 5058.1, 4584.4, 4400.1, 3975.5, 3685.6, 3348.1,
                      3096.8, 2857.8, 2654.8, 2449.2, 2291.7, 2049.4, 1854.1]),
    "gamma0": "0.1%",
}


def find_crossover(omega, Gp, Gpp):
    y = Gp - Gpp
    changes = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(changes) == 0:
        return None
    i = changes[0]
    x1, x2 = np.log10(omega[i]), np.log10(omega[i + 1])
    y1, y2 = y[i], y[i + 1]
    if y2 == y1:
        return float(omega[i])
    x0 = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
    return float(10 ** x0)


genera = [
    ("Cattleya", CATTLEYA_S1, CATTLEYA_S2),
    ("Cymbidium", CYMBIDIUM_S1, CYMBIDIUM_S2),
    ("Phalaenopsis", PHALAENOPSIS_S1, PHALAENOPSIS_S2),
]

fig, axes = plt.subplots(3, 2, figsize=(10, 10.5))
for row, (genus, s1, s2) in enumerate(genera):
    ax_mod = axes[row, 0]
    ax_tan = axes[row, 1]
    c1 = COLORS[genus]
    c2 = darken(c1, 0.72)

    ax_mod.loglog(s1["omega"], s1["Gp"], color=c1, linewidth=1.8,
                  marker="s", markersize=4, markeredgewidth=0.4,
                  markeredgecolor="k", zorder=3)
    ax_mod.loglog(s1["omega"], s1["Gpp"], color=c1, linewidth=1.8,
                  linestyle="--", marker="o", markersize=3.5,
                  markeredgewidth=0.4, markeredgecolor="k", zorder=3)

    ax_mod.loglog(s2["omega"], s2["Gp"], color=c2, linewidth=1.4,
                  marker="^", markersize=4, markeredgewidth=0.4,
                  markeredgecolor="k", zorder=2, alpha=0.9)
    ax_mod.loglog(s2["omega"], s2["Gpp"], color=c2, linewidth=1.4,
                  linestyle="--", marker="v", markersize=3.5,
                  markeredgewidth=0.4, markeredgecolor="k", zorder=2, alpha=0.9)

    for data, col in [(s1, c1), (s2, c2)]:
        oc = find_crossover(data["omega"], data["Gp"], data["Gpp"])
        if oc is not None:
            ax_mod.axvline(oc, color=col, linestyle=":", linewidth=1.0, alpha=0.45, zorder=1)

    ax_mod.set_ylabel("Modulus (Pa)")
    ax_mod.grid(True, which="both", alpha=0.15, linewidth=0.5)
    ax_mod.set_title(f"$\\it{{{genus}}}$", loc="left", fontsize=11, fontweight="bold")

    handles = [
        Line2D([0], [0], color=c1, linewidth=1.8, marker="s", markersize=4,
               markeredgecolor="k", markeredgewidth=0.4,
               label=f"Primary G′ ({s1['gamma0']})"),
        Line2D([0], [0], color=c1, linewidth=1.8, linestyle="--", marker="o",
               markersize=3.5, markeredgecolor="k", markeredgewidth=0.4,
               label="Primary G″"),
        Line2D([0], [0], color=c2, linewidth=1.4, marker="^", markersize=4,
               markeredgecolor="k", markeredgewidth=0.4,
               label=f"Replicate G′ ({s2['gamma0']})"),
        Line2D([0], [0], color=c2, linewidth=1.4, linestyle="--", marker="v",
               markersize=3.5, markeredgecolor="k", markeredgewidth=0.4,
               label="Replicate G″"),
    ]
    ax_mod.legend(handles=handles, loc=("upper left" if genus == "Cattleya" else "lower right"),
                  framealpha=0.9, fontsize=7.5, handlelength=2.5)

    tand_s1 = s1["Gpp"] / s1["Gp"]
    tand_s2 = s2["Gpp"] / s2["Gp"]
    ax_tan.loglog(s1["omega"], tand_s1, color=c1, linewidth=1.8,
                  marker="o", markersize=4, markeredgewidth=0.4,
                  markeredgecolor="k", zorder=3, label=f"Primary ({s1['gamma0']})")
    ax_tan.loglog(s2["omega"], tand_s2, color=c2, linewidth=1.4,
                  marker="^", markersize=4, markeredgewidth=0.4,
                  markeredgecolor="k", zorder=2, alpha=0.9,
                  label=f"Replicate ({s2['gamma0']})")
    ax_tan.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax_tan.set_ylabel("tan δ (—)")
    ax_tan.grid(True, which="both", alpha=0.15, linewidth=0.5)
    ax_tan.legend(loc=("lower left" if genus == "Cattleya" else "upper right"),
                  framealpha=0.9, fontsize=7.5)
    ax_tan.set_title(f"$\\it{{{genus}}}$", loc="left", fontsize=11, fontweight="bold")

    all_omega = np.concatenate([s1["omega"], s2["omega"]])
    xmin = 10 ** (np.floor(np.log10(all_omega.min())) - 0.2)
    xmax = 10 ** (np.ceil(np.log10(all_omega.max())) + 0.2)
    ax_mod.set_xlim(xmin, xmax)
    ax_tan.set_xlim(xmin, xmax)
    ax_tan.set_ylim(0.12, 2.5)

for col in range(2):
    axes[2, col].set_xlabel("Angular frequency ω (rad s⁻¹)")
for row in range(3):
    for col in range(2):
        idx = row * 2 + col
        label = chr(ord('a') + idx)
        ax = axes[row, col]
        if row == 0 and col == 0:
            ax.text(0.97, 0.95, f"({label})", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top", ha="right")
        else:
            ax.text(0.02, 0.95, f"({label})", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")
fig.tight_layout(h_pad=1.5)
fig.savefig(OUT / "Figure_S1_frequency_sweep_replicates.png")
fig.savefig(OUT / "Figure_S1_frequency_sweep_replicates.pdf")
plt.close(fig)

# =====================================================================
# SUPPLEMENTARY FIGURE S2 — DEHYDRATION CONTROL
# =====================================================================

S1 = {
    "omega": np.array([100, 82.9, 68.7, 56.9, 47.1, 39.1, 32.4, 26.8, 22.2,
                        18.4, 15.3, 12.6, 10.5, 8.69, 7.2, 5.96, 4.94, 4.09,
                        3.39, 2.81, 2.33, 1.93, 1.6, 1.33, 1.1, 0.91, 0.754,
                        0.625, 0.518, 0.429, 0.356, 0.295, 0.244, 0.202, 0.168,
                        0.139, 0.115, 0.0954, 0.0791, 0.0655, 0.0543, 0.045,
                        0.0373, 0.0309, 0.0256, 0.0212, 0.0176, 0.0146, 0.0121]),
    "Gp": np.array([64512, 61995, 59608, 57020, 54595, 52094, 49611, 47174, 44764,
                     42397, 40085, 37813, 35592, 33449, 31364, 29342, 27437, 25569,
                     23786, 22073, 20486, 18967, 17489, 16103, 14902, 13592, 12431,
                     11476, 10447, 9327.9, 8655.5, 7754.2, 7057, 6278.5, 5711.7,
                     5175.4, 4673.1, 4228.1, 3728.5, 3456.7, 2990.2, 2731.9,
                     2440.5, 2198.1, 2016.9, 1808.5, 1576.6, 1448.9, 1309.3]),
    "Gpp": np.array([19720, 19971, 20297, 19842, 19781, 19817, 19579, 19344, 19092,
                      18776, 18424, 18040, 17601, 17139, 16650, 16130, 15569, 15023,
                      14450, 13852, 13289, 12686, 12097, 11498, 10938, 10357, 9756.3,
                      9296.7, 8679.2, 8200.4, 7636.3, 7198.3, 6822.9, 6280.3, 5841.1,
                      5456.7, 5058.1, 4584.4, 4400.1, 3975.5, 3685.6, 3348.1,
                      3096.8, 2857.8, 2654.8, 2449.2, 2291.7, 2049.4, 1854.1]),
}
S2 = {
    "omega": np.array([100, 75, 56.2, 42.2, 31.6, 23.7, 17.8, 13.3, 10,
                        7.5, 5.62, 4.22, 3.16, 2.37, 1.78, 1.33, 1.0]),
    "Gp": np.array([70067, 66851, 63245, 59436, 55560, 51627, 47720, 43944, 40265,
                     36728, 33352, 30167, 27137, 24334, 21746, 19291, 17100]),
    "Gpp": np.array([21038, 21360, 21530, 21336, 21027, 20688, 20218, 19673, 19021,
                      18277, 17450, 16576, 15653, 14713, 13807, 12874, 11840]),
}

omega_match = S2["omega"]
Gp_s1_interp = 10**np.interp(np.log10(omega_match), np.log10(S1["omega"][::-1]), np.log10(S1["Gp"][::-1]))
Gpp_s1_interp = 10**np.interp(np.log10(omega_match), np.log10(S1["omega"][::-1]), np.log10(S1["Gpp"][::-1]))
pct_Gp = (S2["Gp"] - Gp_s1_interp) / Gp_s1_interp * 100
pct_Gpp = (S2["Gpp"] - Gpp_s1_interp) / Gpp_s1_interp * 100
tand_s1_match = Gpp_s1_interp / Gp_s1_interp
tand_s2_match = S2["Gpp"] / S2["Gp"]
delta_tand = tand_s2_match - tand_s1_match

c1 = COLORS["Phalaenopsis"]
c2 = darken(c1, 0.68)
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
ax = axes[0, 0]
ax.loglog(S1["omega"], S1["Gp"], color=c1, linewidth=1.8, marker="s", markersize=3.5,
          markeredgewidth=0.4, markeredgecolor="k", label="G′ — sweep 1", zorder=3)
ax.loglog(S1["omega"], S1["Gpp"], color=c1, linewidth=1.8, linestyle="--", marker="o", markersize=3,
          markeredgewidth=0.4, markeredgecolor="k", label="G″ — sweep 1", zorder=3)
ax.loglog(S2["omega"], S2["Gp"], color=c2, linewidth=1.8, marker="^", markersize=4,
          markeredgewidth=0.4, markeredgecolor="k", label="G′ — sweep 2 (+2 h)", zorder=2)
ax.loglog(S2["omega"], S2["Gpp"], color=c2, linewidth=1.8, linestyle="--", marker="v", markersize=3.5,
          markeredgewidth=0.4, markeredgecolor="k", label="G″ — sweep 2 (+2 h)", zorder=2)
ax.set_ylabel("Modulus (Pa)")
ax.set_xlim(8e-3, 200)
ax.set_ylim(800, 1.2e5)
ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
ax.grid(True, which="both", alpha=0.15, linewidth=0.5)

ax = axes[0, 1]
ax.semilogx(S1["omega"], S1["Gpp"] / S1["Gp"], color=c1, linewidth=1.8,
            marker="o", markersize=3.5, markeredgewidth=0.4, markeredgecolor="k", label="sweep 1", zorder=3)
ax.semilogx(S2["omega"], S2["Gpp"] / S2["Gp"], color=c2, linewidth=1.8,
            marker="^", markersize=4, markeredgewidth=0.4, markeredgecolor="k", label="sweep 2 (+2 h)", zorder=2)
ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
ax.set_ylabel("tan δ (—)")
ax.set_xlim(8e-3, 200)
ax.set_ylim(0.2, 1.6)
ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
ax.grid(True, which="both", alpha=0.15, linewidth=0.5)

ax = axes[1, 0]
ax.semilogx(omega_match, pct_Gp, color=c2, linewidth=1.8,
            marker="s", markersize=4, markeredgewidth=0.4, markeredgecolor="k", label="ΔG′ (%)")
ax.semilogx(omega_match, pct_Gpp, color=c2, linewidth=1.8, linestyle="--",
            marker="o", markersize=3.5, markeredgewidth=0.4, markeredgecolor="k", label="ΔG″ (%)", alpha=0.85)
ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
ax.set_xlabel("Angular frequency ω (rad s⁻¹)")
ax.set_ylabel("Modulus change (%)")
ax.set_xlim(0.8, 200)
ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
ax.grid(True, which="both", alpha=0.15, linewidth=0.5)

ax = axes[1, 1]
ax.semilogx(omega_match, delta_tand, color=c2, linewidth=1.8,
            marker="o", markersize=4, markeredgewidth=0.4, markeredgecolor="k")
ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
ax.set_xlabel("Angular frequency ω (rad s⁻¹)")
ax.set_ylabel("Δtan δ (sweep 2 − sweep 1)")
ax.set_xlim(0.8, 200)
ax.grid(True, which="both", alpha=0.15, linewidth=0.5)

for i, ax in enumerate(axes.flat):
    ax.text(0.02, 0.95, f"({chr(ord('a') + i)})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")
fig.tight_layout()
fig.savefig(OUT / "Figure_S2_dehydration_control.png")
fig.savefig(OUT / "Figure_S2_dehydration_control.pdf")
plt.close(fig)

# =====================================================================
# SUPPLEMENTARY FIGURE S3 — COMPLIANCE VS STRESS
# =====================================================================

DATA = {
    "Laelia": {
        "tau": np.array([5, 7, 10, 15, 30, 60]),
        "gpeak_pct": np.array([0.077, 0.107, 0.153, 0.232, 0.515, 2.38]),
        "lvr": 0.180,
    },
    "Phalaenopsis": {
        "tau": np.array([5, 7, 10, 15, 20]),
        "gpeak_pct": np.array([0.527, 0.587, 0.745, 1.04, 1.33]),
        "lvr": 0.676,
    },
    "Cymbidium": {
        "tau": np.array([7, 10, 15, 20]),
        "gpeak_pct": np.array([0.332, 0.494, 0.728, 1.01]),
        "lvr": 0.296,
    },
    "Dendrobium": {
        "tau": np.array([7, 10, 15]),
        "gpeak_pct": np.array([6.98, 8.97, 13.2]),
        "lvr": 2.39,
    },
    "Cattleya": {
        "tau": np.array([10, 40, 60]),
        "gpeak_pct": np.array([0.0218, 0.0616, 0.0835]),
        "lvr": 0.076,
    },
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
for genus in ORDER:
    d = DATA[genus]
    taus = d["tau"]
    J = (d["gpeak_pct"] / 100.0) / taus
    in_lvr = d["gpeak_pct"] <= d["lvr"]
    c, m = COLORS[genus], MARKERS[genus]
    ax1.plot(taus, J, color=c, marker=m, ms=8, lw=1.5, mec="k", mew=0.5, zorder=5, label=genus)
    if np.any(~in_lvr):
        ax1.scatter(taus[~in_lvr], J[~in_lvr], s=200, facecolors="none", edgecolors=c, linewidths=2, zorder=6)
ax1.set_xlabel("Applied stress τ₀ (Pa)")
ax1.set_ylabel("Creep compliance  J(60 s)  (Pa⁻¹)")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend(loc="upper right", framealpha=0.9)
ax1.grid(True, which="both", alpha=0.15)
ax1.set_title("(a) Compliance vs applied stress")

for genus in ORDER:
    d = DATA[genus]
    taus = d["tau"]
    J = (d["gpeak_pct"] / 100.0) / taus
    J_norm = J / J[0]
    in_lvr = d["gpeak_pct"] <= d["lvr"]
    c, m = COLORS[genus], MARKERS[genus]
    ax2.plot(taus, J_norm, color=c, marker=m, ms=8, lw=1.5, mec="k", mew=0.5, zorder=5, label=genus)
    if np.any(~in_lvr):
        ax2.scatter(taus[~in_lvr], J_norm[~in_lvr], s=200, facecolors="none", edgecolors=c, linewidths=2, zorder=6)
ax2.axhline(1.0, color="k", ls="--", lw=1.2, alpha=0.5, zorder=1)
ax2.axhspan(0.95, 1.05, color="k", alpha=0.05, zorder=0)
ax2.set_xlabel("Applied stress τ₀ (Pa)")
ax2.set_ylabel("Normalised compliance  J(60 s) / J(τ₀,min)")
ax2.set_xscale("log")
ax2.set_ylim(0.3, 3.0)
ax2.legend(loc="upper left", framealpha=0.9)
ax2.grid(True, which="both", alpha=0.15)
ax2.set_title("(b) Linearity: (J / Jref = 1 if linear)")

fig.tight_layout()
fig.savefig(OUT / "Figure_S3_compliance_vs_stress.png")
fig.savefig(OUT / "Figure_S3_compliance_vs_stress.pdf")
plt.close(fig)

print(f"Saved corrected supplemental figures to: {OUT}")
