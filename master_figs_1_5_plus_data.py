#!/usr/bin/env python3
"""
Master plotting script — Figures 1–5 + supplemental amplitude tan(delta)
=======================================================================

Outputs:
  Figure_1_amplitude_sweep_moduli.png/.pdf
  Figure_S1_amplitude_sweep_tandelta.png/.pdf
  Figure_2_frequency_sweep_moduli.png/.pdf
  Figure_3_frequency_sweep_tandelta.png/.pdf
  Figure_4_normalised_creep.png/.pdf
  Figure_5_eleutherocyte_morphometry.png/.pdf

Notes
-----
- Figure 1 now contains only amplitude-sweep storage modulus curves.
- The amplitude-sweep tan(delta) panel is exported separately as a supplemental figure.
- Figure 4 uses the normalized creep-recovery representation from the standalone
  Fig. 4 script, with the same embedded data and a colour scheme harmonized across
  all outputs.
- Figure 5 integrates the eleutherocyte morphometry summary provided separately,
  using the same global palette for the shared genera and added colours for
  Oncidium, Vanda, and Coelogyne.
- All data are embedded inline. No external file dependencies.
"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "master_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# =====================================================================
# GLOBAL STYLE + COLOUR MAP
# =====================================================================

GENUS_ORDER = ["Phalaenopsis", "Cattleya", "Cymbidium", "Dendrobium", "Laelia"]

COLORS = {
    "Phalaenopsis": "#7E57C2",  # purple
    "Cattleya":     "#E69F00",  # orange
    "Cymbidium":    "#009E73",  # green
    "Dendrobium":   "#0072B2",  # blue
    "Laelia":       "#CC79A7",  # pink
    "Oncidium":     "#D4AC0D",
    "Vanda":        "#17A521",
    "Coelogyne":    "#C0392B",
}

MARKERS = {
    "Phalaenopsis": "s",
    "Cattleya":     "^",
    "Cymbidium":    "D",
    "Dendrobium":   "v",
    "Laelia":       "o",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# =====================================================================
# POLLINATOR-RELEVANT REFERENCE SCALES (TIMESCALE-ANCHORED)
# ---------------------------------------------------------------------
# Rate-equivalent angular frequency for a mechanical contact lasting t:
#       omega = 2 * pi / t
#
# Insertion regime (deposition / extended contact):
#   t ~ 1–60 s. Lower bound (~1 s) is anchored to bee handling rate
#   (~40 flowers min^-1, BC gov pollination factsheet) and to the
#   mechanical-contact fraction within the 15–25 s total visit
#   duration reported by Lanzino, Ferretti & Ferretti (2022) Plants
#   11:1327. Upper bound (60 s) is NOT directly anchored to a
#   published observation; it is intended as an order-of-magnitude
#   upper estimate covering extended contact in deceptive-orchid
#   visits, slow self-pollination/autogamy, and post-deposition
#   resting. Caption/Discussion should declare this explicitly.
#       -> omega ~ 0.10–6.28 rad/s
#
# Removal regime (rapid pollinarium withdrawal during takeoff):
#   t <= 0.5 s, anchored to bee takeoff dynamics from foraging-flight
#   speeds reported by Spaethe, Tautz & Chittka (2001) PNAS 98:3898
#   and Osborne et al. (1999) J. Appl. Ecol. 36:519.
#       -> omega >= ~12.6 rad/s
#   Note: removal/withdrawal assay is deferred to a companion paper.
# =====================================================================

T_INSERT_LONG  = 25.0   # s, extended-contact upper bound; UNREFERENCED — see note above
T_INSERT_SHORT = 1.0    # s, briefest insertion contact
T_REMOVE_SLOW  = 0.5    # s, slowest takeoff (sets removal-regime onset)

OMEGA_INSERT_MIN = 2.0 * np.pi / T_INSERT_LONG    # ~0.25 rad/s
OMEGA_INSERT_MAX = 2.0 * np.pi / T_INSERT_SHORT   # ~6.28 rad/s
OMEGA_REMOVE     = 2.0 * np.pi / T_REMOVE_SLOW    # ~12.57 rad/s

# Dahlquist tack criterion: E <= 3e5 Pa at 1 Hz, equivalent to G' <= 1e5 Pa
# for an incompressible viscoelastic solid (E = 3 G').
DAHLQUIST_GP_PA = 1.0e5

# =====================================================================
# FIGURE 1 DATA — AMPLITUDE SWEEPS
# =====================================================================

PHALAENOPSIS = {
    "strain_pct": np.array([0.0101, 0.0179, 0.0318, 0.0565, 0.101, 0.179, 0.318, 0.565, 1.01, 1.79, 3.18, 5.65, 10.1, 17.9, 31.8, 56.5, 100], dtype=float),
    "Gp": np.array([23376, 23450, 23357, 23301, 23182, 23066, 22821, 22427, 21771, 20767, 19292, 17218, 14495, 11303, 7923.6, 4843.5, 2560.6], dtype=float),
    "Gpp": np.array([11003, 10893, 10893, 10876, 10867, 10871, 10861, 10831, 10758, 10608, 10289, 9715.4, 8871, 7885.8, 6846.5, 5616.3, 4168.1], dtype=float),
}

LAELIA_SWEEP1 = {
    "strain_pct": np.array([0.0101, 0.0134, 0.0179, 0.0238, 0.0318, 0.0424, 0.0565, 0.0754, 0.101, 0.134, 0.179, 0.238, 0.318, 0.424, 0.565, 0.754, 1.01, 1.34, 1.79, 2.38, 3.18, 4.24, 5.65, 7.54, 10.1], dtype=float),
    "Gp": np.array([32964, 32860, 32882, 32910, 32850, 32725, 32641, 32427, 32247, 31914, 31551, 31023, 30439, 29669, 28729, 27518, 26117, 24635, 23213, 21797, 20426, 19046, 17598, 16055, 14450], dtype=float),
    "Gpp": np.array([13402, 13465, 13410, 13405, 13406, 13388, 13434, 13426, 13484, 13530, 13614, 13736, 13899, 14097, 14330, 14525, 14596, 14536, 14316, 13942, 13484, 13002, 12496, 11974, 11403], dtype=float),
}

LAELIA_SWEEP2 = {
    "strain_pct": np.array([0.0101, 0.0134, 0.0179, 0.0238, 0.0318, 0.0424, 0.0565, 0.0754, 0.101, 0.134, 0.179, 0.238, 0.318, 0.424, 0.565, 0.754, 1.01, 1.34, 1.79, 2.38, 3.18, 4.24, 5.65, 7.54, 10.1], dtype=float),
    "Gp": np.array([32994, 33061, 32944, 32959, 32819, 32774, 32575, 32419, 32163, 31781, 31366, 30775, 30096, 29234, 28235, 27078, 25748, 24285, 22837, 21454, 20117, 18801, 17423, 16001, 14500], dtype=float),
    "Gpp": np.array([13389, 13324, 13347, 13379, 13378, 13396, 13413, 13471, 13506, 13546, 13607, 13665, 13758, 13886, 14069, 14273, 14467, 14476, 14249, 13870, 13420, 12941, 12446, 11921, 11350], dtype=float),
}

LAELIA_SWEEP3 = {
    "strain_pct": np.array([0.0101, 0.0148, 0.0217, 0.0318, 0.0466, 0.0685, 0.101, 0.148, 0.217, 0.318, 0.466, 0.685, 1.01, 1.48, 2.17, 3.18, 4.67, 6.85, 10.1, 14.8, 21.7, 31.8, 46.6, 68.5, 101], dtype=float),
    "Gp": np.array([25284, 25224, 25178, 25098, 25018, 24910, 24704, 24496, 24152, 23748, 23192, 22499, 21567, 20292, 18969, 17557, 15963, 14162, 12239, 10253, 8266.5, 6350.6, 4579, 3053.5, 1889.7], dtype=float),
    "Gpp": np.array([11191, 11201, 11158, 11147, 11148, 11149, 11126, 11139, 11122, 11128, 11106, 11083, 11104, 11132, 10841, 10407, 9840.5, 9149.6, 8373.5, 7567.1, 6783.9, 5999.2, 5184.4, 4301.3, 3375.9], dtype=float),
}

CYMBIDIUM_SWEEP1 = {
    "strain_pct": np.array([0.0101, 0.0148, 0.0217, 0.0318, 0.0467, 0.0685, 0.101, 0.148, 0.217, 0.318, 0.467, 0.685, 1.01, 1.48, 2.17, 3.18, 4.67, 6.85, 10.1, 14.8, 21.7, 31.8, 46.6, 68.5, 101], dtype=float),
    "Gp": np.array([33457, 33600, 33228, 33573, 33167, 33310, 32830, 32682, 32091, 31681, 30930, 29891, 28306, 26556, 24480, 22172, 19597, 16867, 14068, 11389, 8972, 6821.2, 4957.6, 3357.8, 2078.1], dtype=float),
    "Gpp": np.array([20522, 20558, 20370, 20558, 20397, 20562, 20415, 20522, 20411, 20484, 20399, 20243, 19728, 19139, 18266, 17185, 15859, 14392, 12828, 11304, 9906.8, 8612.8, 7412.7, 6182, 4877.6], dtype=float),
}

CATTLEYA_SWEEP1 = {
    "strain_pct": np.array([0.0101, 0.0148, 0.0217, 0.0318, 0.0467, 0.0685, 0.101, 0.148, 0.217, 0.318, 0.466, 0.685, 1.01, 1.48, 2.17, 3.18, 4.67, 6.85, 10.1, 14.8, 21.7, 31.8, 46.7, 68.5, 101], dtype=float),
    "Gp": np.array([590000, 595000, 594000, 593000, 588000, 580000, 566000, 542000, 491000, 329000, 188000, 114000, 71844, 45911, 30523, 20547, 14221, 10367, 7917.5, 6273.2, 4989.6, 4052.3, 3067.6, 2206.7, 1395.4], dtype=float),
    "Gpp": np.array([141000, 141000, 140000, 141000, 142000, 143000, 146000, 155000, 181000, 258000, 236000, 190000, 147000, 113000, 86260, 65801, 49189, 37224, 28132, 21938, 17142, 13580, 10758, 8482.9, 6430.9], dtype=float),
}

CATTLEYA_SWEEP2 = {
    "strain_pct": np.array([0.0101, 0.0148, 0.0217, 0.0318, 0.0467, 0.0685, 0.101, 0.148, 0.217, 0.318, 0.466, 0.685, 1.01, 1.48, 2.17, 3.18, 4.67, 6.85, 10.1, 14.8, 21.7, 31.8, 46.6, 68.5, 100], dtype=float),
    "Gp": np.array([680000, 680000, 678000, 674000, 667000, 652000, 627000, 565000, 345000, 214000, 137000, 87428, 56110, 37032, 24966, 17455, 12752, 9780.4, 7587.9, 6017.1, 4929, 3937.3, 3032.8, 2122.7, 1351.6], dtype=float),
    "Gpp": np.array([142000, 141000, 143000, 143000, 145000, 147000, 155000, 199000, 294000, 266000, 222000, 177000, 138000, 106000, 80460, 61548, 46751, 35769, 27294, 21063, 16503, 12942, 10108, 7933.4, 6163.3], dtype=float),
}

DENDROBIUM_SWEEP1 = {
    "strain_pct": np.array([0.0101, 0.0148, 0.0217, 0.0318, 0.0466, 0.0685, 0.101, 0.148, 0.217, 0.318, 0.467, 0.685, 1, 1.48, 2.17, 3.18, 4.67, 6.85, 10.1, 14.8, 21.7, 31.8, 46.6, 68.5, 101], dtype=float),
    "Gp": np.array([3940.3, 4017.5, 3988.9, 4030.8, 4027.5, 3959.4, 4019.6, 3963.6, 4003.3, 3958.9, 3979.2, 3936.5, 3918.2, 3862.8, 3804.8, 3717.1, 3592.9, 3431.8, 3227, 2966.4, 2654.1, 2293.1, 1883.5, 1436.4, 1014.8], dtype=float),
    "Gpp": np.array([4993.5, 5028, 4947.4, 4989.8, 4994.4, 4969.3, 4989, 4957.7, 4966.3, 4921.2, 4912.7, 4881.8, 4889, 4853.9, 4830.7, 4772, 4687.8, 4570.4, 4413.9, 4215.4, 3981, 3714.7, 3402.6, 3002.4, 2526.3], dtype=float),
}

DENDROBIUM_SWEEP2 = {
    "strain_pct": np.array([0.0101, 0.0148, 0.0217, 0.0318, 0.0467, 0.0685, 0.101, 0.148, 0.217, 0.318, 0.467, 0.685, 1.01, 1.48, 2.17, 3.18, 4.66, 6.85, 10.1, 14.8, 21.7, 31.8, 46.6, 68.5, 101], dtype=float),
    "Gp": np.array([3695.7, 3717.1, 3709.7, 3658.4, 3712.9, 3664.1, 3711.3, 3667.2, 3695.2, 3668, 3678.3, 3647.4, 3638.9, 3604.2, 3555.9, 3488.1, 3394.5, 3272, 3106.1, 2896.5, 2627.2, 2294.6, 1885.3, 1430.6, 1001.8], dtype=float),
    "Gpp": np.array([4482, 4450.7, 4470.9, 4458, 4472, 4469.2, 4492.9, 4476.8, 4492.8, 4475.8, 4483.5, 4469.8, 4470, 4456.1, 4436.9, 4401.5, 4347.5, 4267.6, 4156.7, 4017, 3842.2, 3626.6, 3334.7, 2935.9, 2458.8], dtype=float),
}

DENDROBIUM_SWEEP4_SAMPLE2 = {
    "strain_pct": np.array([0.0101, 0.0148, 0.0217, 0.0318, 0.0467, 0.0685, 0.101, 0.148, 0.217, 0.318, 0.466, 0.686, 1.01, 1.48, 2.17, 3.18, 4.66, 6.85, 10.1, 14.8, 21.7, 31.8, 46.7, 68.5, 101], dtype=float),
    "Gp": np.array([6473.3, 6656.6, 6389.4, 6513.9, 6360, 6439.8, 6380.1, 6398.9, 6368.7, 6329.1, 6325, 6272.5, 6243.1, 6179, 6094.8, 5948.9, 5794.5, 5542.9, 5202.2, 4747.9, 4178.7, 3515.4, 2782.3, 2036.5, 1391.1], dtype=float),
    "Gpp": np.array([6132.5, 6581, 6164.4, 6379.5, 6163.8, 6268.8, 6165, 6204.8, 6181.8, 6178.4, 6198.3, 6202.9, 6197.1, 6192, 6164.4, 6111.8, 6056.5, 5948.8, 5797, 5584.7, 5301.8, 4940.7, 4483.8, 3902.7, 3239.1], dtype=float),
}

AMP_DATASETS = {
    "Phalaenopsis": {"primary": PHALAENOPSIS, "others": []},
    "Cattleya": {"primary": CATTLEYA_SWEEP2, "others": [CATTLEYA_SWEEP1]},
    "Cymbidium": {"primary": CYMBIDIUM_SWEEP1, "others": []},
    "Dendrobium": {"primary": DENDROBIUM_SWEEP1, "others": [DENDROBIUM_SWEEP2, DENDROBIUM_SWEEP4_SAMPLE2]},
    "Laelia": {"primary": LAELIA_SWEEP2, "others": [LAELIA_SWEEP1, LAELIA_SWEEP3]},
}

# =====================================================================
# FIGURES 2–3 DATA — FREQUENCY SWEEPS
# =====================================================================

FREQ_DATA = {
    "Phalaenopsis": {
        "omega": np.array([100, 55.8, 31.2, 17.4, 9.73, 5.43, 3.03, 1.69, 0.946, 0.528, 0.295, 0.165, 0.092, 0.0514, 0.0287, 0.016, 0.00895, 0.005]),
        "Gp": np.array([41768, 33004, 27165, 22578, 18588, 15231, 12429, 10138, 8217.1, 6604.4, 5314, 4304.6, 3482.8, 2798, 2213, 1758, 1385, 1162]),
        "Gpp": np.array([14961, 13509, 12139, 10818, 9666.1, 8547.7, 7486.7, 6503.6, 5619.9, 4711.4, 4050.6, 3384.1, 2828.3, 2358, 1966, 1599, 1368, 1120]),
        "tand": np.array([0.358, 0.409, 0.447, 0.479, 0.520, 0.561, 0.602, 0.642, 0.684, 0.713, 0.762, 0.786, 0.812, 0.843, 0.888, 0.909, 0.988, 0.964]),
    },
    "Cattleya": {
        "omega": np.array([100, 55.8, 31.2, 17.4, 9.73, 5.43, 3.03, 1.69, 0.946, 0.528, 0.295, 0.165, 0.092, 0.0514, 0.0287, 0.016, 0.00895, 0.005]),
        "Gp": np.array([858000, 804000, 755000, 706000, 659000, 611000, 565000, 517000, 475000, 441000, 381000, 332000, 287000, 243000, 197000, 161000, 122000, 92900]),
        "Gpp": np.array([143000, 140000, 139000, 137000, 137000, 135000, 135000, 132000, 136000, 138000, 132000, 140000, 126000, 118000, 110000, 98170, 85630, 82030]),
        "tand": np.array([0.166, 0.175, 0.184, 0.195, 0.207, 0.222, 0.239, 0.255, 0.287, 0.314, 0.347, 0.422, 0.438, 0.487, 0.555, 0.610, 0.704, 0.883]),
    },
    "Cymbidium": {
        "omega": np.array([100, 55.8, 31.2, 17.4, 9.73, 5.43, 3.03, 1.69, 0.946, 0.528, 0.295, 0.165, 0.092, 0.0514, 0.0287, 0.016, 0.00895, 0.005]),
        "Gp": np.array([49246, 40074, 31671, 24413, 18359, 13484, 9687.9, 6841.7, 4763.6, 3310.2, 2304, 1616, 1147.9, 823.01, 609.02, 458.63, 336.56, 262.8]),
        "Gpp": np.array([32608, 27628, 23395, 19542, 16088, 13003, 10260, 7927.2, 5983.8, 4454.2, 3269.9, 2380.5, 1730.2, 1259.3, 923.65, 676.31, 506.89, 376.57]),
        "tand": np.array([0.662, 0.689, 0.739, 0.800, 0.876, 0.964, 1.059, 1.159, 1.256, 1.346, 1.419, 1.473, 1.507, 1.530, 1.517, 1.475, 1.506, 1.433]),
    },
    "Dendrobium": {
        "omega": np.array([300, 188, 118, 73.6, 46, 28.8, 18, 11.3, 7.06, 4.42, 2.77, 1.73, 1.08, 0.678, 0.425, 0.266, 0.166, 0.104, 0.0652, 0.0408, 0.0255]),
        "Gp": np.array([21375, 20155, 16606, 13282, 10405, 8034.6, 6073.8, 4465.4, 3179.2, 2260.5, 1519.2, 1035.6, 666.64, 439.98, 282.75, 189.37, 108.02, 88.036, 45.701, 30.85, 12.379]),
        "Gpp": np.array([9573.7, 10640, 10189, 9315.8, 8329, 7264.3, 6149.6, 5090.7, 4118.7, 3263.8, 2515, 1899.9, 1423.8, 1021.8, 743.26, 539.2, 395.53, 299.29, 210.85, 170.93, 128.84]),
        "tand": np.array([0.448, 0.528, 0.614, 0.701, 0.800, 0.904, 1.012, 1.140, 1.295, 1.444, 1.656, 1.834, 2.136, 2.322, 2.629, 2.847, 3.662, 3.400, 4.614, 5.541, 10.408]),
    },
    "Laelia": {
        "omega": np.array([100, 55.8, 31.2, 17.4, 9.73, 5.43, 3.03, 1.69, 0.946, 0.528, 0.295, 0.165, 0.092, 0.0514, 0.0287, 0.016, 0.00895, 0.005]),
        "Gp": np.array([52817, 48120, 42484, 37003, 31862, 27240, 23093, 19530, 16401, 13760, 11637, 9920.5, 8217.2, 6700.1, 5389.7, 3899.2, 3013.9, 1917.2]),
        "Gpp": np.array([17695, 16998, 15822, 14661, 13400, 12119, 10774, 9487.2, 8255, 7190.3, 5929.5, 5414.4, 4667.8, 4056.8, 3685.8, 3265.1, 2991.1, 2140.6]),
        "tand": np.array([0.335, 0.353, 0.372, 0.396, 0.421, 0.445, 0.467, 0.486, 0.503, 0.523, 0.510, 0.546, 0.568, 0.605, 0.684, 0.837, 0.992, 1.117]),
    },
}

# =====================================================================
# FIGURE 4 DATA — NORMALIZED CREEP-RECOVERY
# =====================================================================

NORM_CREEP_DATA = {
    "Phalaenopsis": {
        "c_pct": np.array([0.0868, 0.127, 0.162, 0.196, 0.23, 0.265, 0.302, 0.342, 0.386, 0.433, 0.485, 0.54, 0.602, 0.669, 0.745]),
        "c_t": np.array([1.0, 2.179, 3.569, 5.207, 7.137, 9.413, 12.1, 15.26, 18.99, 23.38, 28.56, 34.67, 41.86, 50.35, 60.35]),
        "r_pct": np.array([0.0181, 0.0195, 0.0178, 0.0148, 0.0102, 0.00417, -0.00271, -0.0106, -0.0191, -0.0282, -0.0386, -0.0496, -0.061, -0.0734, -0.0865, -0.1, -0.116, -0.133, -0.152, -0.174]),
        "r_t": np.array([1.0, 2.229, 3.739, 5.593, 7.872, 10.67, 14.11, 18.34, 23.53, 29.91, 37.75, 47.38, 59.21, 73.74, 91.6, 113.5, 140.5, 173.6, 214.3, 264.3]),
    },
    "Cymbidium": {
        "c_pct": np.array([0.056, 0.0835, 0.107, 0.129, 0.152, 0.176, 0.202, 0.228, 0.256, 0.286, 0.322, 0.36, 0.401, 0.445, 0.494]),
        "c_t": np.array([1.0, 2.179, 3.569, 5.207, 7.137, 9.413, 12.1, 15.26, 18.99, 23.38, 28.56, 34.67, 41.86, 50.35, 60.35]),
        "r_pct": np.array([-0.0145, -0.0244, -0.0337, -0.0441, -0.055, -0.0664, -0.079, -0.0928, -0.106, -0.121, -0.136, -0.151, -0.167, -0.185, -0.201, -0.219, -0.238, -0.257, -0.275, -0.299]),
        "r_t": np.array([1.0, 2.229, 3.739, 5.593, 7.872, 10.67, 14.11, 18.34, 23.53, 29.91, 37.75, 47.38, 59.21, 73.74, 91.6, 113.5, 140.5, 173.6, 214.3, 264.3]),
    },
    "Dendrobium": {
        "c_pct": np.array([0.398, 0.704, 1.0, 1.33, 1.67, 2.05, 2.48, 2.96, 3.51, 4.14, 4.86, 5.69, 6.62, 7.7, 8.97]),
        "c_t": np.array([1.0, 2.179, 3.569, 5.207, 7.137, 9.413, 12.1, 15.26, 18.99, 23.38, 28.56, 34.67, 41.86, 50.35, 60.35]),
        "r_pct": np.array([0.0133, -0.00594, -0.0352, -0.0711, -0.115, -0.167, -0.224, -0.287, -0.354, -0.423, -0.5, -0.587, -0.675, -0.768, -0.867, -0.971, -1.08, -1.19, -1.3, -1.42]),
        "r_t": np.array([1.0, 2.229, 3.739, 5.593, 7.872, 10.67, 14.11, 18.34, 23.53, 29.91, 37.75, 47.38, 59.21, 73.74, 91.6, 113.5, 140.5, 173.6, 214.3, 264.3]),
    },
    # Laelia: apparent recovery >100% (γ dips slightly below zero after shifting).
    # Attributed to a small time delay between end of creep and start of recovery
    # recording — the instrument-reported γ_peak underestimates the true peak strain,
    # so the shifted trace starts from a baseline slightly too low. The trace shape
    # and relative recovery ranking remain interpretable; the absolute residual
    # value should not be over-interpreted.
    "Laelia": {
        "c_pct": np.array([0.0509, 0.0653, 0.075, 0.0819, 0.087, 0.0924, 0.099, 0.107, 0.114, 0.122, 0.129, 0.137, 0.141, 0.146, 0.153]),
        "c_t": np.array([1.0, 2.179, 3.569, 5.207, 7.137, 9.413, 12.1, 15.26, 18.99, 23.38, 28.56, 34.67, 41.86, 50.35, 60.35]),
        "r_pct": np.array([-0.00169, -0.00416, -0.00738, -0.0113, -0.016, -0.0211, -0.0265, -0.0329, -0.0389, -0.0482, -0.0554, -0.0636, -0.0753, -0.0831, -0.0934, -0.11, -0.122, -0.143, -0.164, -0.188]),
        "r_t": np.array([1.0, 2.229, 3.739, 5.593, 7.872, 10.67, 14.11, 18.34, 23.53, 29.91, 37.75, 47.38, 59.21, 73.74, 91.6, 113.5, 140.5, 173.6, 214.3, 264.3]),
    },
}

# =====================================================================
# HELPERS
# =====================================================================

def compute_gc_5pct(strain_pct: np.ndarray, Gp: np.ndarray, n_plat: int = 3) -> tuple[float, float, int]:
    strain_pct = np.asarray(strain_pct, dtype=float)
    Gp = np.asarray(Gp, dtype=float)
    if len(strain_pct) < max(3, n_plat):
        return float(strain_pct[-1]), float(np.nanmean(Gp)), -1
    plateau = float(np.nanmean(Gp[:n_plat]))
    thresh = 0.95 * plateau
    below = np.where(Gp < thresh)[0]
    if len(below) == 0:
        return float(strain_pct[-1]), plateau, -1
    i = int(below[0])
    if i == 0:
        return float(strain_pct[0]), plateau, 0
    x0, x1 = float(strain_pct[i - 1]), float(strain_pct[i])
    y0, y1 = float(Gp[i - 1]), float(Gp[i])
    if x0 <= 0 or x1 <= 0 or y0 == y1:
        return float(x1), plateau, i
    f = (y0 - thresh) / (y0 - y1)
    logx = np.log10(x0) + f * (np.log10(x1) - np.log10(x0))
    return float(10 ** logx), plateau, i


def interp_in_logx(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if xq <= x.min():
        return float(y[np.argmin(x)])
    if xq >= x.max():
        return float(y[np.argmax(x)])
    i = int(np.searchsorted(x, xq))
    i = max(1, min(i, len(x) - 1))
    x0, x1 = float(x[i - 1]), float(x[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if x0 <= 0 or x1 <= 0:
        f = (xq - x0) / (x1 - x0)
        return float(y0 + f * (y1 - y0))
    f = (np.log10(xq) - np.log10(x0)) / (np.log10(x1) - np.log10(x0))
    return float(y0 + f * (y1 - y0))


def find_crossover(omega: np.ndarray, Gp: np.ndarray, Gpp: np.ndarray) -> float | None:
    y = Gp - Gpp
    changes = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(changes) == 0:
        return None
    i = int(changes[0])
    x1, x2 = np.log10(omega[i]), np.log10(omega[i + 1])
    y1, y2 = y[i], y[i + 1]
    if y2 == y1:
        return float(omega[i])
    x0 = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
    return float(10 ** x0)


# =====================================================================
# FIGURE 1 — AMPLITUDE-SWEEP STORAGE MODULUS ONLY
# =====================================================================

fig1, ax1 = plt.subplots(figsize=(8, 5.8))
amp_summary_rows = []

for genus in GENUS_ORDER:
    bundle = AMP_DATASETS[genus]
    primary = bundle["primary"]
    others = bundle["others"]
    c = COLORS[genus]
    m = MARKERS[genus]

    for ds in others:
        ax1.plot(ds["strain_pct"], ds["Gp"], linestyle="-", marker=m, ms=4, lw=1.0,
                 color="0.7", alpha=0.55, mec="0.7", mfc="0.85", zorder=1)

    ax1.plot(primary["strain_pct"], primary["Gp"], linestyle="-", marker=m, ms=5.5, lw=1.8,
             color=c, mec="k", mew=0.4, mfc=c, zorder=5, label=genus)

    gc, plateau, _ = compute_gc_5pct(primary["strain_pct"], primary["Gp"], n_plat=3)
    thresh = 0.95 * plateau
    ax1.axvline(gc, color=c, linestyle=":", lw=0.9, alpha=0.6, zorder=2)
    ax1.plot(gc, thresh, marker="*", ms=12, color=c, mec="k", mew=0.5, zorder=6)

    amp_summary_rows.append({
        "Genus": genus,
        "gamma_c_pct_5pct_drop": gc,
        "Gp_plateau_Pa_mean_first3": plateau,
        "Gp_at_gamma_c_threshold_Pa": thresh,
    })

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Oscillatory strain amplitude, $\gamma_0$ (%)")
ax1.set_ylabel(r"Storage modulus, $G'$ (Pa)")
ax1.set_xlim(0.008, 120)
ax1.set_ylim(800, 1.2e6)
ax1.grid(True, which="both", alpha=0.2)
ax1.legend(loc="lower left", framealpha=0.9, ncol=2)
fig1.tight_layout()
fig1.savefig(OUT / "Figure_1_amplitude_sweep_moduli.png")
fig1.savefig(OUT / "Figure_1_amplitude_sweep_moduli.pdf")
plt.close(fig1)

# =====================================================================
# SUPPLEMENTARY FIGURE S1 — AMPLITUDE-SWEEP tan(delta)
# =====================================================================

figs1, axs1 = plt.subplots(figsize=(8, 5.5))

for genus in GENUS_ORDER:
    bundle = AMP_DATASETS[genus]
    primary = bundle["primary"]
    others = bundle["others"]
    c = COLORS[genus]
    m = MARKERS[genus]

    for ds in others:
        axs1.plot(ds["strain_pct"], ds["Gpp"] / ds["Gp"], linestyle="-", marker=m, ms=4, lw=1.0,
                  color="0.7", alpha=0.55, mec="0.7", mfc="0.85", zorder=1)

    tan = primary["Gpp"] / primary["Gp"]
    axs1.plot(primary["strain_pct"], tan, linestyle="-", marker=m, ms=5.5, lw=1.8,
              color=c, mec="k", mew=0.4, mfc=c, zorder=5, label=genus)

    gc, _, _ = compute_gc_5pct(primary["strain_pct"], primary["Gp"], n_plat=3)
    tan_at_gc = interp_in_logx(primary["strain_pct"], tan, gc)
    axs1.axvline(gc, color=c, linestyle=":", lw=0.9, alpha=0.6, zorder=2)
    axs1.plot(gc, tan_at_gc, marker="*", ms=10, color=c, mec="k", mew=0.5, zorder=6)

axs1.set_xscale("log")
axs1.set_xlabel(r"Oscillatory strain amplitude, $\gamma_0$ (%)")
axs1.set_ylabel(r"Loss factor, tan $\delta$ = $G^{\prime\prime}/G^{\prime}$")
axs1.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.6, zorder=0)
axs1.set_xlim(0.008, 120)
axs1.set_ylim(0.1, 5.0)
axs1.grid(True, which="both", alpha=0.2)
axs1.legend(loc="upper left", framealpha=0.9, ncol=2)
figs1.tight_layout()
figs1.savefig(OUT / "Figure_S1_amplitude_sweep_tandelta.png")
figs1.savefig(OUT / "Figure_S1_amplitude_sweep_tandelta.pdf")
plt.close(figs1)

# =====================================================================
# FIGURE 2 — FREQUENCY-SWEEP MODULI
# =====================================================================

crossovers: dict[str, float | None] = {}
fig2, ax2 = plt.subplots(figsize=(8.8, 6.2))

# ---- Background regions (pollinator-rate equivalent frequencies) ----
# Insertion regime: handling/contact, t ~ 1–5 s  ->  ~1.3–6.3 rad/s
ax2.axvspan(OMEGA_INSERT_MIN, OMEGA_INSERT_MAX, color="#88BB88", alpha=0.12, zorder=0)
# Removal regime: takeoff, t <= 0.5 s  ->  >=~12.6 rad/s
ax2.axvspan(OMEGA_REMOVE, 1000, color="#D08585", alpha=0.10, zorder=0)
ax2.axvline(OMEGA_REMOVE, color="#A03030", linestyle="-", linewidth=1.0, alpha=0.7, zorder=1)

# ---- Dahlquist tack criterion (horizontal) ----
ax2.axhline(DAHLQUIST_GP_PA, color="0.25", linestyle="-", linewidth=1.2, alpha=0.85, zorder=1)

# ---- Curves ----
for genus in GENUS_ORDER:
    d = FREQ_DATA[genus]
    c = COLORS[genus]
    oc = find_crossover(d["omega"], d["Gp"], d["Gpp"])
    crossovers[genus] = oc

    ax2.loglog(d["omega"], d["Gp"], color=c, linewidth=2.0, marker="s", markersize=6.0,
               markeredgewidth=0.5, markeredgecolor="k", zorder=3)
    ax2.loglog(d["omega"], d["Gpp"], color=c, linewidth=2.0, linestyle="--", marker="o",
               markersize=5.0, markeredgewidth=0.5, markeredgecolor="k", zorder=3)

for genus in GENUS_ORDER:
    oc = crossovers[genus]
    if oc is not None:
        ax2.axvline(oc, color=COLORS[genus], linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)

ax2.set_xlim(3e-3, 400)
ax2.set_ylim(5, 1.5e6)
ax2.set_xlabel(r"Angular frequency $\omega$ (rad s$^{-1}$)")
ax2.set_ylabel("Modulus (Pa)")
ax2.grid(True, which="both", alpha=0.15, linewidth=0.5)

# ---- Region labels ----
# Place at a common y in the clean horizontal band between Cattleya G''
# (~1.4e5 Pa) and Cattleya G' (~5–8e5 Pa).
y_label = 2.5e5
_label_bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9)
# Insertion: centred geometrically inside the (now wider) green band
ax2.text(np.sqrt(OMEGA_INSERT_MIN * OMEGA_INSERT_MAX), y_label,
         "insertion regime\n(handling/contact, ~1–25 s)",
         ha="center", va="center", fontsize=9, color="#2E5E2E",
         fontstyle="italic", zorder=4, bbox=_label_bbox)
# Removal: centred inside the red band, lifted to MATCH insertion y
ax2.text(60, y_label,
         "removal regime\n(takeoff, ≤0.5 s)",
         ha="center", va="center", fontsize=9, color="#7A2020",
         fontstyle="italic", zorder=4, bbox=_label_bbox)
# Dahlquist label: kept in the empty corner below the line at low ω
ax2.text(4e-3, DAHLQUIST_GP_PA * 0.55,
         "Dahlquist tack threshold\n(G′ ≈ 10$^5$ Pa)",
         ha="left", va="top", fontsize=9, color="0.15",
         fontstyle="italic", zorder=4, bbox=_label_bbox)

legend_handles = [Line2D([0], [0], color=COLORS[g], linewidth=2.2, label=g) for g in GENUS_ORDER]
legend_handles += [
    Line2D([0], [0], color="gray", linewidth=1.6, marker="s", markersize=6, markeredgecolor="k", markeredgewidth=0.5, label="G′ (storage)"),
    Line2D([0], [0], color="gray", linewidth=1.6, linestyle="--", marker="o", markersize=5, markeredgecolor="k", markeredgewidth=0.5, label="G″ (loss)"),
    Line2D([0], [0], color="gray", linewidth=1.0, linestyle=":", alpha=0.6, label=r"$\omega_c$ (crossover)"),
]
ax2.legend(handles=legend_handles, ncol=2, loc="lower right", framealpha=0.9, handlelength=2.2, columnspacing=1.0)
fig2.tight_layout()
fig2.savefig(OUT / "Figure_2_frequency_sweep_moduli.png")
fig2.savefig(OUT / "Figure_2_frequency_sweep_moduli.pdf")
plt.close(fig2)

# =====================================================================
# FIGURE 3 — FREQUENCY-SWEEP tan(delta)
# =====================================================================

fig3, ax3 = plt.subplots(figsize=(8.8, 5.8))

# ---- Background regions (pollinator-rate equivalent frequencies) ----
# Same banding as Figure 2 to anchor the rate-dependent reading.
ax3.axvspan(OMEGA_INSERT_MIN, OMEGA_INSERT_MAX, color="#88BB88", alpha=0.12, zorder=0)
ax3.axvspan(OMEGA_REMOVE, 1000, color="#D08585", alpha=0.10, zorder=0)
ax3.axvline(OMEGA_REMOVE, color="#A03030", linestyle="-", linewidth=1.0, alpha=0.7, zorder=1)

for genus in GENUS_ORDER:
    d = FREQ_DATA[genus]
    c = COLORS[genus]
    ax3.loglog(d["omega"], d["tand"], color=c, linewidth=2.2, marker=MARKERS[genus], markersize=6.0,
               markeredgewidth=0.5, markeredgecolor="k", zorder=3, label=genus)
    oc = crossovers[genus]
    if oc is not None:
        ax3.axvline(oc, color=c, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)

ax3.axhline(1.0, color="k", linestyle="--", linewidth=1.3, alpha=0.7, zorder=2, label="tan δ = 1")
ax3.set_xlim(3e-3, 400)
ax3.set_ylim(0.08, 15)
ax3.set_xlabel(r"Angular frequency $\omega$ (rad s$^{-1}$)")
ax3.set_ylabel("Loss tangent tan δ (—)")
ax3.grid(True, which="both", alpha=0.15, linewidth=0.5)

# Region labels (top of plot, above all curves)
y_top = 12.0
ax3.text(np.sqrt(OMEGA_INSERT_MIN * OMEGA_INSERT_MAX), y_top,
         "insertion regime\n(handling/contact, ~1–25 s)",
         ha="center", va="top", fontsize=9, color="#2E5E2E",
         fontstyle="italic", zorder=2)
ax3.text(60, y_top,
         "removal regime\n(takeoff, ≤0.5 s)",
         ha="center", va="top", fontsize=9, color="#7A2020",
         fontstyle="italic", zorder=2)
# Solid-like vs liquid-like annotation for tan δ = 1
ax3.text(4e-3, 1.65, "liquid-like (tan δ > 1)", ha="left", va="bottom",
         fontsize=8.5, color="0.3", fontstyle="italic", zorder=2)
ax3.text(4e-3, 0.55, "solid-like (tan δ < 1)", ha="left", va="top",
         fontsize=8.5, color="0.3", fontstyle="italic", zorder=2)

ax3.legend(loc="lower right", framealpha=0.9, ncol=2)
fig3.tight_layout()
fig3.savefig(OUT / "Figure_3_frequency_sweep_tandelta.png")
fig3.savefig(OUT / "Figure_3_frequency_sweep_tandelta.pdf")
plt.close(fig3)

# =====================================================================
# FIGURE 4 — NORMALISED CREEP-RECOVERY
# =====================================================================

fig4, ax4 = plt.subplots(figsize=(8, 5.5))
fig4_order = ["Phalaenopsis", "Cymbidium", "Dendrobium", "Laelia"]

for genus in fig4_order:
    d = NORM_CREEP_DATA[genus]
    c = COLORS[genus]
    m = MARKERS[genus]
    gpeak_frac = d["c_pct"][-1] / 100.0

    # Recovery phase only: gamma_shifted = gpeak + r_pct/100; normalise by gpeak
    r_shifted_frac = gpeak_frac + d["r_pct"] / 100.0
    r_norm = r_shifted_frac / gpeak_frac

    # Laelia: apparent overshoot (residual > peak) is a data-acquisition timing
    # artifact. Clip the trace to 0 so the visual shows "near-perfect recovery"
    # without the artifact-driven dip below zero. The artifact is annotated.
    if genus == "Laelia":
        r_norm_plot = np.clip(r_norm, 0.0, None)
    else:
        r_norm_plot = r_norm

    # Prepend the (t = 0, gamma/gpeak = 1) anchor point so each curve visibly
    # starts at unity at the moment of stress removal.
    t_plot = np.concatenate([[0.0], d["r_t"]])
    g_plot = np.concatenate([[1.0], r_norm_plot])

    ax4.plot(t_plot, g_plot, color=c, marker=m, ms=6.0, lw=2.0,
             mec="k", mew=0.4, zorder=5, label=genus)

# Reference lines: γ = 1 (peak) and γ = 0 (perfect recovery)
ax4.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.4, zorder=0)
ax4.axhline(0.0, color="k", lw=0.6, alpha=0.5, zorder=0)

# Annotate Laelia clip
laelia_t_end = NORM_CREEP_DATA["Laelia"]["r_t"][-1]
ax4.annotate("Laelia: apparent residual\nbelow 0 clipped (timing artifact)",
             xy=(laelia_t_end, 0.0), xytext=(laelia_t_end - 80, 0.18),
             fontsize=7.5, color=COLORS["Laelia"], fontstyle="italic",
             ha="left", va="bottom",
             arrowprops=dict(arrowstyle="-", color=COLORS["Laelia"], lw=0.7, alpha=0.7))

ax4.set_xlabel("Recovery time after stress removal, $t$ (s)")
ax4.set_ylabel(r"Normalised strain  $\gamma(t) / \gamma_{\mathrm{peak}}$")
ax4.set_xlim(0, max(NORM_CREEP_DATA[g]["r_t"][-1] for g in fig4_order) * 1.05)
ax4.set_ylim(-0.05, 1.1)
ax4.grid(True, which="both", alpha=0.15, linewidth=0.5)
ax4.legend(loc="upper right", framealpha=0.9)
fig4.tight_layout()
fig4.savefig(OUT / "Figure_4_normalised_creep.png")
fig4.savefig(OUT / "Figure_4_normalised_creep.pdf")
plt.close(fig4)


# =====================================================================
# FIGURE 5 — ELEUTHEROCYTE MORPHOMETRY
# =====================================================================

FIG5_GENERA = [
    "Oncidium", "Dendrobium", "Laelia", "Vanda",
    "Cymbidium", "Cattleya", "Coelogyne", "Phalaenopsis",
]

FIG5_AR_MEDIAN = np.array([1.70, 2.38, 2.45, 2.41, 3.03, 3.40, 4.19, 4.68], dtype=float)
FIG5_AR_Q1     = np.array([1.29, 1.80, 2.11, 1.50, 2.37, 3.13, 3.87, 3.94], dtype=float)
FIG5_AR_Q3     = np.array([2.77, 3.35, 3.00, 4.24, 3.58, 3.79, 5.06, 6.94], dtype=float)

FIG5_PHI_MEAN = np.array([36.0, 37.0, 53.9, 30.7, 45.5, 52.0, 57.5, 89.6], dtype=float)
FIG5_PHI_SD   = np.array([ 8.6, 10.6, 12.9,  9.7, 12.1,  5.3,  9.9,  3.8], dtype=float)
FIG5_N_FIELDS = [20, 19, 2, 18, 20, 3, 10, 5]
FIG5_N_CELLS  = [20, 19, 16, 20, 20, 12, 20, 15]

FIG5_SHAPE = ["Spheroid", "Fusiform", "Fusiform", "Fusiform",
              "Fusiform", "Fusiform", "Filamentous", "Filamentous"]
FIG5_CONTENT = ["Clear", "Loaded", "Clear", "Mixed",
                "Loaded", "Clear", "Loaded", "Clear"]

FIG5_SHAPE_MARKER = {"Spheroid": "o", "Fusiform": "D", "Filamentous": "^"}

def _hex_to_rgba_tuple(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex, got {hex_color!r}")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)) + (alpha,)

def fig5_facecolor(i: int):
    genus = FIG5_GENERA[i]
    c = FIG5_CONTENT[i]
    if c == "Loaded":
        return COLORS[genus]
    elif c == "Clear":
        return "white"
    else:
        return _hex_to_rgba_tuple(COLORS[genus], 0.5)

fig5, (ax5a, ax5b) = plt.subplots(
    1, 2, figsize=(7.5, 4.0),
    gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.12}
)
y_pos = np.arange(len(FIG5_GENERA))

for i, genus in enumerate(FIG5_GENERA):
    color = COLORS[genus]
    marker = FIG5_SHAPE_MARKER[FIG5_SHAPE[i]]
    ax5a.plot([FIG5_AR_Q1[i], FIG5_AR_Q3[i]], [y_pos[i], y_pos[i]],
              color=color, linewidth=2.5, solid_capstyle="butt", zorder=2)
    ax5a.plot([FIG5_AR_Q1[i]], [y_pos[i]], "|", color=color, markersize=8,
              markeredgewidth=2, zorder=3)
    ax5a.plot([FIG5_AR_Q3[i]], [y_pos[i]], "|", color=color, markersize=8,
              markeredgewidth=2, zorder=3)
    ax5a.scatter(FIG5_AR_MEDIAN[i], y_pos[i], marker=marker, s=80,
                 facecolors=fig5_facecolor(i), edgecolors=color, linewidths=1.5, zorder=4)

for xv in [2.0, 4.0]:
    ax5a.axvline(x=xv, color="#AAAAAA", linestyle=":", linewidth=0.8, zorder=0)

zone_y = len(FIG5_GENERA) - 0.55
ax5a.text(1.25, zone_y, "Spheroid", fontsize=6, color="#888888", fontstyle="italic", ha="center")
ax5a.text(3.0,  zone_y, "Fusiform", fontsize=6, color="#888888", fontstyle="italic", ha="center")
ax5a.text(5.8,  zone_y, "Filamentous", fontsize=6, color="#888888", fontstyle="italic", ha="center")

ax5a.set_yticks(y_pos)
ax5a.set_yticklabels([f"  {g} ({FIG5_N_CELLS[i]})" for i, g in enumerate(FIG5_GENERA)],
                     fontsize=7.5, fontstyle="italic")
ax5a.set_xlabel("Aspect ratio, median [IQR]", fontsize=8.5)
ax5a.set_xlim(0.5, 8.0)
ax5a.set_ylim(-0.7, len(FIG5_GENERA) - 0.1)
ax5a.set_title("(a)", fontsize=10, fontweight="bold", loc="left")
ax5a.tick_params(axis="x", labelsize=7.5)
ax5a.spines["top"].set_visible(False)
ax5a.spines["right"].set_visible(False)

for i, genus in enumerate(FIG5_GENERA):
    color = COLORS[genus]
    marker = FIG5_SHAPE_MARKER[FIG5_SHAPE[i]]
    ax5b.errorbar(FIG5_PHI_MEAN[i], y_pos[i], xerr=FIG5_PHI_SD[i],
                  fmt="none", ecolor=color, elinewidth=1.8,
                  capsize=3, capthick=1.5, zorder=2)
    ax5b.scatter(FIG5_PHI_MEAN[i], y_pos[i], marker=marker, s=80,
                 facecolors=fig5_facecolor(i), edgecolors=color, linewidths=1.5, zorder=4)
    xtext = FIG5_PHI_MEAN[i] + FIG5_PHI_SD[i] + 2
    if genus == "Phalaenopsis":
        xtext = FIG5_PHI_MEAN[i] - FIG5_PHI_SD[i] - 2
        ha = "right"
    else:
        ha = "left"
    ax5b.text(xtext, y_pos[i], f"n\u2009=\u2009{FIG5_N_FIELDS[i]}", fontsize=5.5,
              color="#888888", va="center", ha=ha)

ax5b.annotate("upper bound\n(cell overlap)", xy=(89.6, y_pos[-1]),
              xytext=(68, y_pos[-1] + 0.55),
              fontsize=5.5, color=COLORS["Phalaenopsis"], fontstyle="italic", ha="center",
              arrowprops=dict(arrowstyle="-", color=_hex_to_rgba_tuple(COLORS["Phalaenopsis"], 0.5), lw=0.7))

ax5b.set_yticks(y_pos)
ax5b.set_yticklabels([""] * len(FIG5_GENERA))
ax5b.set_xlabel("Projected cell area fraction, φ₂D (% ± SD)", fontsize=8.5)
ax5b.set_xlim(10, 105)
ax5b.set_ylim(-0.7, len(FIG5_GENERA) - 0.1)
ax5b.set_title("(b)", fontsize=10, fontweight="bold", loc="left")
ax5b.tick_params(axis="x", labelsize=7.5)
ax5b.spines["top"].set_visible(False)
ax5b.spines["right"].set_visible(False)

fig5_leg_handles = []
for label, mk in [("Spheroid", "o"), ("Fusiform", "D"), ("Filamentous", "^")]:
    fig5_leg_handles.append(
        Line2D([0], [0], marker=mk, color="none", markerfacecolor="#666666",
               markeredgecolor="#444444", markersize=6.5, linestyle="None", label=label)
    )
fig5_leg_handles.append(Line2D([0], [0], linestyle="None", label=""))
fig5_leg_handles.append(
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#666666",
           markeredgecolor="#444444", markersize=6.5, linestyle="None", label="Loaded")
)
fig5_leg_handles.append(
    Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
           markeredgecolor="#444444", markeredgewidth=1.2,
           markersize=6.5, linestyle="None", label="Clear")
)
fig5_leg_handles.append(
    Line2D([0], [0], marker="s", color="none", markerfacecolor=(0.4, 0.4, 0.4, 0.5),
           markeredgecolor="#444444", markersize=6.5, linestyle="None", label="Mixed")
)

fig5.legend(handles=fig5_leg_handles, ncol=7, fontsize=6.5,
            loc="lower center", bbox_to_anchor=(0.5, -0.02),
            frameon=False, columnspacing=1.2, handletextpad=0.3)

plt.subplots_adjust(bottom=0.18)
fig5.savefig(OUT / "Figure_5_eleutherocyte_morphometry.png")
fig5.savefig(OUT / "Figure_5_eleutherocyte_morphometry.pdf")
plt.close(fig5)

# =====================================================================
# CSV EXPORTS
# =====================================================================

with open(OUT / "Table_1_amplitude_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(amp_summary_rows[0].keys()))
    w.writeheader()
    w.writerows(amp_summary_rows)

with open(OUT / "Table_S_frequency_sweep_data.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["genus", "omega_rad_s", "Gp_Pa", "Gpp_Pa", "tan_delta"])
    for genus in GENUS_ORDER:
        d = FREQ_DATA[genus]
        for i in range(len(d["omega"])):
            w.writerow([genus, f"{d['omega'][i]:.4f}", f"{d['Gp'][i]:.2f}", f"{d['Gpp'][i]:.2f}", f"{d['tand'][i]:.4f}"])

with open(OUT / "Table_S_normalised_creep_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["genus", "gamma_peak_pct", "J_60s_Pa_inv", "gamma_resid_pct", "recovery_pct"])
    for genus in fig4_order:
        d = NORM_CREEP_DATA[genus]
        gp = d["c_pct"][-1]
        gp_frac = gp / 100.0
        resid_frac = gp_frac + d["r_pct"][-1] / 100.0
        recov = (gp_frac - resid_frac) / gp_frac * 100.0
        w.writerow([genus, f"{gp:.5f}", f"{gp_frac / 10.0:.6e}", f"{resid_frac * 100.0:.5f}", f"{recov:.2f}"])


with open(OUT / "Table_5_eleutherocyte_morphometry.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["genus", "ar_median", "ar_q1", "ar_q3", "phi2D_mean_pct", "phi2D_sd_pct", "n_fields", "n_cells", "shape", "content"])
    for i, genus in enumerate(FIG5_GENERA):
        w.writerow([
            genus, f"{FIG5_AR_MEDIAN[i]:.2f}", f"{FIG5_AR_Q1[i]:.2f}", f"{FIG5_AR_Q3[i]:.2f}",
            f"{FIG5_PHI_MEAN[i]:.1f}", f"{FIG5_PHI_SD[i]:.1f}",
            FIG5_N_FIELDS[i], FIG5_N_CELLS[i], FIG5_SHAPE[i], FIG5_CONTENT[i]
        ])

print("Saved outputs to:", OUT)