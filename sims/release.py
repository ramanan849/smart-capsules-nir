"""Release model: fraction of doxorubicin out of the carrier vs time and temperature.

Two barriers in series:
  - the thermosensitive lipid bilayer, first-order rate k_L(T): a slow leak at
    body temperature plus a fast release switched on around the lipid
    transition (smooth step of width w at T_m);
  - the PSS/PAH wall, k_shell = 3 P / R_core (Fickian, permeability P, thin wall).
  k = 1 / (1/k_L + 1/k_shell),   dF/dt = k(T(t)) (1 - F).

Anchors:
  slow leak  : Deng 2013 Fig. 6B, doxorubicin from LbL-coated liposomes, ~11% at 24 h
               (their liposome is rigid DSPC/cholesterol, so for a thermosensitive core this
               is a best case; real LTSL leak more)
  fast step  : LTSL literature, 80-100% in 20-40 s at 41.3 C, ~50% in <20 s at 40 C
               (Needham 2000, as quoted by Yarmolenko et al. 2010) - not checked in the primary paper.
Korsmeyer-Peppas fit (M_t/M_inf = k t^n, first 60% only) on Deng's curves.
"""

import os
import json
import numpy as np
from scipy.optimize import curve_fit

import params as prm

HERE = os.path.dirname(os.path.abspath(__file__))

# digitised by hand from Deng et al. ACS Nano 2013 (page images), +/- ~2% and ~1 h
DENG_DOX_6B = np.array([[0, 0], [4, 4], [8, 7], [24, 11], [33, 27], [45, 32], [55, 47], [70, 54]])
DENG_SIRNA_2E = np.array([[0, 0], [1, 5], [2, 8], [4, 12], [8, 17], [12, 22], [24, 28], [36, 57],
                          [48, 92], [55, 100], [71, 100]])

TM = 40.5          # C, middle of the 40-41 C range quoted for LTSL
W = 0.3            # K, width of the step
K_HOT = 0.08       # 1/s, gives ~90% in 30 s at 41.3 C
R_CORE = 50e-7     # cm (100 nm core)
_K37_TARGET = -np.log(1 - prm.v("deng_dox_24h")) / (24 * 3600)
K37 = _K37_TARGET - K_HOT / (1 + np.exp((TM - 37.0) / W))   # so that k(37 C) matches Deng at 24 h


def k_lipid(T):
    T = np.asarray(T, float)
    return K37 + (K_HOT - K37) / (1 + np.exp(-(T - TM) / W))


def k_total(T, P=None):
    kl = k_lipid(T)
    if P is None:
        return kl
    ks = 3 * P / R_CORE
    return 1 / (1 / kl + 1 / ks)


def release(t, T, P=None):
    """t (s) array, T (C) array same length -> released fraction."""
    F = np.zeros(len(t))
    for i in range(1, len(t)):
        k = k_total(0.5 * (T[i] + T[i - 1]), P)
        F[i] = 1 - (1 - F[i - 1]) * np.exp(-k * (t[i] - t[i - 1]))
    return F


def kp_fit(data):
    d = data[(data[:, 1] > 0) & (data[:, 1] <= 60)]
    f = lambda t, k, n: k * t**n
    (k, n), _ = curve_fit(f, d[:, 0], d[:, 1] / 100, p0=(0.05, 0.6))
    return k, n


def run():
    import style as S
    res = {"Tm_C": TM, "width_K": W, "k_hot_per_s": K_HOT, "k37_per_s": K37}
    res["t50_s_at_40C"] = float(np.log(2) / k_lipid(40.0))
    res["t90_s_at_41.3C"] = float(np.log(10) / k_lipid(41.3))
    res["t90_s_at_42C"] = float(np.log(10) / k_lipid(42.0))
    res["fraction_24h_at_37C"] = float(1 - np.exp(-k_lipid(37) * 86400))
    for name, d in (("deng_dox_6B", DENG_DOX_6B), ("deng_siRNA_2E", DENG_SIRNA_2E)):
        k, n = kp_fit(d)
        res["KP_n_" + name] = round(float(n), 2)
    # shell permeability: where does the wall start to slow the triggered release?
    for P in (1e-6, 1e-7, 1e-8):
        res["t90_s_at_42C_P%.0e" % P] = float(np.log(10) / k_total(42.0, P))
    # driven by the Pennes temperature history at the tumour centre and edge
    pr = np.load(os.path.join(HERE, "results", "pennes_run.npz"))
    Fc = release(pr["t"], pr["centre"])
    Fe = release(pr["t"], pr["edge"])
    res["released_centre_after_15min"] = round(float(Fc[pr["t"] == 900][0]), 3)
    res["released_edge_after_15min"] = round(float(Fe[pr["t"] == 900][0]), 3)
    res["time_centre_90pct_min"] = round(float(np.interp(0.9, Fc, pr["t"]) / 60), 2)

    fig, boxes = S.report_halves(2.6)
    a1 = fig.add_axes(boxes[0]); a2 = fig.add_axes(boxes[1])
    th = np.linspace(0, 72, 300)
    a1.plot(DENG_DOX_6B[:, 0], DENG_DOX_6B[:, 1], "o", color=S.CRIMSON, ms=4)
    a1.plot(DENG_SIRNA_2E[:, 0], DENG_SIRNA_2E[:, 1], "^", color=S.GREY, ms=4, mfc="none")
    a1.plot(th, 100 * (1 - np.exp(-k_lipid(37) * th * 3600)), color=S.BLUE, lw=1.8)
    a1.text(70, 62, "Deng: doxorubicin\nfrom coated liposome", color=S.CRIMSON, fontsize=7.5, ha="right")
    a1.text(30, 88, "Deng: siRNA from\nthe film (not drug)", color=S.GREY, fontsize=7.5, ha="right")
    a1.text(70, 6, "our model at 37 °C\n(matched at 24 h)", color=S.BLUE, fontsize=7.5, ha="right")
    a1.set_xlim(0, 72); a1.set_ylim(0, 105)
    a1.set_xticks([0, 24, 48, 72])
    a1.set_xlabel("time (h)"); a1.set_ylabel("released (%)")
    a1.set_title("no laser: slow leak at body temperature")

    ts = np.arange(0, 20 * 60 + 1, 1.0)
    for P, ls, lab in ((None, "-", "lipid only"), (1e-8, "--", "slow wall, P = 10$^{-8}$ cm/s")):
        a2.plot(ts / 60, 100 * release(ts, np.full(len(ts), 42.0), P), color=S.BLUE, ls=ls, lw=1.6, label="42 °C, " + lab)
    a2.plot(pr["t"] / 60, 100 * Fc, color=S.CRIMSON, lw=1.6, label="laser at skin limit, tumour centre")
    a2.set_xlim(0, 15); a2.set_ylim(0, 105); a2.set_xticks([0, 5, 10, 15])
    a2.set_xlabel("time (min)"); a2.set_ylabel("released (%)")
    a2.set_title("with heat: release in minutes")
    a2.legend(loc="lower right", fontsize=7.5)
    S.save_report(fig, "fig_release", panels=[[a1], [a2]])

    json.dump(res, open(os.path.join(HERE, "results", "release.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run()
