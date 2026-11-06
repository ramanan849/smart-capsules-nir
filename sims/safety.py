"""Safety and mechanics checks.

1. Laser limit: CW skin MPE for exposures > 10 s (ANSI Z136.1 / IEC 60825-1, as
   quoted in secondary sources): 0.2 C_A W/cm^2, C_A = 1 (400-700 nm),
   10^(0.002(lam-700)) (700-1050 nm), 5 (1050-1400 nm). Beam areas > 100 cm^2
   get a lower limit; ignored here.
2. Thermal dose: CEM43 = sum R^(43-T) dt, R = 0.25 below 43 C, 0.5 at or above
   (Sapareto & Dewey 1984).
3. Size budget and Reissner validity (h/R < 1/10, Fery & Weinkamer 2007).
"""

import os
import json
import numpy as np

import params as prm

HERE = os.path.dirname(os.path.abspath(__file__))


def mpe_cw_skin(lam_nm):
    lam_nm = np.asarray(lam_nm, float)
    CA = np.where(lam_nm < 700, 1.0, np.where(lam_nm < 1050, 10 ** (0.002 * (lam_nm - 700)), 5.0))
    return 0.2 * CA


def cem43(t, T):
    T = np.asarray(T); dt = np.diff(t, prepend=t[0])
    R = np.where(T >= 43, 0.5, 0.25)
    return np.cumsum(R ** (43 - T) * dt) / 60.0       # minutes


def size_budget(d_core_nm, n_layers, rods=True):
    wall = n_layers * prm.v("layer_dd") * 1e9 / 2          # nm of wall from Deng's diameter growth
    gold = 10.0 if rods else 0.0
    ha = prm.v("layer_dd") * 1e9 / 2
    d_total = d_core_nm + 2 * (wall + gold + ha)
    R = d_core_nm / 2 + wall
    return dict(wall_nm=round(wall, 1), d_total_nm=round(d_total, 1), h_over_R=round(wall / R, 3),
                h_over_R_with_rods=round((wall + gold) / (R + gold), 3))


def run():
    import style as S
    res = {}
    res["MPE_808"] = round(float(mpe_cw_skin(808)), 3)
    res["MPE_830"] = round(float(mpe_cw_skin(830)), 3)
    res["MPE_1064_cw"] = round(float(mpe_cw_skin(1064)), 3)
    res["oneal_over_MPE"] = round(prm.v("I_oneal") / res["MPE_808"], 1)
    res["angelatos_avg_over_MPE"] = round(prm.v("I_angelatos_avg") / res["MPE_1064_cw"], 1)
    res["angelatos_pulse_over_single_pulse_MPE"] = round(0.7 / 0.1, 1)
    res["check_cem43_43C_10min"] = float(cem43(np.arange(0, 601.0), np.full(601, 43.0))[-1])
    pr = np.load(os.path.join(HERE, "results", "pennes_run.npz"))
    t = pr["t"]
    for k in ("centre", "edge", "out2mm"):
        res["CEM43_min_" + k] = round(float(cem43(t, pr[k])[-1]), 3)
    res["size_budget"] = {"%d nm core, %d layers" % (d, n): size_budget(d, n) for d in (100, 120) for n in (4, 5, 6)}

    fig, boxes = S.report_halves(2.6)
    a1 = fig.add_axes(boxes[0]); a2 = fig.add_axes(boxes[1])
    lam = np.linspace(400, 1400, 500)
    a1.semilogy(lam, mpe_cw_skin(lam), color=S.CRIMSON, lw=1.8)
    a1.fill_between(lam, mpe_cw_skin(lam), 1e-2, color=S.BLUE_TINT, lw=0, zorder=0)
    a1.axvspan(800, 830, color=S.BAND, lw=0, zorder=0)
    pts = [(808, prm.v("MPE_cw_808"), "o", S.BLUE, "ours: 0.33 W/cm$^2$", (560, 0.1)),
           (808, prm.v("I_oneal"), "s", S.CRIMSON, "O'Neal 2004: 4 W/cm$^2$, 3 min", (420, 1.3)),
           (1064, prm.v("I_angelatos_avg"), "D", S.CRIMSON, "Angelatos 2005 (average)", (870, 30))]
    for x, y, mk, col, lab, xy in pts:
        a1.plot([x], [y], mk, color=col, ms=5)
        a1.annotate(lab, xy=(x, y), xytext=xy, fontsize=7.5, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
    a1.text(1380, 0.035, "allowed", color=S.BLUE, fontsize=7.5, ha="right")
    a1.text(420, 0.42, "skin limit, > 10 s", color=S.CRIMSON, fontsize=7.5)
    a1.set_xlim(400, 1400); a1.set_ylim(1e-2, 1e2); a1.set_xticks([600, 800, 1000, 1200, 1400])
    a1.set_xlabel("wavelength (nm)"); a1.set_ylabel("irradiance on skin (W/cm$^2$)")
    a1.set_title("continuous-wave laser limit for skin")

    tm = t / 60
    for k, ls, col, lab in (("centre", "-", S.BLUE, "tumour centre"), ("edge", "--", S.BLUE, "tumour edge"),
                            ("out2mm", ":", S.GREY, "2 mm into normal tissue")):
        a2.semilogy(tm[1:], cem43(t, pr[k])[1:], color=col, ls=ls, lw=1.6, label=lab)
    a2.axvspan(0, 15, color=S.CRIMSON_TINT, lw=0, zorder=0)
    a2.set_xlim(0, 20); a2.set_ylim(1e-4, 10)
    a2.set_xlabel("time (min)"); a2.set_ylabel("thermal dose CEM43 (min)")
    a2.set_title("thermal dose, laser on for 15 min")
    a2.text(15.5, 0.9, "laser off", color=S.CRIMSON, fontsize=7.5)
    a2.legend(loc="lower right")
    S.save_report(fig, "fig_safety", panels=[[a1], [a2]])

    json.dump(res, open(os.path.join(HERE, "results", "safety.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run()
