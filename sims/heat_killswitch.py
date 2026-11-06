"""Kill-switch check (run first): can a 120-150 nm carrier reach 41-43 C under a
tissue-safe CW beam at 808 nm?

(a) One object on its own: steady-state surface rise dT = P_abs / (4 pi k R).
(b) Many carriers in a tumour acting together: a sphere of tumour (radius a)
    with carriers spread evenly, heated at Q = c * C_abs * I, cooled by blood
    (Pennes). Steady-state rise at the centre (analytic):
        dT0 = Q/(k m^2) * [1 - (1 + m a) exp(-m a)],   m^2 = w_b c_b / k
    which goes to Q a^2 / (2k) when there is no blood flow.
"""

import os
import json
import numpy as np

import params as prm
import optics as op

HERE = os.path.dirname(os.path.abspath(__file__))

R_CARRIER = 70e-9                       # 140 nm carrier
R_GOLD = 65e-9                          # radius at which the gold sits
I_MPE = prm.v("MPE_cw_808") * 1e4       # W/m^2


def carrier_gold(kind="rods", n_rods=None, Fs=None, n_med=None):
    """Absorption cross-section at 808 nm (m^2) and gold mass (kg) of one carrier."""
    n_med = n_med or prm.v("n_shell")
    if kind == "rods":
        N = n_rods or prm.v("n_rods")
        L, D = prm.v("rod_L"), prm.v("rod_D")
        c1 = op.cabs_rod(808, L, D, n_med, "random")
        V1 = np.pi * D**2 * L / 6
    else:
        r0 = prm.v("au_sphere_r")
        N = op.n_spheres(Fs or prm.v("Fs_spheres"), R_GOLD, r0)
        c1 = op.cabs_qs(808, r0, n_med)     # coupling barely matters at 808 nm (see optics.json)
        V1 = 4 / 3 * np.pi * r0**3
    return N * float(c1), N * V1 * prm.v("rho_au"), N


def dT_single(C_abs, R, I=I_MPE, k=None):
    return C_abs * I / (4 * np.pi * (k or prm.v("k_tissue")) * R)


def dT_tumour_centre(Q, a, w_b=None, k=None):
    k = k or prm.v("k_tissue")
    w_b = prm.v("w_blood") if w_b is None else w_b
    if w_b == 0:
        return Q * a**2 / (2 * k)
    m = np.sqrt(w_b * prm.v("c_blood") / k)
    return Q / (k * m**2) * (1 - (1 + m * a) * np.exp(-m * a))


def gold_to_Q(G_ugpg, C_carrier, m_carrier, I=I_MPE):
    """Tumour gold concentration (ug Au per g tissue) -> heating rate Q (W/m^3)."""
    c = G_ugpg * 1e-6 * prm.v("rho_tissue") / m_carrier     # carriers per m^3
    return c * C_carrier * I, c


def run():
    import style as S
    res = {}
    Cr, mr, Nr = carrier_gold("rods")
    Cs, ms, Ns = carrier_gold("spheres")
    res["rods_per_carrier"] = Nr
    res["spheres_per_carrier_Fs0.2"] = Ns
    res["Cabs_carrier_rods_m2"] = Cr
    res["Cabs_carrier_spheres_m2"] = Cs
    res["I_MPE_Wm2"] = I_MPE
    res["dT_one_carrier_rods_K"] = dT_single(Cr, R_CARRIER)
    res["dT_one_carrier_spheres_K"] = dT_single(Cs, R_CARRIER)

    a = 5e-3            # 1 cm tumour
    for name, C, m in (("rods", Cr, mr), ("spheres", Cs, ms)):
        Q1, c1 = gold_to_Q(1.0, C, m)
        per = dT_tumour_centre(Q1, a)
        res["dT_centre_per_ugpg_%s_K" % name] = per
        res["gold_needed_for_5K_%s_ugpg" % name] = 5.0 / per
        res["mu_a_per_ugpg_%s_cm-1" % name] = c1 * C / 100
        for wb in (0.5, 5.0):
            res["gold_needed_5K_%s_wb%.1f" % (name, wb)] = 5.0 / dT_tumour_centre(Q1, a, w_b=wb)
    res["carriers_per_mL_for_5K_rods"] = gold_to_Q(res["gold_needed_for_5K_rods_ugpg"], Cr, mr)[1] * 1e-6

    # ------------------------------------------------------------------ figure
    fig, boxes = S.report_halves(2.7)
    a1 = fig.add_axes(boxes[0]); a2 = fig.add_axes(boxes[1])

    # (a) one object alone at the MPE
    R = np.logspace(np.log10(3e-9), np.log10(1e-4), 120)
    Rs = R[R <= 60e-9]
    c_sph = np.array([op.cabs_mie(808, r, prm.v("n_water"))[0] for r in Rs])
    a1.loglog(Rs * 1e9, dT_single(c_sph, Rs), color=S.GREY, lw=1.4, ls="--")
    a1.text(10, 4e-9, r"one gold sphere alone ($\propto R^2$)", color=S.GREY, fontsize=7.5, ha="left")
    dens = Nr / (4 * np.pi * R_GOLD**2)            # rods per m^2 of wall, kept fixed
    Rc = R[R >= 50e-9]
    a1.loglog(Rc * 1e9, dT_single(dens * 4 * np.pi * Rc**2 * Cr / Nr, Rc), color=S.BLUE, lw=1.8)
    a1.text(1.2e4, 3e-5, "carrier or capsule\nwith rods in its wall\n" + r"($\propto R$)",
            color=S.BLUE, fontsize=7.5, ha="center")
    a1.axhline(5, color=S.CRIMSON, lw=1.0, ls=":")
    a1.text(5, 7.5, "needed: about +5 K (37 to 42 °C)", color=S.CRIMSON, fontsize=7.5)
    a1.plot([R_CARRIER * 1e9], [res["dT_one_carrier_rods_K"]], "o", color=S.BLUE, ms=5)
    a1.annotate("our 140 nm carrier\n%.0e K" % res["dT_one_carrier_rods_K"],
                xy=(R_CARRIER * 1e9, res["dT_one_carrier_rods_K"]), xytext=(9, 3e-2),
                fontsize=7.5, color=S.BLUE, arrowprops=dict(arrowstyle="-", color=S.BLUE, lw=0.7))
    a1.axvspan(2.5e3, 5e3, color=S.BAND, lw=0, zorder=0)
    a1.text(3.5e3, 1e-7, "Skirtach\ncapsules", ha="center", fontsize=7.5, color=S.GREY)
    a1.set_xlim(3, 1e5); a1.set_ylim(1e-9, 1e2)
    a1.set_xlabel("radius of the heated object (nm)")
    a1.set_ylabel(r"steady temperature rise $\Delta T$ (K)")
    a1.set_title("one object alone, at the skin limit")

    # (b) many carriers in a 1 cm tumour
    G = np.logspace(np.log10(0.3), 3, 200)
    for wb, ls, lab in ((0.5, ":", "low blood flow"), (2.0, "-", "typical"), (5.0, "--", "high blood flow")):
        Q1, _ = gold_to_Q(1.0, Cr, mr)
        a2.loglog(G, G * dT_tumour_centre(Q1, a, w_b=wb), color=S.BLUE, ls=ls, lw=1.6, label="rods, " + lab)
    Q1s, _ = gold_to_Q(1.0, Cs, ms)
    a2.loglog(G, G * dT_tumour_centre(Q1s, a), color=S.CRIMSON, lw=1.4, ls="-.", label="spheres, typical")
    a2.axhline(5, color=S.CRIMSON, lw=1.0, ls=":")
    a2.axvline(prm.v("nunes_gold"), color=S.GREY, lw=0.8)
    a2.text(prm.v("nunes_gold") / 1.12, 2e-3, "Nunes, after IV dose:\n41 $\\mu$g/g", fontsize=7.5,
            color=S.GREY, ha="right")
    a2.text(200, 7, "need +5 K", color=S.CRIMSON, fontsize=7.5, ha="center")
    a2.set_xlim(0.3, 1e3); a2.set_ylim(1e-3, 1e4)
    a2.set_xlabel(r"gold in the tumour ($\mu$g per g tissue)")
    a2.set_ylabel(r"rise at tumour centre $\Delta T$ (K)")
    a2.set_title("many carriers together (1 cm tumour)")
    a2.legend(loc="upper left")
    S.save_report(fig, "fig_killswitch", panels=[[a1], [a2]])

    json.dump(res, open(os.path.join(HERE, "results", "killswitch.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run()
