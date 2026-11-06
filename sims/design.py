"""Sensitivity analysis and the design envelope.

Sensitivity: tumour-centre temperature after 15 min at the skin limit, changing
one input at a time between a low and a high value; and the time to release 90%
at 42 C as the (unknown) wall permeability changes.
Envelope: the ranges of gold, shell, irradiance and dwell time that give
41-43 C at the tumour centre, stay under the skin limit, and release the drug.
"""

import os
import json
import numpy as np

import params as prm
import pennes as pn
import release as rl
import safety as sf

HERE = os.path.dirname(os.path.abspath(__file__))
G0 = 8.0


def centre_T(**kw):
    t, rc, T = pn.simulate(t_on=900, t_end=900, dt=3.0, **kw)
    return float(T[-1, 0]), t, T[:, 0]


def run():
    import style as S
    res = {}
    base, _, _ = centre_T(G=G0)
    res["base_T_centre"] = round(base, 2)
    cases = [
        ("gold in tumour\n4 to 16 $\\mu$g/g", dict(G=G0 / 2), dict(G=G0 * 2)),
        ("light reaching tumour\n50% to 100%", dict(G=G0, light_factor=0.5), dict(G=G0, light_factor=1.0)),
        ("tumour size\n6 to 15 mm", dict(G=G0, a=3e-3), dict(G=G0, a=7.5e-3)),
        ("blood flow\nhigh to low", dict(G=G0, w_b=5.0), dict(G=G0, w_b=0.5)),
        ("index around rods\n1.33 or 1.5 (best 1.45)", dict(G=G0, n_med=1.33), dict(G=G0, n_med=1.5)),
        ("tissue conductivity\n0.6 to 0.4 W/m K", dict(G=G0, k=0.6), dict(G=G0, k=0.4)),
    ]
    rows = []
    for lab, lo, hi in cases:
        tl, _, _ = centre_T(**lo)
        th, _, _ = centre_T(**hi)
        rows.append((lab, tl, th))
    rows.sort(key=lambda r: abs(r[2] - r[1]))
    res["tornado"] = {r[0]: [round(r[1], 2), round(r[2], 2)] for r in rows}

    P = np.logspace(-10, -5, 60)
    t90 = np.log(10) / rl.k_total(42.0, P)
    res["P_where_wall_doubles_t90_cm_s"] = float(np.interp(2 * t90[-1], t90[::-1], P[::-1]))

    fig, boxes = S.report_halves(2.6, left=1.55)
    a1 = fig.add_axes(boxes[0]); a2 = fig.add_axes(boxes[1])
    y = np.arange(len(rows))
    for i, (lab, tl, th) in enumerate(rows):
        a1.barh(i, th - base, left=base, color=S.BLUE, height=0.55)
        a1.barh(i, tl - base, left=base, color=S.CRIMSON, height=0.55)
    a1.axvline(base, color="k", lw=0.8)
    a1.axvspan(41, 43, color=S.BAND, lw=0, zorder=0)
    a1.set_yticks(y); a1.set_yticklabels([r[0] for r in rows], fontsize=7)
    a1.set_xlim(38, 48); a1.set_xticks([38, 41, 43, 46])
    a1.set_xlabel("centre temperature at 15 min (°C)")
    a1.set_title("what the heating depends on")
    a1.spines["left"].set_visible(False); a1.tick_params(axis="y", length=0)

    a2.loglog(P, t90, color=S.BLUE, lw=1.8)
    a2.axhline(120, color=S.GREY, lw=0.9, ls=":")
    a2.text(1.5e-6, 140, "heating time", ha="center", color=S.GREY, fontsize=7.5)
    p_cross = float(np.interp(120, t90[::-1], P[::-1])); res["P_where_t90_equals_heating_time"] = p_cross
    a2.axvspan(1e-10, p_cross, color=S.CRIMSON_TINT, lw=0, zorder=0)
    a2.text(1.3e-10, 12, "wall would\nslow release", color=S.CRIMSON, fontsize=7.5)
    a2.text(1e-6, 14, "lipid sets the pace", color=S.BLUE, fontsize=7.5, ha="center")
    a2.set_xlim(1e-10, 1e-5); a2.set_ylim(10, 1e5); a2.set_xticks([1e-9, 1e-7, 1e-5])
    a2.set_xlabel("doxorubicin permeability of the wall (cm/s)")
    a2.set_ylabel("time to release 90% at 42 °C (s)")
    a2.set_title("the unknown: shell permeability")
    S.save_report(fig, "fig_sensitivity", panels=[[a1], [a2]])

    # ---------------- design envelope --------------------------------
    env = []
    for I in (0.1, 0.2, prm.v("MPE_cw_808")):
        for wb in (0.5, prm.v("w_blood"), 5.0):
            Gs = np.linspace(1, 60, 119)
            Tc = []
            for G in Gs:
                Tc.append(centre_T(G=G, I_Wcm2=I, w_b=wb)[0])
            Tc = np.array(Tc)
            ok = (Tc >= 41) & (Tc <= 43)
            if ok.any():
                g_lo, g_hi = Gs[ok].min(), Gs[ok].max()
                Tmid, t, Tt = centre_T(G=0.5 * (g_lo + g_hi), I_Wcm2=I, w_b=wb)
                t41 = float(np.interp(41, Tt, t)) if Tt.max() > 41 else np.nan
                F = rl.release(t, Tt)
                t90 = float(np.interp(0.9, F, t)) if F.max() > 0.9 else np.nan
                env.append(dict(I_Wcm2=round(I, 2), w_b=wb, gold_ugpg=[round(g_lo, 1), round(g_hi, 1)],
                                t_to_41C_min=round(t41 / 60, 1), t_to_90pct_release_min=round(t90 / 60, 1)))
    res["envelope_rows"] = env
    res["size"] = sf.size_budget(100, 4)

    lines = [
        "DESIGN ENVELOPE (from the models; see simulations_explained.txt for caveats)",
        "",
        "Gold geometry      : nanorods, ~40 x 10 nm, aspect ratio 3.8-4.1 (peak 800-830 nm in n~1.45);",
        "                     spheres cannot reach 800 nm on this carrier (peak <= ~560 nm)",
        "Gold per carrier   : 20-50 rods (Fs ~0.15-0.4); heating depends on total gold per g tissue",
        "Core + shell       : 100 nm core + 4 PSS/PAH layers (~11 nm wall) + rods + HA = ~148 nm;",
        "                     5-6 layers -> 153-159 nm (just over the 150 nm window)",
        "Irradiance         : <= 0.33 W/cm2 at 808 nm (CW skin limit, > 10 s)",
        "",
        "Gold needed in the tumour (ug Au / g tissue) for 41-43 C at the centre, and timing:",
        "  irradiance  blood flow   gold range    time to 41 C   time to 90% released",
    ]
    for r in env:
        lines.append("  %4.2f W/cm2  %-11s  %4.1f-%4.1f     %5.1f min      %5.1f min" % (
            r["I_Wcm2"], {0.5: "low", 2.0: "typical", 5.0: "high"}[r["w_b"]], r["gold_ugpg"][0], r["gold_ugpg"][1],
            r["t_to_41C_min"], r["t_to_90pct_release_min"]))
    lines += ["", "Dwell time         : a few minutes to heat + < 1 min to release; 10-15 min total keeps the",
              "                     centre in 41-43 C with CEM43 ~ a few min (mild hyperthermia)"]
    txt = "\n".join(lines)
    open(os.path.join(HERE, "results", "design_envelope.txt"), "w").write(txt + "\n")
    print(txt)
    json.dump(res, open(os.path.join(HERE, "results", "design.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    run()
