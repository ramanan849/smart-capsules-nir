"""Pennes bioheat in and around a 1 cm tumour loaded with carriers.

    rho c dT/dt = k lap(T) - w_b c_b (T - T_body) + Q(r)

Spherically symmetric: tumour of radius a with carriers spread evenly in it,
heated at Q = mu_a * I (mu_a from the gold concentration); normal tissue around
it has no carriers. Simplification: the light reaching the tumour is taken to
be the skin irradiance (no fall-off with depth); sensitivity.py checks a 2x loss.
"""

import os
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import params as prm
import heat_killswitch as hk

HERE = os.path.dirname(os.path.abspath(__file__))
A_TUM = 5e-3


def Q_from_gold(G, I_Wcm2, light_factor=1.0, n_rods=None, n_med=None):
    C, m, _ = hk.carrier_gold("rods", n_rods=n_rods, n_med=n_med)
    Q, _ = hk.gold_to_Q(G, C, m, I=I_Wcm2 * 1e4 * light_factor)
    return Q


def simulate(G=8.0, I_Wcm2=None, t_on=15 * 60, t_end=20 * 60, dt=1.0, w_b=None, k=None,
             light_factor=1.0, n_med=None, a=A_TUM, r_max=4e-2, n=400):
    I_Wcm2 = prm.v("MPE_cw_808") if I_Wcm2 is None else I_Wcm2
    k = k or prm.v("k_tissue")
    w_b = prm.v("w_blood") if w_b is None else w_b
    rhoc = prm.v("rhoc_tissue")
    edges = np.linspace(0, r_max, n + 1)
    rc = 0.5 * (edges[1:] + edges[:-1])
    vol = (edges[1:]**3 - edges[:-1]**3) / 3
    g = k * edges[1:-1]**2 / (rc[1:] - rc[:-1])
    main = np.zeros(n); main[:-1] += g; main[1:] += g
    main[-1] += k * edges[-1]**2 / (edges[-1] - rc[-1])          # T = T_body far away
    main += w_b * prm.v("c_blood") * vol
    K = sp.diags([main, -g, -g], [0, 1, -1], format="csc")
    Q = np.where(rc < a, Q_from_gold(G, I_Wcm2, light_factor, n_med=n_med), 0.0) * vol
    M = sp.diags(rhoc * vol, format="csc")
    solve = spla.factorized((M + dt * K).tocsc())
    ts = np.arange(0, t_end + dt / 2, dt)
    th = np.zeros(n)
    out = np.zeros((len(ts), n))
    for i in range(1, len(ts)):
        src = Q if ts[i] <= t_on else 0.0 * Q
        th = solve(M @ th + dt * src)
        out[i] = th
    return ts, rc, prm.v("T_body") + out


def at(rc, T, r):
    return np.array([np.interp(r, rc, row) for row in T])


def run():
    import style as S
    res = {}
    # check against the analytic steady state of heat_killswitch
    ts, rc, T = simulate(G=8.0, t_on=4 * 3600, t_end=4 * 3600, dt=20.0)
    Q = Q_from_gold(8.0, prm.v("MPE_cw_808"))
    res["check_centre_numeric_K"] = round(float(T[-1, 0] - 37), 3)
    res["check_centre_analytic_K"] = round(float(hk.dT_tumour_centre(Q, A_TUM)), 3)

    G0 = 8.0
    ts, rc, T = simulate(G=G0)
    c, e, o = at(rc, T, 0.0), at(rc, T, A_TUM), at(rc, T, A_TUM + 2e-3)
    res["G_used_ugpg"] = G0
    res["T_centre_15min"] = round(float(c[ts == 15 * 60][0]), 2)
    res["T_edge_15min"] = round(float(e[ts == 15 * 60][0]), 2)
    res["T_2mm_out_15min"] = round(float(o[ts == 15 * 60][0]), 2)
    res["time_centre_to_41C_min"] = round(float(np.interp(41, c[:15 * 60], ts[:15 * 60]) / 60), 2) if c.max() > 41 else None
    res["time_constant_perfusion_s"] = prm.v("rhoc_tissue") / (prm.v("w_blood") * prm.v("c_blood"))
    # canonical nanoshell condition at the same gold load
    _, _, T4 = simulate(G=G0, I_Wcm2=prm.v("I_oneal"), t_on=3 * 60, t_end=3 * 60)
    res["T_centre_at_4Wcm2_3min_same_gold"] = round(float(T4[-1, 0]), 1)
    np.savez(os.path.join(HERE, "results", "pennes_run.npz"), t=ts, centre=c, edge=e, out2mm=o)

    fig, boxes = S.report_halves(2.6)
    a1 = fig.add_axes(boxes[0]); a2 = fig.add_axes(boxes[1])
    a1.axhspan(41, 43, color=S.BAND, lw=0, zorder=0)
    a1.axvspan(-A_TUM * 1e3, A_TUM * 1e3, color=S.BLUE_TINT, lw=0, zorder=0)
    for G, ls in ((5.0, ":"), (8.0, "-"), (12.0, "--")):
        _, rr, TT = simulate(G=G, t_on=15 * 60, t_end=15 * 60, dt=2.0)
        a1.plot(np.r_[-rr[::-1], rr] * 1e3, np.r_[TT[-1][::-1], TT[-1]], color=S.BLUE, ls=ls, lw=1.6,
                label="%g $\\mu$g/g gold" % G)
    a1.set_xlim(-15, 15); a1.set_ylim(36.5, 46)
    a1.set_xlabel("distance from tumour centre (mm)")
    a1.set_ylabel("temperature (°C)")
    a1.set_title("after 15 min at the skin limit")
    a1.text(0, 36.8, "tumour", color=S.BLUE, fontsize=7.5, ha="center")
    a1.text(14.5, 43.2, "41-43 °C", color=S.GREY, fontsize=7.5, ha="right")
    a1.legend(loc="upper left", fontsize=7.5)

    tm = ts / 60
    a2.axhspan(41, 43, color=S.BAND, lw=0, zorder=0)
    a2.axvspan(0, 15, color=S.CRIMSON_TINT, lw=0, zorder=0)
    a2.plot(tm, c, color=S.BLUE, lw=1.8)
    a2.plot(tm, e, color=S.BLUE, lw=1.4, ls="--")
    a2.plot(tm, o, color=S.GREY, lw=1.4, ls=":")
    a2.text(9, c[9 * 60] + 0.25, "centre", color=S.BLUE, fontsize=7.5, ha="center")
    a2.text(9, e[9 * 60] + 0.25, "tumour edge", color=S.BLUE, fontsize=7.5, ha="center")
    a2.text(9, o[9 * 60] - 0.8, "2 mm into normal tissue", color=S.GREY, fontsize=7.5, ha="center")
    a2.text(15.3, 45.2, "laser off", color=S.CRIMSON, fontsize=7.5)
    a2.set_xlim(0, 20); a2.set_ylim(36.5, 46)
    a2.set_xlabel("time (min)")
    a2.set_ylabel("temperature (°C)")
    a2.set_title("%g $\\mu$g/g gold, 0.33 W/cm$^2$" % G0)
    S.save_report(fig, "fig_pennes", panels=[[a1], [a2]])

    json.dump(res, open(os.path.join(HERE, "results", "pennes.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run()
