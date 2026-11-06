"""Heat model checks against the literature, and the transient time scales.

1. Govorov & Richardson 2007: surface rise of one sphere, dT = C_abs I / (4 pi k R),
   grows as R^2; their Fig. 1b gives ~4.3 K for R = 30 nm at 1e4 W/cm^2 in water.
2. Skirtach 2005: raising Fs from 0.2 to 0.4 raised the heating slope 1.58x.
   Their nearest-neighbour model gives sqrt(2) = 1.41. Adding up every particle
   on the capsule (full superposition) gives 2.0.
3. Transient: a heated sphere of radius R in water; how long until it is close
   to steady state? Should scale as R^2 / kappa.
"""

import os
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import params as prm
import optics as op

HERE = os.path.dirname(os.path.abspath(__file__))


def surface_dT_sphere(R, I, n_med, lam=None, k=None, size_corr=True):
    k = k or prm.v("k_water")
    if lam is None:
        lams = np.arange(480, 600, 1.0)
        c = op.cabs_mie(lams, R, n_med, size_corr)
        i = np.argmax(c)
        return c[i] * I / (4 * np.pi * k * R), lams[i]
    return op.cabs_mie(lam, R, n_med, size_corr)[0] * I / (4 * np.pi * k * R), lam


def skirtach_sum(Fs, R0=2.5e-6, r0=3.75e-9, rng=0):
    """Rise at one particle on the capsule wall, in units of q/(4 pi k), summing
    all particles placed at random on the wall (no-overlap zone 2 r0)."""
    rng = np.random.default_rng(rng)
    N = int(4 * Fs * R0**2 / r0**2)
    p = rng.normal(size=(N, 3))
    p *= R0 / np.linalg.norm(p, axis=1)[:, None]
    me = np.array([0, 0, R0])
    d = np.linalg.norm(p - me, axis=1)
    d = d[d > 2 * r0]
    near = 1 / d.min()
    return 1 / r0 + np.sum(1 / d), 1 / r0 + near


def transient_sphere(R, t_end, kappa=None, n_r=400, n_t=400):
    """Uniform heat source inside r < R switched on at t = 0, same diffusivity
    inside and out (water). Returns t, surface temperature / steady value."""
    kappa = kappa or prm.v("kappa_water")
    r_max = 400 * R
    # graded grid: fine near the particle
    edges = np.concatenate([np.linspace(0, R, 41), R * np.logspace(0, np.log10(r_max / R), n_r)[1:]])
    rc = 0.5 * (edges[1:] + edges[:-1])
    vol = (edges[1:]**3 - edges[:-1]**3) / 3
    area = edges[1:-1]**2
    dr = rc[1:] - rc[:-1]
    g = kappa * area / dr
    n = len(rc)
    main = np.zeros(n); off = -g
    main[:-1] += g; main[1:] += g
    Lap = sp.diags([main, off, off], [0, 1, -1], format="csc")   # outer edge: T = 0 beyond last cell
    main_b = main.copy(); main_b[-1] += kappa * edges[-1]**2 / (edges[-1] - rc[-1])
    Lap = sp.diags([main_b, off, off], [0, 1, -1], format="csc")
    src = np.where(rc < R, 1.0, 0.0) * vol          # source density 1 (units: K/s)
    T_ss_surface = R**2 / (3 * kappa)               # analytic steady value at r = R
    ts = np.concatenate([[0], np.logspace(np.log10(t_end) - 6, np.log10(t_end), n_t)])
    T = np.zeros(n)
    i_s = np.searchsorted(rc, R)
    out = [0.0]
    M = sp.diags(vol, format="csc")
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        T = spla.spsolve(M + dt * Lap, M @ T + dt * src)
        Ts = np.interp(R, rc, T)
        out.append(Ts / T_ss_surface)
    return ts, np.array(out)


def run():
    import style as S
    res = {}

    # ---- 1. Govorov R^2 and Fig. 1b -----------------------------------
    n_gov = np.sqrt(1.8)
    I_gov = 1e4 * 1e4
    dT30, lpk = surface_dT_sphere(30e-9, I_gov, n_gov, size_corr=False)
    res["govorov_R30_dT_K"] = round(float(dT30), 2)
    res["govorov_R30_peak_nm"] = float(lpk)
    res["govorov_fig1b_read_K"] = prm.v("govorov_fig1")
    Rg = np.logspace(np.log10(2e-9), np.log10(60e-9), 25)
    dq = np.array([op.cabs_qs(520, r, n_gov, size_corr=False) * I_gov / (4 * np.pi * prm.v("k_water") * r) for r in Rg])
    dm = np.array([surface_dT_sphere(r, I_gov, n_gov, size_corr=False)[0] for r in Rg])
    dm_sc = np.array([surface_dT_sphere(r, I_gov, n_gov, size_corr=True)[0] for r in Rg])
    m = Rg <= 10e-9
    res["slope_quasistatic"] = round(float(np.polyfit(np.log(Rg), np.log(dq), 1)[0]), 3)
    res["slope_mie_R_le_10nm"] = round(float(np.polyfit(np.log(Rg[m]), np.log(dm[m]), 1)[0]), 3)

    # ---- 2. Skirtach Fs ------------------------------------------------
    Fs = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
    full, nn = zip(*[skirtach_sum(f) for f in Fs])
    full = np.array(full); nn = np.array(nn)
    i2 = 2
    res["skirtach_ratio_full_sum"] = round(float(full[4] / full[i2]), 3)
    res["skirtach_ratio_sqrtFs_model"] = round(float(np.sqrt(2)), 3)
    res["skirtach_ratio_measured"] = prm.v("skirtach_ratio")
    # error on measured ratio, from the error bars in Skirtach Fig. 6b at dE = 48
    # (read off the plot: Au I 5.0 +/- 0.7 K, Au II 7.9 +/- 0.8 K)
    err = 1.58 * np.hypot(0.7 / 5.0, 0.8 / 7.9)
    res["skirtach_ratio_measured_err"] = round(float(err), 2)

    fig, boxes = S.report_halves(2.6)
    a1 = fig.add_axes(boxes[0]); a2 = fig.add_axes(boxes[1])
    a1.loglog(Rg * 1e9, dq, color=S.GREY, lw=1.2, ls="--")
    a1.loglog(Rg * 1e9, dm, color=S.BLUE, lw=1.8)
    a1.loglog(Rg * 1e9, dm_sc, color=S.BLUE, lw=1.0, ls=":")
    a1.plot([30], [prm.v("govorov_fig1")], "D", color=S.CRIMSON, ms=5)
    a1.text(30, prm.v("govorov_fig1") * 2.4, "Govorov\nFig. 1b", color=S.CRIMSON, fontsize=7.5, ha="center")
    a1.text(2.3, 0.6, "slope %.2f, i.e.\n" % res["slope_quasistatic"] + r"$\Delta T \propto R^2$", color=S.GREY, fontsize=7.5)
    a1.text(9, 0.02, "Mie (solid)\nsize-corrected (dotted)", color=S.BLUE, fontsize=7.5)
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    a1.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
    a1.xaxis.set_major_formatter(FixedFormatter(["2", "5", "10", "20", "50"]))
    a1.xaxis.set_minor_formatter(NullFormatter())
    a1.set_xlabel("gold sphere radius R (nm)")
    a1.set_ylabel(r"surface rise $\Delta T$ (K)")
    a1.set_title(r"one sphere, $10^4$ W/cm$^2$, at resonance")
    a1.set_xlim(2, 60)

    a2.plot(Fs, np.sqrt(Fs / 0.2), color=S.GREY, lw=1.4, ls="--")
    a2.plot(Fs, full / full[i2], color=S.BLUE, lw=1.8)
    a2.errorbar([0.2, 0.4], [1.0, 1.58], yerr=[0.7 / 5.0, err], fmt="s", color=S.CRIMSON, ms=4.5,
                capsize=2.5, lw=1.0)
    a2.text(0.34, 0.85, "Skirtach's model " + r"($\sqrt{F_s}$)", color=S.GREY, fontsize=7.5, ha="center")
    a2.text(0.24, 1.75, "adding every particle", color=S.BLUE, fontsize=7.5, ha="right")
    a2.text(0.2, 0.62, "measured", color=S.CRIMSON, fontsize=7.5, ha="center")
    a2.set_xlabel(r"surface filling factor $F_s$")
    a2.set_ylabel(r"heating slope (relative to $F_s$ = 0.2)")
    a2.set_title("5 $\\mu$m capsule, Skirtach 2005")
    a2.set_xlim(0, 0.45); a2.set_ylim(0, 2.3); a2.set_xticks([0.1, 0.2, 0.3, 0.4])
    S.save_report(fig, "fig_heat_checks", panels=[[a1], [a2]])

    # ---- 3. transient -------------------------------------------------
    kap = prm.v("kappa_water")
    t_d, f_d = transient_sphere(1.0, 1e4, kappa=1.0)     # dimensionless: R = 1, kappa = 1
    t63 = float(np.interp(0.63, f_d, t_d))
    t90 = float(np.interp(0.90, f_d, t_d))
    res["t63_over_R2_kappa"] = round(t63, 3)
    res["t90_over_R2_kappa"] = round(t90, 2)
    objs = [("gold particle, R = 3 nm", 3e-9), ("our carrier, R = 70 nm", 70e-9), ("Skirtach capsule, R = 2.5 um", 2.5e-6)]
    res["t63_s"] = {n: t63 * R**2 / kap for n, R in objs}
    res["fraction_of_steady_after_10ns"] = {n: float(np.interp(10e-9 * kap / R**2, t_d, f_d)) for n, R in objs}
    # numerical check of R^2: run two real sizes
    slope_chk = []
    for R in (10e-9, 1e-6):
        tt, ff = transient_sphere(R, 1e4 * R**2 / kap)
        slope_chk.append(np.interp(0.63, ff, tt))
    res["tau_slope_numeric"] = round(float(np.log(slope_chk[1] / slope_chk[0]) / np.log(100)), 3)

    fig, boxes = S.report_halves(2.6)
    b1 = fig.add_axes(boxes[0]); b2 = fig.add_axes(boxes[1])
    Rr = np.logspace(np.log10(2e-9), np.log10(1e-5), 50)
    b1.loglog(Rr * 1e9, t63 * Rr**2 / kap, color=S.BLUE, lw=1.8)
    b1.axhline(10e-9, color=S.CRIMSON, lw=1.0, ls=":")
    b1.text(2.5, 14e-9, "10 ns pulse", color=S.CRIMSON, fontsize=7.5)
    for (n, R), mk in zip(objs, ("o", "s", "D")):
        b1.plot([R * 1e9], [t63 * R**2 / kap], mk, color=S.BLUE, ms=4.5)
    b1.text(5, t63 * (3e-9)**2 / kap * 0.3, "gold", color=S.BLUE, fontsize=7.5)
    b1.text(55, t63 * (70e-9)**2 / kap * 3, "carrier", ha="right", color=S.BLUE, fontsize=7.5)
    b1.text(3e3, t63 * (2.5e-6)**2 / kap * 0.15, "capsule", color=S.BLUE, fontsize=7.5)
    b1.set_xlabel("radius of the heated object (nm)")
    b1.set_ylabel("time to reach 63% of steady state (s)")
    b1.set_title(r"heating time grows as $R^2$")
    b1.set_xlim(2, 1e4)
    tt = np.logspace(-12, -2, 300)
    for (n, R), ls in zip(objs, (":", "-", "--")):
        b2.semilogx(tt, np.interp(tt * kap / R**2, t_d, f_d), color=S.BLUE, ls=ls, lw=1.6)
    b2.text(4e-11, 0.75, "gold\n3 nm", color=S.BLUE, fontsize=7.5, ha="right")
    b2.text(1.5e-7, 0.5, "carrier\n70 nm", color=S.BLUE, fontsize=7.5)
    b2.text(8e-5, 0.25, "capsule\n2.5 $\\mu$m", color=S.BLUE, fontsize=7.5)
    b2.axvspan(1e-12, 10e-9, color=S.CRIMSON_TINT, lw=0, zorder=0)
    b2.text(3e-11, 1.06, "during a 10 ns pulse", color=S.CRIMSON, fontsize=7.5)
    b2.set_ylim(-0.02, 1.15); b2.set_xlim(1e-12, 1e-2)
    b2.set_xlabel("time after the light is switched on (s)")
    b2.set_ylabel("surface rise / steady-state rise")
    b2.set_title("how fast each object heats up")
    S.save_report(fig, "fig_tau_vs_size", panels=[[b1], [b2]])

    json.dump(res, open(os.path.join(HERE, "results", "heat_checks.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run()
