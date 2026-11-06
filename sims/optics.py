"""Optics model: how much light does the gold absorb, and at what wavelength?

1. Gold dielectric function: Johnson & Christy table, with the extra damping
   of small particles, gamma(R) = gamma_bulk + A vF / R.
2. One sphere: Mie theory (miepython) and the quasi-static limit as a check.
3. Rods: Gans theory (quasi-static prolate spheroid, depolarisation factors).
4. Many spheres on one carrier: coupled-dipole approximation (CDA).

Units: SI inside, nm for wavelengths and sizes in the plotting code.
"""

import os
import json
import numpy as np
from scipy.interpolate import CubicSpline
import miepython

import params as prm

HERE = os.path.dirname(os.path.abspath(__file__))
HBAR_EV = 6.582119569e-16      # eV s
C0 = 2.99792458e8

_d = np.loadtxt(os.path.join(HERE, prm.v("eps_au_table")), delimiter=",", comments="#")
_lam_um, _n, _k = _d.T
_eps_jc = (_n + 1j * _k) ** 2
_re = CubicSpline(_lam_um * 1e3, _eps_jc.real)
_im = CubicSpline(_lam_um * 1e3, _eps_jc.imag)


def eps_au(lam_nm, R=None):
    """Gold permittivity at vacuum wavelength lam_nm. With R (m), the Drude part
    gets the surface-scattering damping gamma = gamma_bulk + A vF / R."""
    lam_nm = np.asarray(lam_nm, float)
    e = _re(lam_nm) + 1j * _im(lam_nm)
    if R is None:
        return e
    w = 2 * np.pi * C0 / (lam_nm * 1e-9)
    wp = prm.v("hwp_au") / HBAR_EV
    gb = prm.v("hgamma_au") / HBAR_EV
    gR = gb + prm.v("A_sizecorr") * prm.v("vF_au") / R
    return e + wp**2 / (w**2 + 1j * w * gb) - wp**2 / (w**2 + 1j * w * gR)


# ---------------------------------------------------------------------------
# one sphere
# ---------------------------------------------------------------------------
def cabs_mie(lam_nm, R, n_med=1.33, size_corr=True):
    """Absorption cross-section (m^2) of a gold sphere of radius R (m), Mie theory."""
    lam_nm = np.atleast_1d(lam_nm)
    out = np.empty(len(lam_nm))
    for i, L in enumerate(lam_nm):
        e = eps_au(L, R if size_corr else None)
        m = np.sqrt(e)
        # miepython convention: absorbing spheres have negative imaginary index
        m = complex(m.real, -abs(m.imag))
        qext, qsca, _, _ = miepython.efficiencies(m, 2 * R * 1e9, L, n_env=n_med)
        out[i] = (qext - qsca) * np.pi * R**2
    return out


def alpha_sphere(lam_nm, R, n_med, size_corr=True):
    """Quasi-static polarisability (Gaussian units, volume): R^3 (e - em)/(e + 2 em).
    In SI this is alpha = 4 pi R^3 (e - em)/(e + 2 em), the form in our text."""
    e = eps_au(lam_nm, R if size_corr else None)
    em = n_med**2
    return R**3 * (e - em) / (e + 2 * em)


def cabs_qs(lam_nm, R, n_med=1.33, size_corr=True):
    """Quasi-static absorption: C_abs = k Im(alpha_SI) = 4 pi k Im(alpha_G)."""
    k = 2 * np.pi * n_med / (np.asarray(lam_nm) * 1e-9)
    return 4 * np.pi * k * np.imag(alpha_sphere(lam_nm, R, n_med, size_corr))


# ---------------------------------------------------------------------------
# rods (Gans)
# ---------------------------------------------------------------------------
def depol_prolate(ar):
    """Depolarisation factors (long axis, short axis) of a prolate spheroid."""
    e2 = 1 - 1 / ar**2
    e = np.sqrt(e2)
    PA = (1 - e2) / e2 * (np.log((1 + e) / (1 - e)) / (2 * e) - 1)
    return PA, (1 - PA) / 2


def cabs_rod(lam_nm, L, D, n_med=1.33, orient="random"):
    """Gans absorption cross-section (m^2) of a rod modelled as a prolate
    spheroid of length L and width D. orient: 'random' (average over
    orientations) or 'long' (light polarised along the rod)."""
    ar = L / D
    V = np.pi * D**2 * L / 6
    e = eps_au(lam_nm, D / 2)
    em = n_med**2
    k = 2 * np.pi * n_med / (np.asarray(lam_nm) * 1e-9)
    PA, PB = depol_prolate(ar)
    a = [V / (4 * np.pi) * (e - em) / (em + P * (e - em)) for P in (PA, PB, PB)]
    c = [4 * np.pi * k * np.imag(x) for x in a]
    if orient == "long":
        return c[0]
    return (c[0] + c[1] + c[2]) / 3


# ---------------------------------------------------------------------------
# many spheres on a carrier (coupled dipoles)
# ---------------------------------------------------------------------------
def place_on_sphere(N, Rc, r0, gap_min=1e-9, clustered=False, rng=None, cluster_size=8):
    """N gold centres on a sphere of radius Rc, no two closer than 2 r0 + gap_min.
    clustered=True grows tight clumps of ~cluster_size particles instead."""
    rng = np.random.default_rng(rng)
    dmin = 2 * r0 + gap_min
    pts = []

    def ok(p):
        return all(np.linalg.norm(p - q) >= dmin for q in pts)

    tries = 0
    while len(pts) < N and tries < 200000:
        tries += 1
        if clustered and pts and len(pts) % cluster_size != 0:
            q = pts[-1]
            t = rng.normal(size=3)
            t -= t.dot(q) / Rc**2 * q
            p = q + dmin * 1.02 * t / np.linalg.norm(t)
            p *= Rc / np.linalg.norm(p)
        else:
            p = rng.normal(size=3)
            p *= Rc / np.linalg.norm(p)
        if ok(p):
            pts.append(p)
    return np.array(pts)


def cda_cabs(lam_nm, pos, r0, n_med):
    """Total absorption cross-section (m^2) of dipoles at pos (N x 3), each a gold
    sphere of radius r0, averaged over three incidence/polarisation pairs.
    Gaussian units; radiative-reaction-corrected polarisability (Draine 1988)."""
    N = len(pos)
    lam_nm = np.atleast_1d(lam_nm)
    out = np.empty(len(lam_nm))
    rij = pos[:, None, :] - pos[None, :, :]
    d = np.linalg.norm(rij, axis=-1)
    np.fill_diagonal(d, 1.0)
    rh = rij / d[..., None]
    I3 = np.eye(3)
    rr = rh[..., :, None] * rh[..., None, :]
    cases = [((0, 0, 1), (1, 0, 0)), ((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 1))]
    for il, L in enumerate(lam_nm):
        k = 2 * np.pi * n_med / (L * 1e-9)
        a0 = alpha_sphere(L, r0, n_med)
        a = a0 / (1 - 2j / 3 * k**3 * a0)
        ph = np.exp(1j * k * d) / d
        G = ph[..., None, None] * (k**2 * (I3 - rr) + (1j * k * d - 1)[..., None, None] / d[..., None, None]**2 * (I3 - 3 * rr))
        idx = np.arange(N)
        G[idx, idx] = 0
        A = np.eye(3 * N, dtype=complex) / a - G.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
        rhs = []
        for kd, pol in cases:
            phase = np.exp(1j * k * pos @ np.array(kd, float))
            rhs.append((phase[:, None] * np.array(pol, float)[None, :]).ravel())
        p = np.linalg.solve(A, np.array(rhs).T)
        pa = np.abs(p) ** 2
        c = 4 * np.pi * k * pa.sum(axis=0) * (-np.imag(1 / a) - 2 / 3 * k**3)
        out[il] = c.mean()
    return out


def peak(lam, c):
    i = np.argmax(c)
    if 0 < i < len(c) - 1:      # parabolic refinement
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        dx = lam[1] - lam[0]
        return lam[i] + 0.5 * dx * (y0 - y2) / (y0 - 2 * y1 + y2)
    return lam[i]


def n_spheres(Fs, Rc, r0):
    """Skirtach: Fs = n r0^2 / (4 Rc^2)."""
    return int(round(4 * Fs * Rc**2 / r0**2))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def run():
    import style as S
    res = {}
    lam = np.arange(400, 1101, 2.0)

    # --- validation: Mie vs quasi-static, bare sphere in water ------------
    pk = {}
    for R in (5e-9, 10e-9, 20e-9, 40e-9):
        cm = cabs_mie(lam, R)
        pk[R] = peak(lam, cm)
    res["mie_peak_water_nm"] = {"%d nm radius" % round(R * 1e9): round(p, 1) for R, p in pk.items()}
    c_m = cabs_mie(lam, 5e-9)
    c_q = cabs_qs(lam, 5e-9)
    res["qs_vs_mie_R5nm_peak_ratio"] = round(c_q.max() / c_m.max(), 3)

    # --- rods: resonance vs aspect ratio ----------------------------------
    D = prm.v("rod_D")
    ars = np.linspace(1.5, 6, 46)
    lamr = np.arange(450, 1300, 2.0)
    rod_pk = {}
    for nm in (1.33, prm.v("n_shell"), 1.5):
        rod_pk[nm] = np.array([peak(lamr, cabs_rod(lamr, a * D, D, nm, "long")) for a in ars])
    ar_band = {}
    for nm, pkv in rod_pk.items():
        lo, hi = np.interp([800, 830], pkv, ars)
        ar_band["n=%.2f" % nm] = [round(lo, 2), round(hi, 2)]
    res["rod_AR_for_800_830nm"] = ar_band

    # --- spheres on a carrier: resonance vs filling factor (CDA) ----------
    Rc = 65e-9          # gold sits at the outer wall of a ~130 nm carrier
    r0 = prm.v("au_sphere_r")
    n_med = prm.v("n_shell")
    lamc = np.arange(460, 1000, 4.0)
    Fs_list = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
    sph_pk, sph_pk_cl, sph_808 = [], [], []
    for Fs in Fs_list:
        N = n_spheres(Fs, Rc, r0)
        vals, vals_cl, v808 = [], [], []
        for seed in range(3):
            pos = place_on_sphere(N, Rc, r0, rng=seed)
            c = cda_cabs(lamc, pos, r0, n_med)
            vals.append(peak(lamc, c))
            v808.append(np.interp(808, lamc, c) / len(pos))
            posc = place_on_sphere(N, Rc, r0, clustered=True, rng=seed)
            vals_cl.append(peak(lamc, cda_cabs(lamc, posc, r0, n_med)))
        sph_pk.append(np.mean(vals)); sph_pk_cl.append(np.mean(vals_cl)); sph_808.append(np.mean(v808))
        print("  CDA Fs=%.2f N=%d peak %.0f nm (clustered %.0f nm)" % (Fs, N, sph_pk[-1], sph_pk_cl[-1]))
    res["sphere_peak_vs_Fs"] = {str(F): [round(a, 1), round(b, 1)] for F, a, b in zip(Fs_list, sph_pk, sph_pk_cl)}

    # gold-volume-normalised absorption at 808 nm: spheres vs rods
    Vs = 4 / 3 * np.pi * r0**3
    rod_L = prm.v("rod_L")
    Vr = np.pi * D**2 * rod_L / 6
    c_rod808 = cabs_rod(808, rod_L, D, n_med, "random")
    res["cabs808_per_sphere_m2"] = float(np.mean(sph_808))
    res["cabs808_per_rod_m2"] = float(c_rod808)
    res["rod_peak_nm_40x10"] = round(peak(lamr, cabs_rod(lamr, rod_L, D, n_med, "long")), 1)
    res["abs_per_gold_volume_rod_over_sphere_at808"] = round(float(c_rod808 / Vr / (np.mean(sph_808) / Vs)), 1)

    # ------------------------------------------------------------------
    # Figure A: spectra (validation)
    fig, boxes = S.report_halves(2.6)
    a1 = fig.add_axes(boxes[0]); a2 = fig.add_axes(boxes[1])
    for R, ls in ((5e-9, "-"), (20e-9, "--"), (40e-9, ":")):
        c = cabs_mie(lam, R)
        a1.plot(lam, c / c.max(), color=S.BLUE, ls=ls, lw=1.6, label="Mie, R = %d nm" % round(R * 1e9))
    a1.plot(lam, c_q / c_q.max(), color=S.CRIMSON, lw=1.0, ls="-.", label="quasi-static, R = 5 nm")
    S.band(a1)
    a1.set_xlim(400, 1000); a1.set_ylim(0, 1.15); a1.set_xticks([500, 600, 700, 800, 900, 1000])
    a1.set_xlabel("wavelength (nm)"); a1.set_ylabel("absorption (normalised)")
    a1.set_title("one gold sphere in water")
    a1.legend(loc="upper right")
    a1.text(815, 0.55, "trigger\nwindow", ha="center", va="center", fontsize=7.5, color=S.GREY)
    for ar, ls in ((2, ":"), (3, "--"), (4, "-")):
        c = cabs_rod(lamr, ar * D, D, n_med, "random")
        a2.plot(lamr, c / c.max(), color=S.BLUE, ls=ls, lw=1.6, label="aspect ratio %d" % ar)
    S.band(a2)
    a2.set_xlim(450, 1100); a2.set_ylim(0, 1.5); a2.set_yticks([0, 0.5, 1.0])
    a2.set_xlabel("wavelength (nm)"); a2.set_ylabel("absorption (normalised)")
    a2.set_title("one gold rod (10 nm wide)")
    a2.legend(loc="upper right", ncol=1)
    S.save_report(fig, "fig_optics_spectra", panels=[[a1], [a2]])

    # Figure B: resonance vs loading, spheres and rods
    fig, boxes = S.report_halves(2.6)
    b1 = fig.add_axes(boxes[0]); b2 = fig.add_axes(boxes[1], sharey=b1)
    b1.axhspan(800, 830, color=S.BAND, lw=0, zorder=0)
    b1.plot(Fs_list, sph_pk, "o-", color=S.BLUE, lw=1.6, ms=4, label="spread evenly")
    b1.plot(Fs_list, sph_pk_cl, "s--", color=S.CRIMSON, lw=1.4, ms=4, label="in clumps")
    b1.set_xlabel("surface filling factor $F_s$"); b1.set_ylabel("absorption peak (nm)")
    b1.set_title("10 nm spheres on a 130 nm carrier")
    b1.set_ylim(450, 1000); b1.set_xlim(0, 0.42)
    b1.legend(loc="upper left")
    b1.text(0.21, 845, "800-830 nm target", ha="center", fontsize=7.5, color=S.GREY)
    b2.axhspan(800, 830, color=S.BAND, lw=0, zorder=0)
    for nm, ls in ((1.33, ":"), (prm.v("n_shell"), "-"), (1.5, "--")):
        b2.plot(ars, rod_pk[nm], color=S.BLUE, ls=ls, lw=1.6, label="medium n = %.2f" % nm)
    b2.set_xlabel("rod aspect ratio (length / width)")
    b2.set_title("one rod (Gans)")
    b2.set_xlim(1.5, 6)
    b2.legend(loc="lower right")
    plt_ticks = b2.get_yticklabels()
    for t in plt_ticks:
        t.set_visible(False)
    S.save_report(fig, "fig_resonance_vs_loading", panels=[[b1], [b2]])

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "optics.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run()
