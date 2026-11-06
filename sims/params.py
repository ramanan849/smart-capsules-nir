"""Every input to the BB601 Team 10 models, with where it comes from.

P["name"] = dict(v=value, unit=..., src=..., ok=True/False)
ok=False means we could not check it against a primary source; such numbers
are flagged in the text and should be treated as estimates.
Run this file to print the parameter table (results/parameter_table.txt).
"""

import os

P = {}


def add(name, v, unit, src, ok=True):
    P[name] = dict(v=v, unit=unit, src=src, ok=ok)


# --- thermal ---------------------------------------------------------------
add("k_water", 0.6, "W/(m K)", "standard value for water near 37 C")
add("kappa_water", 1.4e-7, "m^2/s", "standard value for water (k / rho c)")
add("k_tissue", 0.5, "W/(m K)", "typical soft tissue; range 0.4-0.6 used in sensitivity", ok=False)
add("rhoc_tissue", 3.6e6, "J/(m^3 K)", "typical soft tissue (rho ~1050 kg/m3, c ~3500 J/kg K)", ok=False)
add("rho_tissue", 1050.0, "kg/m^3", "typical soft tissue")
add("c_blood", 3617.0, "J/(kg K)", "typical blood heat capacity")
add("w_blood", 2.0, "kg/(m^3 s)", "perfusion; handoff range 0.5-5, middle value used, swept", ok=False)
add("T_body", 37.0, "C", "body temperature")

# --- optics ----------------------------------------------------------------
add("n_water", 1.33, "-", "water")
add("n_shell", 1.45, "-", "polyelectrolyte shell ~1.5, effective value with water; swept 1.33-1.5", ok=False)
add("vF_au", 1.40e6, "m/s", "Fermi velocity of gold (handoff; Kreibig & Vollmer)")
add("A_sizecorr", 1.0, "-", "surface-scattering constant in gamma = gamma_bulk + A vF / R")
add("hwp_au", 9.03, "eV", "Drude plasma energy of gold (standard fit to J&C)", ok=False)
add("hgamma_au", 0.072, "eV", "bulk Drude damping of gold (standard fit to J&C)", ok=False)
add("eps_au_table", "data/au_johnson_christy.csv", "-", "Johnson & Christy, Phys Rev B 1972;6:4370")
add("rho_au", 19300.0, "kg/m^3", "gold density")

# --- the carrier -----------------------------------------------------------
add("d_core", 100e-9, "m", "liposome core, extruded through 100 nm membrane (our fabrication step); "
                           "Deng's core was ~120 nm (Deng 2013 Fig. 2A, CML stand-in)")
add("layer_dd", 5.5e-9, "m", "Deng 2013 Fig. 2A: hydrodynamic DIAMETER grows ~5 nm per layer "
                             "(118 -> 140 nm over 4 layers), i.e. ~2.75 nm of wall per layer")
add("n_layers", 4, "-", "4-6 PSS/PAH layers (Submission 3)")
add("au_sphere_r", 5e-9, "m", "option A sphere radius (5-15 nm diameter range in Submission 3)")
add("au_angelatos_d", 6e-9, "m", "Angelatos 2005: average gold DIAMETER 6 nm (radius 3 nm)")
add("rod_L", 40e-9, "m", "option B rod length, typical seed-mediated rod", ok=False)
add("rod_D", 10e-9, "m", "option B rod width, typical seed-mediated rod", ok=False)
add("n_rods", 30, "-", "rods per carrier (~Fs 0.25 of a 130 nm carrier); swept 10-50")
add("Fs_spheres", 0.2, "-", "surface filling factor for spheres (Skirtach 2005 range 0.05-0.4)")

# --- laser / safety --------------------------------------------------------
add("lam", 808e-9, "m", "trigger wavelength (800-830 nm window)")
add("MPE_cw_808", 0.33, "W/cm^2", "ANSI Z136.1 / IEC 60825-1 skin MPE, t > 10 s: 0.2*C_A W/cm2, "
                                  "C_A = 10^(0.002(lam-700)) for 700-1050 nm -> 0.33 W/cm2 at 808 nm. "
                                  "Formula from the standard as quoted in secondary sources; "
                                  "the 0.33 W/cm2 at 810 nm value is listed in a published protocol", ok=False)
add("I_oneal", 4.0, "W/cm^2", "O'Neal et al., Cancer Lett 2004;209:171 (808 nm, 4 W/cm2, 3 min) - "
                              "abstract-level check only", ok=False)
add("I_angelatos_avg", 7.0, "W/cm^2", "Angelatos 2005: 700 mJ/cm2 x 10 Hz = 7 W/cm2 average, 5 min")

# --- release ---------------------------------------------------------------
add("Tm_ltsl", 41.0, "C", "LTSL (DPPC/lysolipid/DSPE-PEG 90:10:4) release around 39-41 C; "
                          "DPPC main transition ~41 C (Needham 2000; Yarmolenko 2010)", ok=False)
add("t50_40C", 20.0, "s", "LTSL ~50% released in <20 s at 40 C (Needham 2000, as quoted by "
                          "Yarmolenko et al. Int J Hyperthermia 2010)", ok=False)
add("deng_dox_24h", 0.11, "-", "Deng 2013 Fig. 6B: ~11% dox released from LbL liposome at 24 h "
                               "(DMEM; rigid DSPC/Chol/POPG liposome, NOT thermosensitive)")
add("P_shell", 1e-6, "cm/s", "doxorubicin permeability of PSS/PAH wall: UNKNOWN, swept 1e-9 - 1e-5", ok=False)

# --- mechanics -------------------------------------------------------------
add("E_pe", "several hundred", "MPa", "Fery & Weinkamer 2007; no exact value, only for h/R < 1/10")

# --- literature anchors ----------------------------------------------------
add("nunes_gold", 41.0, "ug/g", "Nunes 2019: tumour gold 41 +/- 1 ug/g at 72 h after IV dose")
add("skirtach_ratio", 1.58, "-", "Skirtach 2005: slope ratio for Fs 0.2 -> 0.4 (text p.1375; "
                                "1.5 in our earlier draft)")
add("govorov_fig1", 4.3, "K", "Govorov & Richardson 2007 Fig. 1b: R=30 nm Au, 1e4 W/cm2, water, read off plot")


def v(name):
    return P[name]["v"]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    lines = ["%-16s %-26s %-12s %s %s" % ("name", "value", "unit", "checked", "source")]
    for k, d in P.items():
        lines.append("%-16s %-26s %-12s %-7s %s" % (k, d["v"], d["unit"], "yes" if d["ok"] else "NO", d["src"]))
    txt = "\n".join(lines)
    open(os.path.join(here, "results", "parameter_table.txt"), "w").write(txt + "\n")
    print(txt)
