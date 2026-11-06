"""Regenerate every figure and result from scratch, then check the validation targets.

    python sims/run_all.py
"""
import json
import os

import params, optics, heat_killswitch, heat_checks, pennes, release, safety, design

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
os.system("cd '%s' && %s params.py > /dev/null" % (HERE, __import__("sys").executable))

print("\n== 1. kill-switch (heat, collective)"); k = heat_killswitch.run()
print("\n== 2. optics"); o = optics.run()
print("\n== 3. heat checks + transient"); h = heat_checks.run()
print("\n== 4. Pennes"); p = pennes.run()
print("\n== 5. release"); r = release.run()
print("\n== 6. safety"); s = safety.run()
print("\n== 7. sensitivity + envelope"); d = design.run()

checks = {
    "Mie peak, 5 nm sphere in water, 515-525 nm": 515 <= o["mie_peak_water_nm"]["5 nm radius"] <= 525,
    "rods reach 800-830 nm for some aspect ratio": all(v[0] < v[1] for v in o["rod_AR_for_800_830nm"].values()),
    "quasi-static vs Mie (R = 5 nm) within 5%": abs(o["qs_vs_mie_R5nm_peak_ratio"] - 1) < 0.05,
    "dT ~ R^2: slope 2.00 +/- 0.02": abs(h["slope_quasistatic"] - 2) < 0.02,
    "Govorov Fig. 1b within 30%": abs(h["govorov_R30_dT_K"] / h["govorov_fig1b_read_K"] - 1) < 0.3,
    "transient time ~ R^2 (slope 2)": abs(h["tau_slope_numeric"] - 2) < 0.05,
    "Pennes numeric = analytic (1%)": abs(p["check_centre_numeric_K"] / p["check_centre_analytic_K"] - 1) < 0.01,
    "CEM43 of 43 C for 10 min = 10 min": abs(s["check_cem43_43C_10min"] - 10) < 1e-6,
    "release model matches Deng dox at 24 h": abs(r["fraction_24h_at_37C"] - 0.11) < 0.005,
}
checks = {a: bool(b) for a, b in checks.items()}
json.dump(checks, open(os.path.join(HERE, "results", "validation.json"), "w"), indent=1)
print("\n== validation")
for kk, vv in checks.items():
    print("  %-50s %s" % (kk, "PASS" if vv else "FAIL"))
