# Smart Capsules: NIR-Triggered LbL Carriers for Chemo-Photothermal Therapy <br>
**Team** : Team 10 <br>
**Project** : NIR-Triggered Layer-by-Layer "Smart Capsules" for On-Demand Chemo-Photothermal Therapy of Breast Cancer <br>

| Name | Roll No |
|---|---|
| Gokularamanan R S (Team Lead) | 23B1854 |
| Arash Dev Ahlawat | 23B1817 |
| Arya Joshi | 23B1853 |
| Soumik Sahoo | 23B1825 |
| Johan Sam Baiju | 23B2170 |
| Manaswi Goyal | 23B2259 |
| Yashasvee Taiwade | 23B2232 |
| Dnyaneshwari Kate | 23B2201 |

***
This repo has all the simulation code behind our Submission 3 (materials, methods and characterization) and our final presentation. <br>
The idea in one line - a ~150 nm layer-by-layer carrier with a heat-sensitive liposome core holding doxorubicin, gold nanorods in the shell, and hyaluronic acid on the outside. Shine skin-safe 808 nm light on the tumour, it warms up to 41-43 °C, the liposome melts, and the drug comes out only there. <br>
We are not building this in a lab - it is a course project. What we did instead is a small quantitative design study: how much gold, how many layers, how much light, and for how long. Every model is kept simple on purpose, and every one is checked against a number from the papers we reviewed before we trust it. <br>

***
**What is in here** <br>
1. **[Kill-switch check](sims/heat_killswitch.py)** - we ran this first, since it could have killed the whole idea <br>
      i. One 140 nm carrier at the skin limit warms itself by only ~0.0001 K. So no, a single carrier cannot heat itself. <br>
      ii. But a tumour full of carriers heats up together - about 6-9 µg of rod gold per gram of tissue gives +5 K. <br>
      iii. Figure: [fig_killswitch](sims/figs/fig_killswitch.png) <br>
2. **[Optics: spheres or rods?](sims/optics.py)** <br>
      i. Mie theory (via *miepython*) for spheres, Gans theory for rods, coupled dipoles for many spheres on the carrier. Gold data from Johnson & Christy. <br>
      ii. Spheres on our carrier stay near 530-555 nm no matter how densely we pack them. Rods of aspect ratio ~4 land right in 800-830 nm. So, rods. <br>
      iii. Figures: [fig_resonance_vs_loading](sims/figs/fig_resonance_vs_loading.png), [fig_optics_spectra](sims/figs/fig_optics_spectra.png) <br>
3. **[Heat model checks](sims/heat_checks.py)** <br>
      i. ΔT ∝ R² for a single particle (slope 2.00), Govorov's Fig. 1b case (5.1 K vs ~4.3 K), and Skirtach's heating vs gold filling factor. <br>
      ii. How fast things heat up - our carrier takes ~73 ns, so a 10 ns pulse only heats the gold. That is why we went with continuous-wave light. <br>
      iii. Figures: [fig_heat_checks](sims/figs/fig_heat_checks.png), [fig_tau_vs_size](sims/figs/fig_tau_vs_size.png) <br>
4. **[Temperature in tissue](sims/pennes.py)** <br>
      i. Pennes bioheat equation for a 1 cm tumour in normal tissue, laser on for 15 min. <br>
      ii. Centre settles at ~42.4 °C, tissue 2 mm outside stays below 39 °C. <br>
      iii. Figure: [fig_pennes](sims/figs/fig_pennes.png) <br>
5. **[Drug release](sims/release.py)** <br>
      i. Lipid lets go above ~41 °C, with the polyelectrolyte wall as a second barrier. Matched to Deng's doxorubicin release with no laser (11% in 24 h). <br>
      ii. With the laser on, 90% is out within ~2 min. <br>
      iii. Figure: [fig_release](sims/figs/fig_release.png) <br>
6. **[Safety](sims/safety.py)** <br>
      i. Continuous-wave skin laser limit (0.33 W/cm² at 808 nm), thermal dose (CEM43), and the shell size / stiffness check. <br>
      ii. Figure: [fig_safety](sims/figs/fig_safety.png) <br>
7. **[Sensitivity and the design envelope](sims/design.py)** <br>
      i. Change one input at a time and see what moves the tumour temperature the most (total gold, tumour size, and how much light gets there). <br>
      ii. Final numbers in [design_envelope.txt](sims/results/design_envelope.txt). Figure: [fig_sensitivity](sims/figs/fig_sensitivity.png) <br>

Every input and where it comes from is in [parameter_table.txt](sims/results/parameter_table.txt) (the "NO" ones are the inputs we could not check against a primary source). <br>

***
**Running it** <br>
Python 3.11+. One script rebuilds every figure and number from scratch (takes a few minutes): <br>

```
pip install -r requirements.txt
python sims/run_all.py
```

Figures land in `sims/figs/` (PNG at 300 dpi, plus PDF), numbers in `sims/results/`. At the end it prints 9 validation checks - all of them should say PASS. <br>

***
**Not to forget** <br>
      i. This is a course project, not medical advice, and nothing here has been tested in a lab. <br>
      ii. The biggest unknowns are how much light actually reaches the tumour, and how permeable the polyelectrolyte wall is to doxorubicin. Nobody has measured the second one. <br>
      iii. The papers themselves are not in the repo (copyright) - the full list is in our Submission 3. <br>
