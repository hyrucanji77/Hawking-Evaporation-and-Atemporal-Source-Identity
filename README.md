# Hawking Evaporation and Atemporal Source Identity

Computational companion to **Hawking Evaporation and Atemporal Source Identity: Consistency Conditions, a Conditional Burst Model, and Information in TCGS–SEQUENTION**, by **Henry Arellano-Peña**.

This repository reproduces analytical checks, numerical tables, and four scientific figures. Its calculations connect curvature, evaporation clocks, finite energy budgets, and the photon yields of a specified emission model.

## Scientific context

The paper’s principal contribution is a TCGS–SEQUENTION interpretation of Hawking evaporation. Within its governing source architecture, complete atemporal source identity grounds physically real Shadow configurations; mass loss, outgoing fields, operational clocks, and accessible records describe those configurations and their relations. The ontic Extrinsic Constitutive Law and its formal mathematical representatives have distinct roles. This interpretation preserves the semiclassical radiation problem while separating disappearance of a black-hole configuration, restricted exterior information, and recoverability of a quantum state [1–4].

A luminosity-ceiling model supplies a controlled physical comparison within that account. The ceiling is an **additional emission postulate** whose linked consequences can be calculated. The framework references supply the source arguments and architecture; they are not independent empirical support for this added law. The interpretation of the semiclassical baseline does not depend on that particular prescription.

## Setup and reproduction

Use **Python 3.11 or later** and Git. The pinned direct dependencies are SymPy, mpmath, NumPy, and Matplotlib, as specified in `requirements.txt`. Run the following sequence for your platform. Calling the virtual environment’s interpreter directly avoids activation requirements.

### Linux / macOS

```bash
git clone https://github.com/hyrucanji77/Hawking-Evaporation-and-Atemporal-Source-Identity.git
cd Hawking-Evaporation-and-Atemporal-Source-Identity
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_scalings.py
.venv/bin/python make_figures.py
.venv/bin/python verify_burst_model.py
.venv/bin/python make_burst_figures.py
```

### Windows PowerShell

```powershell
git clone https://github.com/hyrucanji77/Hawking-Evaporation-and-Atemporal-Source-Identity.git
cd Hawking-Evaporation-and-Atemporal-Source-Identity
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_scalings.py
.\.venv\Scripts\python.exe make_figures.py
.\.venv\Scripts\python.exe verify_burst_model.py
.\.venv\Scripts\python.exe make_burst_figures.py
```

The four scripts and `requirements.txt` reside together at the repository root. The verifiers create `data/`; the plotting scripts read those results and create `figures/`. Both output directories are local generated products ignored by Git. Rerunning replaces their named outputs. Matplotlib uses a noninteractive backend, so no graphical desktop or LaTeX installation is needed.

## Scripts and outputs

| Script | Purpose | Principal generated outputs |
|---|---|---|
| `verify_scalings.py` | Exact curvature, temperature, adiabaticity, mass-loss, and energy identities; endpoint approach examples | `data/verified_results.json`, `verified_results.csv`, `endpoint_counterexamples.csv`, `scalings_grid.csv`, `preterminal_history.csv` |
| `verify_burst_model.py` | Ceiling dynamics, finite-cutoff clocks, photon-band integrals, response bounds, and crossover scan | `data/burst_model_results.json`, `burst_model_results.csv`, `burst_window_grid.csv`, `burst_time_grid.csv`, and the `burst_scan_*` products |
| `make_figures.py` | Render the verified scaling and energy illustrations | `figures/figure1_curvature_adiabaticity` and `figure2_preterminal_energy`, each as PDF and 300 dpi PNG |
| `make_burst_figures.py` | Render the ceiling benchmark and crossover surface | `figures/figure3_conditional_burst` and `figure4_crossover_scan`, each as PDF and 300 dpi PNG |

The paper’s **Table 3** is reproduced by `data/burst_table.tex`, with its numerical values in `burst_model_results.csv`. **Table 4** is reproduced by `data/burst_scan_threshold_table.tex` and `burst_scan_crossovers.csv`. Additional selected yields appear in `burst_scan_selected.csv` and `burst_scan_selected_table.tex`.

`burst_scan_grid.csv` stores all **60,491 points**: 241 logarithmically spaced crossover thermal energies from **0.3 to 30 TeV**, crossed with 251 windows from **1 to 150 seconds**. `burst_scan_grid.npz` stores the arrays in `[window index, crossover-temperature index]` order. `burst_scan_slices.csv` contains the 10-, 20-, 100-, and 120-second slices; `burst_scan_results.json` records formulas, assumptions, crossover roots, numerical checks, and runtime versions.

## Models, units, and interpretation

`verify_scalings.py` uses the unreduced Planck mass, $m_P=\sqrt{\hbar c/G}$, and the effective Schwarzschild diagnostic

$$
\widehat B_{K,H}=\frac{\sqrt{3}}{2}\left(\frac{m_P}{M}\right)^2.
$$

The CSV abbreviation `B_H` denotes this effective quantity. Identifying it with a source-pulled admissibility field requires a separate calibration. The crossing $M=(3/4)^{1/4}m_P$ is a selected curvature criterion. Figure 1 compares it with adiabaticity for the explicitly specified photon-blackbody toy luminosity; that toy coefficient is distinct from the high-temperature Standard Model coefficient used in the burst calculation. Figure 2 uses a chosen cutoff $M_c/M_0=0.20$ to illustrate finite preterminal energy. Neither construction determines a physical quantum endpoint.

The burst model retains the reference spectral scale $\Theta=k_BT_H=\hbar c^3/(8\pi GM)$ and defines

$$
P_0=\frac{c^2\gamma_0}{M^2},\qquad
w(M)=\min\!\left[1,\left(\frac{M}{M_a}\right)^2\right],\qquad
P=wP_0,\qquad
\frac{dM}{du}=-\frac{\gamma_0w}{M^2}.
$$

The same weight multiplies every primary emission channel. Secondary spectra inherit it under linear fragmentation and decay. Consequently, suppression cancels the longer residence time when fluence is integrated between the **same masses**. Windows of the same duration generally traverse different mass intervals and yield different counts.

Masses are in kilograms, physical times in seconds, and dimensional powers in watts. The photon template uses energies in **GeV** and spectra in **GeV⁻¹**; reported temperatures are thermal energies in **TeV**, not kelvin. The conversion is $1\,\mathrm{GeV}=1.602176634\times10^{-10}\,\mathrm J$. Dimensionless ratios are labelled explicitly. The burst coefficient follows the adopted reference calibration $\Theta_0(s)=7.8\,\mathrm{TeV}(1\,\mathrm s/s)^{1/3}$ [6,7].

The published cumulative photon approximation is evaluated as an integrated spectrum. Its sharp break is not differentiated to manufacture an instantaneous light curve. The comparison uses a constant response on the true-energy interval **1–15 TeV**, with bounds for any nonnegative, time-independent response supported there. A positive cutoff is retained in the exact mapping and checked against the simplified plotting limit.

For the declared **3 TeV crossover**, the 20-second flat-response yield ratio is approximately **0.281**. At that duration, ratios of 0.5 and 0.9 occur near crossover energies of **3.73 TeV** and **5.79 TeV**. These are template levels, not confidence contours. The temperature–clock index is 2 on the capped branch and 4 in the constant-coefficient reference. The plateau is imposed by the ceiling; the clock and fluence relations follow from the combined assumptions.

## Verification and scope

The verifiers perform **58 exact symbolic checks**: 24 scaling identities, 17 ceiling identities, and 17 scan identities. Additional numerical checks cover endpoint asymptotics, energy conservation, independent band quadratures, transition continuity, crossover roots, monotonicity, and finite-cutoff errors. Reports distinguish exact identities from numerical agreement and record the precision used.

All generated values are analytical or model-derived. The code performs no telescope-data reanalysis and supplies no measured instrumental response or new exclusion limit. The H.E.S.S. comparison is a conditional sensitivity rescaling under its stated yield, volume, background, and exposure assumptions [8]. A horizon-regular quantum state, conserved renormalized stress tensor, and coupled backreaction implementing the ceiling remain to be constructed. Numerical precision does not increase the physical accuracy of the adopted spectral approximation.

## Primary references

1. Arellano-Peña, H. (2026). [Timeless Counterspace & Shadow Gravity—A Unified Framework: 4D-Counterduction, Foundational Consistency, and Cartographic Inquiry](https://www.preprints.org/manuscript/202511.1472/v2). Preprints.org, version 2.
2. Arellano-Peña, H. (2026). [The Metamathematical Extrinsic Constitutive Law: Origin–MIND, Ontological Closure, and Gravity as a Privileged Shadow Expression in TCGS–SEQUENTION](https://philarchive.org/archive/ARETME). Research preprint.
3. Hawking, S. W. (1974). [Black hole explosions?](https://doi.org/10.1038/248030a0). *Nature* **248**, 30–31.
4. Hawking, S. W. (1975). [Particle creation by black holes](https://doi.org/10.1007/BF02345020). *Communications in Mathematical Physics* **43**, 199–220.
5. Barceló, C., Liberati, S., Sonego, S., and Visser, M. (2011). [Hawking-like radiation from evolving black holes and compact horizonless objects](https://arxiv.org/abs/1011.5911). *JHEP* **02**, 003.
6. Ukwatta, T. N., et al. (2016). [Primordial Black Holes: Observational characteristics of the final evaporation](https://arxiv.org/abs/1510.04372). *Astroparticle Physics* **80**, 90–114.
7. LHAASO Collaboration (2025). [All-Sky Search for Individual Primordial Black Hole Bursts with LHAASO](https://doi.org/10.1103/nkby-9cs3). *Physical Review Letters* **135**, 181005; [arXiv:2505.24586](https://arxiv.org/abs/2505.24586).
8. H.E.S.S. Collaboration (2023). [Search for the evaporation of primordial black holes with H.E.S.S.](https://doi.org/10.1088/1475-7516/2023/04/040). *JCAP* **04**, 040; [arXiv:2303.12855](https://arxiv.org/abs/2303.12855).

## Citing the code

Henry Arellano-Peña, *Hawking Evaporation and Atemporal Source Identity: computational companion*, [GitHub repository](https://github.com/hyrucanji77/Hawking-Evaporation-and-Atemporal-Source-Identity). Include the commit identifier used for reproducibility. The associated paper title and author are given at the top of this README.
