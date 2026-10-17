#!/usr/bin/env python3
"""Make publication figures from the verified analytical data tables.

Run verify_scalings.py first. PDFs retain vector graphics and embedded fonts;
PNGs are 300 dpi inspection/share copies. No observational data are plotted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "code" else SCRIPT_DIR
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#24577E"
ORANGE = "#B65B24"
GREEN = "#36725E"
INK = "#252A2D"
GRAY = "#737A80"


def read_columns(filename: str) -> dict[str, np.ndarray]:
    with (DATA / filename).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def save(fig: plt.Figure, stem: str, title: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", metadata={
        "Title": title,
        "Subject": "Analytical toy-model illustration; no observational data or physical endpoint prediction",
        "Creator": "make_figures.py",
        "CreationDate": None,
        "ModDate": None,
    })
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def scalings_figure(results: dict) -> None:
    d = read_columns("scalings_grid.csv")
    mu = d["M_over_mP"]
    mustar = results["Mstar_over_mP"]["value"]
    muad = results["M_ad_over_mP"]["value"]
    fig, ax = plt.subplots(figsize=(7.05, 4.10), constrained_layout=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e4)
    ax.set_ylim(1e-13, 2e6)
    ax.axvspan(1e-3, mustar, color="#EDEFF0", zorder=0)
    ax.axhline(1, color=GRAY, linewidth=0.8, linestyle=(0, (2.5, 3.5)), zorder=1)
    # Dashes identify formal extrapolations beyond the chosen curvature crossing.
    formal = mu < mustar
    controlled = ~formal
    for column, color, label in [
        ("B_H", BLUE, r"$\widehat{B}_{K,H}=\sqrt{3}/(2\mu^2)$"),
        ("epsilon_ad_toy_blackbody", ORANGE, r"$\epsilon_{\rm ad}=1/(3840\pi\mu^2)$"),
    ]:
        ax.plot(mu[controlled], d[column][controlled], color=color, label=label, zorder=3)
        ax.plot(mu[formal], d[column][formal], color=color, linestyle="--", zorder=3)
    ax.axvline(mustar, color=BLUE, linewidth=0.8, linestyle=":", zorder=2)
    ax.axvline(muad, color=ORANGE, linewidth=0.8, linestyle=":", zorder=2)
    ax.scatter([mustar], [1], s=28, facecolors="white", edgecolors=BLUE, linewidths=1.4, zorder=5)
    ax.scatter([muad], [1], s=28, facecolors="white", edgecolors=ORANGE, linewidths=1.4, zorder=5)
    ax.annotate(r"$\widehat{B}_{K,H}=1$" + "\n" + r"$\mu=0.93060$",
                xy=(mustar, 1), xytext=(3.0, 1.2e2), color=BLUE,
                arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.7},
                fontsize=9, ha="left", va="bottom")
    ax.annotate(r"$\epsilon_{\rm ad}=1$" + "\n" + r"$\mu=0.00910$",
                xy=(muad, 1), xytext=(0.0021, 3e-3), color=ORANGE,
                arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.7},
                fontsize=9, ha="left", va="top")
    ax.text(0.055, 2.3e5, "Formal continuation\npast the chosen\ncurvature threshold",
            fontsize=8.5, color=GRAY, ha="left", va="top")
    ax.text(0.975, 0.49,
            r"$\epsilon_{\rm ad}/\widehat{B}_{K,H}=1/(1920\pi\sqrt{3})$",
            transform=ax.transAxes, ha="right", va="center", fontsize=10, color=INK)
    ax.set_xlabel(r"Mass ratio $\mu=M/m_{\rm P}$")
    ax.set_ylabel("Dimensionless diagnostic")
    ax.set_title("Curvature and adiabatic rate conditions are distinct", loc="left", pad=10)
    ax.legend(loc="lower left", frameon=True, facecolor="#EDEFF0", edgecolor="none",
              framealpha=1, handlelength=2.6, borderpad=0.35)
    ax.grid(which="major", color="#E3E6E8", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.tick_params(which="both", top=True, right=True)
    save(fig, "figure1_curvature_adiabaticity", "Curvature and adiabatic rate diagnostics in a specified blackbody toy model")


def preterminal_figure(results: dict) -> None:
    d = read_columns("preterminal_history.csv")
    s = d["s_u_over_uc"]
    cutoff = results["Mc_over_M0_illustration"]["value"]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.95), sharex=True, constrained_layout=True)
    series = [
        ("M_over_M0", GREEN, r"$M/M_0$", "Mass", "(a)", cutoff, "0.20"),
        ("P_over_P0", ORANGE, r"$P/P_0$", "Luminosity", "(b)", cutoff**-2, "25"),
        ("E_emitted_over_M0c2", BLUE, r"$E_{\rm rad}/(M_0c^2)$", "Emitted energy", "(c)", 1-cutoff, "0.80"),
    ]
    for ax, (column, color, ylabel, title, letter, endpoint, endpoint_label) in zip(axes, series):
        ax.plot(s, d[column], color=color)
        ax.scatter([1], [endpoint], s=28, color=color, edgecolors="white", linewidths=0.7, zorder=5)
        ax.axvline(1, color=GRAY, linestyle=(0, (2, 3)), linewidth=0.8)
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{letter}  {title}", loc="left", pad=8)
        ax.set_xlabel(r"$s=(u-u_0)/(u_c-u_0)$", fontsize=10)
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(which="major", color="#E3E6E8", linewidth=0.5)
        ax.annotate(endpoint_label, (1, endpoint), xytext=(-9, 9), textcoords="offset points",
                    color=color, fontsize=10, fontweight="bold", ha="right")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_yticks(np.arange(0, 1.01, 0.2))
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.9, 40)
    axes[1].set_yticks([1, 2, 5, 10, 25])
    axes[1].yaxis.set_major_formatter(ScalarFormatter())
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[2].set_ylim(0, 0.94)
    axes[2].set_yticks(np.arange(0, 0.81, 0.2))
    fig.suptitle(r"Preterminal constant-$\alpha$ illustration: chosen cutoff $M_c/M_0=0.20$",
                 fontsize=10.5, x=0.5)
    save(fig, "figure2_preterminal_energy", "Finite preterminal energy in a constant-alpha illustration with a chosen cutoff")


def main() -> None:
    report = json.loads((DATA / "verified_results.json").read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    scalings_figure(report["results"])
    preterminal_figure(report["results"])
    print(f"Wrote two PDF figures and two 300 dpi PNGs to {FIGURES}")


if __name__ == "__main__":
    main()
