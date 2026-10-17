#!/usr/bin/env python3
"""Render the benchmark and onset-scan figures from verified model tables.

Run verify_burst_model.py first. No observational data are plotted. The upper
curves stop at a declared display temperature, not at either physical endpoint.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "code" else SCRIPT_DIR
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

BLUE = "#24577E"
ORANGE = "#B65B24"
GREEN = "#36725E"
GRAY = "#737A80"


def rows(name: str) -> list[dict]:
    with (DATA/name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def make_onset_scan_figure() -> None:
    """Render the onset scan separately, preserving the original Figure 3."""
    with np.load(DATA/"burst_scan_grid.npz") as archive:
        theta = archive["theta_a_TeV"]
        delta = archive["window_s"]
        ratio = archive["r_flat"]
    curves = rows("burst_scan_slices.csv")
    fig = plt.figure(figsize=(6.2, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=(1.15, 1), hspace=0.06)
    axmap = fig.add_subplot(gs[0])
    axslice = fig.add_subplot(gs[1])
    # contourf retains vector paths in the PDF and avoids a large raster image.
    fill = axmap.contourf(theta, delta, ratio, levels=np.linspace(0, 1, 41),
                         cmap="viridis", vmin=0, vmax=1)
    contours = axmap.contour(theta, delta, ratio, levels=[0.1, 0.5, 0.9],
                            colors=["white", "white", "#263D2A"], linewidths=0.85)
    axmap.clabel(contours, fmt=lambda value: f"{value:.1f}", inline=True, fontsize=9,
                 manual=[(1.6, 50), (3.3, 30), (5.8, 20)])
    clock = 3*(7.8/theta)**3
    visible = (clock >= delta[0]) & (clock <= delta[-1])
    axmap.plot(theta[visible], clock[visible], color="#B94935", linestyle="--",
               linewidth=1.6, label=r"Clock transition $\Delta=3\tau_a$")
    axmap.set_yscale("log")
    axmap.set_ylim(1, 150)
    axmap.set_yticks([1, 3, 10, 30, 100, 150])
    axmap.yaxis.set_major_formatter(ScalarFormatter())
    axmap.yaxis.set_minor_formatter(NullFormatter())
    axmap.set_ylabel(r"Window $\Delta$ before each cutoff (s)")
    axmap.set_title("(a)  Conditional photon-yield ratio across the onset scan",
                    loc="left", pad=8)
    axmap.legend(loc="lower left", frameon=True, facecolor="white",
                 edgecolor="none", framealpha=0.95, fontsize=8.5)
    colorbar = fig.colorbar(fill, ax=axmap, pad=0.025, fraction=0.055,
                           ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    colorbar.set_label(r"$R_{\rm flat}$", labelpad=5)
    colorbar.ax.tick_params(labelsize=8.5)

    styles = [
        (10, BLUE, "-"), (20, ORANGE, "--"),
        (100, GREEN, "-"), (120, "#78537E", "--"),
    ]
    for window, color, linestyle in styles:
        selected = [r for r in curves if int(r["window_s"]) == window]
        xx = np.asarray([float(r["theta_a_TeV"]) for r in selected])
        yy = np.asarray([float(r["r_flat"]) for r in selected])
        axslice.plot(xx, yy, color=color, linestyle=linestyle,
                     linewidth=1.7, label=fr"$\Delta={window}$ s")
    for level in (0.5, 0.9):
        axslice.axhline(level, color=GRAY, linestyle=":", linewidth=0.8, zorder=0)
    axslice.set_ylim(0, 1.025)
    axslice.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1])
    axslice.set_ylabel(r"Ceiling / reference yield $R_{\rm flat}$")
    axslice.set_title("(b)  Fixed-window slices and 50% / 90% yield levels",
                      loc="left", pad=8)
    axslice.grid(which="major", axis="x", color="#E3E6E8", linewidth=0.5)
    axslice.legend(loc="upper left", frameon=True, facecolor="white",
                   edgecolor="none", framealpha=0.95, fontsize=9)
    for ax in (axmap, axslice):
        ax.set_xscale("log")
        ax.set_xlim(0.3, 30)
        ax.set_xticks([0.3, 1, 3, 10, 30])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"Postulated onset temperature $\Theta_a$ (TeV)")
        ax.tick_params(which="both", top=True, right=True)
    fig.suptitle("1–15 TeV photon yields with a constant response\n"
                 "Added luminosity-ceiling model", fontsize=10.5)
    stem = FIGURES/"figure4_crossover_scan"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03, metadata={
        "Title": "Conditional luminosity-ceiling onset scan in a fixed photon band",
        "Subject": "Template photon-yield ratios, not instrument exclusions or statistical distinguishability",
        "Creator": "make_burst_figures.py", "CreationDate": None, "ModDate": None,
    })
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)
    print(f"Wrote {stem.name}.pdf and .png to {FIGURES}")


def main() -> None:
    report = json.loads((DATA/"burst_model_results.json").read_text(encoding="utf-8"))
    times = rows("burst_time_grid.csv")
    windows = rows("burst_window_grid.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.05, 5.65), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(1, 1.05), hspace=0.09)
    axp = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 1], sharex=axp)
    axr = fig.add_subplot(gs[1, :])
    for model, color, linestyle, label in (
        ("reference", BLUE, "--", "Reference emission"),
        ("ceiling", ORANGE, "-", "Added luminosity ceiling"),
    ):
        selected = [row for row in times if row["model"] == model]
        t = np.asarray([float(row["u_since_common_onset_s"]) for row in selected])
        power = np.asarray([float(row["P_over_Pstar"]) for row in selected])
        theta = np.asarray([float(row["theta_TeV"]) for row in selected])
        axp.plot(t, power, color=color, linestyle=linestyle, label=label)
        axt.plot(t, theta, color=color, linestyle=linestyle)
        for ax, value in ((axp, power[-1]), (axt, theta[-1])):
            ax.scatter([t[-1]], [value], s=28, facecolors="white", edgecolors=color, linewidths=1.2, zorder=5)
    for ax in (axp, axt):
        ax.set_xlim(0, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_xlabel(r"Time since common onset $u-u_a$ (s)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="both", top=True, right=True)
        ax.grid(which="major", color="#E3E6E8", linewidth=0.5)
    axp.set_ylim(0.75, 135)
    axp.set_yticks([1, 3, 10, 30, 100])
    axp.set_ylabel(r"Bolometric luminosity $P/P_\ast$")
    axp.set_title("(a)  Plateau from the added ceiling", loc="left", pad=8)
    axp.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    axp.text(27, 1.35, r"$P=P_\ast$", color=ORANGE, fontsize=10)
    axt.set_ylim(2.7, 35)
    axt.set_yticks([3, 5, 10, 20, 30])
    axt.set_ylabel(r"Temperature parameter $\Theta=k_{\rm B}T_H$ (TeV)")
    axt.set_title("(b)  Different temperature evolution", loc="left", pad=8)
    axt.text(0.43, 0.94, r"Open circles: $\Theta=30$ TeV" + "\nEndpoints are not shown",
             transform=axt.transAxes, ha="left", va="top", fontsize=8.5, color=GRAY)

    delta = np.asarray([float(row["window_s"]) for row in windows])
    lower = np.asarray([float(row["r_min"]) for row in windows])
    upper = np.asarray([float(row["r_max"]) for row in windows])
    flat = np.asarray([float(row["r_flat"]) for row in windows])
    axr.set_xscale("log")
    axr.fill_between(delta, lower, upper, color="#9BB5C5", alpha=0.42,
                     label="Any nonnegative response in the band", linewidth=0)
    axr.plot(delta, lower, color=BLUE, linewidth=0.85)
    axr.plot(delta, upper, color=BLUE, linewidth=0.85)
    axr.plot(delta, flat, color=GREEN, linewidth=2, label="Constant effective area")
    axr.axhline(1, color=GRAY, linestyle=":", linewidth=0.8)
    for row in report["windows"][:3]:
        d, ratio = row["window_s"], row["r_flat"]
        axr.scatter([d], [ratio], color=GREEN, edgecolors="white", linewidths=0.65, s=28, zorder=5)
        axr.annotate(f"{d:g} s: {ratio:.3f}", (d, ratio), xytext=(0, 9), textcoords="offset points",
                     ha="center" if d < 100 else "right", va="bottom", color=GREEN, fontsize=9)
    axr.set_xlim(1, 150)
    axr.set_ylim(0, 1.075)
    axr.set_xticks([1, 3, 10, 20, 50, 100])
    axr.xaxis.set_major_formatter(ScalarFormatter())
    axr.xaxis.set_minor_formatter(NullFormatter())
    axr.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axr.set_xlabel(r"Window $\Delta$ before each model's cutoff (s)")
    axr.set_ylabel("Ceiling / reference photon yield")
    axr.set_title("(c)  Finite-window photon yield in the 1–15 TeV band", loc="left", pad=8)
    axr.grid(which="major", color="#E3E6E8", linewidth=0.5)
    axr.tick_params(which="both", top=True, right=True)
    axr.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    fig.suptitle(r"Conditional luminosity ceiling with a declared benchmark onset $\Theta_a=3$ TeV",
                 fontsize=10.5)

    stem = FIGURES/"figure3_conditional_burst"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={
        "Title": "Conditional luminosity-ceiling evaporation model and photon-window predictions",
        "Subject": "Added phenomenological ansatz; analytical template, not observational data or a TCGS-derived prediction",
        "Creator": "make_burst_figures.py",
        "CreationDate": None,
        "ModDate": None,
    })
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {stem.name}.pdf and .png to {FIGURES}")
    make_onset_scan_figure()


if __name__ == "__main__":
    main()
