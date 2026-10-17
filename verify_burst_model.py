#!/usr/bin/env python3
"""Verify a conditional luminosity-ceiling evaporation model.

This is an added phenomenological postulate, not a prediction derived from TCGS.
The multiplier acts on all emission channels and on the total luminosity.
The published LHAASO broken-power-law *integrated* photon template is used only
for integrated counts; it is never differentiated into an instantaneous flux.
No telescope response or observational event data are supplied by this script.

Run from the project directory, or invoke this file by its absolute path:
    python verify_burst_model.py
Dependencies: the existing requirements.txt. Outputs are relative to this
file's project root, not to the current working directory.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "code" else SCRIPT_DIR
DATA = ROOT / "data"
mp.mp.dps = 60

C_LIGHT = mp.mpf("299792458")
G_NEWTON = mp.mpf("6.67430e-11")
HBAR = mp.mpf("1.054571817e-34")
GEV_J = mp.mpf("1.602176634e-10")
M_PLANCK = mp.sqrt(HBAR * C_LIGHT / G_NEWTON)
BETA = HBAR * C_LIGHT**3 / (8 * mp.pi * G_NEWTON)  # M * k_B T, kg J
THETA_A_GEV = mp.mpf(3000)
THETA_ONE_SECOND_GEV = mp.mpf(7800)
TAU_A = (THETA_ONE_SECOND_GEV / THETA_A_GEV)**3  # seconds
M_A = BETA / (THETA_A_GEV * GEV_J)
M_C = (mp.mpf(3) / 4)**mp.mpf("0.25") * M_PLANCK
EPS_C = M_C / M_A
A_MASS = (BETA / (THETA_ONE_SECOND_GEV * GEV_J))**3 / 3  # kg^3 / s
P_CEILING = C_LIGHT**2 * A_MASS / M_A**2
Z_C = TAU_A * EPS_C**3
DELTA_A = 3 * TAU_A * (1 - EPS_C)
B_A = mp.sqrt(3) / 2 * (M_PLANCK / M_A)**2
E_LO_GEV = mp.mpf(1000)
E_HI_GEV = mp.mpf(15000)
TEMPLATE_NORM = mp.mpf("9e35")


def exact(name: str, left: sp.Expr, right: sp.Expr, checks: list[str]) -> None:
    residual = sp.simplify(left - right)
    if residual != 0:
        raise AssertionError(f"{name}: {residual}")
    checks.append(name)


def exact_checks() -> list[str]:
    checks: list[str] = []
    a, c, ma, t, d, tau, e, beta, theta = sp.symbols(
        "a c M_a t Delta tau_a epsilon_c beta Theta", positive=True
    )
    m = ma - a * t / ma**2
    w = (m / ma)**2
    p0 = c**2 * a / m**2
    pstar = c**2 * a / ma**2
    exact("Ceiling luminosity from channel multiplier", w * p0, pstar, checks)
    exact("Ceiling mass-loss law", sp.diff(m, t), -a / ma**2, checks)
    exact("All-species energy conservation", c**2 * sp.diff(m, t) + w*p0, sp.Integer(0), checks)
    tc = ma**3 / a * (1-e)
    exact("Positive cutoff mass", m.subs(t, tc), e*ma, checks)
    exact("Finite-cutoff emitted energy", pstar*tc, c**2*ma*(1-e), checks)
    mref = (ma**3 - 3*a*t)**sp.Rational(1, 3)
    exact("Reference mass law", sp.diff(mref, t), -a/mref**2, checks)
    exact("Delay between the same two masses", tc-(ma**3-(e*ma)**3)/(3*a), ma**3/(3*a)*(2-3*e+e**3), checks)

    zlow = tau*(e+d/(3*tau))**3
    zhigh = d-2*tau+3*tau*e
    da = 3*tau*(1-e)
    exact("Lower map at cutoff", zlow.subs(d, 0), tau*e**3, checks)
    exact("Lower map at onset", zlow.subs(d, da), tau, checks)
    exact("Upper map at onset", zhigh.subs(d, da), tau, checks)
    exact("Equivalent clock derivative equals multiplier", sp.diff(zlow, d), (e+d/(3*tau))**2, checks)
    exact("Equivalent clock is C1 at onset", sp.diff(zlow, d).subs(d, da), sp.diff(zhigh, d), checks)
    exact("Zero-cutoff lower map", zlow.subs(e, 0), d**3/(27*tau**2), checks)
    exact("Zero-cutoff upper map", zhigh.subs(e, 0), d-2*tau, checks)

    source_rate, mult, power0 = sp.symbols("S_0 w P_0", positive=True)
    exact("Mass-interval spectral-fluence cancellation", mult*source_rate*c**2/(mult*power0), c**2*source_rate/power0, checks)
    thdot0 = a*theta**4/beta**3
    thdotcap = a*theta**2/(beta*ma**2)
    exact("Reference hardening logarithmic slope", theta*sp.diff(thdot0, theta)/thdot0, sp.Integer(4), checks)
    exact("Ceiling hardening logarithmic slope", theta*sp.diff(thdotcap, theta)/thdotcap, sp.Integer(2), checks)
    return checks


def z_full(delta: mp.mpf, epsilon: mp.mpf = EPS_C) -> mp.mpf:
    """Reference formal remaining time at the model's mass delta before cutoff."""
    transition = 3*TAU_A*(1-epsilon)
    if delta <= transition:
        return TAU_A*(epsilon+delta/(3*TAU_A))**3
    return delta-2*TAU_A+3*TAU_A*epsilon


def temperature_gev(z: mp.mpf) -> mp.mpf:
    if z <= 0:
        raise ValueError("The integrated template is evaluated only at positive z.")
    return THETA_ONE_SECOND_GEV / mp.root(z, 3)


def photon_template(energy_gev: mp.mpf, z: mp.mpf) -> mp.mpf:
    """dN/dE in GeV^-1, integrated over reference remaining time z to zero."""
    th = temperature_gev(z)
    if energy_gev < th:
        return TEMPLATE_NORM * th**mp.mpf("-1.5") * energy_gev**mp.mpf("-1.5")
    return TEMPLATE_NORM * energy_gev**-3


def power_integral(exponent: mp.mpf, left: mp.mpf, right: mp.mpf) -> mp.mpf:
    if right <= left:
        return mp.mpf(0)
    if exponent == -1:
        return mp.log(right/left)
    return (right**(exponent+1)-left**(exponent+1))/(exponent+1)


def integrated_template(z: mp.mpf, response_power: int = 0,
                        lo: mp.mpf = E_LO_GEV, hi: mp.mpf = E_HI_GEV) -> mp.mpf:
    """Analytic band integral with declared weight (E / 1 TeV)^response_power.

    response_power=0 represents a constant effective area and unit acceptance;
    these illustrative weights are not the measured response of an observatory.
    """
    th = temperature_gev(z)
    p = mp.mpf(response_power)
    low = th**mp.mpf("-1.5") * power_integral(p-mp.mpf("1.5"), lo, min(hi, th))
    high = power_integral(p-3, max(lo, th), hi)
    return TEMPLATE_NORM * (low+high) / E_LO_GEV**p


def quadrature_band(z: mp.mpf, response_power: int = 0) -> mp.mpf:
    th = temperature_gev(z)
    intervals = [E_LO_GEV] + ([th] if E_LO_GEV < th < E_HI_GEV else []) + [E_HI_GEV]
    return mp.quad(lambda energy: photon_template(energy, z)*(energy/E_LO_GEV)**response_power, intervals)


def positive_cutoff_band(delta: mp.mpf, response_power: int = 0,
                         ceiling: bool = True) -> mp.mpf:
    upper = z_full(delta) if ceiling else delta+Z_C
    return integrated_template(upper, response_power)-integrated_template(Z_C, response_power)


def window_result(delta: mp.mpf) -> dict:
    z0 = z_full(delta, mp.mpf(0))
    zexact = z_full(delta)
    den = integrated_template(delta)
    num = integrated_template(z0)
    endpoint_ratios = [photon_template(e, z0)/photon_template(e, delta) for e in (E_LO_GEV, E_HI_GEV)]
    rmin, rmax = endpoint_ratios
    rflat = num/den
    rfinite = positive_cutoff_band(delta)/positive_cutoff_band(delta, ceiling=False)
    if not rmin <= rflat <= rmax:
        raise AssertionError("Flat-response ratio violates the positive-response envelope.")
    # Also inspect strongly soft- and hard-weighted responses without treating
    # these finite checks as a proof of the weighted-mean inequality.
    for p in (-4, 1, 2, 4):
        rp = integrated_template(z0, p)/integrated_template(delta, p)
        if not rmin <= rp <= rmax:
            raise AssertionError(f"Positive-response bound failed for p={p}.")
    photon_budget = M_C*C_LIGHT**2/(E_LO_GEV*GEV_J)
    return {
        "window_s": float(delta),
        "equivalent_standard_remaining_s": float(z0),
        "equivalent_standard_remaining_positive_cutoff_s": float(zexact),
        "theta_standard_TeV": float(temperature_gev(delta)/1000),
        "theta_ceiling_TeV": float(temperature_gev(z0)/1000),
        "r_min": float(rmin),
        "r_flat": float(rflat),
        "r_max": float(rmax),
        "r_weight_E": float(integrated_template(z0, 1)/integrated_template(delta, 1)),
        "r_weight_E2": float(integrated_template(z0, 2)/integrated_template(delta, 2)),
        "r_flat_positive_cutoff": float(rfinite),
        "absolute_r_flat_cutoff_difference": float(abs(rfinite-rflat)),
        "standard_band_photons_emitted": float(den),
        "ceiling_band_photons_emitted": float(num),
        "planck_photon_budget_over_ceiling_band_photons": float(photon_budget/num),
        "bolometric_energy_ratio_zero_cutoff": float(mp.root(z0/delta, 3)),
    }


def write_csv(name: str, rows: list[dict]) -> None:
    with (DATA/name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scan_exact_checks() -> list[str]:
    """Additional identities; kept separate to preserve the benchmark report."""
    checks: list[str] = []
    x, d, tau, e, h, theta, z, energy, y = sp.symbols(
        "x Delta tau epsilon h theta z energy y", positive=True
    )
    hlo, hhi = x**2/3, 1-sp.Rational(2, 3)/x
    exact("Dimensionless lower clock map",
          (d**3/(27*tau**2)/d).subs(d, 3*tau*x), hlo, checks)
    exact("Dimensionless upper clock map",
          ((d-2*tau)/d).subs(d, 3*tau*x), hhi, checks)
    exact("Dimensionless map continuity", hlo.subs(x, 1), hhi.subs(x, 1), checks)
    exact("Dimensionless map derivative continuity",
          sp.diff(hlo, x).subs(x, 1), sp.diff(hhi, x).subs(x, 1), checks)
    exact("Positive lower derivative", sp.diff(hlo, x), 2*x/3, checks)
    exact("Positive upper derivative", sp.diff(hhi, x), 2/(3*x**2), checks)
    exact("Formal small-onset clock limit", sp.limit(hlo, x, 0), sp.Integer(0), checks)
    exact("Formal large-onset clock limit", sp.limit(hhi, x, sp.oo), sp.Integer(1), checks)
    exact("Temperature ratio from the dimensionless clock",
          (d*h)**sp.Rational(-1, 3)/d**sp.Rational(-1, 3),
          h**sp.Rational(-1, 3), checks)
    exact("Stable finite-cutoff lower clock correction",
          tau*((x+e)**3-x**3), tau*(3*x**2*e+3*x*e**2+e**3), checks)
    exact("Stable finite-cutoff crossing correction",
          (3*tau*x-2*tau+3*tau*e)-tau*x**3,
          tau*(3*e-(x-1)**2*(x+2)), checks)
    exact("Stable finite-cutoff upper clock correction",
          (d-2*tau+3*tau*e)-(d-2*tau), 3*tau*e, checks)
    # These derivatives establish a bound on the cumulative fit, not an
    # interpretation of its derivative as an instantaneous photon spectrum.
    phi_low = sp.sqrt(z)*energy**sp.Rational(-3, 2)
    phi_high = energy**-3
    exact("Cumulative low-branch elasticity",
          z*sp.diff(phi_low, z)/phi_low, sp.Rational(1, 2), checks)
    exact("Cumulative high-branch elasticity",
          z*sp.diff(phi_high, z)/phi_high, sp.Integer(0), checks)
    band_low = 2*theta**sp.Rational(-3, 2)*(1-1/sp.sqrt(15))
    band_mid = 2*theta**sp.Rational(-3, 2)-sp.Rational(3, 2)*theta**-2-sp.Rational(1, 450)
    band_high = (1-sp.Rational(1, 225))/2
    exact("Band integral joins at 15 TeV", band_low.subs(theta, 15),
          band_mid.subs(theta, 15), checks)
    exact("Band integral joins at 1 TeV", band_mid.subs(theta, 1), band_high, checks)
    exact("Full matched-mass interval duration ratio",
          3*(1-y)/(1-y**3), 3/(1+y+y**2), checks)
    return checks


def onset_scales(theta_a_tev: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Return tau_a in seconds and epsilon_c for an onset in TeV."""
    tau = (THETA_ONE_SECOND_GEV/(1000*theta_a_tev))**3
    ma = BETA/(1000*theta_a_tev*GEV_J)
    return tau, M_C/ma


def scan_clock(delta: mp.mpf, theta_a_tev: mp.mpf,
               positive_cutoff: bool = False) -> mp.mpf:
    tau, epsilon = onset_scales(theta_a_tev)
    if not positive_cutoff:
        epsilon = mp.mpf(0)
    elif epsilon >= 1:
        raise ValueError("A preterminal ceiling requires M_a > M_c.")
    if delta <= 3*tau*(1-epsilon):
        return tau*(epsilon+delta/(3*tau))**3
    return delta-2*tau+3*tau*epsilon


def scan_ratio(theta_a_tev: mp.mpf, delta: mp.mpf,
               positive_cutoff: bool = False) -> mp.mpf:
    upper = scan_clock(delta, theta_a_tev, positive_cutoff)
    if positive_cutoff:
        terminal = integrated_template(Z_C)
        return (integrated_template(upper)-terminal)/(integrated_template(delta+Z_C)-terminal)
    return integrated_template(upper)/integrated_template(delta)


def band_array(z: np.ndarray) -> np.ndarray:
    """Vectorized analytic 1--15 TeV integral of the same cumulative template."""
    z = np.asarray(z, dtype=float)
    if np.any(z <= 0):
        raise ValueError("All template clocks must be positive.")
    theta = float(THETA_ONE_SECOND_GEV)/np.cbrt(z)
    lo, hi = float(E_LO_GEV), float(E_HI_GEV)
    low = 2*theta**-1.5*np.maximum(0, lo**-0.5-np.minimum(hi, theta)**-0.5)
    high = 0.5*np.maximum(0, np.maximum(lo, theta)**-2-hi**-2)
    return float(TEMPLATE_NORM)*(low+high)


def scan_arrays(theta_a_tev: np.ndarray, delta: np.ndarray) -> dict[str, np.ndarray]:
    """Broadcast inputs; retain the exact positive-cutoff model and a stable bound."""
    theta_a_tev, delta = np.broadcast_arrays(
        np.asarray(theta_a_tev, dtype=float), np.asarray(delta, dtype=float)
    )
    tau = (7.8/theta_a_tev)**3
    epsilon = float(EPS_C)*theta_a_tev/3
    if np.any(epsilon >= 1) or np.any(theta_a_tev <= 0) or np.any(delta <= 0):
        raise ValueError("Require positive windows and 0 < M_c < M_a.")
    x = delta/(3*tau)
    z0 = np.where(x <= 1, delta*x**2/3, delta-2*tau)
    # Evaluate z_full-z0 without subtracting two nearly equal clock values.
    dz = np.where(
        x <= 1-epsilon,
        tau*(3*x**2*epsilon+3*x*epsilon**2+epsilon**3),
        np.where(x <= 1, tau*(3*epsilon-(x-1)**2*(x+2)), 3*tau*epsilon),
    )
    zfinite = z0+dz
    den = band_array(delta)
    terminal = band_array(np.asarray(float(Z_C)))
    r0 = band_array(z0)/den
    rfinite = (band_array(zfinite)-terminal)/(band_array(delta+float(Z_C))-terminal)
    # 0 <= d log C(z)/d log z <= 1/2, including continuous branch joins.
    # Rationalized square roots preserve corrections below machine epsilon.
    rel_clock = dz/z0
    eps_num = rel_clock/(np.sqrt(1+rel_clock)+1)
    rel_ref = float(Z_C)/delta
    eps_den = rel_ref/(np.sqrt(1+rel_ref)+1)
    eta = terminal/den
    error_bound = (r0*(eps_num+eps_den)+(1+r0)*eta)/(1-eta)
    return {
        "x": x, "z0_s": z0, "z_positive_cutoff_s": zfinite,
        "r_flat": r0, "r_flat_positive_cutoff": rfinite,
        "cutoff_error_absolute_bound": error_bound,
    }


def scan_point(theta_a_tev: mp.mpf, delta: mp.mpf) -> dict:
    tau, epsilon = onset_scales(theta_a_tev)
    z0 = scan_clock(delta, theta_a_tev)
    zfinite = scan_clock(delta, theta_a_tev, True)
    ratio = scan_ratio(theta_a_tev, delta)
    finite_ratio = scan_ratio(theta_a_tev, delta, True)
    rel_clock = (zfinite-z0)/z0
    eps_num = rel_clock/(mp.sqrt(1+rel_clock)+1)
    rel_ref = Z_C/delta
    eps_den = rel_ref/(mp.sqrt(1+rel_ref)+1)
    eta = integrated_template(Z_C)/integrated_template(delta)
    bound = (ratio*(eps_num+eps_den)+(1+ratio)*eta)/(1-eta)
    correction = abs(finite_ratio-ratio)
    if correction > bound:
        raise AssertionError("Positive-cutoff correction exceeds its analytic bound.")
    return {
        "theta_a_TeV": float(theta_a_tev), "window_s": float(delta),
        "tau_a_s": float(tau), "epsilon_c": float(epsilon),
        "x": float(delta/(3*tau)), "h": float(z0/delta),
        "z0_s": float(z0), "z_positive_cutoff_s": float(zfinite),
        "theta_reference_TeV": float(temperature_gev(delta)/1000),
        "theta_ceiling_TeV": float(temperature_gev(z0)/1000),
        "r_flat": float(ratio), "r_flat_positive_cutoff": float(finite_ratio),
        "absolute_cutoff_correction": float(correction),
        "cutoff_error_absolute_bound": float(bound),
    }


def solve_scan_crossover(delta: mp.mpf, target: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Monotone bisection, independent of the plotting grid."""
    left, right = mp.mpf("0.3"), mp.mpf(30)
    if not scan_ratio(left, delta) < target < scan_ratio(right, delta):
        raise AssertionError("Crossover is not bracketed by the declared scan.")
    for _ in range(200):
        middle = (left+right)/2
        if scan_ratio(middle, delta) < target:
            left = middle
        else:
            right = middle
        if right-left < mp.mpf("1e-45"):
            break
    root = (left+right)/2
    residual = abs(scan_ratio(root, delta)-target)
    if residual > mp.mpf("1e-40"):
        raise AssertionError("Crossover bisection did not converge.")
    return root, residual


def generate_onset_scan() -> None:
    """Produce separate scan outputs without changing the 3 TeV benchmark."""
    checks = scan_exact_checks()
    theta_axis = np.geomspace(0.3, 30, 241)
    delta_axis = np.geomspace(1, 150, 251)
    grid = scan_arrays(theta_axis[None, :], delta_axis[:, None])
    ratios = grid["r_flat"]
    increments = np.diff(ratios, axis=1)
    violations = int(np.count_nonzero(increments < -1e-13))
    if violations or np.any(ratios <= 0) or np.any(ratios > 1):
        raise AssertionError("Scan violates monotonicity or 0 < R <= 1.")
    if np.any(grid["cutoff_error_absolute_bound"] < 0):
        raise AssertionError("Negative finite-cutoff error bound.")
    max_index = np.unravel_index(
        np.argmax(grid["cutoff_error_absolute_bound"]), ratios.shape
    )

    with mp.workdps(80):
        windows = [mp.mpf(d) for d in (10, 20, 100, 120)]
        selected_thetas = [mp.mpf(t) for t in ("0.5", "1", "2", "3", "5", "10", "20")]
        selected = [scan_point(t, d) for t in selected_thetas for d in windows]
        crossovers, roots, transition_checks = [], {}, []
        for delta in windows:
            theta50, error50 = solve_scan_crossover(delta, mp.mpf("0.5"))
            theta90, error90 = solve_scan_crossover(delta, mp.mpf("0.9"))
            transition = mp.mpf("7.8")*mp.root(3/delta, 3)
            roots[(int(delta), 50)] = theta50
            roots[(int(delta), 90)] = theta90
            low = lambda t: integrated_template(
                delta**3/(27*onset_scales(t)[0]**2)
            )/integrated_template(delta)
            high = lambda t: integrated_template(
                delta-2*onset_scales(t)[0]
            )/integrated_template(delta)
            value_gap = abs(low(transition)-high(transition))
            derivative_gap = abs(mp.diff(low, transition)-mp.diff(high, transition))
            if max(value_gap, derivative_gap) > mp.mpf("1e-40"):
                raise AssertionError("Yield or its onset derivative is discontinuous.")
            transition_checks.append({
                "window_s": float(delta), "value_gap": float(value_gap),
                "derivative_gap_per_TeV": float(derivative_gap),
            })
            crossovers.append({
                "window_s": float(delta),
                "theta_a_at_r50_TeV": float(theta50),
                "theta_a_at_r90_TeV": float(theta90),
                "theta_a_clock_transition_TeV": float(transition),
                "r_flat_at_clock_transition": float(scan_ratio(transition, delta)),
                "x_at_r50": float(delta/(3*onset_scales(theta50)[0])),
                "x_at_r90": float(delta/(3*onset_scales(theta90)[0])),
                "r50_root_absolute_residual": float(error50),
                "r90_root_absolute_residual": float(error90),
                "r50_positive_cutoff_correction": scan_point(theta50, delta)["absolute_cutoff_correction"],
                "r90_positive_cutoff_correction": scan_point(theta90, delta)["absolute_cutoff_correction"],
            })

        # Quadrature uses the original piecewise spectrum, independently of
        # both the closed band integral and the vectorized scan implementation.
        audit_points = [(mp.mpf(t), mp.mpf(d)) for t, d in (
            ("0.3", 1), ("0.3", 150), ("3", 1), ("3", 10),
            ("3", 20), ("3", 100), ("30", 1), ("30", 150),
        )]
        audit_points += [
            (mp.mpf("7.8")*mp.root(3/windows[0], 3), windows[0]),
            (mp.mpf("7.8")*mp.root(3/windows[-1], 3), windows[-1]),
            (roots[(20, 50)], mp.mpf(20)),
            (roots[(100, 90)], mp.mpf(100)),
        ]
        terminal_quad = quadrature_band(Z_C)
        quadrature_audits = []
        float_errors = []
        for theta_a, delta in audit_points:
            z0, zfinite = scan_clock(delta, theta_a), scan_clock(delta, theta_a, True)
            qzero = quadrature_band(z0)/quadrature_band(delta)
            qfinite = ((quadrature_band(zfinite)-terminal_quad) /
                       (quadrature_band(delta+Z_C)-terminal_quad))
            analytic_zero = scan_ratio(theta_a, delta)
            analytic_finite = scan_ratio(theta_a, delta, True)
            zero_error = abs(qzero-analytic_zero)/analytic_zero
            finite_error = abs(qfinite-analytic_finite)/analytic_finite
            if max(zero_error, finite_error) > mp.mpf("1e-40"):
                raise AssertionError("Scan quadrature audit failed.")
            array_ratio = scan_arrays(float(theta_a), float(delta))["r_flat"]
            array_error = abs(float(array_ratio)-float(analytic_zero))/float(analytic_zero)
            float_errors.append(array_error)
            quadrature_audits.append({
                "theta_a_TeV": float(theta_a), "window_s": float(delta),
                "zero_cutoff_relative_error": float(zero_error),
                "positive_cutoff_relative_error": float(finite_error),
                "numpy_relative_error": array_error,
            })
        if max(float_errors) > 1e-13:
            raise AssertionError("Vectorized scan disagrees with high-precision evaluation.")

        limit_checks = []
        for delta in windows:
            low = scan_ratio(mp.mpf("1e-6"), delta)
            high = scan_ratio(mp.mpf("1e6"), delta)
            cubic_ratio = scan_ratio(mp.mpf("1e-3"), delta)/scan_ratio(mp.mpf("1e-4"), delta)
            if low > mp.mpf("1e-16") or 1-high > mp.mpf("1e-14"):
                raise AssertionError("Formal onset limits failed.")
            if abs(cubic_ratio-1000) > mp.mpf("1e-40"):
                raise AssertionError("Small-onset cubic yield scaling failed.")
            limit_checks.append({
                "window_s": float(delta), "r_at_theta_1e_minus6_TeV": float(low),
                "one_minus_r_at_theta_1e6_TeV": float(1-high),
                "r_at_1e_minus3_over_r_at_1e_minus4_TeV": float(cubic_ratio),
            })
        max_bound_point = scan_point(
            mp.mpf(str(theta_axis[max_index[1]])),
            mp.mpf(str(delta_axis[max_index[0]])),
        )
        # The 3 TeV values are also regenerated through the independent-onset
        # path, without rewriting or extending the earlier benchmark report.
        benchmark_agreement = max(
            abs(scan_ratio(mp.mpf(3), d) -
                integrated_template(z_full(d, mp.mpf(0)))/integrated_template(d))
            for d in windows
        )
        if benchmark_agreement > mp.mpf("1e-40"):
            raise AssertionError("The original benchmark changed.")

    write_csv("burst_scan_selected.csv", selected)
    write_csv("burst_scan_crossovers.csv", crossovers)
    slices = []
    for delta in (10, 20, 100, 120):
        curve = scan_arrays(theta_axis, np.asarray(float(delta)))
        for j, theta_a in enumerate(theta_axis):
            slices.append({
                "theta_a_TeV": float(theta_a), "window_s": delta,
                "r_flat": float(curve["r_flat"][j]),
                "r_flat_positive_cutoff": float(curve["r_flat_positive_cutoff"][j]),
            })
    write_csv("burst_scan_slices.csv", slices)
    grid_fields = ["theta_a_TeV", "window_s", "x", "z0_s", "z_positive_cutoff_s",
                   "r_flat", "r_flat_positive_cutoff", "cutoff_error_absolute_bound"]
    with (DATA/"burst_scan_grid.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=grid_fields)
        writer.writeheader()
        for i, delta in enumerate(delta_axis):
            for j, theta_a in enumerate(theta_axis):
                writer.writerow({
                    "theta_a_TeV": float(theta_a), "window_s": float(delta),
                    **{key: float(grid[key][i, j]) for key in grid_fields[2:]},
                })
    np.savez_compressed(
        DATA/"burst_scan_grid.npz", theta_a_TeV=theta_axis, window_s=delta_axis, **grid
    )
    threshold_table = [
        "% Generated by verify_burst_model.py; template yield features, not exclusion limits.",
        r"\begin{tabular}{@{}rrrr@{}}", r"\toprule",
        r"\(\Delta\) (s) & \(\Theta_a^{(50)}\) (TeV) & \(\Theta_a^{(90)}\) (TeV) & \(\Theta_a^{\rm tr}\) (TeV) \\",
        r"\midrule",
    ]
    for row in crossovers:
        threshold_table.append(
            f"{row['window_s']:.0f} & {row['theta_a_at_r50_TeV']:.3f} & "
            f"{row['theta_a_at_r90_TeV']:.3f} & "
            f"{row['theta_a_clock_transition_TeV']:.3f}" + r" \\"
        )
    threshold_table += [r"\bottomrule", r"\end{tabular}"]
    (DATA/"burst_scan_threshold_table.tex").write_text(
        "\n".join(threshold_table)+"\n", encoding="utf-8"
    )
    selected_table = [
        "% Generated by verify_burst_model.py; 1--15 TeV constant-response yield ratios.",
        r"\begin{tabular}{@{}rrrrr@{}}", r"\toprule",
        r"\(\Theta_a\) (TeV) & \(R_{\rm flat}(10\,{\rm s})\) & \(R_{\rm flat}(20\,{\rm s})\) & \(R_{\rm flat}(100\,{\rm s})\) & \(R_{\rm flat}(120\,{\rm s})\) \\",
        r"\midrule",
    ]
    for theta_a in (0.5, 1, 2, 3, 5, 10, 20):
        selected_rows = [r for r in selected if r["theta_a_TeV"] == theta_a]
        selected_table.append(
            f"{theta_a:g} & " + " & ".join(f"{r['r_flat']:.5f}" for r in selected_rows) + r" \\"
        )
    selected_table += [r"\bottomrule", r"\end{tabular}"]
    (DATA/"burst_scan_selected_table.tex").write_text(
        "\n".join(selected_table)+"\n", encoding="utf-8"
    )

    report = {
        "title": "Conditional luminosity-ceiling onset scan",
        "scope": "Quantitative features of the adopted cumulative photon template with a constant response in 1-15 TeV. No instrument exclusions, statistical distinguishability, or source-derived TCGS predictions are calculated.",
        "assumptions": [
            "Retain the same additional all-species ceiling multiplier w=min(1,(M/M_a)^2), energy conservation, constant gamma_0, and adopted approximate 7.8 TeV at 1 s reference calibration.",
            "Vary the independently postulated onset Theta_a; no curvature rule fixes it.",
            "Use the original cumulative photon fit only as an integrated template; do not infer instantaneous photon rates by differentiating it.",
            "Each window ends at its own model's fixed positive cutoff M_c=(3/4)^(1/4)*m_P.",
            "The plotted zero-cutoff ratio approximates the exact subtraction of the common integrated template at z_c. This template calculation does not predict endpoint emission.",
            "Numerical root precision is computational precision, not physical precision of the approximate spectral fit or calibration.",
            "Formal Theta_a limits outside the displayed range are mathematical checks, not applications of the Standard Model calibration at arbitrarily low temperatures or physical continuation beyond M_a=M_c.",
        ],
        "formulas": {
            "tau_a_seconds": "(7.8 / theta_a_TeV)^3",
            "x": "Delta / (3*tau_a)",
            "h": "x^2/3 for x<=1; 1-2/(3*x) for x>=1",
            "z0": "Delta*h",
            "temperature_ratio": "Theta_ceiling/Theta_reference = h^(-1/3)",
            "dimensionless_band_yield": "Y(vartheta)=2*vartheta^(-3/2)*(1-15^(-1/2)) above 15; 2*vartheta^(-3/2)-(3/2)*vartheta^(-2)-1/450 between 1 and 15; (1-15^(-2))/2 below 1. vartheta=Theta/(1 TeV); N_band=9e29*Y.",
            "r_flat": "C(z0)/C(Delta), with C the analytic integral of the original cumulative template in 1-15 TeV.",
            "band_dependence": "R_flat depends on x and on the band edges relative to Theta_reference; it is not a universal function of x alone.",
            "epsilon_c": "M_c/M_a; varies in direct proportion to Theta_a while z_c=M_c^3/(3*gamma_0) is constant.",
            "z_full": "tau_a*(epsilon_c+Delta/(3*tau_a))^3 for Delta<=3*tau_a*(1-epsilon_c); Delta-2*tau_a+3*tau_a*epsilon_c otherwise.",
            "r_positive_cutoff": "[C(z_full)-C(z_c)]/[C(Delta+z_c)-C(z_c)]",
            "clock_transition_theta_TeV": "7.8*(3/Delta)^(1/3), for the plotted zero-cutoff approximation.",
            "cumulative_elasticity_bound": "0 <= d log C(z)/d log z <= 1/2; follows because each positive spectral contribution is either proportional to sqrt(z) or constant and joins continuously.",
            "cutoff_error_bound": "|R_epsilon-R0| <= [R0*(e_N+e_D)+(1+R0)*eta]/(1-eta), where e_N=sqrt(z_full/z0)-1, e_D=sqrt(1+z_c/Delta)-1, eta=C(z_c)/C(Delta). Rationalized square roots and a polynomial clock correction avoid cancellation.",
            "matched_mass_fluence": "Identical spectra integrated between the same two masses when the same positive w scales every emission channel and luminosity.",
            "matched_mass_duration_ratio": "t_ceiling/t_reference=3/(1+y+y^2) for the interval M_a to y*M_a, 0<y<1. Equal full-interval fluence does not imply equal duration.",
            "formal_low_onset_behavior": "At fixed Delta, R_flat is proportional to Theta_a^3 once the modified temperature exceeds the band's upper edge, and tends to zero.",
            "formal_high_onset_behavior": "At fixed Delta, z0 tends to Delta and R_flat tends to one.",
        },
        "grid": {
            "theta_a_range_TeV": [0.3, 30], "window_range_s": [1, 150],
            "theta_count": len(theta_axis), "window_count": len(delta_axis),
            "point_count": int(ratios.size), "spacing": "logarithmic on both axes",
            "array_order": "[window index, onset-temperature index]",
            "r_flat_min": float(np.min(ratios)), "r_flat_max": float(np.max(ratios)),
            "B_hat_onset_range": [float(B_A)*0.1**2, float(B_A)*10**2],
            "M_c_kg": float(M_C), "z_c_s": float(Z_C),
        },
        "checks": {
            "exact_identity_count": len(checks), "exact_identities_passed": checks,
            "grid_range_checks": int(ratios.size),
            "theta_monotonicity_comparison_count": int(increments.size),
            "theta_monotonicity_violations": violations,
            "minimum_theta_grid_increment": float(np.min(increments)),
            "independent_quadrature_case_count": len(quadrature_audits),
            "independent_band_quadrature_count": 4*len(quadrature_audits)+1,
            "quadrature_audits": quadrature_audits,
            "maximum_numpy_relative_error_in_audit": max(float_errors),
            "root_count": 2*len(crossovers),
            "transition_value_and_derivative_checks": transition_checks,
            "formal_limit_checks": limit_checks,
            "maximum_cutoff_error_bound_on_grid": float(np.max(grid["cutoff_error_absolute_bound"])),
            "maximum_bound_grid_point_high_precision_audit": max_bound_point,
            "largest_observed_double_precision_cutoff_difference": float(np.max(abs(grid["r_flat_positive_cutoff"]-ratios))),
            "floating_point_scope": "Float64 cannot accurately resolve every tiny cutoff correction. The whole-grid analytic bound and 80-digit evaluations supply the cutoff consistency check.",
            "benchmark_maximum_absolute_ratio_difference": float(benchmark_agreement),
        },
        "crossovers": crossovers,
        "selected_yields": selected,
        "runtime": {"python": platform.python_version(), "sympy": sp.__version__,
                    "mpmath": mp.__version__, "numpy": np.__version__, "audit_digits": 80},
        "outputs": [
            "burst_scan_results.json", "burst_scan_grid.csv", "burst_scan_grid.npz",
            "burst_scan_selected.csv", "burst_scan_selected_table.tex",
            "burst_scan_crossovers.csv", "burst_scan_threshold_table.tex",
            "burst_scan_slices.csv",
        ],
    }
    (DATA/"burst_scan_results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"SCAN PASS: {ratios.size:,} grid points; {increments.size:,} monotonicity comparisons; {len(checks)} new exact identities.")
    print(f"SCAN PASS: {len(quadrature_audits)} independent quadrature cases; {2*len(crossovers)} crossover roots.")
    for row in crossovers:
        print(f"Delta={row['window_s']:g}s: Theta50={row['theta_a_at_r50_TeV']:.9f}, Theta90={row['theta_a_at_r90_TeV']:.9f}, transition={row['theta_a_clock_transition_TeV']:.9f} TeV")
    print(f"Maximum scan cutoff-error bound: {np.max(grid['cutoff_error_absolute_bound']):.6g}")
    print(f"Wrote separate onset-scan tables to {DATA}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    checks = exact_checks()
    quadrature_errors = []
    for th in (mp.mpf(500), mp.mpf(3000), mp.mpf(20000)):
        z = (THETA_ONE_SECOND_GEV/th)**3
        for response_power in (0, 1, 2):
            analytic = integrated_template(z, response_power)
            numerical = quadrature_band(z, response_power)
            error = abs(analytic-numerical)/abs(analytic)
            quadrature_errors.append(float(error))
            if error > mp.mpf("1e-40"):
                raise AssertionError("Analytic broken-power-law integral disagrees with quadrature.")

    rows = [window_result(mp.mpf(d)) for d in (10, 20, 100, 120)]
    write_csv("burst_model_results.csv", rows)
    table_lines = ["% Generated by code/verify_burst_model.py; declared template and flat-response benchmark."]
    for row in rows[:3]:
        table_lines.append(" & ".join([
            f"{row['window_s']:.0f}",
            f"{row['equivalent_standard_remaining_s']:.3f}",
            f"{row['theta_standard_TeV']:.3f}",
            f"{row['theta_ceiling_TeV']:.3f}",
            f"{row['r_flat']:.3f}",
            f"{row['r_min']:.3f}",
            f"{row['r_max']:.3f}",
        ]) + r" \\")
    (DATA/"burst_table_rows.tex").write_text("\n".join(table_lines)+"\n", encoding="utf-8")
    full_table = [
        table_lines[0],
        r"\begin{tabular}{@{}rrrrrrr@{}}",
        r"\toprule",
        r"\(\Delta\) & \(z_0\) & \(\Theta_r\) & \(\Theta_m\) & \(R_{\rm flat}\) & \(r_{\min}\) & \(r_{\max}\) \\",
        r"\(\mathrm{s}\) & \(\mathrm{s}\) & \(\mathrm{TeV}\) & \(\mathrm{TeV}\) & & & \\",
        r"\midrule",
        *table_lines[1:],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (DATA/"burst_table.tex").write_text("\n".join(full_table)+"\n", encoding="utf-8")
    window_grid = [window_result(mp.mpf(10)**(mp.mpf(i)/400*mp.log10(150))) for i in range(401)]
    write_csv("burst_window_grid.csv", window_grid)

    time_grid = []
    # Common onset clock, deliberately displayed only until Theta=30 TeV.
    # The physical positive cutoff is far below these displayed masses.
    for i in range(501):
        x = mp.mpf(1)-mp.mpf("0.9")*i/500
        for model in ("reference", "ceiling"):
            time = TAU_A*(1-x**3) if model == "reference" else 3*TAU_A*(1-x)
            power = x**-2 if model == "reference" else mp.mpf(1)
            time_grid.append({
                "model": model,
                "u_since_common_onset_s": float(time),
                "M_over_Ma": float(x),
                "P_over_Pstar": float(power),
                "theta_TeV": float(THETA_A_GEV/(1000*x)),
                "emitted_energy_over_Ma_c2": float(1-x),
            })
    write_csv("burst_time_grid.csv", time_grid)
    hess = rows[-1]
    conditional_hess = {
        "conditional_template_only": True,
        "not_a_reanalysed_HESS_limit": True,
        "window_s": 120,
        "published_reference_bound_pc3yr": 2000,
        "published_confidence_percent": 95,
        "q_flat_illustrative": hess["r_flat"],
        "q_response_lower": hess["r_min"],
        "q_response_upper": hess["r_max"],
        "q_low_energy": float(mp.sqrt(z_full(mp.mpf(120), mp.mpf(0))/120)),
        "rho_low_energy_pc3yr": float(mp.mpf(2000)*(z_full(mp.mpf(120), mp.mpf(0))/120)**mp.mpf("-0.75")),
        "rho_rescaled_flat_pc3yr": 2000/hess["r_flat"]**1.5,
        "rho_rescaled_response_lower_pc3yr": 2000/hess["r_max"]**1.5,
        "rho_rescaled_response_upper_pc3yr": 2000/hess["r_min"]**1.5,
        "scope": "Illustrative rho_UL/q^(3/2) only under a common fixed-count threshold, Euclidean volume, unchanged background procedure and exposure, and a distance-independent q. Not an instrument-response calculation or a new H.E.S.S. confidence limit.",
    }
    report = {
        "title": "Conditional luminosity-ceiling model: analytical clocks and photon-window predictions",
        "status": "Added ansatz; no source-derived TCGS prediction, state completion, or telescope-data reanalysis.",
        "assumptions": [
            "P_0=c^2*a/M^2 with the approximately constant high-temperature Standard Model mass-loss coefficient a fixed by the adopted 7.8 TeV at 1 s calibration.",
            "Theta=k_B*T_H=beta/M and beta=hbar*c^3/(8*pi*G).",
            "A new finite luminosity ceiling Pstar is postulated; w=min(1,Pstar/P_0)=min(1,(M/M_a)^2).",
            "The same w multiplies all instantaneous asymptotic particle-emission spectra and the all-species luminosity, preserving their relative spectral shapes at fixed mass.",
            "The underlying inclusive energy accounting has the quasistationary worldtube scope stated in the manuscript; it is not an all-species Bondi identity.",
            "The integrated photon fit is LHAASO 2025 equation (1), including primary and secondary photons; no instantaneous spectrum is obtained by differentiating that fit.",
            "The 1-15 TeV flat-response benchmark uses constant effective area and unit acceptance. Positive-response bounds do not specify an observatory's actual response.",
            "Distances, overall response normalization, and the integrated-spectrum normalization cancel in the listed ratios; energy-dependent response shapes need not cancel.",
            "Tables use the zero-cutoff mapping as an approximation to a positive unit-curvature cutoff. Its exact correction and an independent residual photon-energy budget are reported.",
            "A residual photon-count bound additionally assumes nonnegative outgoing particle energies, no extra reservoir, and photon energy at least 1 TeV; it is stronger in premises than a bound on net quantum stress-energy flux.",
            "The ideal sharp ceiling is continuous at onset but not differentiable there. A smooth transition changes its detailed onset profile.",
            "The common-onset plots stop at Theta=30 TeV, far above M_c; neither model's physical endpoint is depicted.",
        ],
        "primary_source": {"key": "lhaaso2025", "url": "https://arxiv.org/pdf/2505.24586", "locator": "equations (1)-(2), PDF page 4"},
        "scales": {
            "theta_a_TeV": 3,
            "theta_at_one_second_reference_TeV": 7.8,
            "tau_a_s": float(TAU_A),
            "ceiling_onset_to_formal_zero_s": float(3*TAU_A),
            "ceiling_onset_to_positive_cutoff_s": float(DELTA_A),
            "reference_onset_to_positive_cutoff_s": float(TAU_A*(1-EPS_C**3)),
            "M_a_kg": float(M_A),
            "M_a_over_mPlanck": float(M_A/M_PLANCK),
            "M_c_kg": float(M_C),
            "epsilon_c": float(EPS_C),
            "a_kg3_per_s": float(A_MASS),
            "gamma_0_kg3_per_s": float(A_MASS),
            "Pstar_W": float(P_CEILING),
            "M_a_c2_J": float(M_A*C_LIGHT**2),
            "B_hat_at_onset": float(B_A),
            "reference_zc_s": float(Z_C),
            "ceiling_cutoff_clock_shift_s": float(3*TAU_A*EPS_C),
            "Planck_residual_available_energy_J": float(M_C*C_LIGHT**2),
            "Planck_residual_max_photons_above_1TeV": float(M_C*C_LIGHT**2/(E_LO_GEV*GEV_J)),
            "display_min_mass_over_Ma": 0.1,
            "display_max_theta_TeV": 30,
            "onset_equal_to_cutoff_has_no_preterminal_suppression": True,
        },
        "exact_identities_passed": checks,
        "exact_identity_count": len(checks),
        "quadrature_check_count": len(quadrature_errors),
        "maximum_quadrature_relative_error": max(quadrature_errors),
        "windows": rows,
        "conditional_hess_120s": conditional_hess,
        "endpoint_extension": {
            "q_zero": "Include w or F with positive nonzero boundary limit: finite arrival, nonzero one-sided boundary luminosity, no post-boundary continuation selected.",
            "q_positive": "The existing F_q family forces zero boundary luminosity for every q>0; it classifies approach rates within that chosen boundary behavior.",
            "q_negative_optional": "For negative q, formally divergent boundary luminosity can still have integrable finite total energy. Such rates are smooth only on the open domain and do not have a finite continuous boundary extension.",
        },
        "runtime": {"python": platform.python_version(), "sympy": sp.__version__, "mpmath": mp.__version__},
        "outputs": ["burst_model_results.json", "burst_model_results.csv", "burst_table_rows.tex", "burst_table.tex", "burst_window_grid.csv", "burst_time_grid.csv"],
    }
    (DATA/"burst_model_results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"PASS: {len(checks)} exact identities and {len(quadrature_errors)} independent band-integral quadratures.")
    print(f"Benchmark: tau_a={float(TAU_A):.6f} s, 3*tau_a={float(3*TAU_A):.6f} s, B_a={float(B_A):.6g}.")
    for row in rows:
        print(f"Delta={row['window_s']:g}s: z={row['equivalent_standard_remaining_s']:.6g}s, ratios=[{row['r_min']:.6f}, {row['r_flat']:.6f}, {row['r_max']:.6f}]")
    print(f"Wrote burst-model tables to {DATA}")
    generate_onset_scan()


if __name__ == "__main__":
    main()
