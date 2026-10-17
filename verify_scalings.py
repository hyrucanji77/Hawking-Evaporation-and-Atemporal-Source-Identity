#!/usr/bin/env python3
"""Reproduce exact identities and dimensionless illustrations for the paper.

All tabulated curves are analytical model evaluations, not observational data.
The luminosity P = hbar*c**6/(15360*pi*G**2*M**2) is a deliberately
specified photon-blackbody toy model. It is not the full greybody-corrected
Standard Model luminosity. Curvature and rate thresholds do not determine a
physical endpoint. The endpoint examples demonstrate non-identifiability.

Run from the project directory (or use the absolute script path):
    python verify_scalings.py

Dependencies: SymPy and mpmath; see requirements.txt and README.md.
Outputs are written to data/ in the repository root, independently of the
working directory.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from pathlib import Path

import mpmath as mp
import sympy as sp


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "code" else SCRIPT_DIR
DATA = ROOT / "data"
mp.mp.dps = 60


def identity(name: str, left: sp.Expr, right: sp.Expr, checks: list[str]) -> None:
    """Demand an exact symbolic equality; do not silently accept float agreement."""
    residual = sp.simplify(left - right)
    if residual != 0:
        raise AssertionError(f"{name}: nonzero residual {residual}")
    checks.append(name)


def symbolic_checks() -> list[str]:
    checks: list[str] = []
    G, c, hbar, kb, M, r, mu = sp.symbols(
        "G c hbar k_B M r mu", positive=True
    )
    ellp2 = hbar * G / c**3
    mp2 = hbar * c / G
    rs = 2 * G * M / c**2
    K = 48 * G**2 * M**2 / (c**4 * r**6)
    rootK = sp.sqrt(K)
    identity("Kretschmann square root", rootK, 4 * sp.sqrt(3) * G * M / (c**2 * r**3), checks)
    BH = sp.simplify((ellp2 * rootK).subs(r, rs))
    identity("Horizon curvature selector", BH, sp.sqrt(3) * mp2 / (2 * M**2), checks)
    Bmu = sp.simplify(BH.subs(M, mu * sp.sqrt(mp2)))
    identity("Dimensionless horizon selector", Bmu, sp.sqrt(3) / (2 * mu**2), checks)
    mustar = (sp.Rational(3, 4)) ** sp.Rational(1, 4)
    identity("Unit-curvature crossing", Bmu.subs(mu, mustar), sp.Integer(1), checks)
    identity("Horizon radius at crossing", (rs / sp.sqrt(ellp2)).subs(M, mustar * sp.sqrt(mp2)), 2 * mustar, checks)

    TH = hbar * c**3 / (8 * sp.pi * G * kb * M)
    TP = sp.sqrt(mp2) * c**2 / kb
    Tmu = sp.simplify((TH / TP).subs(M, mu * sp.sqrt(mp2)))
    identity("Dimensionless Hawking temperature", Tmu, 1 / (8 * sp.pi * mu), checks)
    identity("Temperature-curvature relation", Bmu, 32 * sp.sqrt(3) * sp.pi**2 * Tmu**2, checks)
    identity("Reduced thermal wavelength", hbar * c / (kb * TH), 4 * sp.pi * rs, checks)

    P = hbar * c**6 / (15360 * sp.pi * G**2 * M**2)
    PP = c**5 / G
    # kappa is an inverse-time peeling rate, normalized to the asymptotic u.
    kappa = c**3 / (4 * G * M)
    dMdu = -P / c**2
    dkappadu = sp.diff(kappa, M) * dMdu
    epsilon = sp.simplify(dkappadu / kappa**2)  # positive in evaporation
    identity("Adiabaticity-power relation", epsilon, 4 * P / PP, checks)
    epsmu = sp.simplify(epsilon.subs(M, mu * sp.sqrt(mp2)))
    identity("Toy adiabaticity parameter", epsmu, 1 / (3840 * sp.pi * mu**2), checks)
    identity("Adiabaticity-curvature ratio", epsmu / Bmu, 1 / (1920 * sp.pi * sp.sqrt(3)), checks)
    muad = 1 / sp.sqrt(3840 * sp.pi)
    identity("Unit-adiabaticity crossing", epsmu.subs(mu, muad), sp.Integer(1), checks)

    alpha, M0, Mc, u = sp.symbols("alpha M_0 M_c u", positive=True)
    mass = (M0**3 - 3 * alpha * u / c**2) ** sp.Rational(1, 3)
    identity("Cubic mass-loss solution", sp.diff(mass, u), -alpha / (c**2 * mass**2), checks)
    energy = c**2 * (M0 - mass)
    identity("Energy conservation along the trajectory", sp.diff(energy, u), alpha / mass**2, checks)
    uc = c**2 * (M0**3 - Mc**3) / (3 * alpha)
    identity("Finite-cutoff mass", mass.subs(u, uc), Mc, checks)
    identity("Finite-cutoff emitted energy", energy.subs(u, uc), c**2 * (M0 - Mc), checks)
    # Independent endpoint-energy integral, using remaining time x and its
    # positive real cube root; exponent -2/3 is integrable at x=0.
    x, delta = sp.symbols("x delta", positive=True)
    p_remaining = alpha * (3 * alpha * x / c**2) ** (-sp.Rational(2, 3))
    integral = sp.integrate(p_remaining, (x, 0, delta))
    identity("Integrable formal power divergence", integral, c**2 * (3 * alpha * delta / c**2) ** sp.Rational(1, 3), checks)

    # Normalize M_c=1 and alpha/c^2=1 for illustrative endpoint counterexamples.
    # For x>1, elapsed time is the integral of x^2/(1-x^-2)^q.
    z = sp.symbols("z", positive=True)
    y = sp.symbols("y", positive=True)
    antiderivatives = {
        sp.Rational(1, 2): (z**2 - 1) ** sp.Rational(3, 2) / 3 + sp.sqrt(z**2 - 1),
        sp.Integer(1): z**3 / 3 + z + sp.log((z - 1) / (z + 1)) / 2,
        sp.Integer(2): z**3 / 3 + 2 * z + 5 * sp.log((z - 1) / (z + 1)) / 4 - z / (2 * (z**2 - 1)),
    }
    for q, antiderivative in antiderivatives.items():
        # Substitute z=sqrt(y^2+1) to enforce z>1 and avoid square-root branches.
        rate_integrand = z**2 / (1 - z**-2) ** q
        residual = (sp.diff(antiderivative, z) - rate_integrand).subs(z, sp.sqrt(y**2 + 1))
        identity(f"Endpoint time antiderivative q={q}", residual, sp.Integer(0), checks)
        suppression = (1 - (1 + delta)**-2) ** q
        leading = sp.limit(suppression / delta**q, delta, 0, dir="+")
        identity(f"Endpoint flux exponent q={q}", leading, 2**q, checks)
    identity("Finite arrival q=1/2", antiderivatives[sp.Rational(1, 2)].subs(z, 2), 2 * sp.sqrt(3), checks)
    return checks


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def endpoint_data() -> tuple[list[dict], dict]:
    """Exact-antiderivative verification of finite and divergent arrival times.

    x=M/M_c, x0=2, and theta=(alpha/(c^2*M_c^3))*(u-u0).
    A positive lower cutoff delta=x-1 is used only to evaluate an improper
    integral. The analytic integral test, rather than finite numerical rows,
    establishes finite versus infinite arrival time.
    """
    def primitive(q: mp.mpf, x: mp.mpf) -> mp.mpf:
        w = x*x - 1
        if q == mp.mpf("0.5"):
            return mp.sqrt(w) * (w / 3 + 1)
        if q == 1:
            return x**3 / 3 + x + mp.log((x - 1) / (x + 1)) / 2
        if q == 2:
            return x**3 / 3 + 2*x + mp.mpf("1.25") * mp.log((x - 1) / (x + 1)) - x / (2*w)
        raise ValueError(q)

    rows: list[dict] = []
    qs = (mp.mpf("0.5"), mp.mpf(1), mp.mpf(2))
    values: dict[str, list[mp.mpf]] = {}
    half_limit = 2 * mp.sqrt(3)
    for q in qs:
        times = []
        for exponent in range(1, 13):
            delta = mp.mpf(10) ** (-exponent)
            t = primitive(q, mp.mpf(2)) - primitive(q, 1 + delta)
            times.append(t)
            rows.append({
                "q": str(q),
                "delta_M_over_Mc": mp.nstr(delta, 18),
                "M_over_Mc": mp.nstr(1 + delta, 25),
                "dimensionless_elapsed_time": mp.nstr(t, 25),
                "analytic_arrival": "finite" if q < 1 else "infinite",
                "finite_limit_if_applicable": mp.nstr(half_limit, 25) if q < 1 else "",
            })
        values[str(q)] = times
    tail_ratio = (half_limit - values["0.5"][-1]) / mp.sqrt(mp.mpf("1e-12"))
    log_increment = values["1.0"][-1] - values["1.0"][-2]
    inverse_coefficient = mp.mpf("1e-12") * values["2.0"][-1]
    if not abs(tail_ratio - mp.sqrt(2)) < mp.mpf("1e-9"):
        raise AssertionError("q=1/2 finite-tail coefficient failed")
    if not abs(log_increment - mp.log(10)/2) < mp.mpf("1e-9"):
        raise AssertionError("q=1 logarithmic divergence coefficient failed")
    if not abs(inverse_coefficient - mp.mpf("0.25")) < mp.mpf("1e-9"):
        raise AssertionError("q=2 inverse-distance divergence coefficient failed")
    audit = {
        "normalization": "M_c=1, M_0=2; theta=alpha*(u-u0)/(c^2*M_c^3)",
        "scope": "Constructed effective counterexamples; no physical endpoint inference.",
        "regularity": "F_q is smooth for B<b_c and continuous at b_c. The q=1/2 example is not Lipschitz at b_c.",
        "q_half_arrival_time_exact": "2*sqrt(3)",
        "q_half_arrival_time": float(half_limit),
        "q_half_tail_over_sqrt_delta_limit": "sqrt(2)",
        "q_half_tail_over_sqrt_delta_at_1e_12": float(tail_ratio),
        "q_one_decade_time_increment_limit": "log(10)/2",
        "q_one_decade_time_increment_at_1e_12": float(log_increment),
        "q_two_delta_times_elapsed_time_limit": "1/4",
        "q_two_delta_times_elapsed_time_at_1e_12": float(inverse_coefficient),
        "proof": "At M_c>0 with positive continuous regular prefactor, elapsed time behaves as integral delta^(-q) ddelta; finite exactly for 0<q<1.",
    }
    return rows, audit


def main() -> None:
    checks = symbolic_checks()
    DATA.mkdir(parents=True, exist_ok=True)
    mstar = (3/4)**0.25
    muad = 1 / math.sqrt(3840*math.pi)
    cutoff = 0.20
    results = [
        ("Mstar_over_mP", "(3/4)^(1/4)", mstar, "Chosen B_H=1 curvature crossing"),
        ("rsstar_over_ellP", "2*(3/4)^(1/4)", 2*mstar, "Formal Schwarzschild value"),
        ("THstar_over_TP", "1/(8*pi*(3/4)^(1/4))", 1/(8*math.pi*mstar), "Formal semiclassical temperature at chosen crossing"),
        ("thermal_reduced_wavelength_over_ellP_at_star", "8*pi*(3/4)^(1/4)", 8*math.pi*mstar, "hbar*c/(k_B*T_H), not a spectral mean"),
        ("epsilon_ad_at_BH_1", "1/(1920*pi*sqrt(3))", 1/(1920*math.pi*math.sqrt(3)), "Specified toy luminosity only"),
        ("M_ad_over_mP", "1/sqrt(3840*pi)", muad, "Formal epsilon_ad=1 crossing outside chosen geometric control"),
        ("BH_at_epsilon_ad_1", "1920*pi*sqrt(3)", 1920*math.pi*math.sqrt(3), "Specified toy luminosity only"),
        ("Mc_over_M0_illustration", "1/5", cutoff, "Chosen illustration; no measured or predicted endpoint"),
        ("uc_over_ustar_illustration", "1-(1/5)^3", 1-cutoff**3, "Constant-alpha mass-loss model"),
        ("P_at_cutoff_over_P0", "(1/5)^(-2)", cutoff**-2, "Finite pre-cutoff luminosity ratio"),
        ("emitted_energy_at_cutoff_over_M0c2", "1-1/5", 1-cutoff, "Conservation result"),
        ("remaining_energy_at_cutoff_over_M0c2", "1/5", cutoff, "Conservation result"),
    ]
    result_rows = [dict(zip(("quantity", "expression", "value", "interpretation"), row)) for row in results]
    write_csv(DATA / "verified_results.csv", list(result_rows[0]), result_rows)
    endpoint_rows, endpoint_audit = endpoint_data()
    write_csv(DATA / "endpoint_counterexamples.csv", list(endpoint_rows[0]), endpoint_rows)

    scaling_rows = []
    for i in range(2001):
        mu = 10**(-3 + 7*i/2000)
        scaling_rows.append({
            "M_over_mP": mu,
            "B_H": math.sqrt(3)/(2*mu**2),
            "epsilon_ad_toy_blackbody": 1/(3840*math.pi*mu**2),
            "TH_over_TP": 1/(8*math.pi*mu),
            "P_toy_over_PPlanck": 1/(15360*math.pi*mu**2),
        })
    write_csv(DATA / "scalings_grid.csv", list(scaling_rows[0]), scaling_rows)

    history_rows = []
    for i in range(1001):
        s = i/1000
        fraction = (1-(1-cutoff**3)*s)**(1/3)
        power_ratio = fraction**-2
        emitted = 1-fraction
        if not math.isclose(fraction + emitted, 1, rel_tol=0, abs_tol=2e-15):
            raise AssertionError("Preterminal energy balance failed")
        history_rows.append({
            "s_u_over_uc": s,
            "u_over_ustar": (1-cutoff**3)*s,
            "M_over_M0": fraction,
            "P_over_P0": power_ratio,
            "E_emitted_over_M0c2": emitted,
        })
    if not math.isclose(history_rows[-1]["M_over_M0"], cutoff, abs_tol=2e-14):
        raise AssertionError("Preterminal cutoff mass failed")
    write_csv(DATA / "preterminal_history.csv", list(history_rows[0]), history_rows)

    report = {
        "title": "Dimensionless verification of Hawking scaling relations and endpoint counterexamples",
        "data_status": "Analytical model evaluations; not observations or TCGS endpoint predictions.",
        "conventions": {
            "Planck_mass": "m_P=sqrt(hbar*c/G), unreduced",
            "Planck_length": "ell_P=sqrt(hbar*G/c^3)",
            "Planck_temperature": "T_P=m_P*c^2/k_B",
            "Planck_power": "P_P=c^5/G",
            "curvature": "K=R_abcd R^abcd=48*G^2*M^2/(c^4*r^6) in Schwarzschild",
            "selector": "widehat{B}_{K,H}=ell_P^2*sqrt(K(r_s))=sqrt(3)/(2*mu^2); CSV column B_H abbreviates this effective Schwarzschild diagnostic and is not an intrinsic source field",
            "clock": "u is a specified asymptotically normalized retarded-time coordinate",
            "peeling_rate": "kappa=c^3/(4*G*M), units inverse time",
            "adiabaticity": "epsilon_ad=abs(dkappa/du)/kappa^2",
            "toy_luminosity": "P=hbar*c^6/(15360*pi*G^2*M^2)",
            "figure2_normalization": "s=(u-u0)/(u_c-u0); M_c/M0=0.20 is chosen only to illustrate a finite cutoff",
        },
        "symbolic_checks_passed": checks,
        "symbolic_check_count": len(checks),
        "results": {row["quantity"]: {"expression": row["expression"], "value": row["value"], "interpretation": row["interpretation"]} for row in result_rows},
        "endpoint_checks": endpoint_audit,
        "runtime": {"python": platform.python_version(), "sympy": sp.__version__, "mpmath": mp.__version__},
    }
    (DATA / "verified_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(checks)} exact symbolic checks, endpoint asymptotics, and preterminal energy balance.")
    print(f"Wrote dimensionless results and model grids to {DATA}")


if __name__ == "__main__":
    main()
