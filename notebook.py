# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20",
#     "numpy>=1.26",
#     "matplotlib>=3.8",
# ]
# ///
"""Rare Earth Separation Lab: a four-page marimo notebook.

Pages: 1 Concepts and governing equations, 2 Interactive lab, 3 Python code,
4 Attribution. Only marimo, numpy and matplotlib are used so the notebook runs
unchanged in molab and in a WebAssembly (Pyodide) export.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Rare Earth Separation Lab")


# ----------------------------------------------------------------------------
# Imports and page navigation
# ----------------------------------------------------------------------------
@app.cell(hide_code=True)
def _():
    import inspect
    import linecache
    import textwrap

    import marimo as mo
    import numpy as np
    from matplotlib.figure import Figure

    return Figure, inspect, linecache, mo, np, textwrap


@app.cell(hide_code=True)
def _(mo):
    # The four pages, in order. The tab bar is created once in its own cell so
    # that the selected page survives every re-run of the lab cells.
    PAGES = [
        "1 · Concepts and governing equations",
        "2 · Interactive lab",
        "3 · Python code",
        "4 · Attribution",
    ]
    nav = mo.ui.tabs({_p: "" for _p in PAGES}, value=PAGES[0])
    return PAGES, nav


@app.cell(hide_code=True)
def _(mo, nav):
    mo.vstack(
        [
            mo.md(
                "# Rare Earth Separation Lab\n"
                "Solvent extraction of La, Ce, Pr, Nd, Sm and Y with PC88A / D2EHPA: "
                "equilibrium, staging, cascades, mixer-settler dynamics and PI control. "
                "*Choose a page:*"
            ),
            nav,
        ]
    )
    return


# ============================================================================
# CODE CELLS (shown in full on page 3)
# ============================================================================
@app.cell
def _(np):
    # ---------------------------------------------------------------- chemistry
    # Cation exchange with a dimeric acidic extractant (PC88A or D2EHPA):
    #     M3+ + 3 (HA)2  <=>  M(HA2)3 + 3 H+
    # gives  log10 D = idx + 3 (pH - 2.0) + 3 log10(C / 0.5)
    # where idx is log10 D of the element at pH 2.0 and C = 0.5 mol/L,
    # expressed relative to Nd (idx_Nd = 0).

    ELEMENTS = ["La", "Ce", "Pr", "Nd", "Sm", "Y"]  # order of extraction

    IDX = {
        "La": -1.0792,  # log10(1/12):  Ce/La = 4  (US 10,227,676)
        "Ce": -0.4771,  # log10(1/3):   Pr/Ce = 2  (US 10,227,676)
        "Pr": -0.1761,  # log10(1/1.5): Nd/Pr = 1.5 (US 10,227,676)
        "Nd": 0.0,      # reference element
        "Sm": +0.8451,  # log10(7):     Sm/Nd = 7, TEACHING ESTIMATE
        "Y": +2.4472,   # log10(280):   Y/Sm = 40, TEACHING ESTIMATE
    }
    IDX_SOURCE = {
        "Ce/La": "US 10,227,676",
        "Pr/Ce": "US 10,227,676",
        "Nd/Pr": "US 10,227,676",
        "Sm/Nd": "teaching estimate",
        "Y/Sm": "teaching estimate",
    }
    PH_REF = 2.0      # reference pH of the idx values
    C_REF = 0.5       # reference extractant concentration, mol/L (as HA)
    CE4_SHIFT = 4.0   # illustrative: Ce(IV) is extracted ~10^4 times more strongly

    def log10_D(el, pH, C=C_REF, ce4=False):
        """log10 of the distribution ratio D = [M]org / [M]aq."""
        value = IDX[el] + 3.0 * (np.asarray(pH, float) - PH_REF) + 3.0 * np.log10(C / C_REF)
        if ce4 and el == "Ce":
            value = value + CE4_SHIFT  # Ce oxidised to Ce(IV): extracted far earlier
        return value

    def distribution_ratio(el, pH, C=C_REF, ce4=False):
        """Distribution ratio D (dimensionless)."""
        return 10.0 ** log10_D(el, pH, C, ce4)

    def pH50(el, C=C_REF, ce4=False):
        """pH at which D = 1 (50 % extraction at O/A = 1)."""
        shift = CE4_SHIFT if (ce4 and el == "Ce") else 0.0
        return PH_REF - (IDX[el] + shift) / 3.0 - np.log10(C / C_REF)

    def percent_extracted(D, OA=1.0):
        """One equilibrium stage: %E = 100 E / (1 + E), extraction factor E = D * O/A."""
        E = np.asarray(D, float) * OA
        return 100.0 * E / (1.0 + E)

    def separation_factor(heavy, light, ce4=False):
        """beta = D_heavy / D_light (independent of pH and C in this model)."""
        return float(distribution_ratio(heavy, PH_REF, ce4=ce4) / distribution_ratio(light, PH_REF, ce4=ce4))

    def adjacent_factors(ce4=False):
        """Separation factors of neighbours in the extraction order."""
        return [(f"{h}/{l}", separation_factor(h, l, ce4)) for l, h in zip(ELEMENTS[:-1], ELEMENTS[1:])]

    def fenske_nmin(beta, purity=0.999):
        """Minimum stages at total reflux for `purity` in both products.
        With purity 0.999 this is ln(999^2) / ln(beta)."""
        b = max(beta, 1.0 / beta)  # an inverted pair separates just as well
        if b <= 1.0 + 1e-12:
            return float("inf")
        ratio = purity / (1.0 - purity)
        return float(np.log(ratio * ratio) / np.log(b))

    return (
        CE4_SHIFT,
        C_REF,
        ELEMENTS,
        IDX,
        IDX_SOURCE,
        PH_REF,
        adjacent_factors,
        distribution_ratio,
        fenske_nmin,
        log10_D,
        pH50,
        percent_extracted,
        separation_factor,
    )


@app.cell
def _(np):
    # ------------------------------------------- Kremser and McCabe-Thiele
    def kremser_raffinate_fraction(E, N):
        """Fraction of the feed left in the raffinate after N ideal
        counter-current stages: (E - 1) / (E^(N+1) - 1); 1/(N+1) when E = 1."""
        E = np.asarray(E, float)
        N = np.asarray(N, float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            phi = np.where(
                np.abs(E - 1.0) < 1e-9,
                1.0 / (N + 1.0),
                (E - 1.0) / (E ** (N + 1.0) - 1.0),
            )
        return phi

    def kremser_recovery(E, N):
        """Extracted fraction 1 - phi."""
        return 1.0 - kremser_raffinate_fraction(E, N)

    def kremser_ceiling(E):
        """Recovery with infinitely many stages: E if E < 1, otherwise 1."""
        return np.minimum(np.asarray(E, float), 1.0)

    def isotherm(x, D, Q):
        """Loading-limited isotherm y = D x / (1 + D x / Q), Q = C/6."""
        return D * x / (1.0 + D * x / Q)

    def mccabe_thiele(xF, D, OA, N, Q, tol=1e-12):
        """Counter-current extraction with fresh solvent.

        Stage n has (x_n, y_n) on the isotherm. A balance over stages n..N gives
        the operating line x_{n-1} = x_N + (O/A) y_n. Guess the raffinate x_N,
        step stage by stage up to x_0, and bisect on x_N until x_0 = xF.
        Returns (x_N, [x_N, x_{N-1}, ..., x_0], [y_N, ..., y_1])."""

        def march(xN):
            xs, ys, x = [xN], [], xN
            for _ in range(N):
                y = isotherm(x, D, Q)
                ys.append(y)
                x = xN + OA * y
                xs.append(x)
            return x, xs, ys

        lo, hi = 0.0, xF
        for _ in range(200):  # x_0 rises monotonically with x_N
            mid = 0.5 * (lo + hi)
            if march(mid)[0] > xF:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol * max(xF, 1e-30):
                break
        xN = 0.5 * (lo + hi)
        _, xs, ys = march(xN)
        return xN, xs, ys

    return (
        isotherm,
        kremser_ceiling,
        kremser_raffinate_fraction,
        kremser_recovery,
        mccabe_thiele,
    )


@app.cell
def _(C_REF, distribution_ratio, np):
    # ------------------------------------------- steady extraction/scrub cascade
    # Stages 1..N, N = n_ext + n_scr. Fresh solvent S enters stage 1 and leaves
    # loaded from stage N (heavy product = extract). Scrub W enters stage N and
    # the aqueous phase leaves from stage 1 (light product = raffinate). The feed F
    # enters at stage f = n_ext, so the aqueous flow is W + F in stages 1..n_ext
    # and W in stages n_ext+1..N. Flows are per unit feed when F = 1.

    PAIR_DEFAULTS = {  # S, W, pH_ext, pH_scr, n_ext, n_scr
        "Pr|Nd": dict(S=2.9, W=0.45, pH_ext=1.9, pH_scr=1.8, n_ext=10, n_scr=10),
        "La|Ce": dict(S=2.9, W=0.55, pH_ext=2.2, pH_scr=2.0, n_ext=4, n_scr=4),
        "Ce|Pr": dict(S=2.9, W=0.55, pH_ext=2.0, pH_scr=1.9, n_ext=8, n_scr=8),
        "La|Nd": dict(S=2.9, W=0.75, pH_ext=2.2, pH_scr=1.9, n_ext=4, n_scr=4),
        "Nd|Sm": dict(S=2.9, W=1.4, pH_ext=1.9, pH_scr=1.7, n_ext=3, n_scr=3),
        "Sm|Y": dict(S=2.9, W=0.95, pH_ext=1.6, pH_scr=1.1, n_ext=3, n_scr=3),
    }
    CF_DEFAULT = 100.0  # total feed concentration, mmol/L (a modelling choice)

    def thomas_solve(a, b, c, d):
        """Tridiagonal solve: a = sub-, b = main-, c = super-diagonal."""
        n = len(b)
        cp, dp = np.zeros(n), np.zeros(n)
        cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
        for i in range(1, n):
            m = b[i] - a[i] * cp[i - 1]
            cp[i] = c[i] / m if i < n - 1 else 0.0
            dp[i] = (d[i] - a[i] * dp[i - 1]) / m
        x = np.zeros(n)
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    def cascade_flows(n_ext, n_scr, W, F=1.0):
        """Aqueous flow leaving each stage: W+F up to the feed stage, W after."""
        N = n_ext + n_scr
        L = np.array([W + F if n <= n_ext else W for n in range(1, N + 1)], float)
        return N, L

    def cascade_steady(pair, n_ext, n_scr, S, W, pH_ext, pH_scr, F=1.0,
                       cF=CF_DEFAULT, frac_heavy=0.5, C=C_REF):
        """Ideal-stage steady state. For each element and stage n:
           S D(n-1) x(n-1) - (S D(n) + L(n)) x(n) + L(n+1) x(n+1) = -F cF [n = f]
        which is tridiagonal, one system per element (constant D, no coupling)."""
        light, heavy = pair.split("|")
        N, L = cascade_flows(n_ext, n_scr, W, F)
        pH = np.array([pH_ext if n <= n_ext else pH_scr for n in range(1, N + 1)])
        f = n_ext - 1                     # 0-based feed stage
        Q = 1000.0 * C / 6.0              # loading capacity, mmol/L
        out = {"N": N, "L": L, "pH": pH, "Q": Q, "light": light, "heavy": heavy,
               "n_ext": n_ext, "S": S}
        for el, frac in ((light, 1.0 - frac_heavy), (heavy, frac_heavy)):
            D = distribution_ratio(el, pH, C)
            a, b, c, d = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
            for i in range(N):
                b[i] = -(S * D[i] + L[i])
                if i > 0:
                    a[i] = S * D[i - 1]   # organic arriving from stage i-1
                if i < N - 1:
                    c[i] = L[i + 1]       # aqueous arriving from stage i+1
            d[f] = -F * cF * frac
            x = thomas_solve(a, b, c, d)
            out[el] = {"x": x, "y": D * x, "D": D,
                       "raff": L[0] * x[0],             # leaves stage 1 in the aqueous
                       "ext": S * D[-1] * x[-1],        # leaves stage N in the organic
                       "feed": F * cF * frac}
        rl, rh = out[light]["raff"], out[heavy]["raff"]
        el_, eh = out[light]["ext"], out[heavy]["ext"]
        out["raff_purity"] = 100.0 * rl / (rl + rh)    # light element in raffinate
        out["ext_purity"] = 100.0 * eh / (el_ + eh)    # heavy element in extract
        out["rec_light"] = 100.0 * rl / out[light]["feed"]
        out["rec_heavy"] = 100.0 * eh / out[heavy]["feed"]
        out["balance_error"] = abs(rl + rh + el_ + eh - F * cF) / (F * cF)
        out["peak_loading"] = float(np.max(out[light]["y"] + out[heavy]["y"]))
        out["loading_ok"] = out["peak_loading"] <= Q   # validity check
        return out

    return CF_DEFAULT, PAIR_DEFAULTS, cascade_flows, cascade_steady, thomas_solve


@app.cell
def _(np):
    # ------------------------------------------------------ banded linear solver
    # Matrices are stored as ab[batch, i, KL + j - i] = A[i, j], i.e. one row per
    # equation and one column per diagonal. The dynamic model has its non-zeros
    # within 6 diagonals of the main one, so a half-bandwidth of 6 is used.
    # LU without pivoting is safe here: the matrices are diagonally dominant
    # M-matrices (outflow + transfer >= inflow terms in every column).
    KL = KU = 6

    def band_lu(ab, kl=KL, ku=KU):
        """In-place-style banded LU (Doolittle), batched over the first axis."""
        ab = ab.copy()
        n = ab.shape[1]
        for k in range(n):
            pivot = ab[:, k, kl]
            rows = np.arange(k + 1, min(n, k + kl + 1))
            if rows.size == 0:
                break
            lcol = kl + k - rows                       # column of A[r, k] in band storage
            mult = ab[:, rows, lcol] / pivot[:, None]
            ab[:, rows, lcol] = mult                   # store L below the diagonal
            width = min(ku, n - 1 - k)
            if width > 0:
                urow = ab[:, k, kl + 1: kl + 1 + width]   # U[k, k+1 .. k+width]
                cols = (kl + k - rows + 1)[:, None] + np.arange(width)[None, :]
                ab[:, rows[:, None], cols] -= mult[:, :, None] * urow[:, None, :]
        return ab

    def band_solve(lu, rhs, kl=KL, ku=KU):
        """Forward then back substitution with the factors from band_lu."""
        n = lu.shape[1]
        x = np.array(rhs, float, copy=True)
        for i in range(1, n):
            k0 = max(0, i - kl)
            x[:, i] -= np.einsum("bj,bj->b", lu[:, i, kl + k0 - i: kl], x[:, k0:i])
        for i in range(n - 1, -1, -1):
            k1 = min(n, i + ku + 1)
            if k1 > i + 1:
                x[:, i] -= np.einsum("bj,bj->b", lu[:, i, kl + 1: kl + k1 - i], x[:, i + 1:k1])
            x[:, i] /= lu[:, i, kl]
        return x

    def band_to_dense(ab, kl=KL, ku=KU):
        """Expand band storage to dense matrices (used only by the tests)."""
        B, n, _ = ab.shape
        A = np.zeros((B, n, n))
        for i in range(n):
            for j in range(max(0, i - kl), min(n, i + ku + 1)):
                A[:, i, j] = ab[:, i, kl + j - i]
        return A

    return KL, KU, band_lu, band_solve, band_to_dense


@app.cell
def _(CF_DEFAULT, C_REF, KL, KU, band_lu, band_solve, distribution_ratio, np):
    # ---------------------------------------------- mixer-settler dynamics
    # Each stage = perfectly mixed mixer (states xm, ym) followed by a lumped
    # settler (states xs, ys). Unknowns per element are ordered
    # [xm1, ym1, xs1, ys1, xm2, ...], so mixer n couples to settler n+1
    # (offset +6) and settler n-1 (offset -2): a band of half-width 6.
    # Units: flows L/min, volumes L, concentrations mmol/L, time min.

    def make_plant(pair, n_ext, n_scr, S, W, pH_ext, pH_scr, F=1.0, V_mix=8.0,
                   V_set=40.0, tau_mt=0.2, cF=CF_DEFAULT, frac_heavy=0.5, C=C_REF):
        """Collect plant data; phase volumes split by the NOMINAL flow ratio."""
        light, heavy = pair.split("|")
        N = n_ext + n_scr
        L = np.array([W + F if n <= n_ext else W for n in range(1, N + 1)], float)
        return dict(
            pair=pair, light=light, heavy=heavy, N=N, n_ext=n_ext, n_scr=n_scr,
            S=S, W=W, F=F, pH_ext=pH_ext, pH_scr=pH_scr, V_mix=V_mix, V_set=V_set,
            tau_mt=tau_mt, cF=cF, frac_heavy=frac_heavy, C=C,
            phi_o=S / (S + L),                         # organic volume fraction
            tau_mix=V_mix / (S + L),                   # tau = V / (S + L_n)
            tau_set=V_set / (S + L),
            eta=(V_mix / (S + L)) / (V_mix / (S + L) + tau_mt),  # approach to equilibrium
            holdup=N * (V_mix + V_set),
        )

    def assemble(plant, S=None, W=None, F=None, frac_heavy=None, dpH_ext=0.0):
        """Build  M du/dt = -A u + b  for both elements (batch axis 0).
        Optional arguments override the nominal flows/feed/pH (disturbances)."""
        p = plant
        S = p["S"] if S is None else S
        W = p["W"] if W is None else W
        F = p["F"] if F is None else F
        fh = p["frac_heavy"] if frac_heavy is None else frac_heavy
        N, n_ext = p["N"], p["n_ext"]
        L = np.array([W + F if n <= n_ext else W for n in range(1, N + 1)], float)
        pH = np.array([p["pH_ext"] + dpH_ext if n <= n_ext else p["pH_scr"]
                       for n in range(1, N + 1)])
        n = 4 * N
        ab = np.zeros((2, n, KL + KU + 1))
        b = np.zeros((2, n))
        m = np.zeros(n)

        def put(e, i, j, v):
            ab[e, i, KL + j - i] += v

        for e, (el, frac) in enumerate(((p["light"], 1 - fh), (p["heavy"], fh))):
            D = distribution_ratio(el, pH, p["C"])
            for s in range(N):
                xm, ym, xs, ys = 4 * s, 4 * s + 1, 4 * s + 2, 4 * s + 3
                Vmo = p["V_mix"] * p["phi_o"][s]          # organic volume in mixer
                k = Vmo / p["tau_mt"]                       # transfer: k (D xm - ym)
                # mixer aqueous: in from settler s+1 (+ feed), out L_s, transfer out
                put(e, xm, xm, L[s] + k * D[s])
                put(e, xm, ym, -k)
                if s < N - 1:
                    put(e, xm, 4 * (s + 1) + 2, -L[s + 1])
                if s == n_ext - 1:
                    b[e, xm] = F * p["cF"] * frac
                # mixer organic: in from settler s-1, out S, transfer in
                put(e, ym, ym, S + k)
                put(e, ym, xm, -k * D[s])
                if s > 0:
                    put(e, ym, 4 * (s - 1) + 3, -S)
                # settler phases: plug-free lumped holdups, no further transfer
                put(e, xs, xs, L[s])
                put(e, xs, xm, -L[s])
                put(e, ys, ys, S)
                put(e, ys, ym, -S)
                m[xm] = p["V_mix"] * (1 - p["phi_o"][s])
                m[ym] = Vmo
                m[xs] = p["V_set"] * (1 - p["phi_o"][s])
                m[ys] = p["V_set"] * p["phi_o"][s]
        return ab, b, m, L, S

    def implicit_euler_step(ab, b, m, u, dt):
        """(M/dt + A) u_new = M/dt u + b.  A huge dt returns the steady state."""
        A = ab.copy()
        A[:, :, KL] += m / dt
        return band_solve(band_lu(A), b + (m / dt) * u)

    def product_flows(u, L1, S):
        """Metal flows (mmol/min) of raffinate (settler 1 aqueous) and extract
        (settler N organic), per element."""
        return L1 * u[:, 2], S * u[:, -1]

    def steady_state(plant, **overrides):
        ab, b, m, L, S = assemble(plant, **overrides)
        return implicit_euler_step(ab, b, m, np.zeros_like(b), 1e12)

    def simulate_startup(plant, t_end=None, nsteps=1440):
        """Start from metal-free phases; feed switched on at t = 0.
        Purities are masked until each product carries 5 % of its steady flow.
        Settling time: last time either purity is outside +/-1 point of steady."""
        if t_end is None:
            t_end = float(np.clip(3.0 * plant["holdup"] / plant["F"], 48 * 60.0, 240 * 60.0))
        dt = t_end / nsteps
        ab, b, m, L, S = assemble(plant)
        A = ab.copy()
        A[:, :, KL] += m / dt
        lu = band_lu(A)                       # constant matrix: factor once
        u_ss = steady_state(plant)
        r_ss, e_ss = product_flows(u_ss, L[0], S)
        pr_ss, pe_ss = 100 * r_ss[0] / r_ss.sum(), 100 * e_ss[1] / e_ss.sum()
        t = np.arange(nsteps + 1) * dt
        pr = np.full(nsteps + 1, np.nan)
        pe = np.full(nsteps + 1, np.nan)
        u = np.zeros_like(b)
        for k in range(1, nsteps + 1):
            u = band_solve(lu, b + (m / dt) * u)
            r, e = product_flows(u, L[0], S)
            if r.sum() >= 0.05 * r_ss.sum():
                pr[k] = 100 * r[0] / r.sum()
            if e.sum() >= 0.05 * e_ss.sum():
                pe[k] = 100 * e[1] / e.sum()
        bad = ~np.isfinite(pr) | ~np.isfinite(pe) | (np.abs(pr - pr_ss) > 1) | (np.abs(pe - pe_ss) > 1)
        idx = np.nonzero(bad)[0]
        if idx.size == 0:
            t_settle = 0.0
        elif idx[-1] == nsteps:
            t_settle = float("nan")           # not settled within the horizon
        else:
            t_settle = float(t[idx[-1] + 1])
        return dict(t=t, pr=pr, pe=pe, pr_ss=pr_ss, pe_ss=pe_ss, t_settle=t_settle, u_ss=u_ss)

    return (
        assemble,
        implicit_euler_step,
        make_plant,
        product_flows,
        simulate_startup,
        steady_state,
    )


@app.cell
def _(KL, assemble, band_lu, band_solve, np, steady_state):
    # --------------------------------------------------------------- PI control
    # Controlled variable: heavy-element share (%) in the raffinate, seen through
    # an analyzer dead time. Manipulated variable: solvent flow S, limited to
    # +/-50 % of nominal. Reverse-acting PI with conditional-integration anti-windup.

    DISTURBANCES = {
        "Feed flow +20%": dict(F=1.2),
        "Feed composition 50:50 → 60:40 heavy:light": dict(frac_heavy=0.6),
        "Extraction pH −0.05": dict(dpH_ext=-0.05),
        "Scrub flow −20%": dict(W=0.8),
    }

    def heavy_share_raffinate(u, L1):
        r = L1 * u[:, 2]
        return 100.0 * r[1] / r.sum()

    def process_gain(plant, rel=0.05):
        """Kp = d(CV)/dS from a +/-5 % steady-state perturbation of S
        (percentage points per L/min)."""
        cv = []
        for f in (1 + rel, 1 - rel):
            ab, b, m, L, S = assemble(plant, S=plant["S"] * f)
            cv.append(heavy_share_raffinate(steady_state(plant, S=plant["S"] * f), L[0]))
        return (cv[0] - cv[1]) / (2 * rel * plant["S"])

    def disturbance_overrides(plant, name):
        d = DISTURBANCES[name]
        kw = {}
        if "F" in d:
            kw["F"] = plant["F"] * d["F"]
        if "W" in d:
            kw["W"] = plant["W"] * d["W"]
        if "frac_heavy" in d:
            kw["frac_heavy"] = d["frac_heavy"]
        if "dpH_ext" in d:
            kw["dpH_ext"] = d["dpH_ext"]
        return kw

    def simulate_disturbance(plant, name, closed_loop, LG=2.0, Ti=120.0, delay=10.0,
                             t_end=None, nsteps=None, limit=0.5):
        """Step disturbance at t = 0 from the nominal steady state."""
        if t_end is None:
            t_end = float(np.clip(2.25 * plant["holdup"] / plant["F"], 36 * 60.0, 200 * 60.0))
        if nsteps is None:
            nsteps = int(round(t_end / 2.0))      # dt = 2 min at the default size
        dt = t_end / nsteps
        kw = disturbance_overrides(plant, name)
        S0 = plant["S"]
        u = steady_state(plant)
        L1_nominal = assemble(plant)[3][0]
        sp = heavy_share_raffinate(u, L1_nominal)  # setpoint = nominal steady value
        Kp = process_gain(plant)
        Kc = LG / abs(Kp)                          # loop gain LG = Kc |Kp|
        nd = int(round(delay / dt))
        t = np.arange(nsteps + 1) * dt
        cv = np.zeros(nsteps + 1)
        s_hist = np.zeros(nsteps + 1)
        cv[0], s_hist[0] = sp, S0
        integral, S = 0.0, S0
        lo, hi = (1 - limit) * S0, (1 + limit) * S0
        cached_S, lu = None, None
        for k in range(1, nsteps + 1):
            if S != cached_S:                       # refactor only when S moves
                ab, b, m, L, _ = assemble(plant, S=S, **kw)
                A = ab.copy()
                A[:, :, KL] += m / dt
                lu, cached_S = band_lu(A), S
            u = band_solve(lu, b + (m / dt) * u)
            cv[k] = heavy_share_raffinate(u, L[0])
            s_hist[k] = S
            if closed_loop:
                measured = cv[max(0, k - nd)]       # analyzer dead time
                e = measured - sp                   # too much heavy -> more solvent
                trial = S0 + Kc * (e + (integral + e * dt) / Ti)
                if lo < trial < hi or (trial >= hi and e < 0) or (trial <= lo and e > 0):
                    integral += e * dt              # conditional integration
                S = float(np.clip(S0 + Kc * (e + integral / Ti), lo, hi))
        return dict(t=t, cv=cv, S=s_hist, sp=sp, Kp=Kp, Kc=Kc,
                    peak=float(np.max(np.abs(cv - sp))), final=float(cv[-1]))

    return DISTURBANCES, heavy_share_raffinate, process_gain, simulate_disturbance


@app.cell
def _(
    ELEMENTS,
    PAIR_DEFAULTS,
    band_lu,
    band_solve,
    band_to_dense,
    cascade_steady,
    distribution_ratio,
    fenske_nmin,
    implicit_euler_step,
    assemble,
    kremser_raffinate_fraction,
    make_plant,
    mccabe_thiele,
    np,
    pH50,
    separation_factor,
    simulate_disturbance,
    simulate_startup,
):
    # -------------------------------------------------------------------- tests
    def run_tests():
        """Return a list of (name, expected, got, tolerance, passed)."""
        results = []

        def check(name, expected, got, tol):
            results.append((name, expected, float(got), tol, bool(abs(got - expected) <= tol)))

        # chemistry reference values
        for el, v in zip(ELEMENTS, [2.360, 2.159, 2.059, 2.000, 1.718, 1.184]):
            check(f"pH50 {el} (C = 0.5 M)", v, pH50(el), 0.0005)
        for el, v in zip(ELEMENTS, [0.6619, 2.648, 5.295, 7.943, 55.60, 2224]):
            check(f"D {el} at pH 2.3", v, distribution_ratio(el, 2.3), 6e-4 * v)
        for pair, v in (("Ce/La", 4.0), ("Pr/Ce", 2.0), ("Nd/Pr", 1.5), ("Sm/Nd", 7.0), ("Y/Sm", 40.0)):
            h, l = pair.split("/")
            check(f"separation factor {pair}", v, separation_factor(h, l), 0.002 * v)
        check("Fenske N_min, beta = 1.5", np.log(999**2) / np.log(1.5), fenske_nmin(1.5), 1e-9)

        # Kremser limits
        check("Kremser E = 1, N = 4 -> 1/5", 0.2, float(kremser_raffinate_fraction(1.0, 4)), 1e-12)
        check("Kremser E = 0.5, N = 200 -> 1 - E", 0.5, float(kremser_raffinate_fraction(0.5, 200)), 1e-9)

        # McCabe-Thiele, Nd, 40 mmol/L, O/A 1, pH 2.3, Q = 500/6 mmol/L
        D_nd = float(distribution_ratio("Nd", 2.3))
        for N, v in zip(range(1, 5), [6.910, 1.122, 0.152, 0.019]):
            check(f"McCabe-Thiele Nd raffinate N = {N} (mmol/L)", v,
                  mccabe_thiele(40.0, D_nd, 1.0, N, 500.0 / 6.0)[0], 0.0005)

        # banded solver against numpy dense solve
        plant = make_plant("Pr|Nd", **PAIR_DEFAULTS["Pr|Nd"])
        ab, b, m, L, S = assemble(plant)
        A = ab.copy()
        A[:, :, 6] += m / 1.0
        x_band = band_solve(band_lu(A), b)
        x_dense = np.linalg.solve(band_to_dense(A), b[..., None])[..., 0]
        check("banded LU vs dense solve (max abs error)", 0.0, np.max(np.abs(x_band - x_dense)), 1e-9)

        # equilibrium cascade
        cas = cascade_steady("Pr|Nd", **PAIR_DEFAULTS["Pr|Nd"])
        check("Pr|Nd cascade raffinate purity (%)", 79.6, cas["raff_purity"], 0.1)
        check("Pr|Nd cascade extract purity (%)", 79.6, cas["ext_purity"], 0.1)
        check("cascade mass balance closure", 0.0, cas["balance_error"], 1e-10)

        # dynamic model with very fast transfer reduces to the ideal cascade
        fast = make_plant("Pr|Nd", **PAIR_DEFAULTS["Pr|Nd"], tau_mt=1e-6)
        ab, b, m, L, S = assemble(fast)
        u = implicit_euler_step(ab, b, m, np.zeros_like(b), 1e12)
        r = L[0] * u[:, 2]
        check("dynamic steady state -> ideal cascade (tau_mt -> 0)", cas["raff_purity"],
              100 * r[0] / r.sum(), 1e-3)

        # mixer-settler reference case: F = 1 L/min, 8 L, 40 L, tau_mt = 12 s
        check("tau_mix extraction zone (min)", 1.84, plant["tau_mix"][0], 0.005)
        check("tau_set extraction zone (min)", 9.2, plant["tau_set"][0], 0.005)
        check("tau_mix scrub zone (min)", 2.39, plant["tau_mix"][-1], 0.005)
        check("tau_set scrub zone (min)", 11.94, plant["tau_set"][-1], 0.005)
        check("hold-up (L)", 960.0, plant["holdup"], 1e-9)
        st = simulate_startup(plant)
        check("steady extract purity, Nd (%)", 78.8, st["pe_ss"], 0.1)
        check("steady raffinate purity, Pr (%)", 78.1, st["pr_ss"], 0.1)
        check("start-up settling time (h)", 23.0, st["t_settle"] / 60.0, 0.5)

        # disturbance and control
        ol = simulate_disturbance(plant, "Feed flow +20%", False)
        cl = simulate_disturbance(plant, "Feed flow +20%", True, LG=2.0, Ti=120.0, delay=10.0)
        check("raffinate heavy share before feed +20% (%)", 21.9, ol["sp"], 0.1)
        check("raffinate heavy share after feed +20%, open loop (%)", 28.9, ol["final"], 0.1)
        check("open-loop peak deviation (points)", 6.9, ol["peak"], 0.1)
        check("PI peak deviation, LG 2, Ti 120, delay 10 (points)", 1.8, cl["peak"], 0.1)
        return results

    return (run_tests,)


@app.cell
def _(Figure):
    # ---------------------------------------------------------- plotting helpers
    # Fixed categorical colour per element (never re-ordered); every line also
    # carries a direct label, so identity never depends on colour alone.
    COLORS = {"La": "#2a78d6", "Ce": "#eb6834", "Pr": "#1baf7a",
              "Nd": "#eda100", "Sm": "#e87ba4", "Y": "#008300"}
    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    ACCENT1, ACCENT2 = "#2a78d6", "#eb6834"

    def new_figure(ncols=1, width=7.5, height=3.6):
        fig = Figure(figsize=(width, height), dpi=110, facecolor=SURFACE, layout="constrained")
        axes = fig.subplots(1, ncols, squeeze=False)[0]
        for ax in axes:
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK2)
            ax.tick_params(colors=INK2, labelsize=8)
        return fig, axes

    def label_axes(ax, title, xlabel, ylabel):
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)

    return ACCENT1, ACCENT2, COLORS, INK, INK2, label_axes, new_figure


# ============================================================================
# PAGE 1: Concepts and governing equations
# ============================================================================
@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[0])
    mo.md(
        r"""
## 1 · Concepts and governing equations

### Why rare earths are hard to separate
La, Ce, Pr, Nd, Sm and Y are all trivalent in acid solution and almost the same size, so no single
chemical step separates them. Industry uses **solvent extraction**: an aqueous chloride solution is
mixed with kerosene holding an acidic organophosphorus extractant (PC88A, also sold as P507 or HEH/EHP; or
D2EHPA). Smaller, heavier lanthanides bind slightly more strongly, so each contact enriches the
organic phase in the heavier element. Many contacts in counter-current cascades of
mixer-settlers turn small differences into high purity.

### Extraction chemistry
The extractant exists as a hydrogen-bonded dimer $(\mathrm{HA})_2$. A metal ion swaps its charge for three protons:

$$\mathrm{M^{3+}_{(aq)}} + 3\,(\mathrm{HA})_{2\,(org)} \rightleftharpoons \mathrm{M(HA_2)_{3\,(org)}} + 3\,\mathrm{H^+_{(aq)}}$$

The distribution ratio $D = [\mathrm{M}]_{org}/[\mathrm{M}]_{aq} = K_{ex}[(\mathrm{HA})_2]^3/[\mathrm{H^+}]^3$, so
$\log_{10}D$ rises by 3 per pH unit and by 3 per decade of extractant. The lab uses

$$\log_{10} D = \mathrm{idx} + 3\,(\mathrm{pH} - 2.0) + 3\,\log_{10}\!\left(\frac{C}{0.5}\right)$$

where $C$ is the extractant concentration (mol/L as HA) and idx is $\log_{10}D$ at pH 2.0 and
$C = 0.5$ M, relative to Nd:

| Element | La | Ce | Pr | Nd | Sm | Y |
|---|---|---|---|---|---|---|
| idx | −1.0792 | −0.4771 | −0.1761 | 0 | +0.8451 | +2.4472 |

Adjacent **separation factors** $\beta = D_{heavy}/D_{light} = 10^{\Delta \mathrm{idx}}$ are therefore
Ce/La = 4, Pr/Ce = 2 and Nd/Pr = 1.5 (from US patent 10,227,676), and **Sm/Nd = 7 and Y/Sm = 40, which are teaching
estimates**. The literature search on page 4 found PC‑88A Nd/Pr ≈ 1.4 (close to 1.5), and
β values that vary strongly with acid, pH and loading (Ce/La from about 4 to more than 50). None of the sources
gave a usable PC88A Sm/Nd or Y/Sm under comparable conditions, so **no idx value was changed**.

**pH of half extraction** ($D = 1$):
$$\mathrm{pH}_{50} = 2 - \frac{\mathrm{idx}}{3} - \log_{10}\!\left(\frac{C}{0.5}\right)$$

**One equilibrium stage** with organic-to-aqueous ratio O/A:
$$E = D\cdot\frac{O}{A},\qquad \%\text{extracted} = \frac{100\,E}{1+E}$$

**Ce(IV) switch (illustrative).** Oxidising cerium to Ce(IV) makes it extract far more strongly. The lab
models this by adding $+4$ to $\log_{10}D_{Ce}$. This inverts the Ce/Pr order and makes Ce removal easy.

### Staging
**Fenske (total reflux).** The minimum number of stages for 99.9 % purity in both products is
$$N_{min} = \frac{\ln(999^2)}{\ln \beta}$$
which gives about 10 stages for $\beta = 4$ and about 34 for $\beta = 1.5$.

**Kremser (counter-current extraction, fresh solvent, constant D).** The fraction of a metal left in the raffinate after $N$ stages is
$$\varphi = \frac{E-1}{E^{N+1}-1}\qquad(\varphi = \tfrac{1}{N+1}\ \text{at}\ E=1)$$
When $E<1$, even infinitely many stages recover only a fraction $E$, because $\varphi \to 1-E$. This is the *E < 1 ceiling*.

**McCabe–Thiele with loading.** The extractant saturates at $Q = C/6$ (six HA per metal), so the isotherm is
$$y = \frac{D x}{1 + D x / Q}$$
The operating line from a balance over stages $n..N$ is $x_{n-1} = x_N + (O/A)\,y_n$. The raffinate $x_N$ is
found by bisection: guess $x_N$, step up the stages to $x_0$, and repeat until $x_0 = x_F$.

### Extraction / scrub cascade
Stages $1..N$ with $N = n_{Ext} + n_{Scr}$. Fresh solvent $S$ enters stage 1 and leaves loaded at stage $N$ (heavy product).
Scrub liquor $W$ enters stage $N$, and the aqueous phase leaves at stage 1 (light product, the raffinate). The feed $F$
(concentration $c_F$) enters at stage $f = n_{Ext}$. The aqueous flow is $L(n) = W + F$ for $n \le f$ and $W$ after. $D$
is evaluated at the extraction pH in stages $1..f$ and at the scrub pH after. Each element obeys

$$S\,D(n-1)\,x(n-1) - \big(S\,D(n)+L(n)\big)\,x(n) + L(n+1)\,x(n+1) = -F\,c_F\,[n=f]$$

This is a tridiagonal system per element, solved with the Thomas algorithm. **Validity check:** if the peak total
organic loading $\sum_i D_i x_i$ exceeds $Q$, the constant‑D assumption is broken and the result is flagged.

### Mixer–settlers
Residence times are $\tau = V/(S + L_n)$ for mixer and settler. Transfer in the mixer is first order with time
constant $\tau_{mt}$, so a mixer approaches equilibrium to
$$\eta = \frac{\tau_{mix}}{\tau_{mix} + \tau_{mt}}$$

**Dynamic model.** Each stage is a perfectly mixed mixer (states $x_m, y_m$) followed by a lumped settler
(states $x_s, y_s$). Phase volumes are split by the nominal flow ratio $\phi_o = S/(S+L_n)$. For each element:

$$V_a^{m}\dot x_m = L_{n+1}x_s^{(n+1)} + F c_F[n=f] - L_n x_m - V_o^{m}\,\frac{D x_m - y_m}{\tau_{mt}}$$
$$V_o^{m}\dot y_m = S\,y_s^{(n-1)} - S\,y_m + V_o^{m}\,\frac{D x_m - y_m}{\tau_{mt}}$$
$$V_a^{s}\dot x_s = L_n (x_m - x_s),\qquad V_o^{s}\dot y_s = S(y_m - y_s)$$

With the unknowns ordered $[x_m, y_m, x_s, y_s]$ stage by stage, the matrix is banded with half-bandwidth 6.
It is integrated by **implicit Euler**, $(M/\Delta t + A)u^{k+1} = (M/\Delta t)u^k + b$, using a banded LU solver.
The steady state is one step with a huge $\Delta t$. Start-up begins with metal-free phases. Product purities are
masked until each product carries 5 % of its steady metal flow. The settling time is the last time either purity
is outside ±1 point of its steady value.

### PI control
* **Controlled variable:** heavy-element share (%) in the raffinate, measured by an on-line analyzer with dead time $\theta$.
* **Manipulated variable:** solvent flow $S$, limited to ±50 % of nominal.
* **Controller:** $S = S_0 + K_c\big(e + \tfrac{1}{T_i}\int e\,dt\big)$ with $e = y_{meas} - y_{sp}$ (more heavy element in the
  raffinate calls for more solvent). The integral is frozen while the output is saturated and the error would drive it further (**conditional-integration anti-windup**).
* **Tuning:** $K_p$ is found from a ±5 % steady-state perturbation of $S$, and $K_c = LG/|K_p|$, where $LG$ is the chosen loop gain.
* **Disturbances (steps at t = 0):** feed flow +20 %; feed composition 50:50 → 60:40 heavy:light; extraction pH −0.05; scrub flow −20 %.

### Assumptions
* pH is fixed in each stage (perfect acid/base control), and $D$ is constant (no loading or activity effects beyond the check).
* Stages are ideal in the steady cascade. In the dynamic model, stage efficiency comes only from $\tau_{mt}$.
* There is no reflux of product back to the cascade, no entrainment and no third phase.
* $\tau_{mt}$ = 12 s, the volumes (8 L mixer, 40 L settler), feed concentration $c_F$ = 100 mmol/L and the controller limits
  are **teaching estimates**, not plant data.
* The Sm/Nd and Y/Sm separation factors and the Ce(IV) shift are illustrative.
"""
    )
    return


# ============================================================================
# PAGE 2: Interactive lab — controls (created once, displayed on page 2)
# ============================================================================
@app.cell(hide_code=True)
def _(ELEMENTS, mo):
    # Section 1 and 2 controls
    sl_C = mo.ui.slider(0.1, 1.0, 0.05, value=0.5, label="Extractant C (mol/L)", show_value=True)
    sl_OA1 = mo.ui.slider(0.2, 5.0, 0.1, value=1.0, label="O/A", show_value=True)
    ms_els1 = mo.ui.multiselect(ELEMENTS, value=ELEMENTS, label="Elements")
    sw_ce4 = mo.ui.switch(value=False, label="Ce(IV) switch (+4 on log D for Ce, illustrative)")
    sl_purity = mo.ui.slider(0.99, 0.9999, 0.0001, value=0.999, label="Target purity (both products)", show_value=True)
    # Section 3 controls
    sl_pH3 = mo.ui.slider(1.0, 3.0, 0.05, value=2.0, label="pH", show_value=True)
    sl_OA3 = mo.ui.slider(0.2, 5.0, 0.1, value=1.0, label="O/A", show_value=True)
    ms_els3 = mo.ui.multiselect(ELEMENTS, value=["La", "Ce", "Pr", "Nd"], label="Elements")
    # Section 4 controls
    dd_el4 = mo.ui.dropdown(ELEMENTS, value="Nd", label="Element")
    sl_xF4 = mo.ui.slider(5, 80, 1, value=40, label="Feed (mmol/L)", show_value=True)
    sl_OA4 = mo.ui.slider(0.25, 4.0, 0.05, value=1.0, label="O/A", show_value=True)
    sl_pH4 = mo.ui.slider(1.5, 3.0, 0.05, value=2.3, label="pH", show_value=True)
    sl_N4 = mo.ui.slider(1, 8, 1, value=3, label="Stages N", show_value=True)
    sl_C4 = mo.ui.slider(0.1, 1.0, 0.05, value=0.5, label="Extractant C (mol/L)", show_value=True)
    return (
        dd_el4,
        ms_els1,
        ms_els3,
        sl_C,
        sl_C4,
        sl_N4,
        sl_OA1,
        sl_OA3,
        sl_OA4,
        sl_pH3,
        sl_pH4,
        sl_purity,
        sl_xF4,
        sw_ce4,
    )


@app.cell(hide_code=True)
def _(PAIR_DEFAULTS, mo):
    dd_pair = mo.ui.dropdown(list(PAIR_DEFAULTS), value="Pr|Nd", label="Pair (light|heavy)")
    return (dd_pair,)


@app.cell(hide_code=True)
def _(CF_DEFAULT, PAIR_DEFAULTS, dd_pair, mo):
    # Re-created whenever the pair changes, so every slider jumps to that pair's defaults.
    _d = PAIR_DEFAULTS[dd_pair.value]
    sl_next = mo.ui.slider(1, 30, 1, value=_d["n_ext"], label="nExt", show_value=True, debounce=True)
    sl_nscr = mo.ui.slider(1, 30, 1, value=_d["n_scr"], label="nScr", show_value=True, debounce=True)
    sl_S = mo.ui.slider(0.5, 6.0, 0.05, value=_d["S"], label="S (per unit F)", show_value=True, debounce=True)
    sl_W = mo.ui.slider(0.05, 3.0, 0.05, value=_d["W"], label="W (per unit F)", show_value=True, debounce=True)
    sl_pHe = mo.ui.slider(0.5, 3.5, 0.05, value=_d["pH_ext"], label="pH extraction", show_value=True, debounce=True)
    sl_pHs = mo.ui.slider(0.5, 3.5, 0.05, value=_d["pH_scr"], label="pH scrub", show_value=True, debounce=True)
    sl_cF = mo.ui.slider(10, 300, 5, value=CF_DEFAULT, label="Feed total (mmol/L)", show_value=True, debounce=True)
    sl_fh = mo.ui.slider(0.1, 0.9, 0.05, value=0.5, label="Feed heavy fraction", show_value=True, debounce=True)
    return sl_S, sl_W, sl_cF, sl_fh, sl_next, sl_nscr, sl_pHe, sl_pHs


@app.cell(hide_code=True)
def _(DISTURBANCES, mo):
    sl_F = mo.ui.slider(0.2, 5.0, 0.1, value=1.0, label="Feed flow F (L/min)", show_value=True, debounce=True)
    sl_Vm = mo.ui.slider(1, 50, 1, value=8, label="Mixer volume (L)", show_value=True, debounce=True)
    sl_Vs = mo.ui.slider(5, 200, 5, value=40, label="Settler volume (L)", show_value=True, debounce=True)
    sl_tmt = mo.ui.slider(1, 120, 1, value=12, label="τ_mt (s)", show_value=True, debounce=True)
    dd_dist = mo.ui.dropdown(list(DISTURBANCES), value="Feed flow +20%", label="Disturbance")
    sl_LG = mo.ui.slider(0.25, 5.0, 0.25, value=2.0, label="Loop gain LG", show_value=True, debounce=True)
    sl_Ti = mo.ui.slider(10, 600, 10, value=120, label="Integral time Ti (min)", show_value=True, debounce=True)
    sl_delay = mo.ui.slider(0, 60, 2, value=10, label="Analyzer delay (min)", show_value=True, debounce=True)
    return dd_dist, sl_F, sl_LG, sl_Ti, sl_Vm, sl_Vs, sl_delay, sl_tmt


@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[1])
    mo.md(
        "## 2 · Interactive lab\n"
        "Six sections, from one equilibrium contact up to a controlled mixer-settler plant. "
        "Every chart updates live. Equations are on page 1, and the code is on page 3."
    )
    return


# --------------------------------------------------- Section 1: % extracted vs pH
@app.cell(hide_code=True)
def _(
    COLORS,
    ELEMENTS,
    INK,
    PAGES,
    distribution_ratio,
    label_axes,
    mo,
    ms_els1,
    nav,
    new_figure,
    np,
    pH50,
    percent_extracted,
    sl_C,
    sl_OA1,
    sw_ce4,
):
    mo.stop(nav.value != PAGES[1])
    _pH = np.linspace(0.0, 3.5, 351)
    _fig, (_ax,) = new_figure(1, 7.5, 3.8)
    _rows = []
    for _k, _el in enumerate([e for e in ELEMENTS if e in ms_els1.value]):
        _D = distribution_ratio(_el, _pH, sl_C.value, sw_ce4.value)
        _ax.plot(_pH, percent_extracted(_D, sl_OA1.value), color=COLORS[_el], lw=2)
        # 50 % point at the chosen O/A: D * O/A = 1
        _p50 = pH50(_el, sl_C.value, sw_ce4.value) - np.log10(sl_OA1.value) / 3.0
        if 0.0 <= _p50 <= 3.5:
            _ax.plot([_p50], [50], "o", ms=7, color=COLORS[_el], mec="white", mew=1.5)
            # stagger labels above/below so close neighbours (Pr, Nd) do not collide
            _ax.annotate(_el, (_p50, 50), xytext=(5, 8 if _k % 2 == 0 else -16), textcoords="offset points",
                         fontsize=9, color=INK)
        _rows.append({"Element": _el, "pH50 (O/A = 1)": round(float(pH50(_el, sl_C.value, sw_ce4.value)), 3),
                      "pH at 50 % (this O/A)": round(float(_p50), 3),
                      "D at pH 2.3": float(f"{float(distribution_ratio(_el, 2.3, sl_C.value, sw_ce4.value)):.4g}")})
    _ax.set_xlim(0, 3.5)
    _ax.set_ylim(0, 100)
    label_axes(_ax, "Single-stage extraction (dots mark 50 %)", "Equilibrium pH", "% extracted")
    mo.vstack([
        mo.md("### 1 · % extracted vs pH"),
        mo.hstack([sl_C, sl_OA1, sw_ce4], justify="start", gap=1.5, wrap=True),
        ms_els1,
        _fig,
        mo.ui.table(_rows, selection=None, label="Values (each line is direct-labelled at its 50 % point)"),
    ])
    return


# --------------------------------------------------- Section 2: separation factors
@app.cell(hide_code=True)
def _(
    ACCENT1,
    IDX_SOURCE,
    INK,
    PAGES,
    adjacent_factors,
    fenske_nmin,
    label_axes,
    mo,
    nav,
    new_figure,
    sl_purity,
    sw_ce4,
):
    mo.stop(nav.value != PAGES[1])
    _pairs = adjacent_factors(sw_ce4.value)
    _fig, (_ax,) = new_figure(1, 7.5, 3.6)
    _rows = []
    for _i, (_name, _beta) in enumerate(_pairs):
        _b = max(_beta, 1 / _beta)
        _lab = _name if _beta >= 1 else "/".join(reversed(_name.split("/"))) + " (inverted)"
        _est = IDX_SOURCE[_name] == "teaching estimate"
        _ax.bar(_i, _b, width=0.6, color=ACCENT1, alpha=0.55 if _est else 1.0,
                hatch="//" if _est else None, edgecolor="white", linewidth=2)
        _n = fenske_nmin(_beta, sl_purity.value)
        _ax.annotate(f"β={_b:.3g}\nN_min={_n:.1f}", (_i, _b), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=8, color=INK)
        _rows.append({"Pair": _lab, "β": float(f"{_b:.4g}"), "N_min": round(_n, 2), "Source": IDX_SOURCE[_name]
                      + (" + Ce(IV) switch" if sw_ce4.value and "Ce" in _name else "")})
    _ax.set_yscale("log")
    _ax.set_xticks(range(len(_pairs)), [r["Pair"] for r in _rows], fontsize=8)
    _ax.set_ylim(1, max(r["β"] for r in _rows) * 8)
    label_axes(_ax, "Adjacent separation factors (hatched = teaching estimate)", "", "β (log scale)")
    mo.vstack([
        mo.md(f"### 2 · Separation factors and Fenske minimum stages\n"
              f"$N_{{min}} = \\ln\\big((P/(1-P))^2\\big)/\\ln\\beta$ with P = {sl_purity.value:.4f} "
              f"(P = 0.999 gives ln(999²)/ln β)."),
        mo.hstack([sl_purity, sw_ce4], justify="start", gap=1.5, wrap=True),
        _fig,
        mo.ui.table(_rows, selection=None),
    ])
    return


# --------------------------------------------------- Section 3: Kremser
@app.cell(hide_code=True)
def _(
    COLORS,
    ELEMENTS,
    INK,
    PAGES,
    distribution_ratio,
    kremser_ceiling,
    kremser_recovery,
    label_axes,
    mo,
    ms_els3,
    nav,
    new_figure,
    np,
    sl_OA3,
    sl_pH3,
):
    mo.stop(nav.value != PAGES[1])
    _N = np.arange(0, 21)
    _fig, (_ax,) = new_figure(1, 7.5, 3.8)
    _rows = []
    for _el in [e for e in ELEMENTS if e in ms_els3.value]:
        _E = float(distribution_ratio(_el, sl_pH3.value)) * sl_OA3.value
        _rec = 100 * kremser_recovery(_E, _N)
        _ax.plot(_N, _rec, "-o", ms=4, lw=2, color=COLORS[_el])
        if _E < 1:
            _ax.axhline(100 * _E, color=COLORS[_el], lw=1, ls="--")
        _ax.annotate(f"{_el} (E={_E:.3g})", (_N[-1], _rec[-1]), xytext=(4, 0), textcoords="offset points",
                     fontsize=8, color=INK, va="center")
        _rows.append({"Element": _el, "E": float(f"{_E:.4g}"), "N=1 (%)": round(float(_rec[1]), 2),
                      "N=5 (%)": round(float(_rec[5]), 2), "N=20 (%)": round(float(_rec[20]), 2),
                      "Ceiling N→∞ (%)": round(100 * float(kremser_ceiling(_E)), 2)})
    _ax.set_xlim(0, 24)
    _ax.set_ylim(0, 102)
    label_axes(_ax, "Counter-current recovery (dashed = E < 1 ceiling)", "Number of stages N", "% extracted")
    mo.vstack([
        mo.md("### 3 · Kremser recovery vs stages\nFraction left in the raffinate: (E−1)/(E^(N+1)−1), with E = D·O/A."),
        mo.hstack([sl_pH3, sl_OA3], justify="start", gap=1.5, wrap=True),
        ms_els3,
        _fig,
        mo.ui.table(_rows, selection=None),
    ])
    return


# --------------------------------------------------- Section 4: McCabe-Thiele
@app.cell(hide_code=True)
def _(
    ACCENT1,
    COLORS,
    INK,
    INK2,
    PAGES,
    dd_el4,
    distribution_ratio,
    isotherm,
    label_axes,
    mccabe_thiele,
    mo,
    nav,
    new_figure,
    np,
    sl_C4,
    sl_N4,
    sl_OA4,
    sl_pH4,
    sl_xF4,
):
    mo.stop(nav.value != PAGES[1])
    _el, _xF, _OA, _N = dd_el4.value, float(sl_xF4.value), sl_OA4.value, sl_N4.value
    _D = float(distribution_ratio(_el, sl_pH4.value, sl_C4.value))
    _Q = 1000 * sl_C4.value / 6
    _xN, _xs, _ys = mccabe_thiele(_xF, _D, _OA, _N, _Q)
    _fig, (_ax,) = new_figure(1, 7.5, 4.2)
    _xx = np.linspace(0, _xF * 1.05, 300)
    _ax.plot(_xx, isotherm(_xx, _D, _Q), color=COLORS[_el], lw=2)
    _xl = min(_xF * 0.5, _xs[-2] if len(_xs) > 1 else _xF) * 0.35
    _ax.annotate(f"isotherm, {_el}", (_xl, isotherm(_xl, _D, _Q)), xytext=(6, 0),
                 textcoords="offset points", ha="left", fontsize=8, color=INK)
    _ax.plot([_xN, _xF], [0, (_xF - _xN) / _OA], color=INK2, lw=2)
    _ax.annotate("operating line", (_xF, (_xF - _xN) / _OA), xytext=(-4, -12), textcoords="offset points",
                 ha="right", fontsize=8, color=INK)
    # stair steps: from (x0, y1) across to the isotherm, then down to the operating line
    _sx, _sy = [_xF], [_ys[-1]]
    _x_seq = _xs[::-1]  # x0, x1, ..., xN
    _y_seq = _ys[::-1]  # y1, ..., yN
    for _n in range(_N):
        _sx.append(_x_seq[_n + 1]); _sy.append(_y_seq[_n])
        _next_y = _y_seq[_n + 1] if _n + 1 < _N else 0.0
        _sx.append(_x_seq[_n + 1]); _sy.append(_next_y)
    _ax.plot(_sx, _sy, color=ACCENT1, lw=1.5)
    for _n in range(_N):
        _ax.annotate(str(_n + 1), (_x_seq[_n + 1], _y_seq[_n]), xytext=(-8, 3), textcoords="offset points",
                     fontsize=8, color=INK)
    _ax.axhline(_Q, color=INK2, lw=1, ls=":")
    _ax.annotate(f"Q = C/6 = {_Q:.1f}", (0, _Q), xytext=(4, -10), textcoords="offset points", fontsize=8, color=INK2)
    _ax.set_xlim(0, _xF * 1.05)
    _ax.set_ylim(0, min(_Q * 1.08, max(_ys) * 1.25 + 1))
    label_axes(_ax, f"{_el}: D = {_D:.4g}, O/A = {_OA}, {_N} stages", "Aqueous x (mmol/L)", "Organic y (mmol/L)")
    _table = [{"N": _k, "Raffinate x_N (mmol/L)": float(f"{mccabe_thiele(_xF, _D, _OA, _k, _Q)[0]:.4g}")}
              for _k in range(1, 9)]
    mo.vstack([
        mo.md("### 4 · McCabe–Thiele with stair steps\n"
              f"Raffinate **{_xN:.4g} mmol/L**, recovery **{100 * (1 - _xN / _xF):.2f} %** "
              "(the raffinate is found by bisection)."),
        mo.hstack([dd_el4, sl_xF4, sl_OA4], justify="start", gap=1.5, wrap=True),
        mo.hstack([sl_pH4, sl_N4, sl_C4], justify="start", gap=1.5, wrap=True),
        mo.hstack([_fig, mo.ui.table(_table, selection=None)], widths=[3, 1]),
    ])
    return


# --------------------------------------------------- Section 5: cascade
@app.cell(hide_code=True)
def _(
    PAGES,
    cascade_steady,
    dd_pair,
    mo,
    nav,
    sl_S,
    sl_W,
    sl_cF,
    sl_fh,
    sl_next,
    sl_nscr,
    sl_pHe,
    sl_pHs,
):
    mo.stop(nav.value != PAGES[1])
    cas = cascade_steady(dd_pair.value, sl_next.value, sl_nscr.value, sl_S.value, sl_W.value,
                         sl_pHe.value, sl_pHs.value, 1.0, sl_cF.value, sl_fh.value)
    return (cas,)


@app.cell(hide_code=True)
def _(
    COLORS,
    INK,
    INK2,
    cas,
    dd_pair,
    fenske_nmin,
    label_axes,
    mo,
    new_figure,
    np,
    separation_factor,
    sl_S,
    sl_W,
    sl_cF,
    sl_fh,
    sl_next,
    sl_nscr,
    sl_pHe,
    sl_pHs,
):
    _l, _h = cas["light"], cas["heavy"]
    _st = np.arange(1, cas["N"] + 1)
    _fig, _axes = new_figure(2, 9.5, 3.6)
    for _ax, _key, _title, _yl in ((_axes[0], "x", "Aqueous profile", "x (mmol/L)"),
                                    (_axes[1], "y", "Organic profile", "y (mmol/L)")):
        _ax.axvspan(cas["n_ext"] + 0.5, cas["N"] + 0.5, color="#ecebe7", zorder=0)
        for _el in (_l, _h):
            _v = cas[_el][_key]
            _ax.plot(_st, _v, "-o", ms=4, lw=2, color=COLORS[_el])
            _ax.annotate(_el, (_st[-1], _v[-1]), xytext=(5, 0), textcoords="offset points", fontsize=9,
                         color=INK, va="center")
        _ax.axvline(cas["n_ext"], color=INK2, lw=1, ls="--")
        _ax.annotate("feed", (cas["n_ext"], 0), xytext=(3, 3), textcoords="offset points", fontsize=8, color=INK2)
        _ax.set_xlim(0.5, cas["N"] + 1.5)
        label_axes(_ax, _title + " (shaded = scrub)", "Stage (1 = raffinate end)", _yl)
    _axes[1].axhline(cas["Q"], color=INK2, lw=1, ls=":")
    _beta = separation_factor(_h, _l)
    _warn = (mo.callout(mo.md(f"**Validity check failed:** peak organic loading {cas['peak_loading']:.1f} mmol/L "
                              f"exceeds Q = C/6 = {cas['Q']:.1f} mmol/L. Constant D is not valid here."), kind="danger")
             if not cas["loading_ok"] else
             mo.callout(mo.md(f"Loading check OK: peak {cas['peak_loading']:.1f} ≤ Q = {cas['Q']:.1f} mmol/L."), kind="success"))
    mo.vstack([
        mo.md("### 5 · Extraction / scrub cascade (ideal stages)"),
        mo.hstack([dd_pair, sl_next, sl_nscr], justify="start", gap=1.5, wrap=True),
        mo.hstack([sl_S, sl_W, sl_cF], justify="start", gap=1.5, wrap=True),
        mo.hstack([sl_pHe, sl_pHs, sl_fh], justify="start", gap=1.5, wrap=True),
        mo.hstack([
            mo.stat(f"{cas['raff_purity']:.2f} %", label=f"Raffinate purity ({_l})", bordered=True),
            mo.stat(f"{cas['ext_purity']:.2f} %", label=f"Extract purity ({_h})", bordered=True),
            mo.stat(f"{cas['rec_light']:.2f} %", label=f"{_l} recovery", bordered=True),
            mo.stat(f"{cas['rec_heavy']:.2f} %", label=f"{_h} recovery", bordered=True),
            mo.stat(f"{fenske_nmin(_beta):.1f}", label=f"Fenske N_min (β={_beta:.3g})", bordered=True),
        ], widths="equal"),
        _warn,
        _fig,
    ])
    return


# --------------------------------------------------- Section 6: dynamics & control
@app.cell(hide_code=True)
def _(
    PAGES,
    dd_pair,
    make_plant,
    mo,
    nav,
    simulate_startup,
    sl_F,
    sl_S,
    sl_Vm,
    sl_Vs,
    sl_W,
    sl_cF,
    sl_fh,
    sl_next,
    sl_nscr,
    sl_pHe,
    sl_pHs,
    sl_tmt,
):
    mo.stop(nav.value != PAGES[1])
    # S and W from section 5 are per unit feed; scale them by F.
    plant = make_plant(dd_pair.value, sl_next.value, sl_nscr.value, sl_S.value * sl_F.value,
                       sl_W.value * sl_F.value, sl_pHe.value, sl_pHs.value, sl_F.value,
                       float(sl_Vm.value), float(sl_Vs.value), sl_tmt.value / 60.0,
                       float(sl_cF.value), sl_fh.value)
    startup = simulate_startup(plant)
    return plant, startup


@app.cell(hide_code=True)
def _(dd_dist, plant, simulate_disturbance):
    open_loop = simulate_disturbance(plant, dd_dist.value, closed_loop=False)
    return (open_loop,)


@app.cell(hide_code=True)
def _(dd_dist, plant, simulate_disturbance, sl_LG, sl_Ti, sl_delay):
    closed_loop = simulate_disturbance(plant, dd_dist.value, closed_loop=True, LG=sl_LG.value,
                                       Ti=float(sl_Ti.value), delay=float(sl_delay.value))
    return (closed_loop,)


@app.cell(hide_code=True)
def _(
    ACCENT1,
    ACCENT2,
    COLORS,
    INK,
    INK2,
    closed_loop,
    dd_dist,
    label_axes,
    mo,
    new_figure,
    np,
    open_loop,
    plant,
    sl_F,
    sl_LG,
    sl_Ti,
    sl_Vm,
    sl_Vs,
    sl_delay,
    sl_tmt,
    startup,
):
    _N = plant["N"]
    _st = np.arange(1, _N + 1)
    # residence times per stage (grouped bars, 2 px surface gap)
    _f1, (_a1,) = new_figure(1, 9.5, 3.0)
    _a1.bar(_st - 0.2, plant["tau_mix"], width=0.38, color=ACCENT1, edgecolor="white", lw=1, label="τ_mix")
    _a1.bar(_st + 0.2, plant["tau_set"], width=0.38, color=ACCENT2, edgecolor="white", lw=1, label="τ_set")
    _a1.axvline(plant["n_ext"] + 0.5, color=INK2, lw=1, ls="--")
    _a1.legend(frameon=False, fontsize=8, loc="upper left")
    label_axes(_a1, "Residence time per stage (dashed line = feed/scrub boundary)", "Stage", "min")
    # start-up
    _f2, (_a2,) = new_figure(1, 9.5, 3.2)
    _th = startup["t"] / 60
    for _v, _ss, _el, _name, _dy in ((startup["pr"], startup["pr_ss"], plant["light"], "raffinate", -14),
                                     (startup["pe"], startup["pe_ss"], plant["heavy"], "extract", 8)):
        _a2.plot(_th, _v, color=COLORS[_el], lw=2)
        _a2.axhspan(_ss - 1, _ss + 1, color=COLORS[_el], alpha=0.12, lw=0)
        _ok = np.isfinite(_v)
        _a2.annotate(f"{_name} ({_el})", (_th[-1], _v[_ok][-1] if _ok.any() else _ss), xytext=(-4, _dy),
                     textcoords="offset points", ha="right", fontsize=8, color=INK)
    if np.isfinite(startup["t_settle"]):
        _a2.axvline(startup["t_settle"] / 60, color=INK2, lw=1, ls="--")
        _a2.annotate(f"settled {startup['t_settle'] / 60:.1f} h", (startup["t_settle"] / 60, 1.0),
                     xycoords=("data", "axes fraction"), xytext=(4, -12),
                     textcoords="offset points", fontsize=8, color=INK2)
    _vals = np.concatenate([startup["pr"][np.isfinite(startup["pr"])], startup["pe"][np.isfinite(startup["pe"])]])
    _a2.set_ylim(max(0.0, (_vals.min() if _vals.size else 0.0) - 5), 100)
    label_axes(_a2, "Start-up purities (masked below 5 % of steady flow; bands = ±1 point)", "Time (h)", "Purity (%)")
    # disturbance response: CV and MV on separate axes (no dual axis)
    _f3, (_a3, _a4) = new_figure(2, 9.5, 3.4)
    _tt = open_loop["t"] / 60
    _a3.plot(_tt, open_loop["cv"], color=INK2, lw=2)
    _a3.plot(_tt, closed_loop["cv"], color=ACCENT1, lw=2)
    _a3.axhline(open_loop["sp"], color=INK2, lw=1, ls=":")
    _a3.annotate("open loop", (_tt[-1], open_loop["cv"][-1]), xytext=(-4, 6), textcoords="offset points",
                 ha="right", fontsize=8, color=INK)
    _a3.annotate("PI", (_tt[-1], closed_loop["cv"][-1]), xytext=(-4, -12), textcoords="offset points",
                 ha="right", fontsize=8, color=INK)
    label_axes(_a3, "Heavy-element share in raffinate (CV)", "Time after step (h)", "%")
    _a4.plot(_tt, open_loop["S"], color=INK2, lw=2)
    _a4.plot(_tt, closed_loop["S"], color=ACCENT1, lw=2)
    _a4.axhspan(0.5 * plant["S"], 1.5 * plant["S"], color="#ecebe7", zorder=0)
    label_axes(_a4, "Solvent flow S (MV; shaded = ±50 % limits)", "Time after step (h)", "L/min")
    _ext, _scr = 0, _N - 1
    mo.vstack([
        mo.md("### 6 · Mixer-settler dynamics and PI control\n"
              "Uses the pair, stages, S, W (× F) and pH values from section 5."),
        mo.hstack([sl_F, sl_Vm, sl_Vs, sl_tmt], justify="start", gap=1.5, wrap=True),
        mo.hstack([
            mo.stat(f"{plant['tau_mix'][_ext]:.2f} / {plant['tau_set'][_ext]:.2f} min", label="Extraction τ_mix / τ_set", bordered=True),
            mo.stat(f"{plant['tau_mix'][_scr]:.2f} / {plant['tau_set'][_scr]:.2f} min", label="Scrub τ_mix / τ_set", bordered=True),
            mo.stat(f"{plant['eta'][_ext]:.3f} / {plant['eta'][_scr]:.3f}", label="η extraction / scrub", bordered=True),
            mo.stat(f"{plant['holdup']:.0f} L", label="Hold-up", bordered=True),
        ], widths="equal"),
        _f1,
        mo.hstack([
            mo.stat(f"{startup['pr_ss']:.2f} %", label=f"Steady raffinate purity ({plant['light']})", bordered=True),
            mo.stat(f"{startup['pe_ss']:.2f} %", label=f"Steady extract purity ({plant['heavy']})", bordered=True),
            mo.stat(f"{startup['t_settle'] / 60:.1f} h" if np.isfinite(startup["t_settle"]) else "> horizon",
                    label="Settling time (±1 point)", bordered=True),
        ], widths="equal"),
        _f2,
        mo.hstack([dd_dist, sl_LG, sl_Ti, sl_delay], justify="start", gap=1.5, wrap=True),
        mo.hstack([
            mo.stat(f"{open_loop['sp']:.1f} → {open_loop['final']:.1f} %", label="CV open loop (start → end)", bordered=True),
            mo.stat(f"{open_loop['peak']:.2f}", label="Open-loop peak deviation (points)", bordered=True),
            mo.stat(f"{closed_loop['peak']:.2f}", label="PI peak deviation (points)", bordered=True),
            mo.stat(f"{open_loop['Kp']:.2f} / {closed_loop['Kc']:.3f}", label="Kp (pts per L/min) / Kc", bordered=True),
        ], widths="equal"),
        _f3,
    ])
    return


# ============================================================================
# PAGE 3: Python code
# ============================================================================
@app.cell(hide_code=True)
def _(
    PAGES,
    assemble,
    band_lu,
    cascade_steady,
    inspect,
    linecache,
    log10_D,
    mccabe_thiele,
    mo,
    nav,
    new_figure,
    run_tests,
    simulate_disturbance,
    textwrap,
):
    mo.stop(nav.value != PAGES[2])

    def _cell_source(fn):
        """Whole source of the cell that defines `fn` (marimo registers each
        cell's code with linecache, also under Pyodide)."""
        try:
            _lines = linecache.getlines(inspect.getsourcefile(fn) or "")
            _src = "".join(_lines) if _lines else inspect.getsource(fn)
        except (OSError, TypeError):
            return "# source not available in this runtime; see notebook.py"
        return textwrap.dedent(_src).strip()

    _groups = [
        ("Chemistry: D, pH50, % extracted, separation factors, Fenske", log10_D),
        ("Kremser and McCabe–Thiele", mccabe_thiele),
        ("Steady extraction/scrub cascade (tridiagonal)", cascade_steady),
        ("Banded LU solver (half-bandwidth 6)", band_lu),
        ("Mixer-settler dynamics (implicit Euler)", assemble),
        ("PI control and disturbances", simulate_disturbance),
        ("Tests", run_tests),
        ("Plotting helpers", new_figure),
    ]
    mo.vstack(
        [mo.md("## 3 · Python code\nEvery function behind pages 1 and 2, in full, exactly as it runs. "
               "The test results are at the bottom of the page.")]
        + [mo.vstack([mo.md(f"### {_t}"), mo.ui.code_editor(_cell_source(_f), language="python", disabled=True)])
           for _t, _f in _groups]
    )
    return


@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[2])
    btn_tests = mo.ui.run_button(label="Re-run tests")
    return (btn_tests,)


@app.cell(hide_code=True)
def _(btn_tests, mo, run_tests):
    btn_tests  # re-runs this cell when clicked; tests also run when page 3 opens
    _res = run_tests()
    _n_pass = sum(r[4] for r in _res)
    _rows = [{"Test": r[0], "Expected": float(f"{r[1]:.6g}"), "Got": float(f"{r[2]:.6g}"),
              "Tolerance": float(f"{r[3]:.3g}"), "Result": "PASS" if r[4] else "FAIL"} for r in _res]
    mo.vstack([
        mo.md(f"### Test results: {_n_pass} / {len(_res)} passed"),
        mo.callout(mo.md("All tests passed." if _n_pass == len(_res) else
                         f"**{len(_res) - _n_pass} test(s) failed.** See the table."),
                   kind="success" if _n_pass == len(_res) else "danger"),
        btn_tests,
        mo.ui.table(_rows, selection=None, page_size=50),
    ])
    return


# ============================================================================
# PAGE 4: Attribution
# ============================================================================
@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[3])
    mo.md(
        r"""
## 4 · Attribution

> **This is an AI-generated project.** The notebook code, text, equations, tests, README and
> web export were written by an AI model from a detailed specification supplied by the author.
> All numbers are for teaching. They are not validated plant data.

### Author
> **✏️ PLACEHOLDER: author name**
>
> **✏️ PLACEHOLDER: author's contribution** (for example: specification of the model, equations,
> parameter values and reference checks; review of the output).

### Models and services used
| Item | Role |
|---|---|
| Anthropic Claude (large language model), run through **Claude Code** in a cloud session | Wrote all code and text, ran the tests, built the export and committed to GitHub. *(The exact model version is not recorded in this file. ✏️ Author: fill in if required.)* |
| Anthropic web search tool (inside Claude Code) | Literature search for separation factors (search excerpts only, see below) |
| marimo | Notebook format, UI widgets, `marimo export html-wasm` |
| NumPy, Matplotlib | Numerics and charts |
| Pyodide (loaded from a CDN in the browser) | Runs the WebAssembly export |
| molab, GitHub, GitHub Pages | Hosting and running the notebook |

### References
Status key: **Full**: the full text was read. **Excerpt**: only a search-engine result/excerpt was seen. Direct page
fetching was blocked by the session's network policy, so **no reference was read in full**.

| # | Reference | Used for | Status |
|---|---|---|---|
| 1 | US Patent 10,227,676, *Process of solvent extraction for separating rare earth elements based on separation factor*: [patents.justia.com/patent/10227676](https://patents.justia.com/patent/10227676) | Ce/La 4, Pr/Ce 2, Nd/Pr 1.5. These were supplied in the specification. The search excerpt confirms the title and scope but **did not show these numbers**. | Excerpt |
| 2 | Xie F., Zhang T.A., Dreisinger D., Doyle F., *A critical review on solvent extraction of rare earths from aqueous solutions*, Minerals Engineering 56 (2014) 10–28: [doi:10.1016/j.mineng.2013.10.021](https://doi.org/10.1016/j.mineng.2013.10.021) | Background on D2EHPA, HEHEHP (PC88A) and cascade configurations | Excerpt |
| 3 | Kumari A. et al., C. R. Chimie (2018), PC88A separation of rare earths from monazite leach: [doi:10.1016/j.crci.2018.09.005](https://comptes-rendus.academie-sciences.fr/chimie/articles/10.1016/j.crci.2018.09.005/) | Reports Ce/La 60.85, Nd/La 72.17 and Pr/La 98.33 at pH 2. This shows that β depends strongly on conditions. | Excerpt |
| 4 | US patent (USPTO full-text PDF 8177881), *Method for extracting and separating rare earth elements* (diglycolamic acid vs PC‑88A): [image-ppubs.uspto.gov/…/8177881](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8177881) | PC‑88A Nd/Pr ≈ 1.4 (supports 1.5); PC‑88A Dy/Tb 2.1 | Excerpt |
| 5 | US patent (USPTO full-text PDF 8802040), *Method for extracting and separating light rare earth element*: [image-ppubs.uspto.gov/…/8802040](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8802040) | PC‑88A Nd/Pr 1.4 under the same conditions as a DGA extractant | Excerpt |
| 6 | Rare Metals (2024), light rare earth separation study: [doi:10.1007/s12598-024-03019-7](https://link.springer.com/article/10.1007/s12598-024-03019-7) | A search summary listed β(Sm/Nd) = 20, β(Sm/Ce) = 54 and β(Ce/La) = 12 for an extraction system. **It was not clear which source gave these numbers**, so they were not used. | Excerpt (attribution uncertain) |
| 7 | Agarwal V., Safarzadeh M.S. et al., Y(III) extraction with PC88A from sulfate, nitrate and chloride media, Mineral Processing and Extractive Metallurgy Review (2018): [doi:10.1080/08827508.2017.1415210](https://doi.org/10.1080/08827508.2017.1415210) | Y separation from light and heavy rare earths (qualitative only, no β) | Excerpt |
| 8 | *Solvent extraction of Sm from chloride solution with PC88A*, Materials Transactions 46(1) (2005) 64–68: [jstage.jst.go.jp](https://www.jstage.jst.go.jp/article/matertrans/46/1/46_1_64/_article/-char/ja/) | Sm–PC88A equilibrium (no Sm/Nd β in the excerpt) | Excerpt |
| 9 | *Solvent extraction and separation of Ce(III) and Sm(III) from mixed rare earth solutions using PC88A*: [onemine.org](https://onemine.org/documents/solvent-extraction-and-separation-of-cerium-iii-and-samarium-iii-from-mixed-rare-earth-solutions-using-pc88a) | Ce/Sm selectivity is best in H₂SO₄ at pH 2 (qualitative) | Excerpt |
| 10 | Fenske, Kremser and McCabe–Thiele relations, as in standard separation-process textbooks (e.g. Seader, Henley & Roper, *Separation Process Principles*, Wiley) | Staging equations on page 1 | Not consulted in this session (from the model's background knowledge) |
| 11 | marimo documentation: [docs.marimo.io](https://docs.marimo.io) | Notebook API, WebAssembly export | Not consulted in this session (from the model's background knowledge) |

**Flags on parameter values.** No idx value was changed. Ce/La 4, Pr/Ce 2 and Nd/Pr 1.5 are kept as given (attributed to
ref. 1, and Nd/Pr is consistent with refs. 4 and 5). **Sm/Nd = 7 and Y/Sm = 40 remain teaching estimates.** No source in
this session gave a comparable PC88A value: the only Sm/Nd figure found (20, ref. 6) has uncertain attribution and
conditions. Published β for Ce/La range from about 4 to above 50 depending on the medium, so every β here should be read as illustrative.

### Choices made without asking (as instructed)
* **Plotting library:** Matplotlib (ships with Pyodide). The charts are static images: they redraw on every slider change but have no hover tooltips.
* **Pages:** a tab bar at the top. Each page's cells render only when that page is selected, so the expensive lab cells do not run while other pages are open.
* **Feed concentration:** $c_F$ = 100 mmol/L total in the cascade and dynamics (adjustable). Extractant C = 0.5 M in sections 3, 5 and 6.
* **Flows in section 5 are per unit feed.** Section 6 multiplies S and W by F (L/min).
* **Time stepping:** start-up is integrated in 1,440 implicit-Euler steps over max(48 h, 3 × hold-up/F), up to 240 h. Disturbances are integrated in 2‑min steps over 36 h at the default plant size. The PI output updates every step.
* **Settling time:** both product purities must stay within ±1 point. The ±1 point bands are drawn on the chart.
* **Disturbance step:** applied at t = 0 from the nominal steady state, which is also the setpoint.
* **Process gain:** central difference over S ± 5 %. Kc = LG/|Kp|, reverse acting.
* **Phase volumes** in the dynamic model are fixed at the nominal split, even when the flows change.
* **Ce(IV) switch:** applies to sections 1 and 2 only.
* **Test tolerances:** set to the number of decimals quoted in the specification (for example ±0.0005 for pH50 and ±0.1 point for purities).
"""
    )
    return


if __name__ == "__main__":
    app.run()
