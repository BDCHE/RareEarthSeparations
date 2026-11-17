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

    def efficiency_sweep(plant, tau_mt_values):
        """Steady product purities (raffinate light %, extract heavy %) for a range
        of mass-transfer times: slower transfer -> lower stage efficiency."""
        _, _, _, L, S = assemble(plant)
        rows = []
        for tmt in tau_mt_values:
            u = steady_state(dict(plant, tau_mt=float(tmt)))
            r, e = product_flows(u, L[0], S)
            rows.append((100 * r[0] / r.sum(), 100 * e[1] / e.sum()))
        return np.array(rows)

    def simulate_startup(plant, t_end=None, nsteps=1440, snap_hours=(2, 6, 12, 24), frame_every=15):
        """Start from metal-free phases; feed switched on at t = 0.
        Purities are masked until each product carries 5 % of its steady flow.
        Settling time: last time either purity is outside +/-1 point of steady.
        Stage profiles are stored at the times in `snap_hours`, and the full state
        every `frame_every` steps (for the P&ID animation)."""
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
        snap_steps = {int(round(h * 60 / dt)): h for h in snap_hours if h * 60 <= t_end}
        snaps = {}
        u = np.zeros_like(b)
        frames, frame_k = [u.copy()], [0]
        for k in range(1, nsteps + 1):
            u = band_solve(lu, b + (m / dt) * u)
            if k % frame_every == 0:
                frames.append(u.copy())
                frame_k.append(k)
            if k in snap_steps:
                snaps[snap_steps[k]] = u.copy()
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
        return dict(t=t, pr=pr, pe=pe, pr_ss=pr_ss, pe_ss=pe_ss, t_settle=t_settle,
                    u_ss=u_ss, snaps=snaps, frames=np.array(frames), frame_k=np.array(frame_k))

    return (
        assemble,
        efficiency_sweep,
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
    # +/-50 % of nominal. Direct-acting PI (output rises when the CV rises, because the
    # process gain is negative) with conditional-integration anti-windup.

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

    def step_test(plant, rel=0.05, t_end=None, nsteps=None):
        """Open-loop step of the solvent flow S by `rel` (fraction) at t = 0,
        starting from the nominal steady state. Returns the CV response."""
        if t_end is None:
            t_end = float(np.clip(2.25 * plant["holdup"] / plant["F"], 36 * 60.0, 200 * 60.0))
        if nsteps is None:
            nsteps = int(round(t_end / 2.0))
        dt = t_end / nsteps
        L1 = assemble(plant)[3][0]
        u = steady_state(plant)
        y0 = heavy_share_raffinate(u, L1)
        S1 = plant["S"] * (1 + rel)
        ab, b, m, L, _ = assemble(plant, S=S1)
        A = ab.copy()
        A[:, :, KL] += m / dt
        lu = band_lu(A)
        t = np.arange(nsteps + 1) * dt
        y = np.zeros(nsteps + 1)
        y[0] = y0
        for k in range(1, nsteps + 1):
            u = band_solve(lu, b + (m / dt) * u)
            y[k] = heavy_share_raffinate(u, L[0])
        y_inf = heavy_share_raffinate(steady_state(plant, S=S1), L[0])
        return dict(t=t, y=y, y0=y0, y_inf=y_inf, dS=S1 - plant["S"])

    def fopdt_smith(t, y, y0, y_inf):
        """Smith's two-point method: fit y = y0 + K du (1 - exp(-(t - theta)/tau))
        from the times at which 28.3 % and 63.2 % of the final change is reached:
        tau = 1.5 (t63 - t28), theta = t63 - tau."""
        frac = (np.asarray(y) - y0) / (y_inf - y0)

        def first_crossing(level):
            i = int(np.argmax(frac >= level))
            if i == 0:
                return float("nan")       # never reached within the horizon
            return float(t[i - 1] + (level - frac[i - 1]) * (t[i] - t[i - 1]) / (frac[i] - frac[i - 1]))

        t28, t63 = first_crossing(0.283), first_crossing(0.632)
        tau = 1.5 * (t63 - t28)
        return dict(t28=t28, t63=t63, tau=tau, theta=max(t63 - tau, 0.0))

    def fopdt_response(t, y0, K, du, tau, theta):
        """First-order-plus-dead-time step response."""
        t = np.asarray(t, float)
        return y0 + K * du * np.where(t > theta, 1.0 - np.exp(-(t - theta) / tau), 0.0)

    def simc_tuning(tau, theta_total, tau_c=None):
        """SIMC PI rule (Skogestad, 2003) for an FOPDT process, written as a loop
        gain: LG = Kc |Kp| = tau / (tau_c + theta), Ti = min(tau, 4 (tau_c + theta)).
        The default closed-loop time constant is tau_c = theta (the 'tight' choice)."""
        tau_c = theta_total if tau_c is None else tau_c
        return dict(LG=tau / (tau_c + theta_total), Ti=min(tau, 4.0 * (tau_c + theta_total)), tau_c=tau_c)

    CONTROL_MODES = ["PI", "P only", "PI + feedforward ratio (S/F)"]

    def simulate_disturbance(plant, name, closed_loop, LG=2.0, Ti=120.0, delay=10.0,
                             t_end=None, nsteps=None, limit=0.5, mode="PI", frame_every=15):
        """Step disturbance at t = 0 from the nominal steady state.

        mode = "PI": feedback PI on the delayed analyzer signal.
        mode = "P only": proportional feedback, no integral (leaves an offset).
        mode = "PI + feedforward ratio (S/F)": the solvent flow follows the measured
               feed flow (S = S0 F/F0, a static feedforward / ratio controller)
               and PI feedback trims it.
        Also records the (uncontrolled) extract purity, IAE, ISE and the fraction of
        time the solvent flow sits on a limit."""
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
        ext = np.zeros(nsteps + 1)
        cv[0], s_hist[0] = sp, S0
        e_flow = S0 * u[:, -1]
        ext[0] = 100.0 * e_flow[1] / e_flow.sum()
        integral, S = 0.0, S0
        lo, hi = (1 - limit) * S0, (1 + limit) * S0
        use_i = mode != "P only"
        # feedforward: the feed flow is measured without delay; S/F held at its nominal ratio
        S_base = S0 * kw.get("F", plant["F"]) / plant["F"] if mode.startswith("PI +") else S0
        cached_S, lu = None, None
        frames, frame_k = [u.copy()], [0]           # full states for the P&ID animation
        for k in range(1, nsteps + 1):
            if S != cached_S:                       # refactor only when S moves
                ab, b, m, L, _ = assemble(plant, S=S, **kw)
                A = ab.copy()
                A[:, :, KL] += m / dt
                lu, cached_S = band_lu(A), S
            u = band_solve(lu, b + (m / dt) * u)
            if k % frame_every == 0:
                frames.append(u.copy())
                frame_k.append(k)
            cv[k] = heavy_share_raffinate(u, L[0])
            s_hist[k] = S
            e_flow = S * u[:, -1]
            ext[k] = 100.0 * e_flow[1] / e_flow.sum()   # heavy purity of the extract
            if closed_loop:
                measured = cv[max(0, k - nd)]       # analyzer dead time
                e = measured - sp                   # too much heavy -> more solvent
                if use_i:
                    trial = S_base + Kc * (e + (integral + e * dt) / Ti)
                    if lo < trial < hi or (trial >= hi and e < 0) or (trial <= lo and e > 0):
                        integral += e * dt          # conditional integration
                S = float(np.clip(S_base + Kc * (e + (integral / Ti if use_i else 0.0)), lo, hi))
        dev = np.abs(cv - sp)
        on_limit = (np.abs(s_hist - lo) < 1e-9) | (np.abs(s_hist - hi) < 1e-9)
        return dict(t=t, cv=cv, S=s_hist, ext=ext, sp=sp, Kp=Kp, Kc=Kc,
                    peak=float(np.max(dev)), t_peak=float(t[int(np.argmax(dev))]),
                    final=float(cv[-1]),
                    iae=float(np.sum(0.5 * (dev[1:] + dev[:-1])) * dt / 60.0),  # point-hours
                    ise=float(np.sum(0.5 * (dev[1:] ** 2 + dev[:-1] ** 2)) * dt / 60.0),
                    sat_frac=float(np.mean(on_limit)), frames=np.array(frames),
                    frame_k=np.array(frame_k), nd=nd, dt=dt, F_run=kw.get("F", plant["F"]),
                    W_run=kw.get("W", plant["W"]), mode=mode, closed_loop=closed_loop)

    return (
        CONTROL_MODES,
        DISTURBANCES,
        fopdt_response,
        fopdt_smith,
        heavy_share_raffinate,
        process_gain,
        simc_tuning,
        simulate_disturbance,
        step_test,
    )


@app.cell
def _(np):
    # ------------------------------------------- frequency response and tuning rules
    # The loop is modelled as FOPDT process x PI controller (x analyzer dead time):
    #     L(s) = LG (1 + 1/(Ti s)) exp(-theta s) / (tau s + 1),   LG = Kc |Kp|
    # (the controller's direct action cancels the negative process gain, so the
    # loop is an ordinary negative-feedback loop). theta = process + analyzer dead time.

    def loop_frequency_response(w, LG, Ti, tau, theta, integral=True):
        """Magnitude |L(jw)| and phase (degrees, continuous) of the open loop."""
        w = np.asarray(w, float)
        mag = LG / np.sqrt(1.0 + (w * tau) ** 2)
        phase = -w * theta - np.arctan(w * tau)
        if integral:
            mag = mag * np.sqrt(1.0 + 1.0 / (w * Ti) ** 2)
            phase = phase - np.arctan(1.0 / (w * Ti))
        return mag, np.degrees(phase)

    def _bisect(f, lo, hi, iters=200):
        flo = f(lo)
        for _ in range(iters):
            mid = np.sqrt(lo * hi)                 # bisection on a log scale
            fm = f(mid)
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi = mid
        return np.sqrt(lo * hi)

    def ultimate_point(tau, theta):
        """P-only stability limit of an FOPDT loop: the frequency where the phase
        is -180 deg (w_u theta + atan(w_u tau) = pi) and the loop gain that puts
        |L| = 1 there. Returns (w_u, LG_u, P_u)."""
        if theta <= 0:
            return float("inf"), float("inf"), 0.0
        wu = _bisect(lambda w: w * theta + np.arctan(w * tau) - np.pi, 1e-9, 1e3 / theta)
        return float(wu), float(np.sqrt(1.0 + (wu * tau) ** 2)), float(2 * np.pi / wu)

    def stability_margins(LG, Ti, tau, theta, integral=True):
        """Gain margin (factor), phase margin (deg), gain-crossover and
        phase-crossover frequencies (rad/min) of the FOPDT + PI loop."""
        mag = lambda w: loop_frequency_response(w, LG, Ti, tau, theta, integral)[0]
        ph = lambda w: loop_frequency_response(w, LG, Ti, tau, theta, integral)[1]
        w_lo, w_hi = 1e-7, 1e3 / max(theta, 1e-3)
        # gain crossover: |L| = 1 (|L| falls monotonically with w)
        wc = _bisect(lambda w: mag(w) - 1.0, w_lo, w_hi) if mag(w_lo) > 1 > mag(w_hi) else float("nan")
        pm = 180.0 + ph(wc) if np.isfinite(wc) else float("inf")
        # phase crossover: first w where the phase reaches -180 deg
        grid = np.logspace(np.log10(w_lo), np.log10(w_hi), 4000)
        below = np.nonzero(ph(grid) <= -180.0)[0]
        if below.size == 0 or below[0] == 0:
            w180, gm = float("nan"), float("inf")
        else:
            i = below[0]
            w180 = _bisect(lambda w: ph(w) + 180.0, grid[i - 1], grid[i])
            gm = 1.0 / mag(w180)
        return dict(GM=float(gm), PM=float(pm), wc=float(wc), w180=float(w180))

    def tuning_rules(tau, theta, tau_c=None):
        """Classic PI rules for an FOPDT model, written as loop gain LG = Kc|Kp|
        and integral time Ti (both independent of the sign and size of Kp)."""
        r = theta / tau
        wu, LGu, Pu = ultimate_point(tau, theta)
        lam = theta if tau_c is None else tau_c
        return {
            "Ziegler–Nichols (ultimate gain)": (0.45 * LGu, Pu / 1.2),
            "Ziegler–Nichols (reaction curve)": (0.9 / r, 3.33 * theta),
            "Cohen–Coon": ((1 / r) * (0.9 + r / 12.0), theta * (30 + 3 * r) / (9 + 20 * r)),
            "Lambda / IMC (λ = θ)": (tau / (lam + theta), tau),
            "SIMC (τc = θ)": (tau / (lam + theta), min(tau, 4 * (lam + theta))),
        }

    return loop_frequency_response, stability_margins, tuning_rules, ultimate_point


@app.cell
def _(
    ELEMENTS,
    PAIR_DEFAULTS,
    band_lu,
    band_solve,
    band_to_dense,
    cascade_steady,
    distribution_ratio,
    efficiency_sweep,
    fenske_nmin,
    fopdt_response,
    fopdt_smith,
    implicit_euler_step,
    assemble,
    kremser_raffinate_fraction,
    make_plant,
    mccabe_thiele,
    np,
    pH50,
    process_gain,
    separation_factor,
    simc_tuning,
    simulate_disturbance,
    pid_svg,
    simulate_startup,
    stability_margins,
    stage_trend,
    step_test,
    steady_state,
    tuning_rules,
    ultimate_point,
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
        check("PI removes the offset (final CV - setpoint, points)", 0.0, cl["final"] - cl["sp"], 0.05)
        check("PI integral of absolute error below open loop (IAE ratio < 1)", 0.0,
              max(0.0, cl["iae"] / ol["iae"] - 1.0), 1e-12)

        # stage efficiency: slower mass transfer must lower both purities
        sweep = efficiency_sweep(plant, [0.2, 0.5, 1.0])
        check("efficiency sweep at tau_mt = 12 s matches start-up steady state", st["pr_ss"], sweep[0, 0], 1e-9)
        check("purity falls as tau_mt grows (monotone)", 0.0,
              float(np.sum(np.diff(sweep, axis=0) > 0)), 0.0)

        # process identification and tuning
        up = step_test(plant, +0.05)
        down = step_test(plant, -0.05)
        k_up = (up["y_inf"] - up["y0"]) / up["dS"]
        k_down = (down["y_inf"] - down["y0"]) / down["dS"]
        check("mean of +/-5 % step gains = central-difference Kp", process_gain(plant),
              0.5 * (k_up + k_down), 1e-9)
        check("step test reaches its new steady state (points)", up["y_inf"], up["y"][-1], 0.05)
        tt = np.linspace(0, 600, 6001)
        fit = fopdt_smith(tt, fopdt_response(tt, 0.0, 2.0, 1.0, 100.0, 20.0), 0.0, 2.0)
        check("Smith two-point fit recovers tau = 100 min", 100.0, fit["tau"], 0.5)
        check("Smith two-point fit recovers theta = 20 min", 20.0, fit["theta"], 0.5)
        simc = simc_tuning(10.0, 2.0)
        check("SIMC loop gain, tau 10, theta 2", 2.5, simc["LG"], 1e-12)
        check("SIMC integral time, tau 10, theta 2", 10.0, simc["Ti"], 1e-12)

        # controller modes: P-only leaves an offset close to dy/(1 + LG); feedforward helps feed-flow steps
        p_only = simulate_disturbance(plant, "Feed flow +20%", True, LG=2.0, delay=10.0, mode="P only")
        check("P-only offset vs linear theory dy/(1+LG) (points)", (ol["final"] - ol["sp"]) / 3.0,
              p_only["final"] - p_only["sp"], 0.35)
        ff = simulate_disturbance(plant, "Feed flow +20%", True, LG=2.0, Ti=120.0, delay=10.0,
                                  mode="PI + feedforward ratio (S/F)")
        check("feedforward ratio lowers the feed-flow peak below PI alone (excess, points)", 0.0,
              max(0.0, ff["peak"] - cl["peak"]), 0.0)

        # frequency response: FOPDT with tau = theta = 1 has w_u = 2.0288 rad/time, K_u = 2.2618
        wu, LGu, Pu = ultimate_point(1.0, 1.0)
        check("FOPDT ultimate frequency, tau = theta = 1", 2.0288, wu, 1e-3)
        check("FOPDT ultimate loop gain, tau = theta = 1", 2.2618, LGu, 1e-3)
        gm_p = stability_margins(1.0, 1.0, 1.0, 1.0, integral=False)["GM"]
        check("gain margin of P-only loop with LG = 1 equals LG_u", LGu, gm_p, 1e-6)
        m_sel = stability_margins(2.5, 1e9, 1.0, 0.0, integral=False)
        check("phase margin of LG/(tau s + 1), LG 2.5, no dead time (deg)",
              180.0 - np.degrees(np.arctan(np.sqrt(2.5 ** 2 - 1))), m_sel["PM"], 1e-6)
        rules = tuning_rules(10.0, 2.0)
        check("Ziegler–Nichols reaction-curve PI: LG = 0.9 tau/theta", 4.5,
              rules["Ziegler–Nichols (reaction curve)"][0], 1e-12)
        check("Cohen–Coon PI integral time (tau 10, theta 2)", 2.0 * (30 + 0.6) / (9 + 4.0),
              rules["Cohen–Coon"][1], 1e-12)
        theta_tot = fopdt_smith(up["t"], up["y"], up["y0"], up["y_inf"])["theta"] + 10.0
        tau_fit = fopdt_smith(up["t"], up["y"], up["y0"], up["y_inf"])["tau"]
        lgu_model = ultimate_point(tau_fit, theta_tot)[1]
        at_ku = simulate_disturbance(plant, "Feed flow +20%", True, LG=lgu_model, Ti=1e9, delay=10.0,
                                     limit=0.95, mode="P only", t_end=24 * 60.0)
        tail = at_ku["cv"][at_ku["t"] > at_ku["t"][-1] - 360.0]
        check("plant P-only at the model's LG_u still settles (late swing, points)", 0.0, float(np.ptp(tail)), 0.01)

        # P&ID: one stage symbol per stage for any configuration; frames stored for the animation
        cols = {el: "#000000" for el in ELEMENTS}
        flows0 = dict(S=plant["S"], W=plant["W"], F=plant["F"])
        ctrl0 = dict(mode="AUTO", pv=21.9, meas=21.9, sp=21.9, delay=10.0)
        svg20 = pid_svg(plant, st["u_ss"], cols, 150.0, 40.0, flows0, ctrl0, "test")
        check("P&ID draws one stage per stage (Pr|Nd, 10 + 10)", 20.0, float(svg20.count('class="stage"')), 0.0)
        small = make_plant("Sm|Y", **PAIR_DEFAULTS["Sm|Y"])
        svg6 = pid_svg(small, steady_state(small), cols, 150.0, 40.0, flows0, ctrl0, "test")
        check("P&ID draws one stage per stage (Sm|Y, 3 + 3)", 6.0, float(svg6.count('class="stage"')), 0.0)
        check("start-up frames: last stored frame is the final time step", 1440.0, float(st["frame_k"][-1]), 0.0)
        check("start-up frames: first frame is the metal-free plant", 0.0, float(np.abs(st["frames"][0]).max()), 0.0)
        tr = stage_trend(st["frames"][4], st["frames"][3], 0.5)
        check("stage trend: every stage is filling 1.5–2 h into start-up (count of ▲)", 20.0,
              float(np.sum(tr == 1)), 0.0)

        # actuator limits and anti-windup: a +/-20 % limit forces saturation
        hard = simulate_disturbance(plant, "Extraction pH −0.05", True, LG=5.0, Ti=30.0,
                                    delay=10.0, limit=0.2)
        check("solvent flow never exceeds +20 % limit (excess, L/min)", 0.0,
              max(0.0, float(hard["S"].max()) - 1.2 * plant["S"]), 1e-12)
        check("solvent flow actually saturates in that case (1 = yes)", 1.0,
              float(hard["sat_frac"] > 0.1), 0.0)
        return results

    return (run_tests,)


@app.cell
def _(np):
    # --------------------------------------------------------------- P&ID drawing
    # Builds an SVG piping & instrumentation diagram of the cascade: one
    # mixer-settler box per stage (organic layer on top, aqueous below) with bars
    # for the light and heavy element, the process streams, and ISA-style
    # instrument bubbles for every control loop. Hover a stage for exact values.
    from html import escape as _esc

    def stage_trend(u_now, u_prev, dt_h, threshold=0.02):
        """+1 / -1 / 0 per stage: total metal held in the stage's settler rising,
        falling, or steady (relative change above `threshold` per hour)."""
        if u_prev is None or dt_h <= 0:
            return np.zeros(u_now.shape[1] // 4, int)
        tot_now = (u_now[:, 2::4] + u_now[:, 3::4]).sum(axis=0)
        tot_prev = (u_prev[:, 2::4] + u_prev[:, 3::4]).sum(axis=0)
        rate = (tot_now - tot_prev) / dt_h / np.maximum(np.maximum(tot_now, tot_prev), 1e-9)
        return np.where(rate > threshold, 1, np.where(rate < -threshold, -1, 0))

    def _bubble(x, y, tag, sub="", active=True, r=19, sub_at=None, anchor="middle"):
        """ISA instrument bubble; the horizontal line means 'control-room (DCS) mounted'.
        `sub` is an optional reading, drawn below the bubble or at `sub_at` = (x, y)."""
        sx, sy = sub_at if sub_at else (x, y + r + 12)
        col = "#0b0b0b" if active else "#b5b3ad"
        a, b_ = (tag.split("-") + [""])[:2]
        return (f'<g><circle cx="{x}" cy="{y}" r="{r}" fill="#ffffff" stroke="{col}" stroke-width="1.5"/>'
                f'<line x1="{x - r}" y1="{y}" x2="{x + r}" y2="{y}" stroke="{col}" stroke-width="1"/>'
                f'<text x="{x}" y="{y - 5}" font-size="10" text-anchor="middle" fill="{col}" font-weight="600">{_esc(a)}</text>'
                f'<text x="{x}" y="{y + 12}" font-size="9" text-anchor="middle" fill="{col}">{_esc(b_)}</text>'
                + (f'<text x="{sx}" y="{sy}" font-size="9" text-anchor="{anchor}" fill="#52514e">{_esc(sub)}</text>' if sub else "")
                + "</g>")

    def _valve(x, y, active=True):
        """Control-valve bow-tie on a horizontal line."""
        col = "#0b0b0b" if active else "#b5b3ad"
        return (f'<path d="M{x - 7},{y - 6} L{x + 7},{y + 6} L{x + 7},{y - 6} L{x - 7},{y + 6} Z" '
                f'fill="#ffffff" stroke="{col}" stroke-width="1.3"/>')

    def _signal(points, active=True):
        """Dashed instrument signal line."""
        col = "#52514e" if active else "#cfcdc6"
        pts = " ".join(f"{x},{y}" for x, y in points)
        return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.2" stroke-dasharray="5,3"/>'

    def pid_svg(plant, u, colors, scale_aq, scale_org, flows, ctrl, time_label="", trend=None):
        """Return the P&ID as an SVG string.

        plant  : dict from make_plant (N, n_ext, light, heavy, pH values)
        u      : (2, 4N) state [xm, ym, xs, ys per stage] for light (0) and heavy (1)
        scale_*: concentration (mmol/L) that fills a layer's full bar height
        flows  : dict S, W, F (actual flows at this instant, L/min)
        ctrl   : dict mode ('MAN', 'AUTO', 'AUTO+FF'), pv, meas, sp, delay (min)
        trend  : optional +1/-1/0 per stage from stage_trend()"""
        N, f = plant["N"], plant["n_ext"]
        light, heavy = plant["light"], plant["heavy"]
        W_svg, H_svg = 1100, 560
        x0, x1 = 190, W_svg - 190                    # stage area
        pitch = (x1 - x0) / N
        bw = pitch * 0.78                            # box width
        top, mid, bot = 200, 285, 370                # organic layer top..mid, aqueous mid..bot
        yo, ya = (top + mid) / 2, (mid + bot) / 2    # organic / aqueous stream levels
        cx = lambda n: x0 + (n - 0.5) * pitch        # centre of stage n (1-based)
        out = [f'<svg viewBox="0 0 {W_svg} {H_svg}" width="100%" xmlns="http://www.w3.org/2000/svg" '
               f'font-family="Inter, Helvetica, Arial, sans-serif" role="img" '
               f'aria-label="P&amp;ID of a {N}-stage extraction/scrub cascade">',
               f'<rect x="0" y="0" width="{W_svg}" height="{H_svg}" fill="#fcfcfb"/>',
               '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#0b0b0b"/></marker></defs>']
        mode = ctrl.get("mode", "MAN")
        auto, ff = mode != "MAN", mode == "AUTO+FF"

        # ---- section brackets and pH controllers
        for lo_n, hi_n, name, ph, tag in ((1, f, "Extraction", plant["pH_ext"], "pHIC-201"),
                                          (f + 1, N, "Scrub", plant["pH_scr"], "pHIC-202")):
            if hi_n < lo_n:
                continue
            xa, xb = cx(lo_n) - bw / 2, cx(hi_n) + bw / 2
            out.append(f'<path d="M{xa},{top - 22} L{xa},{top - 30} L{xb},{top - 30} L{xb},{top - 22}" '
                       f'fill="none" stroke="#52514e" stroke-width="1"/>')
            out.append(f'<text x="{(xa + xb) / 2}" y="{top - 36}" font-size="11" text-anchor="middle" fill="#0b0b0b">'
                       f'{name} section · stages {lo_n}–{hi_n} · pH {ph:.2f}</text>')
            # bubble at the outer end of the bracket, clear of the centred title and the feed line
            bx = xa + 22 if name == "Extraction" else xb - 22
            out.append(_bubble(bx, top - 80, tag, ""))
            out.append(f'<text x="{bx + (24 if name == "Extraction" else -24)}" y="{top - 84}" font-size="9" '
                       f'text-anchor="{"start" if name == "Extraction" else "end"}" fill="#52514e">acid/alkali dosing (ideal)</text>')
            out.append(_signal([(bx, top - 61), (bx, top - 30)]))

        # ---- stages
        tr = trend if trend is not None else np.zeros(N, int)
        hl = (mid - top) - 8                          # max bar height in a layer
        for n in range(1, N + 1):
            s = n - 1
            xs_ = u[:, 4 * s + 2]                      # settler aqueous (light, heavy)
            ys_ = u[:, 4 * s + 3]                      # settler organic
            xm_, ym_ = u[:, 4 * s], u[:, 4 * s + 1]
            share = 100 * xs_[1] / max(xs_.sum(), 1e-12)
            tip = (f"Stage {n} ({'extraction' if n <= f else 'scrub'}{', feed stage' if n == f else ''})\n"
                   f"Settler organic: {light} {ys_[0]:.2f}, {heavy} {ys_[1]:.2f} mmol/L\n"
                   f"Settler aqueous: {light} {xs_[0]:.2f}, {heavy} {xs_[1]:.2f} mmol/L\n"
                   f"Mixer aqueous:   {light} {xm_[0]:.2f}, {heavy} {xm_[1]:.2f} mmol/L\n"
                   f"Mixer organic:   {light} {ym_[0]:.2f}, {heavy} {ym_[1]:.2f} mmol/L\n"
                   f"Heavy share of aqueous metal: {share:.1f} %")
            xl = cx(n) - bw / 2
            out.append(f'<g class="stage"><title>{_esc(tip)}</title>')
            out.append(f'<rect x="{xl}" y="{top}" width="{bw}" height="{mid - top}" fill="#f4efe1" stroke="#52514e" stroke-width="1"/>')
            out.append(f'<rect x="{xl}" y="{mid}" width="{bw}" height="{bot - mid}" fill="#e7eff6" stroke="#52514e" stroke-width="1"/>')
            barw = bw * 0.36
            for j, (vals, y_base, sc) in enumerate(((ys_, mid - 4, scale_org), (xs_, bot - 4, scale_aq))):
                for e in range(2):
                    h = min(hl, hl * vals[e] / max(sc, 1e-12))
                    bx_ = xl + bw * (0.1 + 0.44 * e)
                    out.append(f'<rect x="{bx_:.2f}" y="{y_base - h:.2f}" width="{barw:.2f}" height="{h:.2f}" '
                               f'fill="{colors[light if e == 0 else heavy]}" rx="1"/>')
            out.append("</g>")
            if N <= 30 or n % max(1, N // 15) == 0 or n in (1, f, N):
                out.append(f'<text x="{cx(n)}" y="{bot + 14}" font-size="9" text-anchor="middle" fill="#52514e">{n}</text>')
            if tr[n - 1] != 0:
                sym, col = ("▲", "#b4521f") if tr[n - 1] > 0 else ("▼", "#2a78d6")
                out.append(f'<text x="{cx(n)}" y="{top - 6}" font-size="10" text-anchor="middle" fill="{col}">{sym}</text>')
            # inter-stage streams: organic to the right, aqueous to the left
            if n < N:
                gx0, gx1 = cx(n) + bw / 2, cx(n + 1) - bw / 2
                out.append(f'<line x1="{gx0}" y1="{yo}" x2="{gx1}" y2="{yo}" stroke="#0b0b0b" stroke-width="1.2"/>')
                out.append(f'<line x1="{gx1}" y1="{ya}" x2="{gx0}" y2="{ya}" stroke="#0b0b0b" stroke-width="1.2"/>')

        # ---- feed line with FT-102 (and ratio relay FY-102)
        xf = cx(f)
        out.append(f'<line x1="{xf}" y1="40" x2="{xf}" y2="{top}" stroke="#0b0b0b" stroke-width="2" marker-end="url(#ah)"/>')
        out.append(f'<text x="{xf + 8}" y="40" font-size="11" fill="#0b0b0b">Feed F = {flows["F"]:.2f} L/min '
                   f'({light}+{heavy}, {plant["cF"]:.0f} mmol/L)</text>')
        out.append(f'<line x1="{xf}" y1="62" x2="{xf - 30}" y2="62" stroke="#52514e" stroke-width="1"/>')
        out.append(_bubble(xf - 50, 62, "FT-102", ""))

        # ---- solvent inlet (left, organic level) with FIC-101 and valve
        out.append(f'<line x1="20" y1="{yo}" x2="{cx(1) - bw / 2}" y2="{yo}" stroke="#0b0b0b" stroke-width="2" marker-end="url(#ah)"/>')
        out.append(f'<text x="20" y="{yo - 8}" font-size="11" fill="#0b0b0b">Fresh solvent S</text>')
        out.append(_valve(120, yo))
        out.append(_bubble(120, yo - 62, "FIC-101", f"S = {flows['S']:.2f} L/min", sub_at=(120, yo + 20)))
        out.append(_signal([(120, yo - 43), (120, yo - 6)]))
        # ratio feedforward FT-102 -> FY-102 (x) -> FIC-101 setpoint
        out.append(_bubble(45, 62, "FY-102", "S/F ratio", active=ff, sub_at=(68, 94), anchor="start"))
        out.append(_signal([(xf - 69, 62), (64, 62)], active=ff))
        out.append(_signal([(45, 81), (45, yo - 62), (101, yo - 62)], active=ff))

        # ---- raffinate outlet (left, aqueous level) with AT-101 -> AIC-101 -> FIC-101
        out.append(f'<line x1="{cx(1) - bw / 2}" y1="{ya}" x2="20" y2="{ya}" stroke="#0b0b0b" stroke-width="2" marker-end="url(#ah)"/>')
        out.append(f'<text x="20" y="{ya + 18}" font-size="11" fill="#0b0b0b">Raffinate ({light} product)</text>')
        out.append(f'<line x1="140" y1="{ya}" x2="140" y2="{ya + 50}" stroke="#52514e" stroke-width="1"/>')
        out.append(_bubble(140, ya + 69, "AT-101", f"delay {ctrl.get('delay', 0):.0f} min",
                           sub_at=(115, ya + 73), anchor="end"))
        aic_txt = (f"PV {ctrl['pv']:.2f} % · meas {ctrl['meas']:.2f} % · SP {ctrl['sp']:.2f} %"
                   if ctrl.get("pv") is not None else "")
        out.append(_bubble(70, ya + 150, "AIC-101", "", active=auto))
        out.append(f'<text x="20" y="{ya + 186}" font-size="9" fill="#52514e">{_esc(aic_txt)}</text>')
        out.append(f'<text x="20" y="{ya + 198}" font-size="9" fill="{"#0b0b0b" if auto else "#b4521f"}" '
                   f'font-weight="600">{_esc({"MAN": "MANUAL (loop open)", "AUTO": "AUTO (PI)", "AUTO-P": "AUTO (P only)", "AUTO+FF": "AUTO (PI) + ratio FF"}.get(mode, mode))}</text>')
        out.append(_signal([(140, ya + 88), (140, ya + 150), (89, ya + 150)]))
        out.append(_signal([(51, ya + 150), (12, ya + 150), (12, yo - 85), (101, yo - 70)], active=auto))

        # ---- extract outlet (right, organic level) and scrub inlet with FIC-103
        xr = cx(N) + bw / 2
        out.append(f'<line x1="{xr}" y1="{yo}" x2="{W_svg - 20}" y2="{yo}" stroke="#0b0b0b" stroke-width="2" marker-end="url(#ah)"/>')
        out.append(f'<text x="{W_svg - 20}" y="{yo - 8}" font-size="11" text-anchor="end" fill="#0b0b0b">Loaded organic → extract ({heavy})</text>')
        out.append(f'<line x1="{W_svg - 20}" y1="{ya}" x2="{xr}" y2="{ya}" stroke="#0b0b0b" stroke-width="2" marker-end="url(#ah)"/>')
        out.append(f'<text x="{W_svg - 20}" y="{ya + 18}" font-size="11" text-anchor="end" fill="#0b0b0b">Scrub liquor W</text>')
        out.append(_valve(W_svg - 110, ya))
        out.append(_bubble(W_svg - 110, ya + 69, "FIC-103", f"W = {flows['W']:.2f} L/min"))
        out.append(_signal([(W_svg - 110, ya + 50), (W_svg - 110, ya + 6)]))
        # interface level control, drawn once on the last settler (one per settler in practice)
        out.append(_bubble(xr + 40, mid - 2, "LIC-1xx", "", r=15))
        out.append(f'<text x="{xr + 40}" y="{mid + 26}" font-size="8" text-anchor="middle" fill="#52514e">each settler</text>')

        # ---- legend
        ly = H_svg - 52
        out.append(f'<text x="200" y="{ly}" font-size="11" fill="#0b0b0b" font-weight="600">{_esc(time_label)}</text>')
        items = [(colors[light], f"{light} (light)"), (colors[heavy], f"{heavy} (heavy)"),
                 ("#f4efe1", f"organic layer, full bar = {scale_org:.0f} mmol/L"),
                 ("#e7eff6", f"aqueous layer, full bar = {scale_aq:.0f} mmol/L")]
        lx = 200
        for colr, txt in items:
            out.append(f'<rect x="{lx}" y="{ly + 12}" width="12" height="12" fill="{colr}" stroke="#52514e" stroke-width="0.5"/>'
                       f'<text x="{lx + 17}" y="{ly + 22}" font-size="10" fill="#52514e">{_esc(txt)}</text>')
            lx += 40 + 6.2 * len(txt)
        out.append(f'<text x="200" y="{ly + 42}" font-size="10" fill="#52514e">○ instrument (line = control room) · '
                   '– – – signal · ⋈ control valve · ▲/▼ stage metal rising/falling &gt; 2 %/h · greyed = not active · '
                   'hover a stage for values</text>')
        out.append("</svg>")
        return "".join(out)

    return pid_svg, stage_trend


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

This page is written as a short textbook in two parts of equal weight.

* **Part A, Separations (chapters A1–A9)** starts from first principles of liquid–liquid extraction and works up to
  the design of a counter-current extraction/scrub cascade for separating neighbouring rare earths.
* **Part B, Process dynamics and control (chapters B1–B17)** covers process-control theory: modelling, transfer
  functions, identification, P/PI/PID control, stability and frequency response, tuning, feedforward and
  multivariable strategies. It then applies them to a mixer-settler plant.

Each chapter ends in the lab section where you can try it. **Worked examples** use the lab's default numbers, so you
can reproduce every figure on page 2.

| Lab section (page 2) | Chapters on this page |
|---|---|
| 1 · % extracted vs pH | A1–A4 |
| 2 · Separation factors and Fenske | A5 |
| 3 · Kremser recovery vs stages | A6 |
| 4 · McCabe–Thiele with stair steps | A7 |
| 5 · Extraction / scrub cascade | A8 |
| 6 · Mixer-settler hydraulics and stage efficiency | B5, B6 |
| 7 · Start-up dynamics | B3, B4, B7, B8 |
| 8 · Step test, process model and tuning rules | B4, B9, B12 |
| 9 · Frequency response and stability margins | B11 |
| 10 · Disturbance rejection: P, PI and feedforward | B10, B13, B15, B16 |
| 11 · Interactive P&ID | B2, B13, B14 (reading the P&ID), B8, B15 |
| (background for all of Part B) | B1, B2, B14, B17 |
"""
    )
    return


@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[0])
    mo.md(
        r"""
---
## Part A · Separations

### A1 · Fundamentals of liquid–liquid extraction

#### A1.1 What it is and where it is used
**Liquid–liquid extraction** (LLE), or **solvent extraction** (SX), transfers a solute from one liquid phase to a
second liquid phase that is immiscible with the first. The separation rests on the solute having a different
affinity for the two phases, and on different solutes having *different* affinities. In hydrometallurgy, SX is the
standard way to purify and concentrate metals from leach solutions. Examples include copper (hydroxyoxime extractants),
uranium, nickel/cobalt, the PUREX process for spent nuclear fuel (TBP in nitric acid) and, above all, the rare earths. For the
rare earths it is the only practical route to individual high-purity elements at industrial scale, because their
chemistry is so similar that precipitation or crystallisation can only enrich them a little per step.

#### A1.2 The aqueous phase
The **feed** is an acidic aqueous solution of the metals, usually a chloride liquor from leaching the ore
(nitrate and sulfate media are also used). Its important properties are:

* **acidity (pH)**, which controls cation-exchange extraction (A3);
* **total metal concentration** and **composition**, which set the loading and the separation duty;
* **anions and ionic strength**, which change metal activity and can form weak complexes such as $\mathrm{MCl^{2+}}$.

#### A1.3 The organic phase
The **solvent** is a mixture:

| Component | Role | Examples |
|---|---|---|
| Extractant | Reacts with or solvates the metal and carries it into the organic phase | PC88A (P507, HEH/EHP), D2EHPA (P204), Cyanex 272, TBP, Aliquat 336 |
| Diluent | Dissolves the extractant and sets density, viscosity and flash point; must be almost insoluble in water | Aliphatic kerosene |
| Modifier (optional) | Prevents a **third phase** (a heavy, extractant-rich layer) and improves phase separation | Long-chain alcohols (e.g. isodecanol), TBP |

The extractant concentration $C$ (here 0.5 mol/L as monomer HA) sets the solvent's capacity (A7). The diluent
makes the organic lighter than water (density about 0.8 against about 1.0–1.2 for the aqueous), so the organic
floats in the settler.

#### A1.4 Distribution ratio and the single batch contact
At equilibrium a metal M divides between the phases in the **distribution ratio**

$$D = \frac{y}{x} = \frac{[\mathrm{M}]_{org}}{[\mathrm{M}]_{aq}}$$

summing all chemical forms of M in each phase. (The *partition coefficient* is the same idea for one defined species.)
Contact an aqueous volume $V_a$ at concentration $x_0$ with metal-free organic $V_o$. A metal balance and
equilibrium give

$$V_a x_0 = V_a x + V_o y,\qquad y = D x \;\Rightarrow\; x = \frac{x_0}{1+E},\qquad E = D\,\frac{V_o}{V_a}$$

$E$ is the **extraction factor**: the ratio of metal carried in the organic to metal left in the aqueous. The fraction
extracted is $E/(1+E)$. Doubling either $D$ or the phase ratio O/A has the same effect on one contact.

#### A1.5 Extraction mechanisms
* **Cation exchange (acidic extractants).** The extractant's proton is exchanged for the metal ion:
  $\mathrm{M^{3+}} + 3\,\overline{(\mathrm{HA})_2} \rightleftharpoons \overline{\mathrm{M(HA_2)_3}} + 3\mathrm{H^+}$
  (the overbar means the organic phase). Extraction is strongly **pH dependent**. D2EHPA, PC88A and Cyanex 272 work
  this way, and they are the workhorses for rare earths in chloride media. **This lab uses this mechanism.**
* **Solvation (neutral extractants).** A neutral salt is carried by donor molecules, for example
  $\mathrm{M^{3+}} + 3\mathrm{NO_3^-} + 3\,\overline{\mathrm{TBP}} \rightleftharpoons \overline{\mathrm{M(NO_3)_3(TBP)_3}}$.
  Extraction depends on the anion concentration (salting-out), not on pH.
* **Anion exchange (amines, quaternary ammonium).** Anionic complexes such as $\mathrm{M(NO_3)_4^-}$ swap with the
  extractant's anion.
* **Chelation**, in which a ring-forming ligand (for example hydroxyoximes for copper) binds the metal, is a special case of cation exchange.

#### A1.6 Mass transfer between the phases
Equilibrium is the destination. **Mass transfer** sets how fast the system gets there. In the **two-film model**, each phase
has a thin, stagnant film at the interface, and the flux of metal through the aqueous film is
$N = k_a\,(x - x_i)$, through the organic film $N = k_o\,(y_i - y)$, with $y_i = D x_i$ at the interface. Eliminating
the interface values gives one overall coefficient:

$$N = K_a\,(x - y/D),\qquad \frac{1}{K_a} = \frac{1}{k_a} + \frac{1}{D\,k_o}$$

The rate per unit volume of dispersion is $N\,a$, where $a$ is the **interfacial area per unit volume**. For
droplets with volume fraction $\phi$ and Sauter mean diameter $d_{32}$,

$$a = \frac{6\,\phi}{d_{32}}$$

Mixing breaks the dispersed phase into fine droplets. A higher impeller speed gives a smaller $d_{32}$ (in dilute dispersions roughly $d_{32}\propto We^{-0.6}$,
where $We$ is the impeller Weber number) and a larger $a$. Writing the transfer rate as
$V_o(Dx - y)/\tau_{mt}$ lumps $K_a a$ into a single **mass-transfer time constant**
$\tau_{mt} \sim 1/(K_a a)$, which is the form used in the dynamic model (B6). With acidic extractants, the chemical
reaction at the interface can be slower than diffusion. The heavier lanthanides in particular can be slow, and that
also shows up as a longer $\tau_{mt}$.

> **Worked example.** With $\phi = 0.5$ and $d_{32} = 0.2$ mm, $a = 6(0.5)/(2\times10^{-4}) = 15\,000\ \mathrm{m^2/m^3}$.
> A $\tau_{mt}$ of 12 s then corresponds to an overall coefficient of about
> $1/(12 \times 15\,000) \approx 6\times10^{-6}$ m/s. Agitation that halved $d_{32}$ would roughly halve $\tau_{mt}$.

#### A1.7 Dispersion, phase separation and practical problems
After mixing, the dispersion must **disengage**. The droplets settle and then coalesce into a continuous layer. A
small droplet's settling velocity follows **Stokes' law**:

$$v = \frac{\Delta\rho\, g\, d^2}{18\,\mu_c}$$

Here $\mu_c$ is the viscosity of the continuous phase. For $\Delta\rho = 200$ kg/m³ in water, a 0.2 mm drop settles at about
4 mm/s, but a 0.05 mm drop settles 16 times slower. **Intense mixing helps mass transfer and hurts phase separation**,
which is the central design compromise of a mixer-settler. In the settler the dispersion forms a **dispersion band**
whose thickness grows with the flow per unit settler area, which is how settlers are sized.

* **Phase continuity.** Either phase can be continuous in the mixer. Organic-continuous operation is often chosen to
  limit losses of expensive organic in the raffinate. It is held by recycling one phase inside the mixer so the
  local O/A stays on the right side of **phase inversion**.
* **Entrainment** is droplets of one phase leaving with the other. It causes solvent loss and cross-contamination of streams.
* **Crud** is a stable emulsion stabilised by fine solids or degradation products. It collects at the interface.
* **Third phase** is a separate extractant-rich layer that can form at high loading. Modifiers suppress it.

The models in this lab assume perfect phase separation, so none of these effects appear.

#### A1.8 Contacting patterns and equipment
* **Single contact (batch).** Recovery is $E/(1+E)$ (Lab 1).
* **Cross-current.** The same aqueous meets fresh solvent repeatedly. Recovery is high, but the solvent use is large
  and the extract dilute.
* **Counter-current.** The phases move through a train of stages in opposite directions. The leanest aqueous meets the
  freshest solvent, so the driving force is kept in every stage. This gives the highest recovery for a given solvent
  flow (A6, A7, Labs 3–5) and is universal in industry.

| Equipment | Residence time | Inventory | Typical use |
|---|---|---|---|
| **Mixer-settler** (a pump-mixer plus a gravity settler per stage) | Minutes in the mixer, many minutes in the settler | Large | Rare earths, copper and most metals SX. It is robust, easy to stage (often tens to over a hundred stages for rare earths) and tolerant of flow upsets |
| Pulsed or agitated column | Minutes | Medium | Nuclear reprocessing, some metals; small floor area |
| Centrifugal contactor | Seconds | Very small | Nuclear and research; fast, low inventory, but costly |

#### A1.9 A complete SX circuit
A working circuit is a closed loop for the solvent:

1. **Extraction.** The loaded organic takes up the more extractable metals, and the aqueous raffinate leaves with the less extractable ones.
2. **Scrubbing.** A small aqueous stream, often dilute acid or a portion of pure product solution, washes weakly held
   metals out of the loaded organic and back towards extraction. Scrubbing is what makes a sharp **two-product
   separation** possible (A8).
3. **Stripping.** Strong acid (for example several mol/L HCl) reverses the extraction reaction and recovers the
   extracted metals as a concentrated solution. That solution is then usually precipitated as oxalate or carbonate
   and calcined to oxide.
4. **Regeneration and saponification.** The stripped organic, now back in its acid form, is often partly
   **saponified**: pre-neutralised with NaOH or ammonia, $\overline{\mathrm{HA}} + \mathrm{Na^+} \rightarrow \overline{\mathrm{NaA}} + \mathrm{H^+}$.
   The metal then exchanges for $\mathrm{Na^+}$ instead of $\mathrm{H^+}$, so the aqueous does not acidify as metal loads,
   and both loading and $D$ stay high. Ammonia-based saponification produces ammonium effluent, an environmental
   issue that drives alternative schemes.

This lab models only **extraction + scrubbing at fixed pH**. Fixed pH is the idealisation of perfect saponification
or acid control, and the solvent is assumed to arrive metal-free from a perfect stripping section.

#### A1.10 How a rare-earth refinery is organised
A mixed rare-earth feed is separated by a sequence of **binary cuts**. Each cascade divides the feed between two
**key elements** that are neighbours in the extraction order. For example, a first cascade might split light rare earths
(La–Nd) from medium and heavy ones (Sm onwards) at the Nd|Sm cut, and later cascades take Pr|Nd, La|Ce and so on. The
lab's six pairs are such cuts. Each cut is designed with the tools in A5–A8.

### A2 · Rare-earth chemistry that matters for separation
* **Trivalent and alike.** La–Lu and Y all form $\mathrm{M^{3+}}$ in acid solution with very similar chemistry.
* **Lanthanide contraction.** From La to Lu the $4f$ shell fills while shielding the nucleus poorly, so the ionic radius
  shrinks steadily (about 1.03 Å for La³⁺ to about 0.86 Å for Lu³⁺ in six-fold coordination). A smaller ion has a
  higher charge density and binds the hard oxygen donor atoms of organophosphorus extractants more strongly. $D$ therefore
  rises along the series, La < Ce < Pr < Nd < (Pm) < Sm < …, by a factor of order 1.5–4 per neighbour pair.
* **Yttrium** is not a lanthanide, but its ionic radius is close to that of holmium. It therefore extracts among the heavy
  rare earths, far beyond Sm (the lab uses Y/Sm = 40).
* **Redox exceptions.** Cerium can be oxidised to $\mathrm{Ce^{4+}}$ and europium reduced to $\mathrm{Eu^{2+}}$. The
  changed charge alters their chemistry drastically and gives an easy separation route (A4).

### A3 · Extraction equilibrium of acidic organophosphorus extractants  → *Lab 1*
**Derivation.** For the cation-exchange reaction, the equilibrium constant (with activities taken as concentrations) is

$$K_{ex} = \frac{[\overline{\mathrm{M(HA_2)_3}}]\,[\mathrm{H^+}]^3}{[\mathrm{M^{3+}}]\,[\overline{(\mathrm{HA})_2}]^3}
\quad\Rightarrow\quad D = K_{ex}\,\frac{[\overline{(\mathrm{HA})_2}]^3}{[\mathrm{H^+}]^3}$$

$$\log_{10} D = \log_{10}K_{ex} + 3\log_{10}[\overline{(\mathrm{HA})_2}] + 3\,\mathrm{pH}$$

* **Slope 3 with pH.** One pH unit changes $D$ by $10^3$. Acid therefore drives the reaction backwards, which is why
  scrubbing uses a lower pH and stripping uses strong acid.
* **Slope 3 with extractant.** At low loading, $[\overline{(\mathrm{HA})_2}] \approx C/2$, so $\log_{10}D$ rises by 3 per decade of $C$.
* **Loading.** Each metal ion takes three dimers (six HA). Free dimer falls as metal loads,
  $[\overline{(\mathrm{HA})_2}] = (C - 6y)/2$, so $D = D_0\,(1 - 6y/C)^3$, and loading stops at $y = C/6 = Q$.
  The lab captures this limit with the simpler isotherm of A7.
* **Selectivity.** For two metals at the same pH and extractant level, all the terms except $K_{ex}$ cancel:
  $\beta = D_2/D_1 = K_{ex,2}/K_{ex,1}$. The separation factor is set by chemistry alone (A5).

**The lab's form.** Relative to a reference state (pH 2.0, $C$ = 0.5 mol/L):

$$\log_{10} D = \mathrm{idx} + 3\,(\mathrm{pH} - 2.0) + 3\,\log_{10}\!\left(\frac{C}{0.5}\right)$$

| Element | La | Ce | Pr | Nd | Sm | Y |
|---|---|---|---|---|---|---|
| idx = $\log_{10}D$ at reference, relative to Nd | −1.0792 | −0.4771 | −0.1761 | 0 | +0.8451 | +2.4472 |
| $D$ at pH 2.3 | 0.6619 | 2.648 | 5.295 | 7.943 | 55.60 | 2224 |

**What the model leaves out:** activity coefficients (high ionic strength), aqueous chloride complexation,
temperature (extraction is exothermic or endothermic depending on system), and the change of extractant aggregation at
high loading.

### A4 · Single-stage extraction, pH50 and the Ce(IV) switch  → *Lab 1*
From A1.4, one contact extracts $100\,E/(1+E)$ %, with $E = D\cdot O/A$. Plotted against pH, this is an S-curve that
rises from 10 % to 90 % over about 0.64 pH units (E goes from 1/9 to 9, a factor of 81, and $\log D$ has slope 3). Its midpoint, the **pH of half
extraction** at O/A = 1, is

$$\mathrm{pH}_{50} = 2 - \frac{\mathrm{idx}}{3} - \log_{10}\!\left(\frac{C}{0.5}\right)$$

At another phase ratio the midpoint moves by $-\log_{10}(O/A)/3$, because 50 % needs $D\cdot O/A = 1$. Neighbouring
curves are separated horizontally by $\Delta\mathrm{idx}/3$ pH units: about 0.06 pH for Pr/Nd but 0.53 pH for Sm/Y.

> **Worked example: why one stage is not enough.** Take a 50:50 Pr/Nd feed at pH 2.0 and O/A = 1. Then $D_{Nd} = 1$
> and $D_{Pr} = 0.667$, so 50 % of the Nd and 40 % of the Pr are extracted. The extract is only
> 50/(50 + 40) = **55.6 % Nd**, and the raffinate is 54.5 % Pr. Reaching 99.9 % needs many stages (A5–A8).

**Ce(IV) switch (illustrative).** Oxidised cerium is far more extractable. Adding +4 to $\log_{10}D_{Ce}$ moves the
Ce curve $4/3$ pH units to the left, past Pr and Nd. Ce can then be removed in a few stages, and the remaining
La/Pr/Nd separation becomes easier. Real oxidation chemistry and Ce(IV) stability are not modelled.

### A5 · Separation factor and the Fenske equation  → *Lab 2*
The **separation factor** $\beta = D_{heavy}/D_{light} = 10^{\Delta\mathrm{idx}}$ measures how much one ideal stage
enriches the heavy element relative to the light one.

| Pair | Ce/La | Pr/Ce | Nd/Pr | Sm/Nd | Y/Sm |
|---|---|---|---|---|---|
| β | 4 | 2 | 1.5 | 7 | 40 |
| Source | US 10,227,676 | US 10,227,676 | US 10,227,676 | teaching estimate | teaching estimate |

The literature search found PC‑88A Nd/Pr ≈ 1.4, consistent with 1.5, and showed that β depends on the medium,
pH and loading (Ce/La from about 4 to over 50). No usable PC88A Sm/Nd or Y/Sm was found, so those two are hatched in
Lab 2 and remain estimates. No idx value was changed.

**Derivation of the Fenske equation.** In an ideal stage the outgoing phases are in equilibrium, so their
heavy/light ratios differ by exactly β: $(y_h/y_l) = \beta\,(x_h/x_l)$. At **total reflux** (all product returned,
the limit of infinite internal flow), the stream leaving one stage has the same composition as the other phase
entering it. The ratio is therefore multiplied by β at every stage. Over $N$ stages,

$$\left(\frac{h}{l}\right)_{\text{heavy end}} = \beta^{N}\left(\frac{h}{l}\right)_{\text{light end}}$$

For purity $P$ in both products, the heavy end has $h/l = P/(1-P)$ and the light end $h/l = (1-P)/P$, so

$$N_{min} = \frac{\ln\!\big[(P/(1-P))^2\big]}{\ln \beta}\qquad\Big(P = 0.999:\ N_{min} = \frac{\ln 999^2}{\ln\beta}\Big)$$

This is the **fewest** stages possible. Any real flowsheet needs more.

> **Worked example.** At P = 99.9 %, Ce/La (β = 4) needs at least 9.96 stages, Nd/Pr (β = 1.5) 34.1 and Y/Sm (β = 40) 3.74.
> The cost of a separation scales with $1/\ln\beta$, which is why Pr/Nd is the most expensive cut in the lab.

With the Ce(IV) switch on, Pr/Ce falls below 1 because Ce now extracts first. Lab 2 then plots
$\max(\beta, 1/\beta)$ and labels the pair "inverted".

### A6 · Counter-current extraction: the Kremser equation  → *Lab 3*
**Set-up.** There are $N$ ideal stages. The aqueous enters stage 1 at $x_0 = x_F$ and leaves stage $N$ as raffinate $x_N$.
Fresh solvent enters stage $N$ ($y_{N+1} = 0$), and the flows $A$, $O$ and $D$ are constant.

**Derivation.** A balance on stage $n$ is $A x_{n-1} + O y_{n+1} = A x_n + O y_n$. With $y = Dx$ and $E = DO/A$:

$$x_{n-1} - (1+E)\,x_n + E\,x_{n+1} = 0$$

This linear difference equation has characteristic roots $r = 1$ and $r = 1/E$, so $x_n = c_1 + c_2 E^{-n}$.
Applying $x_0 = x_F$ and $x_{N+1} = 0$ (no metal in the fresh solvent) gives the **Kremser equation**:

$$\varphi = \frac{x_N}{x_F} = \frac{E-1}{E^{N+1}-1}\qquad\Big(\varphi = \tfrac{1}{N+1}\ \text{at}\ E = 1\Big),\qquad \text{recovery} = 1 - \varphi$$

* $E > 1$: recovery approaches 1 geometrically as stages are added.
* $E < 1$: as $N\to\infty$, $\varphi \to 1 - E$, so recovery can never exceed $E$. The solvent cannot carry more
  metal than its flow allows: $O\,y_1 \le O\,D\,x_F = E\,A\,x_F$. This is the **ceiling** (dashed lines in Lab 3).
* **Separation by staging.** Two metals with $E_{heavy} > 1 > E_{light}$ are recovered completely and capped at $E$
  respectively. Adding stages sharpens that contrast.

> **Worked example.** Nd at pH 2.0 and O/A = 1 has $E = 1$, so 5 stages recover 5/6 = 83.3 %. At the same conditions
> La has $E = 0.083$ and can never exceed 8.3 % extracted, however many stages are used.

### A7 · Loading limits and the McCabe–Thiele method  → *Lab 4*
Kremser needs a constant $D$. When the solvent approaches saturation, $D$ falls, and a **graphical stage-by-stage**
method is needed. The lab uses a Langmuir-type isotherm with the stoichiometric limit $Q = C/6$ (A3):

$$y = \frac{D\,x}{1 + D\,x/Q}$$

It is linear with slope $D$ at low $x$ and flattens to $Q$ (83.3 mmol/L at C = 0.5 M).

* **Operating line.** A balance from stage $n$ to the raffinate end, with fresh solvent, gives
  $A(x_{n-1} - x_N) = O\,y_n$, that is $x_{n-1} = x_N + (O/A)\,y_n$. It is straight, with slope $A/O$ on a $y$-vs-$x$ plot,
  from $(x_N, 0)$ to $(x_F, y_1)$.
* **Stepping.** Each ideal stage is a step: horizontally from the operating line to the isotherm (equilibrium), then
  vertically to the operating line (the balance). The number of steps between $x_F$ and $x_N$ is the number of stages.
* **Minimum solvent.** The operating line cannot cross the isotherm. As O/A falls, the line pivots about $(x_N, 0)$
  until it touches the isotherm at the feed end (a **pinch**), where infinitely many stages are needed:
  $(O/A)_{min} = (x_F - x_N)/y^*(x_F)$.
* **Solution for a given N.** Guess $x_N$, step up $N$ stages to compute $x_0$, and **bisect** on $x_N$ until $x_0 = x_F$.
  This works because $x_0$ increases monotonically with $x_N$.

> **Worked example.** Nd at 40 mmol/L, pH 2.3, O/A = 1 has $D = 7.943$. The raffinate is 6.910, 1.122, 0.152 and
> 0.019 mmol/L for N = 1–4. The isotherm gives $y^*(40) = 66.0$ mmol/L, so complete extraction needs
> $(O/A)_{min} = 40/66.0 = 0.61$. Below it, even 30 stages leave Nd in the raffinate (try O/A = 0.6).

### A8 · The extraction / scrub cascade  → *Lab 5*
To split a binary feed into **two** pure products, the feed enters the **middle** of a cascade:

* **Numbering.** Stages $1..N$, $N = n_{Ext} + n_{Scr}$. The **extraction section** is $1..n_{Ext}$ and the
  **scrub section** is $n_{Ext}+1..N$. The feed is at $f = n_{Ext}$.
* **Flows.** Fresh solvent $S$ enters stage 1 and leaves stage $N$ as the **extract** (heavy product). Scrub liquor
  $W$ enters stage $N$, and the aqueous leaves stage 1 as the **raffinate** (light product). The aqueous flow is
  $L = W + F$ in extraction and $W$ in scrub. S and W are quoted **per unit feed flow**.
* **pH.** $D$ is evaluated at the extraction pH in stages $1..f$ and at the lower scrub pH after.

**Stage balance.** What enters stage $n$ (organic from $n-1$, aqueous from $n+1$, feed) equals what leaves:

$$S\,D(n-1)\,x(n-1) - \big(S\,D(n)+L(n)\big)\,x(n) + L(n+1)\,x(n+1) = -F\,c_F\,[n=f]$$

$c_F$ is that element's feed concentration (**Feed total** × its fraction, set by **Feed heavy fraction**). Each element
gives a **tridiagonal** system, solved by the Thomas algorithm (forward elimination, back substitution, $O(N)$ work).
With constant $D$ the elements do not interact.

**The extraction-factor window.** In stage $n$, $E_n = S D_n/L_n$. Separation requires:

* in the **extraction section**, $E_{heavy} > 1$, so the heavy element is carried up into the organic, and
  $E_{light} < 1$, so the light element is left in the aqueous;
* in the **scrub section**, $E_{light} < 1$, so the scrub liquor can pull the light element out of the organic, and
  $E_{heavy} > 1$, so the heavy element stays in the organic.

Since $E_{light} = E_{heavy}/\beta$, the window between the two is a factor of exactly β wide. For Pr/Nd, β = 1.5 and
the window is narrow. The lower scrub pH and the smaller scrub flow move the scrub-section factors, so the two sections
can be tuned separately.

> **Worked example (Pr|Nd default).** Extraction section: $E_{Pr} = 0.668$, $E_{Nd} = 1.002$. Scrub section:
> $E_{Pr} = 1.079$, $E_{Nd} = 1.619$. Nd sits on the edge in extraction, and Pr is slightly above 1 in scrub. Both
> products are therefore only **79.6 %** pure with 20 stages. Fenske's minimum for 99.9 % is 34 stages at total reflux.

**What Lab 5 reports.**
* **Purity**: light/(light + heavy) in the raffinate and heavy/(light + heavy) in the extract.
* **Recovery**: the fraction of each element's feed reaching its own product. Purity and recovery trade off. Moving S up
  raises heavy recovery but drags light element into the extract.
* **Profiles**: both elements peak near the feed stage, above the feed concentration. Metal recirculates: the scrub
  returns it down, extraction picks it up again, and an internal inventory builds up. This inventory is what makes
  start-up slow (B8).
* **Fenske $N_{min}$** as a benchmark, and the **extraction-factor table**.
* **Loading check**: if the peak total organic loading $\sum_i D_i x_i$ exceeds $Q = C/6$, the constant‑D
  model is invalid and a red warning appears. A higher feed concentration raises loading proportionally.

**Reflux.** Industrial circuits return part of a product as scrub liquor (heavy-product reflux) or recycle loaded
organic (light-product reflux). This raises internal flows towards the Fenske limit and achieves high purity in fewer
stages. The lab has no reflux, so its purities are lower than a refluxed plant would reach with the same stages.

### A9 · Assumptions of Part A
* $D$ follows the slope-3 law with constant separation factors. Activity, complexation, temperature and loading effects
  are ignored, except for the loading check and the McCabe–Thiele isotherm.
* The pH is fixed in each stage (perfect saponification or acid control).
* Stages are ideal: equilibrium is reached and phase separation is perfect, with no entrainment, crud or third phase.
* There is no reflux and no stripping section (the solvent enters metal-free).
* The Sm/Nd and Y/Sm separation factors, the Ce(IV) shift and the 100 mmol/L feed are teaching estimates.
"""
    )
    return


@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[0])
    mo.md(
        r"""
---
## Part B · Process dynamics and control

### B1 · Why processes need control
A process plant is designed for a **steady state**, but it never stays there. Feeds change, pumps drift, acid strength
varies and equipment fouls. **Process control** keeps the plant at its intended operating point despite these
**disturbances**, and moves it smoothly when the operating point itself changes. The objectives, roughly in order of
priority, are:

1. **Safety and environment:** keeping pressures, levels and temperatures within limits and avoiding releases. In SX
   this includes a flammable diluent.
2. **Equipment protection and operability:** no overflowing settlers, no pump cavitation, and stable phase continuity.
3. **Product quality:** each product meets its purity specification.
4. **Production rate and economics:** maximum throughput, minimum reagent and energy use, minimum off-spec product.

Control is organised as a **hierarchy**. *Measurement and actuation* sit at the bottom. **Regulatory control**
(single-loop PID on flows, levels, pH) responds in seconds to minutes. **Advanced or supervisory control** (composition
control, feedforward, model predictive control) responds in minutes to hours. **Optimisation** chooses setpoints over
hours to days, and **planning and scheduling** works over days to weeks. Each layer gives setpoints to the one below.

**Why an SX plant is a hard control problem.** The lab's Pr|Nd plant holds **960 L** (20 stages × 48 L), equal to 16 h of
feed. It responds over many hours, its main product measurement comes from an analyzer with a dead time, and its two
products cannot both be held by one manipulated variable. An upset that is not corrected can produce a day of off-spec
product, all of which must be reprocessed.

### B2 · Terminology, instrumentation and block diagrams
| Term | Meaning | In this lab |
|---|---|---|
| **Controlled variable (CV)**, process variable (PV) | The output to be held at a target | Heavy-element share (%) in the raffinate |
| **Setpoint (SP)** | The target for the CV | Its nominal steady value, 21.9 % for Pr|Nd |
| **Manipulated variable (MV)**, controller output (CO) | The input the controller adjusts | Solvent flow S |
| **Disturbance variable (DV)** | An input that upsets the CV and is not adjusted by the controller | Feed flow, feed composition, extraction pH, scrub flow |
| **Error** | $e = \text{PV} - \text{SP}$ here (sign conventions vary) | Positive means too much heavy in the raffinate |
| **Sensor / transmitter** | Measures the CV and sends a signal (4–20 mA or digital) | On-line analyzer (AT) with a dead time |
| **Final control element** | Physically changes the MV | Solvent pump with variable-speed drive, under flow control (FIC) |

On a **P&ID** (piping and instrumentation diagram), instruments carry ISA tags. The first letter is the measured
variable and the following letters are the functions: **FT/FIC** (flow transmitter / indicating controller), **LT/LIC**
(level), **AT/AIC** (analyzer, here composition), **pHT/pHIC**. The lab's loop would be drawn as
**AIC → (setpoint of) FIC → solvent pump**.

**Open versus closed loop.** In **open loop** (manual), the MV is fixed and disturbances pass straight to the CV. In
**closed loop**, the measured CV is compared with the setpoint and the MV is adjusted automatically. This is **feedback**.

**The feedback block diagram.** Write the process as $Y = G_p U + G_d D$, the controller as
$U = G_c\,(Y_{sp} - G_m Y)$ (the sign of the controller's action is absorbed into $G_c$) and the measurement
as $G_m$ (here a dead time). Eliminating $U$ gives

$$Y = \underbrace{\frac{G_c G_p}{1 + G_c G_p G_m}}_{\text{servo}}\,Y_{sp} \;+\; \underbrace{\frac{G_d}{1 + G_c G_p G_m}}_{\text{regulator}}\,D$$

The **loop transfer function** $L = G_c G_p G_m$ appears in both denominators. When $|L|$ is large, disturbances are
divided down and the output tracks the setpoint. When $1 + L$ approaches zero, the loop is near instability (B11).
Following a changed setpoint is the **servo** problem. Rejecting disturbances at a fixed setpoint is the **regulator**
problem, which is the one studied in the lab.

**Action (sign).** Here more solvent means less heavy element in the raffinate, so the process gain is negative.
The controller must then *increase* its output when the CV *rises*. In industrial terms this is a **direct-acting**
controller on a negative-gain process, and the net loop is negative feedback. Getting this sign wrong turns
feedback into positive feedback, and the loop runs away.

### B3 · Dynamic modelling: balances, deviation variables and linearisation
Dynamic models come from **conservation balances** written with an accumulation term:

$$\frac{d}{dt}(\text{inventory}) = \text{in} - \text{out} + \text{generation}$$

**Example: one perfectly mixed tank.** Volume $V$, flow $q$, inlet concentration $c_{in}$:
$V\,\dot c = q\,(c_{in} - c)$. At steady state $\bar c = \bar c_{in}$. Defining **deviation variables**
$c' = c - \bar c$ and $c_{in}' = c_{in} - \bar c_{in}$ gives $\tau\,\dot c' + c' = c_{in}'$, with time constant $\tau = V/q$
(the residence time). This is the **first-order system**, the building block of almost every process model.

**Linearisation.** Most models are nonlinear. Near a steady state, a Taylor expansion keeps the first-order terms:

$$\dot x = f(x,u) \;\approx\; \underbrace{\frac{\partial f}{\partial x}\Big|_{ss}}_{A}\,x' + \underbrace{\frac{\partial f}{\partial u}\Big|_{ss}}_{B}\,u'$$

In the mixer-settler model (B7), every equation is linear in the concentrations, but the flow S multiplies the
concentrations ($S\,y$ terms). The model is therefore **bilinear** in the MV. Its response to S depends on direction
and size. This is why a +5 % and a −5 % step in S give different gains in Lab 8 (−20.35 and −22.75 points per L/min).

**Degrees of freedom.** Each MV used for control removes one degree of freedom. One MV (S) can hold **one** CV, so
with only S the plant cannot hold both product purities (B13, Lab 10).

### B4 · Transfer functions and the standard dynamic responses
The **Laplace transform** $Y(s) = \int_0^\infty y(t)\,e^{-st}dt$ turns linear ODEs in deviation variables into
algebra, because $\mathcal{L}\{\dot y\} = sY(s)$ when the system starts at steady state. The **transfer function**
$G(s) = Y(s)/U(s)$ is the ratio of output to input transforms. The **final value theorem**,
$y(\infty) = \lim_{s\to0} sY(s)$, gives steady-state results directly. The **steady-state gain** is $K = G(0)$.

| System | $G(s)$ | Step response (input step of size $M$) | Notes |
|---|---|---|---|
| First order | $\dfrac{K}{\tau s + 1}$ | $KM\,(1 - e^{-t/\tau})$ | 63.2 % at $t=\tau$, 95 % at $3\tau$, 98 % at $4\tau$ |
| Integrating | $\dfrac{K}{s}$ | $KMt$ (ramps without limit) | Liquid levels, e.g. a settler interface with unequal in and out flows |
| Dead time | $e^{-\theta s}$ | Delayed copy of the input | Transport, sampling, analyzers |
| $n$ lags in series | $\dfrac{K}{(\tau s+1)^n}$ | S-shaped, slow start | Stages, tanks in series, cascades |
| Second order | $\dfrac{K}{\tau^2 s^2 + 2\zeta\tau s + 1}$ | Oscillates if $\zeta < 1$; overshoot $e^{-\pi\zeta/\sqrt{1-\zeta^2}}$, decay ratio = overshoot² | Closed loops with integral action behave like this |
| Inverse response | Zero with positive real part | Starts the wrong way | Two competing effects |

**Poles and stability.** The roots of the denominator of $G(s)$, the **poles**, are the eigenvalues of the system's
state matrix. A linear system is stable if every pole has a negative real part. Each pole $p$ contributes a mode
$e^{pt}$ with time constant $1/|p|$, so the **slowest pole dominates** the long-term response.

**The FOPDT approximation.** High-order systems such as many stages in series are usually approximated by a
**first-order-plus-dead-time** (FOPDT) model:

$$G(s) \approx \frac{K\,e^{-\theta s}}{\tau s + 1}$$

The long lags become $\tau$, and the short lags plus true delays become an **apparent dead time** $\theta$. By
**Skogestad's half rule**, with lags $\tau_1 \ge \tau_2 \ge \dots$, take $\tau \approx \tau_1 + \tau_2/2$ and
$\theta \approx \theta_0 + \tau_2/2 + \sum_{i\ge3}\tau_i$. The ratio $\theta/\tau$ measures how hard a loop is to control:
small values ("lag-dominant") are easy, large values ("dead-time-dominant") are hard. The lab's plant has
$\theta/\tau \approx 24\ \text{min}/7\ \text{h} \approx 0.06$, which is easy *if* the dead time is short compared with the
disturbances.

> **Worked example: the plant's poles.** At the Pr|Nd defaults, each element's linear model has 80 poles. They range
> from about −13.5 /min (mass transfer in the mixers, time constant about 4 s) to about −0.0021 /min (time constant
> about **8 h**, the slow redistribution of metal along the whole cascade). The ratio of the extremes, the **stiffness
> ratio**, is about 5,200. The dominant 7–8 h mode is what Lab 8's step test measures as $\tau$.
"""
    )
    return


@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[0])
    mo.md(
        r"""
### B5 · Mixer-settler hydraulics  → *Lab 6*
A mixer-settler stage has a small agitated **mixer** (volume $V_{mix}$) and a larger, quiet **settler**
($V_{set}$). The organic flows from settler $n$ to mixer $n+1$, and the aqueous flows from settler $n$ to mixer $n-1$,
which gives counter-current flow. Each unit's **mean residence time** is its volume over the total flow through it:

$$\tau_{mix} = \frac{V_{mix}}{S + L_n},\qquad \tau_{set} = \frac{V_{set}}{S + L_n}$$

$L_n$ is larger in the extraction section ($W+F$) than in the scrub section ($W$), so scrub stages hold the liquid longer
(the step in Lab 6's bars). S and W from Lab 5 are per unit feed and are multiplied by the **feed flow F**, so doubling F
halves every residence time.

> **Worked example.** F = 1 L/min, S = 2.9, W = 0.45 L/min. In extraction, $S + L = 4.35$ L/min, so
> $\tau_{mix} = 8/4.35 = 1.84$ min and $\tau_{set} = 40/4.35 = 9.20$ min. In scrub, $S + L = 3.35$ L/min, giving 2.39 and
> 11.94 min. Hold-up is $20 \times 48 = 960$ L, and hold-up ÷ F = 16 h.

**Residence-time distribution.** A perfectly mixed vessel has an exponential residence-time distribution,
$E(t) = e^{-t/\tau}/\tau$. Real settlers are closer to **plug flow** (a dead time) with some back-mixing. The lab
treats both units as perfectly mixed (B17).

**Phase volumes and interface level.** Each unit holds both phases. The lab splits volumes in the nominal flow ratio,
with organic fraction $\phi_o = S/(S+L_n)$. In a real settler the **interface level** is set by weirs, and it is an
**integrating** variable: any imbalance between aqueous in and out moves the level steadily. It needs its own level
control (B14), which the lab assumes is perfect.

### B6 · Mass-transfer rate and stage efficiency  → *Lab 6*
Real stages do not reach equilibrium. From A1.6, transfer in the mixer is modelled as **first order**:

$$\text{rate to organic} = V_o^{m}\,\frac{D\,x_m - y_m}{\tau_{mt}}$$

**Derivation of the approach to equilibrium.** Take any quantity $z$ in a well-mixed vessel. It enters at $z_{in}$, is
washed out with residence time $\tau$, and relaxes towards equilibrium $z^*$ with time constant $\tau_{mt}$. At steady state
$0 = (z_{in} - z)/\tau + (z^* - z)/\tau_{mt}$, so the fraction of the possible change achieved is

$$\eta = \frac{z - z_{in}}{z^* - z_{in}} = \frac{\tau_{mix}}{\tau_{mix} + \tau_{mt}}$$

This plays the role of a **Murphree stage efficiency**, $E_M = (y_{out} - y_{in})/(y^* - y_{in})$. At the defaults
$\eta$ = 0.902 in extraction and 0.923 in scrub. Lower efficiency acts like fewer ideal stages: Lab 6's sweep shows both
purities falling from the ideal 79.6 % to 78.1 % (raffinate) and 78.8 % (extract) at $\tau_{mt}$ = 12 s, and further as
$\tau_{mt}$ grows. Larger mixers or more intense mixing raise $\eta$. Larger settlers do not, but they add hold-up and slow
the dynamics.

### B7 · The plant model, eigenvalues and numerical integration  → *Labs 6–10*
Each stage has a mixer ($x_m, y_m$) and a settler ($x_s, y_s$) for each element. The aqueous enters mixer $n$ from settler
$n+1$ plus the feed at stage $f$, and the organic enters from settler $n-1$:

$$V_a^{m}\dot x_m = L_{n+1}x_s^{(n+1)} + F c_F[n=f] - L_n x_m - V_o^{m}\,\frac{D x_m - y_m}{\tau_{mt}}$$
$$V_o^{m}\dot y_m = S\,y_s^{(n-1)} - S\,y_m + V_o^{m}\,\frac{D x_m - y_m}{\tau_{mt}}$$
$$V_a^{s}\dot x_s = L_n (x_m - x_s),\qquad V_o^{s}\dot y_s = S\,(y_m - y_s)$$

**State-space form.** Stacking the $4N$ states gives $M\dot u = -A\,u + b$, with $M$ the diagonal matrix of phase
volumes. Its poles are the eigenvalues of $-M^{-1}A$ (B4).

**Banded structure.** Ordering the unknowns $[x_m, y_m, x_s, y_s]$ stage by stage keeps every non-zero within six
places of the diagonal (**half-bandwidth 6**): mixer $n$ couples to settler $n+1$ at offset +6 and to settler $n-1$ at
offset −2. A banded LU factorisation costs $O(n\,k_l\,k_u)$ instead of $O(n^3)$. It needs no pivoting here because
the matrix is a diagonally dominant **M-matrix** (every outflow is at least as large as the matching inflows).

**Stiffness and the choice of integrator.**
* **Explicit Euler**, $u^{k+1} = u^k + \Delta t\,f(u^k)$, multiplies each mode by $1 + \lambda\Delta t$ per step. It is
  stable only if $|1 + \lambda\Delta t| < 1$, that is $\Delta t < 2/|\lambda_{max}|$. With $|\lambda_{max}| \approx 13.5$ /min
  this means $\Delta t < 0.15$ min, or about 19,500 steps for 48 h, even though nothing of interest happens that fast.
* **Implicit Euler**, $(M/\Delta t + A)\,u^{k+1} = (M/\Delta t)\,u^k + b$, multiplies each mode by $1/(1 - \lambda\Delta t)$.
  That factor is below 1 for every stable $\lambda$ and every $\Delta t$ (**A-stable**). The lab uses $\Delta t$ = 2 min. The fast modes
  are damped rather than resolved, which is harmless because they settle in seconds.
* **Steady state** is a single implicit step with an enormous $\Delta t$, which reduces to $A\,u = b$. As
  $\tau_{mt}\to0$ it reproduces the ideal cascade of A8 (one of the tests on page 3).

### B8 · Start-up dynamics  → *Lab 7*
At $t = 0$ the plant is full of metal-free liquor and the feed is switched on. Metal spreads out from the feed stage, the
internal profiles of A8 build up (Lab 7's snapshots, light to dark with time), and the products slowly approach
specification.

* **Masking.** At first the products carry almost no metal, so their "purity" is meaningless. Each purity is shown only once that
  product carries 5 % of its steady metal flow.
* **Settling time** is the last moment either purity is outside **±1 point** of its final value. The default plant
  settles in **23 h**. That is about 1.4 × (hold-up ÷ F) and about 3 × the dominant 7–8 h time constant (B4): a first-order
  mode reaches 95 % in $3\tau$. The internal metal inventory near the feed must be built up from the feed itself, and that
  is what sets the pace.
* **Practice.** Plants shorten start-up by pre-loading stages with solutions of the right composition, starting at total
  reflux, or recycling off-spec product until the profiles are established.

### B9 · Process identification from step tests  → *Lab 8*
A controller is designed from a **model** of how the CV responds to the MV. The simplest experiment is an **open-loop step
test**:

1. Wait for a steady state.
2. Step the MV by an amount large enough to see clearly above measurement noise, but small enough to stay nearly linear
   (a few %).
3. Record the CV until it settles.
4. Repeat in the opposite direction to reveal nonlinearity.

**Steady-state gain.** $K_p = \Delta y_\infty/\Delta u$. In Lab 8, a +5 % step in S gives −20.35 points per L/min and a
−5 % step gives −22.75. The controller uses the average of the two, −21.55, which equals the central difference of the
±5 % steady states.

**Smith's two-point method.** For an FOPDT model, $y(t) = 1 - e^{-(t-\theta)/\tau}$ (normalised) reaches 28.3 % at
$t = \theta + \tau/3$ and 63.2 % at $t = \theta + \tau$. Subtracting the two:

$$\tau = 1.5\,(t_{63} - t_{28}),\qquad \theta = t_{63} - \tau$$

> **Worked example.** For the +5 % step, $t_{28}$ = 2.55 h and $t_{63}$ = 7.19 h, so $\tau$ = 1.5 × 4.64 = **6.96 h** and
> $\theta$ = 7.19 − 6.96 h = **14 min**. The analyzer adds 10 min, so the controller sees $\theta_{tot}$ = 24 min.

Other methods are the tangent (reaction-curve) method, least-squares fitting, pseudo-random binary input
sequences, and **relay auto-tuning**. In relay auto-tuning, an on/off relay of amplitude $h$ makes the loop oscillate
with amplitude $a$, giving $K_u \approx 4h/(\pi a)$ and the ultimate period $P_u$ directly (B11).
"""
    )
    return


@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[0])
    mo.md(
        r"""
### B10 · Feedback control laws: on/off, P, PI and PID  → *Lab 10*
**On/off control** switches the MV fully on or fully off. It is simple, but the CV cycles constantly. It suits
thermostats, not product quality.

**Proportional (P) control:** $u = \bar u + K_c\,e$. The output is proportional to the error. The
**proportional band** is $PB = 100\%/K_c$ when signals are in % of span. **P control leaves an offset.** For a
disturbance that would move the CV by $\Delta y_{OL}$ in open loop, the final value theorem applied to the regulator
equation of B2 gives

$$\Delta y_{\infty} = \frac{\Delta y_{OL}}{1 + K_c |K_p|} = \frac{\Delta y_{OL}}{1 + LG}$$

so a higher gain reduces the offset but never removes it.

> **Worked example (Lab 10, P only).** The +20 % feed step moves the CV by 6.97 points in open loop. With LG = 2,
> linear theory predicts an offset of 6.97/3 = 2.32 points. The simulation gives **2.55**. The difference comes from the
> bilinear plant (B3): the gain is not quite constant.

**Integral (I) action** adds the accumulated error:

$$u = \bar u + K_c\Big(e + \frac{1}{T_i}\int_0^t e\,dt\Big)\qquad\Longleftrightarrow\qquad G_c(s) = K_c\Big(1 + \frac{1}{T_i s}\Big)$$

The integral keeps moving the MV as long as any error remains, so at steady state $e = 0$: **no offset**. The cost is
phase lag (up to −90° at low frequency), which pushes the loop towards oscillation (B11). A smaller **integral time**
$T_i$ means stronger integral action.

**Derivative (D) action**, $K_c T_d\,de/dt$, anticipates the error from its slope and adds phase lead. It also
amplifies measurement noise, so it needs a filter $T_d s/(\alpha T_d s + 1)$ and is usually applied to the measurement
rather than the error, to avoid a "kick" on setpoint changes. On slow, sampled, noisy analyzer loops like this one, D action
gains little, so **PI is the industrial standard** and is what the lab uses.

**PID forms.** *Ideal (ISA):* $K_c(1 + 1/(T_i s) + T_d s)$. *Series:* $K_c'(1 + 1/(T_i' s))(1 + T_d' s)$. Parameters
are not interchangeable between forms. In a digital system, the **velocity form**
$\Delta u_k = K_c\big[(e_k - e_{k-1}) + (\Delta t/T_i)\,e_k\big]$ is common. Sampling with a zero-order hold adds an
effective dead time of about $\Delta t/2$.

**Loop gain.** The lab specifies the dimensionless **loop gain** $LG = K_c\,|K_p|$ and computes $K_c = LG/|K_p|$ from the
±5 % gain (−21.55 points per L/min). LG 2 gives $K_c$ = 0.093 L/min per point.

**Windup and anti-windup.** When the MV hits a limit (here S may move only ±50 %, the **Solvent-flow limit**), the
integral keeps accumulating error it cannot act on. When the error finally reverses, the swollen integral holds the MV
on the limit too long, causing a large overshoot. Remedies:
* **Conditional integration (clamping)**, used here: freeze the integral while the output is saturated and the error
  would drive it further into saturation.
* **Back-calculation:** feed the difference between the saturated and unsaturated output back into the integrator.

The lab's test with a ±20 % limit and the pH disturbance keeps S on its limit 98 % of the time. The CV cannot return to
setpoint (an offset of about 8 points), but the integral does not wind up. **Bumpless transfer** initialises the integral
when switching from manual to automatic so that the output does not jump.

### B11 · Closed-loop stability and frequency response  → *Lab 9*
**Characteristic equation.** From B2, the closed-loop poles are the roots of $1 + L(s) = 0$, where
$L = G_c G_p G_m$. With dead time this equation is transcendental, so algebraic tests such as Routh's do not apply
directly. **Frequency response** handles it naturally.

**Frequency response.** A stable linear system driven by $\sin\omega t$ settles to a sinusoid of the same frequency,
scaled by $|G(j\omega)|$ (the **amplitude ratio**) and shifted by $\angle G(j\omega)$ (the **phase**):

| Element | Amplitude ratio | Phase |
|---|---|---|
| $K/(\tau s + 1)$ | $K/\sqrt{1 + \omega^2\tau^2}$ | $-\arctan(\omega\tau)$, never below −90° |
| $e^{-\theta s}$ | 1 | $-\omega\theta$, **without limit** |
| PI: $K_c(1 + 1/(T_i s))$ | $K_c\sqrt{1 + 1/(\omega T_i)^2}$ | $-\arctan(1/(\omega T_i))$, between −90° and 0 |

For the lab's loop, $L(j\omega) = LG\,(1 + \frac{1}{j\omega T_i})\,\frac{e^{-j\omega\theta}}{1 + j\omega\tau}$, with
$\theta$ = process dead time + analyzer delay. Magnitudes multiply and phases add. On a **Bode plot** (log magnitude and
phase against log frequency, the two panels of Lab 9), each element's contribution is easy to see.

**Bode stability criterion** (for open-loop-stable loops with monotonic phase): the closed loop is stable if $|L| < 1$ at
the **phase-crossover frequency** $\omega_{180}$, where $\angle L = -180°$. A signal that goes round the loop
inverted (−180°) and amplified (|L| > 1) grows without limit.

* **Gain margin** $GM = 1/|L(j\omega_{180})|$: the factor by which the gain could rise before instability.
* **Phase margin** $PM = 180° + \angle L(j\omega_c)$ at the **gain-crossover frequency** $\omega_c$, where $|L| = 1$:
  the extra lag (for example extra dead time $= PM/\omega_c$) the loop could tolerate.
* A common guideline is **GM ≥ 1.7–2** and **PM ≥ 30–45°**. Larger margins mean more robust but slower control.

**The ultimate gain** is the P-only loop gain that puts the loop exactly at the stability limit. For FOPDT,
$\omega_u$ solves $\omega_u\theta + \arctan(\omega_u\tau) = \pi$. Then $LG_u = \sqrt{1 + \omega_u^2\tau^2}$ and the
**ultimate period** is $P_u = 2\pi/\omega_u$. Dead time is what makes $\omega_u$ finite: without it the phase of a
first-order loop never reaches −180°. **Dead time limits performance**, because achievable bandwidth is roughly
$\omega_c \lesssim 1/\theta$.

> **Worked example.** With $\tau$ = 6.96 h and $\theta_{tot}$ = 24 min, $\omega_u$ = 0.067 rad/min, $LG_u$ ≈ **28** and
> $P_u$ ≈ **94 min**. The default PI (LG 2, $T_i$ 120 min) has GM ≈ 12.7 and PM ≈ 49°, which is very robust.

**Model error, and why it matters.** These margins are computed on the FOPDT *model*. Simulating the full plant with
P-only control shows sustained oscillation only at a loop gain between **60 and 65**, with a period of about 46–48 min.
That is twice the model's ultimate gain and half its period. The real plant has less phase lag at these frequencies than
the FOPDT suggests: the solvent enters stage 1, next to the raffinate outlet, so part of the response is almost
immediate, and Smith's method lumps that fast part into an apparent dead time. Here the model errs on the **safe**
side. In practice the error can go either way, which is why margins, not just nominal performance, guide tuning. One
of the page 3 tests checks that the plant still settles at the model's $LG_u$.

**Other tools.** The **Nyquist criterion** generalises the Bode test to any loop. **Padé approximations**
$e^{-\theta s} \approx (1 - \theta s/2)/(1 + \theta s/2)$ allow root-locus analysis. A **Smith predictor** uses a process
model to remove the dead time from the feedback path when $\theta/\tau$ is large.

### B12 · Controller tuning  → *Labs 8, 9*
Tuning chooses $K_c$ and $T_i$ to balance **speed** (small peak deviation, fast recovery) against **robustness** (margins,
tolerance of model error) and **MV effort** (how hard the pumps and reagents are worked). The classic PI rules for an
FOPDT model ($K$, $\tau$, $\theta$), written as loop gain $LG = K_c|K|$, are:

| Rule | $LG$ | $T_i$ | Character |
|---|---|---|---|
| Ziegler–Nichols, ultimate gain (1942) | $0.45\,LG_u$ | $P_u/1.2$ | Aims at quarter-amplitude decay; aggressive |
| Ziegler–Nichols, reaction curve | $0.9\,\tau/\theta$ | $3.33\,\theta$ | Aggressive |
| Cohen–Coon (1953) | $(\tau/\theta)(0.9 + \theta/12\tau)$ | $\theta\dfrac{30 + 3\theta/\tau}{9 + 20\theta/\tau}$ | Aggressive, for larger $\theta/\tau$ |
| Lambda / IMC | $\tau/(\lambda + \theta)$ | $\tau$ | Smooth; $\lambda$ sets closed-loop speed |
| **SIMC** (Skogestad 2003) | $\tau/(\tau_c + \theta)$ | $\min(\tau,\ 4(\tau_c + \theta))$ | $\tau_c = \theta$ for tight but robust control |

> **Worked example ($\tau$ = 6.96 h, $\theta_{tot}$ = 24 min; margins on the FOPDT model; plant response to feed +20 %).**
>
> | Rule | LG | $T_i$ (min) | GM | PM | Plant peak (points) | Plant IAE (point·h) |
> |---|---|---|---|---|---|---|
> | Default | 2.0 | 120 | 12.7 | 49° | 1.83 | 12.1 |
> | ZN ultimate | 12.6 | 78 | 1.9 | 28° | 0.55 | 1.1 |
> | ZN reaction curve | 15.7 | 80 | 1.5 | 22° | 0.49 | 0.86 |
> | Cohen–Coon | 15.8 | 71 | 1.5 | 19° | 0.48 | 0.78 |
> | Lambda (λ = θ) | 8.7 | 418 | 3.1 | 61° | 0.83 | 6.6 |
> | SIMC | 8.7 | 192 | 3.1 | 53° | 0.77 | 3.1 |
>
> The ZN and Cohen–Coon settings perform best on *this* plant, but only because the real plant is more forgiving than
> its model (B11). On the model their phase margins are below 30°, which is too little for a plant with an uncertain
> analyzer delay. Lambda tuning sets $T_i = \tau$ = 7 h, so it removes the integrated error slowly. That suits setpoint
> changes but not disturbances on a lag-dominant process, and it is exactly why SIMC caps $T_i$ at $4(\tau_c + \theta)$.
> **SIMC is the recommended starting point:** good margins and a peak less than half the default's.
"""
    )
    return


@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[0])
    mo.md(
        r"""
### B13 · Beyond single-loop feedback  → *Lab 10*
Feedback acts only after the CV has already deviated. Better strategies use more information.

**Feedforward control.** If a disturbance $D$ is *measured*, the MV can be moved before the CV is affected. From
$Y = G_p U + G_d D$, perfect cancellation needs

$$G_{ff}(s) = -\frac{G_d(s)}{G_p(s)}$$

In practice a **static** feedforward ($G_{ff}(0)$) or a lead-lag approximation is used, always **combined with
feedback** to correct for model error and unmeasured disturbances.

**Ratio control** is the most common static feedforward. The MV is held in a fixed ratio to a measured flow. In SX the
natural form is **S/F (organic/aqueous) ratio control**: when the feed flow rises, the solvent rises with it, which keeps the
extraction factors $E = SD/L$ roughly constant (the scrub flow does not scale, so the compensation is not perfect).
The lab's mode **PI + feedforward ratio (S/F)** sets $S = S_0\,F/F_0$ instantly, and PI trims the rest.

> **Worked example (feed +20 %).** PI alone: peak 1.83 points, IAE 12.1 point·h. PI + S/F ratio: peak **0.63**,
> IAE **4.6**. For the composition, pH and scrub disturbances the feedforward has nothing to measure, so the result
> equals PI alone. Feedforward only helps with disturbances it can see.

**Cascade control.** The composition controller (master, slow) does not drive the pump directly. It sends a setpoint to
a **flow controller** (slave, fast) on the solvent line, and the slave rejects pump and pressure upsets within seconds,
before they reach the composition. The lab assumes an ideal slave: S equals its setpoint instantly. A cascade
works when the inner loop is several times faster than the outer one.

**Multivariable control.** The plant has two products (raffinate and extract purity) and several possible MVs (S, W,
extraction pH, scrub pH). Each MV affects both products: this is **interaction**. Lab 10 shows it. Holding the raffinate
with S makes extract purity fall from 78.8 % to 73.6 % after the feed step. Tools:
* The **relative gain array (RGA)**: for a 2×2 system, $\lambda_{11} = 1/(1 - K_{12}K_{21}/(K_{11}K_{22}))$. Pair each MV with
  the CV whose $\lambda$ is closest to 1, and avoid negative $\lambda$.
* **Decouplers**, which are feedforward from one controller's output to the other's MV.
* **Model predictive control (MPC)**, which uses a dynamic model to optimise all MV moves over a horizon while respecting
  constraints such as pump limits and the loading limit $Q$. MPC is the industrial standard for interacting, constrained
  plants.

**Inferential and sensitive-stage control.** A product analyzer sees the effect of an upset only after it has travelled
through the cascade. Measuring the composition at an **intermediate stage**, where the profile is steep and responds
early, gives a faster CV. This is the SX analogue of controlling a sensitive tray temperature in distillation, with the
product analyzer trimming the setpoint. **Soft sensors** infer composition from faster measurements such as flows, pH
and absorbance.

**Other structures.** **Override (selector) control** lets a constraint controller take over the MV near a limit, for
example to keep organic loading below $Q$. **Gain scheduling** changes controller gains with the operating point to
handle nonlinearity.

### B14 · Instrumentation and control of real SX plants
* **Flow control on every stream.** Magnetic or Coriolis flow meters with pump speed control (FIC) on feed, solvent,
  scrub and strip. O/A ratio controllers tie solvent to feed.
* **Acidity.** pH is a strongly nonlinear variable: the titration curve is steep near neutrality. Electrodes also foul in
  organic-contaminated streams. Plants often control the **saponification degree** (alkali flow in ratio to solvent flow) and the
  acid strength of scrub and strip liquors instead of measuring pH in every stage. The lab's fixed pH per stage idealises
  this layer.
* **Settler interface level.** This is an integrating variable, controlled by adjustable weirs or by the aqueous
  outflow. Losing it sends one phase out with the other.
* **Mixer phase continuity**, held by internal recycle, and **temperature**, because $D$ and phase separation both depend
  on it.
* **Composition analyzers.** These include on-line X-ray fluorescence, and visible-light spectrophotometry: Pr³⁺ and Nd³⁺
  absorb at characteristic wavelengths, which makes on-line monitoring of Pr/Nd circuits possible. Laboratory ICP analysis
  is slow. Sample lines and analysis cycles cause the **analyzer dead time** in Lab 9 and Lab 10.
* **Solvent inventory and quality:** make-up, entrainment losses and degradation.
* **Alarms, interlocks and safety systems.** The diluent is flammable, so ventilation, low static generation and
  fire protection are part of the design.

#### Reading the lab's P&ID  → *Lab 11*
A **piping and instrumentation diagram** (P&ID) shows the process equipment, the process streams and every instrument
and control loop. Its conventions follow ISA-5.1:

| Symbol | Meaning |
|---|---|
| Solid line with arrow | Process stream (organic to the right, aqueous to the left) |
| Dashed line | Instrument signal (measurement or controller output) |
| Circle ("bubble") with a horizontal line | Instrument function mounted in the control room / DCS |
| Tag letters | First letter = measured variable (F flow, A analysis, L level, pH); next letters = function (T transmitter, I indicator, C controller, Y relay/computation) |
| Tag number | Loop number: every instrument in one loop shares it (AT-101 → AIC-101 → FIC-101) |
| Bow-tie | Control valve, or a pump speed drive acting as the final control element |

**Loops drawn in Lab 11:**

* **FIC-101 (solvent flow)** is the fast inner loop. It holds S at its setpoint by adjusting the solvent valve or pump.
* **AT-101 → AIC-101 → FIC-101** is the composition loop. The analyzer on the raffinate (with its dead time) feeds the
  PI controller, whose output is the **setpoint** of FIC-101. This is **cascade control** (B13). In *MANUAL* the AIC is
  greyed and S stays fixed, which is how start-up and the open-loop disturbance are run.
* **FT-102 → FY-102 → FIC-101** is the feed-ratio feedforward. FY-102 multiplies the measured feed flow by the S/F ratio.
  It is active (black) only in the "PI + feedforward ratio" mode.
* **FIC-103** holds the scrub flow W. In the "Scrub flow −20 %" disturbance it is the scrub flow itself that is
  stepped, as if this loop's setpoint had been changed wrongly.
* **pHIC-201/202** hold the acidity of the extraction and scrub sections. **LIC** loops (one per settler, drawn once)
  hold each interface level. The lab treats both as perfect (B14, B17).

**The stage symbols are live.** Each stage is drawn as a settler with its organic layer on top and its aqueous layer
below. The bars show the light and heavy element concentrations leaving that stage. Change nExt or nScr in section 5 and the
number of stages redraws. ▲/▼ marks stages whose metal content is still rising or falling by more than 2 % per hour.
During start-up (B8) the arrows show the profile filling outward from the feed stage. After a disturbance (B15) they show
the upset travelling through the cascade, and in closed loop the controller pulling it back.

### B15 · The disturbances in this lab  → *Lab 10*
Each is applied as a step at $t = 0$ from steady state.

| Disturbance | Physical effect | Open loop (Pr|Nd) | Can feedforward help? |
|---|---|---|---|
| **Feed flow +20 %** | More aqueous flow and metal in extraction, so $E = SD/L$ falls | Heavy share in raffinate 21.9 → 28.9 % | Yes: feed flow is measured (S/F ratio) |
| **Feed composition 50:50 → 60:40** | More heavy element at the same flows | → about 29.5 % | Only with a feed analyzer |
| **Extraction pH −0.05** | $D$ falls by $10^{0.15} \approx 1.41$ in every extraction stage | → about 37 %, the largest upset | Better acidity control is the real fix |
| **Scrub flow −20 %** | Less wash liquor, so less metal is returned from the organic | Raffinate *cleaner* (about 18 %), extract dirtier | The scrub flow controller should prevent it |

The last row shows why a single CV can mislead: the controlled raffinate improves while the uncontrolled extract
degrades.

### B16 · Measuring control performance  → *Lab 10*
| Measure | Definition | What it rewards |
|---|---|---|
| Peak deviation | $\max |y - y_{sp}|$ | Fast initial response |
| IAE | $\int |e|\,dt$ (point·h) | Small and short deviations; the lab's main index |
| ISE | $\int e^2\,dt$ | Penalises large errors most |
| ITAE | $\int t\,|e|\,dt$ | Penalises errors that persist |
| Offset | $y(\infty) - y_{sp}$ | Zero with integral action, unless the MV saturates |
| Overshoot, decay ratio, settling time | From the shape of the response (B4) | Smoothness |
| MV effort and time on a limit | Movement of S and saturation | Gentle use of equipment; feasibility |
| Other product | Extract purity under control | Exposes interaction (B13) |

### B17 · Assumptions of Part B
* The pH is fixed in each stage, and $D$ is constant: the model is linear in concentrations and bilinear in flows.
* Mixers and settlers are perfectly mixed. There is no plug flow in settlers, no hydraulic (level) dynamics, no transfer in
  settlers, and phase volumes stay fixed at the nominal split.
* $\tau_{mt}$ = 12 s, the 8 L mixer and 40 L settler, the ±50 % MV limit and the analyzer (a pure dead time, continuous,
  noise-free) are **teaching estimates**.
* The flow slave loop is ideal, feedforward is static, and the controller acts every 2‑min integration step.
* The process gain comes from steady-state perturbation and the FOPDT model from Smith's method. Both approximate a
  nonlinear, 80-state plant. The frequency-response margins are therefore model-based (see B11 for how far they differ).
* Implicit Euler with 2‑min steps is accurate for the slow dynamics but damps anything faster than a few minutes.
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
def _(CONTROL_MODES, DISTURBANCES, mo):
    sl_F = mo.ui.slider(0.2, 5.0, 0.1, value=1.0, label="Feed flow F (L/min)", show_value=True, debounce=True)
    sl_Vm = mo.ui.slider(1, 50, 1, value=8, label="Mixer volume (L)", show_value=True, debounce=True)
    sl_Vs = mo.ui.slider(5, 200, 5, value=40, label="Settler volume (L)", show_value=True, debounce=True)
    sl_tmt = mo.ui.slider(1, 120, 1, value=12, label="τ_mt (s)", show_value=True, debounce=True)
    dd_dist = mo.ui.dropdown(list(DISTURBANCES), value="Feed flow +20%", label="Disturbance")
    sl_LG = mo.ui.slider(0.25, 5.0, 0.25, value=2.0, label="Loop gain LG", show_value=True, debounce=True)
    sl_Ti = mo.ui.slider(10, 600, 10, value=120, label="Integral time Ti (min)", show_value=True, debounce=True)
    sl_delay = mo.ui.slider(0, 60, 2, value=10, label="Analyzer delay (min)", show_value=True, debounce=True)
    sl_limit = mo.ui.slider(10, 100, 5, value=50, label="Solvent-flow limit (± % of nominal)", show_value=True, debounce=True)
    sl_step = mo.ui.slider(-20, 20, 1, value=5, label="Step in S (% of nominal)", show_value=True, debounce=True)
    dd_mode = mo.ui.dropdown(CONTROL_MODES, value="PI", label="Controller mode")
    return (
        dd_dist,
        dd_mode,
        sl_F,
        sl_LG,
        sl_Ti,
        sl_Vm,
        sl_Vs,
        sl_delay,
        sl_limit,
        sl_step,
        sl_tmt,
    )


@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[1])
    mo.md(
        "## 2 · Interactive lab\n"
        "The lab has two halves of equal weight. **Part A, Separations** (sections 1–5) goes from one equilibrium "
        "contact to a full extraction/scrub cascade. **Part B, Process dynamics and control** (sections 6–11) turns that "
        "cascade into a mixer-settler plant and controls it. Every chart updates live. Each section has a matching "
        "concept section on page 1, and the code is on page 3.\n\n"
        "---\n## Part A · Separations"
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
    _efac = [{"Section": _sec, f"E {_l} (light)": round(float(sl_S.value * cas[_l]["D"][_i] / cas["L"][_i]), 3),
              f"E {_h} (heavy)": round(float(sl_S.value * cas[_h]["D"][_i] / cas["L"][_i]), 3)}
             for _sec, _i in (("Extraction", 0), ("Scrub", cas["N"] - 1))]
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
        mo.ui.table(_efac, selection=None, label="Extraction factors E = S·D/L (separation needs heavy > 1 > light)"),
        _fig,
    ])
    return


# ============================================================================
# PAGE 2, PART B: Process dynamics and control (sections 6-9)
# ============================================================================
@app.cell(hide_code=True)
def _(PAGES, mo, nav):
    mo.stop(nav.value != PAGES[1])
    mo.md(
        "---\n## Part B · Process dynamics and control\n"
        "Sections 6–11 build the section 5 cascade as a plant of real mixer-settlers. They use the same pair, stages, "
        "S and W (scaled by the feed flow F) and pH values. You can then watch the plant fill, find "
        "its step response, tune a PI controller and reject disturbances. The concepts are in **Part B** of page 1."
    )
    return


# --------------------------------------------------- Section 6: hydraulics and stage efficiency
@app.cell(hide_code=True)
def _(
    PAGES,
    dd_pair,
    make_plant,
    mo,
    nav,
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
    # S and W from section 5 are per unit feed; scale them by F (L/min).
    plant = make_plant(dd_pair.value, sl_next.value, sl_nscr.value, sl_S.value * sl_F.value,
                       sl_W.value * sl_F.value, sl_pHe.value, sl_pHs.value, sl_F.value,
                       float(sl_Vm.value), float(sl_Vs.value), sl_tmt.value / 60.0,
                       float(sl_cF.value), sl_fh.value)
    return (plant,)


@app.cell(hide_code=True)
def _(
    ACCENT1,
    ACCENT2,
    COLORS,
    INK,
    INK2,
    efficiency_sweep,
    label_axes,
    mo,
    new_figure,
    np,
    plant,
    sl_F,
    sl_Vm,
    sl_Vs,
    sl_tmt,
):
    _N = plant["N"]
    _st = np.arange(1, _N + 1)
    _f1, (_a1,) = new_figure(1, 9.5, 3.0)
    _a1.bar(_st - 0.2, plant["tau_mix"], width=0.38, color=ACCENT1, edgecolor="white", lw=1, label="τ_mix (mixer)")
    _a1.bar(_st + 0.2, plant["tau_set"], width=0.38, color=ACCENT2, edgecolor="white", lw=1, label="τ_set (settler)")
    _a1.axvline(plant["n_ext"] + 0.5, color=INK2, lw=1, ls="--")
    _a1.legend(frameon=False, fontsize=8, loc="upper left")
    label_axes(_a1, "Residence time per stage, τ = V/(S + L_n) (dashed = feed/scrub boundary)", "Stage", "min")
    # stage efficiency: steady purities against the mass-transfer time
    _grid_s = np.unique(np.r_[np.linspace(1, 120, 40), sl_tmt.value])
    _sw = efficiency_sweep(plant, _grid_s / 60.0)
    _ideal = efficiency_sweep(plant, [1e-6])[0]
    _now = efficiency_sweep(plant, [sl_tmt.value / 60.0])[0]
    _f2, (_a2,) = new_figure(1, 9.5, 3.2)
    for _j, _el, _name, _dy in ((0, plant["light"], "raffinate", -12), (1, plant["heavy"], "extract", 6)):
        _a2.plot(_grid_s, _sw[:, _j], color=COLORS[_el], lw=2)
        _a2.axhline(_ideal[_j], color=COLORS[_el], lw=1, ls=":")
        _a2.annotate(f"{_name} ({_el})", (_grid_s[-1], _sw[-1, _j]), xytext=(-4, _dy),
                     textcoords="offset points", ha="right", fontsize=8, color=INK)
    _a2.axvline(sl_tmt.value, color=INK2, lw=1, ls="--")
    _a2.annotate(f"τ_mt = {sl_tmt.value} s", (sl_tmt.value, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK2)
    label_axes(_a2, "Steady purity vs mass-transfer time (dotted = ideal stages)", "τ_mt (s)", "Purity (%)")
    mo.vstack([
        mo.md("### 6 · Mixer-settler hydraulics and stage efficiency"),
        mo.hstack([sl_F, sl_Vm, sl_Vs, sl_tmt], justify="start", gap=1.5, wrap=True),
        mo.hstack([
            mo.stat(f"{plant['tau_mix'][0]:.2f} / {plant['tau_set'][0]:.2f} min", label="Extraction τ_mix / τ_set", bordered=True),
            mo.stat(f"{plant['tau_mix'][-1]:.2f} / {plant['tau_set'][-1]:.2f} min", label="Scrub τ_mix / τ_set", bordered=True),
            mo.stat(f"{plant['eta'][0]:.3f} / {plant['eta'][-1]:.3f}", label="η extraction / scrub", bordered=True),
            mo.stat(f"{plant['holdup']:.0f} L", label="Hold-up", bordered=True),
            mo.stat(f"{plant['holdup'] / plant['F'] / 60:.1f} h", label="Hold-up ÷ feed flow", bordered=True),
        ], widths="equal"),
        _f1,
        mo.md(f"With τ_mt = {sl_tmt.value} s the plant reaches **{_now[0]:.2f} % / {_now[1]:.2f} %** "
              f"(raffinate / extract purity). Ideal stages would give **{_ideal[0]:.2f} % / {_ideal[1]:.2f} %**."),
        _f2,
    ])
    return


# --------------------------------------------------- Section 7: start-up
@app.cell(hide_code=True)
def _(plant, simulate_startup):
    startup = simulate_startup(plant)
    return (startup,)


@app.cell(hide_code=True)
def _(COLORS, INK, INK2, label_axes, mo, new_figure, np, plant, startup):
    _th = startup["t"] / 60
    _f1, (_a1,) = new_figure(1, 9.5, 3.2)
    for _v, _ss, _el, _name, _dy in ((startup["pr"], startup["pr_ss"], plant["light"], "raffinate", -14),
                                     (startup["pe"], startup["pe_ss"], plant["heavy"], "extract", 8)):
        _a1.plot(_th, _v, color=COLORS[_el], lw=2)
        _a1.axhspan(_ss - 1, _ss + 1, color=COLORS[_el], alpha=0.12, lw=0)
        _ok = np.isfinite(_v)
        _a1.annotate(f"{_name} ({_el})", (_th[-1], _v[_ok][-1] if _ok.any() else _ss), xytext=(-4, _dy),
                     textcoords="offset points", ha="right", fontsize=8, color=INK)
    if np.isfinite(startup["t_settle"]):
        _a1.axvline(startup["t_settle"] / 60, color=INK2, lw=1, ls="--")
        _a1.annotate(f"settled {startup['t_settle'] / 60:.1f} h", (startup["t_settle"] / 60, 1.0),
                     xycoords=("data", "axes fraction"), xytext=(4, -12),
                     textcoords="offset points", fontsize=8, color=INK2)
    _vals = np.concatenate([startup["pr"][np.isfinite(startup["pr"])], startup["pe"][np.isfinite(startup["pe"])]])
    _a1.set_ylim(max(0.0, (_vals.min() if _vals.size else 0.0) - 5), 100)
    label_axes(_a1, "Start-up purities (hidden below 5 % of steady flow; bands = ±1 point)", "Time (h)", "Purity (%)")
    # profile snapshots: one hue per element, light -> dark as time advances
    _st = np.arange(1, plant["N"] + 1)
    _f2, _axs = new_figure(2, 9.5, 3.3)
    _times = sorted(startup["snaps"])
    for _ax, _el, _e, _off, _what in ((_axs[0], plant["light"], 0, 2, "aqueous x"),
                                      (_axs[1], plant["heavy"], 1, 3, "organic y")):
        for _i, _h in enumerate(_times):
            _ax.plot(_st, startup["snaps"][_h][_e, _off::4], color=COLORS[_el], lw=2,
                     alpha=0.25 + 0.6 * _i / max(1, len(_times) - 1), label=f"{_h} h")
        _ax.plot(_st, startup["u_ss"][_e, _off::4], color=INK, lw=1.5, ls="--", label="steady")
        _ax.axvline(plant["n_ext"], color=INK2, lw=1, ls=":")
        _ax.legend(frameon=False, fontsize=7, loc="best")
        label_axes(_ax, f"{_el} settler {_what} profile filling up", "Stage (dotted = feed)", "mmol/L")
    mo.vstack([
        mo.md("### 7 · Start-up dynamics\nThe plant starts full of metal-free liquor, and the feed is switched on at t = 0."),
        mo.hstack([
            mo.stat(f"{startup['pr_ss']:.2f} %", label=f"Steady raffinate purity ({plant['light']})", bordered=True),
            mo.stat(f"{startup['pe_ss']:.2f} %", label=f"Steady extract purity ({plant['heavy']})", bordered=True),
            mo.stat(f"{startup['t_settle'] / 60:.1f} h" if np.isfinite(startup["t_settle"]) else "> horizon",
                    label="Settling time (±1 point)", bordered=True),
            mo.stat(f"{startup['t_settle'] / 60 / (plant['holdup'] / plant['F'] / 60):.2f}"
                    if np.isfinite(startup["t_settle"]) else "n/a",
                    label="Settling time ÷ (hold-up ÷ F)", bordered=True),
        ], widths="equal"),
        _f1,
        _f2,
    ])
    return


# --------------------------------------------------- Section 8: step test, model and tuning rules
@app.cell(hide_code=True)
def _(fopdt_smith, plant, simc_tuning, sl_delay, sl_step, step_test):
    if sl_step.value == 0:
        stp, fit, simc = None, None, None
    else:
        stp = step_test(plant, sl_step.value / 100.0)
        fit = fopdt_smith(stp["t"], stp["y"], stp["y0"], stp["y_inf"])
        simc = simc_tuning(fit["tau"], fit["theta"] + float(sl_delay.value))
    return fit, simc, stp


@app.cell(hide_code=True)
def _(
    ACCENT1,
    INK,
    INK2,
    fit,
    fopdt_response,
    label_axes,
    mo,
    new_figure,
    np,
    plant,
    process_gain,
    simc,
    sl_delay,
    sl_step,
    stability_margins,
    stp,
    tuning_rules,
):
    _head = mo.md("### 8 · Step test, process model and tuning rules")
    _ctrl = mo.hstack([sl_step, sl_delay], justify="start", gap=1.5, wrap=True)
    if stp is None:
        _out = mo.vstack([_head, _ctrl, mo.callout(mo.md("Choose a non-zero step size."), kind="warn")])
    else:
        _K = (stp["y_inf"] - stp["y0"]) / stp["dS"]
        _th = stp["t"] / 60
        _f, (_a,) = new_figure(1, 9.5, 3.4)
        _a.plot(_th, stp["y"], color=ACCENT1, lw=2)
        _a.plot(_th, fopdt_response(stp["t"], stp["y0"], _K, stp["dS"], fit["tau"], fit["theta"]),
                color=INK2, lw=1.5, ls="--")
        for _lvl, _tx in ((0.283, fit["t28"]), (0.632, fit["t63"])):
            if np.isfinite(_tx):
                _yy = stp["y0"] + _lvl * (stp["y_inf"] - stp["y0"])
                _a.plot([_tx / 60], [_yy], "o", ms=7, color=ACCENT1, mec="white", mew=1.5)
                _a.annotate(f"{_lvl * 100:.1f} % at {_tx / 60:.2f} h", (_tx / 60, _yy), xytext=(6, 4),
                            textcoords="offset points", fontsize=8, color=INK)
        _a.axhline(stp["y_inf"], color=INK2, lw=1, ls=":")
        _a.annotate("simulated plant", (_th[-1], stp["y"][-1]), xytext=(-4, 6 if _K > 0 else -12),
                    textcoords="offset points", ha="right", fontsize=8, color=INK)
        label_axes(_a, f"Open-loop response to a {sl_step.value:+d} % step in S (dashed = fitted FOPDT model)",
                   "Time after step (h)", "Heavy share in raffinate (%)")
        _theta_tot = fit["theta"] + float(sl_delay.value)
        _rows = []
        for _name, (_lg, _ti) in tuning_rules(fit["tau"], _theta_tot).items():
            _m = stability_margins(_lg, _ti, fit["tau"], _theta_tot)
            _rows.append({"Rule": _name, "LG = Kc·|Kp|": round(_lg, 2), "Ti (min)": round(_ti, 1),
                          "Gain margin": round(_m["GM"], 2), "Phase margin (°)": round(_m["PM"], 1)})
        _out = mo.vstack([
            _head,
            _ctrl,
            mo.hstack([
                mo.stat(f"{_K:.2f}", label="Kp from this step (points per L/min)", bordered=True),
                mo.stat(f"{process_gain(plant):.2f}", label="Kp from ±5 % steady states", bordered=True),
                mo.stat(f"{fit['tau'] / 60:.2f} h", label="Time constant τ", bordered=True),
                mo.stat(f"{fit['theta']:.1f} min", label="Apparent dead time θ", bordered=True),
                mo.stat(f"{_theta_tot:.1f} min", label="θ + analyzer delay", bordered=True),
            ], widths="equal"),
            _f,
            mo.callout(mo.md(
                f"**SIMC suggestion** (τ_c = θ + analyzer delay = {_theta_tot:.1f} min): "
                f"loop gain **LG ≈ {simc['LG']:.1f}**, **Ti ≈ {simc['Ti']:.0f} min**. "
                "The table compares the classic PI rules on the fitted model. Margins are computed on the FOPDT "
                "model, which is conservative for this plant (see page 1, B11)."), kind="info"),
            mo.ui.table(_rows, selection=None, label="PI tuning rules for the fitted FOPDT model"),
        ])
    _out
    return


# --------------------------------------------------- Section 9: frequency response and margins
@app.cell(hide_code=True)
def _(
    ACCENT1,
    ACCENT2,
    INK,
    INK2,
    dd_mode,
    fit,
    label_axes,
    loop_frequency_response,
    mo,
    new_figure,
    np,
    sl_LG,
    sl_Ti,
    sl_delay,
    stability_margins,
    ultimate_point,
):
    _head = mo.md("### 9 · Frequency response and stability margins")
    _ctrl = mo.hstack([dd_mode, sl_LG, sl_Ti, sl_delay], justify="start", gap=1.5, wrap=True)
    if fit is None:
        _out = mo.vstack([_head, _ctrl, mo.callout(mo.md("Set a non-zero step in section 8 to fit the model."),
                                                   kind="warn")])
    else:
        _tau, _theta = fit["tau"], fit["theta"] + float(sl_delay.value)
        _integral = dd_mode.value != "P only"
        _m = stability_margins(sl_LG.value, float(sl_Ti.value), _tau, _theta, _integral)
        _wu, _LGu, _Pu = ultimate_point(_tau, _theta)
        _w = np.logspace(-5, 0.5, 600)
        _mag, _ph = loop_frequency_response(_w, sl_LG.value, float(sl_Ti.value), _tau, _theta, _integral)
        _f, (_a1, _a2) = new_figure(2, 10.0, 3.5)
        _a1.loglog(_w, _mag, color=ACCENT1, lw=2)
        _a1.axhline(1.0, color=INK2, lw=1, ls=":")
        _a2.semilogx(_w, _ph, color=ACCENT2, lw=2)
        _a2.axhline(-180.0, color=INK2, lw=1, ls=":")
        _a2.set_ylim(max(-400, float(_ph.min())), 0)
        if np.isfinite(_m["wc"]):
            for _ax in (_a1, _a2):
                _ax.axvline(_m["wc"], color=INK2, lw=1, ls="--")
            _a2.annotate(f"PM = {_m['PM']:.0f}°", (_m["wc"], -180 + _m["PM"]), xytext=(6, 0),
                         textcoords="offset points", fontsize=8, color=INK)
            _a2.plot([_m["wc"], _m["wc"]], [-180, -180 + _m["PM"]], color=INK, lw=2.5)
        if np.isfinite(_m["w180"]):
            _g180 = 1.0 / _m["GM"]
            _a1.plot([_m["w180"], _m["w180"]], [_g180, 1.0], color=INK, lw=2.5)
            _a1.annotate(f"GM = {_m['GM']:.2f}", (_m["w180"], np.sqrt(_g180)), xytext=(6, 0),
                         textcoords="offset points", fontsize=8, color=INK)
            _a2.axvline(_m["w180"], color=INK2, lw=1, ls="-.")
        label_axes(_a1, "Loop magnitude |L(jω)| (dotted = 1)", "ω (rad/min)", "|L|")
        label_axes(_a2, "Loop phase ∠L(jω) (dotted = −180°)", "ω (rad/min)", "degrees")
        _ok = _m["GM"] > 1 and _m["PM"] > 0
        _out = mo.vstack([
            _head,
            mo.md("Bode plot of the open loop $L(s) = LG\\,(1 + 1/(T_i s))\\,e^{-\\theta s}/(\\tau s + 1)$, using the "
                  "FOPDT model fitted in section 8 plus the analyzer delay."),
            _ctrl,
            mo.hstack([
                mo.stat(f"{_m['GM']:.2f}" if np.isfinite(_m["GM"]) else "∞", label="Gain margin", bordered=True),
                mo.stat(f"{_m['PM']:.1f}°" if np.isfinite(_m["PM"]) else "∞", label="Phase margin", bordered=True),
                mo.stat(f"{_m['wc']:.4f}" if np.isfinite(_m["wc"]) else "n/a", label="Crossover ω_c (rad/min)", bordered=True),
                mo.stat(f"{_LGu:.1f} / {_Pu:.0f} min", label="Ultimate LG_u / period P_u", bordered=True),
                mo.stat(f"{sl_LG.value / _LGu:.2f}", label="LG ÷ LG_u", bordered=True),
            ], widths="equal"),
            mo.callout(mo.md("Model loop is **stable**." if _ok else "Model loop is **unstable**: GM < 1 or PM < 0."),
                       kind="success" if _ok else "danger"),
            _f,
        ])
    _out
    return


# --------------------------------------------------- Section 10: disturbances and control
@app.cell(hide_code=True)
def _(dd_dist, plant, simulate_disturbance):
    open_loop = simulate_disturbance(plant, dd_dist.value, closed_loop=False)
    return (open_loop,)


@app.cell(hide_code=True)
def _(dd_dist, dd_mode, plant, simulate_disturbance, sl_LG, sl_Ti, sl_delay, sl_limit):
    closed_loop = simulate_disturbance(plant, dd_dist.value, closed_loop=True, LG=sl_LG.value,
                                       Ti=float(sl_Ti.value), delay=float(sl_delay.value),
                                       limit=sl_limit.value / 100.0, mode=dd_mode.value)
    return (closed_loop,)


@app.cell(hide_code=True)
def _(
    ACCENT1,
    INK,
    INK2,
    closed_loop,
    dd_dist,
    dd_mode,
    label_axes,
    mo,
    new_figure,
    open_loop,
    plant,
    sl_LG,
    sl_Ti,
    sl_delay,
    sl_limit,
):
    _tt = open_loop["t"] / 60
    _f, (_a1, _a2, _a3) = new_figure(3, 10.5, 3.3)
    _lab = "P" if dd_mode.value == "P only" else ("PI+FF" if dd_mode.value.startswith("PI +") else "PI")
    for _ax, _key, _title, _yl in ((_a1, "cv", "CV: heavy share in raffinate", "%"),
                                   (_a2, "ext", "Extract purity (not controlled)", "%"),
                                   (_a3, "S", "MV: solvent flow S", "L/min")):
        _ax.plot(_tt, open_loop[_key], color=INK2, lw=2)
        _ax.plot(_tt, closed_loop[_key], color=ACCENT1, lw=2)
        _ol_up = open_loop[_key][-1] >= closed_loop[_key][-1]   # put the higher curve's label above
        _ax.annotate("open loop", (_tt[-1], open_loop[_key][-1]), xytext=(-4, 6 if _ol_up else -12),
                     textcoords="offset points", ha="right", fontsize=8, color=INK)
        _ax.annotate(_lab, (_tt[-1], closed_loop[_key][-1]), xytext=(-4, -12 if _ol_up else 6),
                     textcoords="offset points", ha="right", fontsize=8, color=INK)
        label_axes(_ax, _title, "Time after step (h)", _yl)
    _a1.axhline(open_loop["sp"], color=INK2, lw=1, ls=":")
    _lim = sl_limit.value / 100.0
    _a3.axhspan((1 - _lim) * plant["S"], (1 + _lim) * plant["S"], color="#ecebe7", zorder=0)
    _p_offset = (open_loop["final"] - open_loop["sp"]) / (1.0 + sl_LG.value)
    mo.vstack([
        mo.md("### 10 · Disturbance rejection: P, PI and feedforward\n"
              "The grey band on the S chart is the allowed range of the solvent flow."
              + (f" Linear theory predicts a P-only offset of Δy/(1 + LG) ≈ {_p_offset:+.2f} points."
                 if dd_mode.value == "P only" else "")),
        mo.hstack([dd_dist, dd_mode, sl_LG], justify="start", gap=1.5, wrap=True),
        mo.hstack([sl_Ti, sl_delay, sl_limit], justify="start", gap=1.5, wrap=True),
        mo.hstack([
            mo.stat(f"{open_loop['sp']:.1f} → {open_loop['final']:.1f} %", label="CV open loop (start → end)", bordered=True),
            mo.stat(f"{open_loop['peak']:.2f} / {closed_loop['peak']:.2f}", label=f"Peak deviation, open / {_lab} (points)", bordered=True),
            mo.stat(f"{open_loop['iae']:.1f} / {closed_loop['iae']:.1f}", label=f"IAE, open / {_lab} (point·h)", bordered=True),
            mo.stat(f"{closed_loop['final'] - closed_loop['sp']:+.2f}", label=f"{_lab} offset at end (points)", bordered=True),
        ], widths="equal"),
        mo.hstack([
            mo.stat(f"{open_loop['Kp']:.2f}", label="Kp (points per L/min)", bordered=True),
            mo.stat(f"{closed_loop['Kc']:.3f}", label="Kc = LG/|Kp| (L/min per point)", bordered=True),
            mo.stat(f"{100 * closed_loop['sat_frac']:.0f} %", label="Time with S on a limit", bordered=True),
            mo.stat(f"{open_loop['ext'][0]:.1f} → {closed_loop['ext'][-1]:.1f} %", label=f"Extract purity under {_lab}", bordered=True),
        ], widths="equal"),
        _f,
    ])
    return


# --------------------------------------------------- Section 11: interactive P&ID
@app.cell(hide_code=True)
def _(mo):
    PID_SCENARIOS = ["Steady state", "Start-up (feed switched on at t = 0)",
                     "Disturbance, open loop (section 10)", "Disturbance, closed loop (section 10)"]
    dd_pid = mo.ui.dropdown(PID_SCENARIOS, value=PID_SCENARIOS[1], label="Scenario")
    sw_play = mo.ui.switch(value=False, label="Play")
    refresher = mo.ui.refresh(options=["0.5s", "1s", "2s"], default_interval="1s", label="Speed")
    get_frame, set_frame = mo.state(0)
    return PID_SCENARIOS, dd_pid, get_frame, refresher, set_frame, sw_play


@app.cell(hide_code=True)
def _(
    PID_SCENARIOS,
    closed_loop,
    dd_mode,
    dd_pid,
    np,
    open_loop,
    plant,
    set_frame,
    sl_delay,
    startup,
):
    # Collect the frames, times, flows and controller readings for the chosen scenario.
    def _heavy_share(u, L1):
        r = L1 * u[:, 2]
        return float(100 * r[1] / r.sum()) if r.sum() > 1e-9 else None

    _L1 = plant["W"] + plant["F"]
    _sp = _heavy_share(startup["u_ss"], _L1)
    _sc = dd_pid.value
    if _sc == PID_SCENARIOS[0]:
        _frames, _t = startup["u_ss"][None], np.array([0.0])
        _flows = [dict(S=plant["S"], W=plant["W"], F=plant["F"])]
        _mode = {"P only": "AUTO-P", "PI": "AUTO"}.get(dd_mode.value, "AUTO+FF")
        _ctrl = [dict(mode=_mode, pv=_sp, meas=_sp, sp=_sp, delay=sl_delay.value)]
    elif _sc == PID_SCENARIOS[1]:
        _frames, _t = startup["frames"], startup["t"][startup["frame_k"]] / 60.0
        _flows = [dict(S=plant["S"], W=plant["W"], F=plant["F"])] * len(_frames)
        _ctrl = [dict(mode="MAN", pv=_heavy_share(_u, _L1), meas=_heavy_share(_u, _L1), sp=_sp,
                      delay=sl_delay.value) for _u in _frames]
        _ctrl = [dict(_c, pv=None) if _c["pv"] is None else _c for _c in _ctrl]
    else:
        _run = open_loop if _sc == PID_SCENARIOS[2] else closed_loop
        _frames, _k = _run["frames"], _run["frame_k"]
        _t = _run["t"][_k] / 60.0
        if not _run["closed_loop"]:
            _mode = "MAN"
        else:
            _mode = {"P only": "AUTO-P", "PI": "AUTO"}.get(_run["mode"], "AUTO+FF")
        _flows = [dict(S=float(_run["S"][k]), W=_run["W_run"] if k > 0 else plant["W"],
                       F=_run["F_run"] if k > 0 else plant["F"]) for k in _k]
        _ctrl = [dict(mode=_mode, pv=float(_run["cv"][k]), meas=float(_run["cv"][max(0, k - _run["nd"])]),
                      sp=_run["sp"], delay=sl_delay.value) for k in _k]
    pid_data = dict(frames=_frames, t=_t, flows=_flows, ctrl=_ctrl, scenario=_sc,
                    scale_aq=float(np.ceil(_frames[:, :, 2::4].max() / 10) * 10 or 10),
                    scale_org=float(np.ceil(_frames[:, :, 3::4].max() / 10) * 10 or 10))
    set_frame(0)
    return (pid_data,)


@app.cell(hide_code=True)
def _(pid_data, refresher, set_frame, sw_play):
    # Advance one frame on every timer tick while Play is on.
    refresher.value
    if sw_play.value and len(pid_data["frames"]) > 1:
        set_frame(lambda i: (i + 1) % len(pid_data["frames"]))
    return


@app.cell(hide_code=True)
def _(get_frame, mo, pid_data, set_frame):
    _n = len(pid_data["frames"])
    sl_frame = mo.ui.slider(0, max(_n - 1, 1), 1, value=min(get_frame(), _n - 1), on_change=set_frame,
                            label="Time", show_value=False, debounce=True, disabled=_n == 1)
    return (sl_frame,)


@app.cell(hide_code=True)
def _(
    COLORS,
    dd_pid,
    get_frame,
    mo,
    pid_data,
    pid_svg,
    plant,
    refresher,
    sl_frame,
    stage_trend,
    sw_play,
):
    _i = min(get_frame(), len(pid_data["frames"]) - 1)
    _u = pid_data["frames"][_i]
    _prev = pid_data["frames"][_i - 1] if _i > 0 else None
    _dt_h = float(pid_data["t"][_i] - pid_data["t"][_i - 1]) if _i > 0 else 0.0
    _fl, _ct = pid_data["flows"][_i], pid_data["ctrl"][_i]
    _r = (_fl["W"] + _fl["F"]) * _u[:, 2]
    _e = _fl["S"] * _u[:, -1]
    _pr = f"{100 * _r[0] / _r.sum():.1f} %" if _r.sum() > 1e-6 else "–"
    _pe = f"{100 * _e[1] / _e.sum():.1f} %" if _e.sum() > 1e-6 else "–"
    _label = (f"{pid_data['scenario'].split(' (')[0]} · t = {pid_data['t'][_i]:.1f} h · "
              f"raffinate {plant['light']} purity {_pr} · extract {plant['heavy']} purity {_pe}")
    _svg = pid_svg(plant, _u, COLORS, pid_data["scale_aq"], pid_data["scale_org"], _fl, _ct,
                   _label, stage_trend(_u, _prev, _dt_h))
    mo.vstack([
        mo.md("### 11 · Interactive P&ID\n"
              f"The cascade from section 5 ({plant['N']} stages: {plant['n_ext']} extraction + {plant['n_scr']} scrub) "
              "drawn as a piping and instrumentation diagram. Bars show the settler concentrations of each element "
              "in the organic (top) and aqueous (bottom) layer of every stage. **Hover over a stage** for exact "
              "values. Choose a scenario, then drag the time slider or press **Play**. The disturbance, controller mode and tuning "
              "come from section 10. Symbols are explained on page 1, B14."),
        mo.hstack([dd_pid, sw_play, refresher], justify="start", gap=1.5, wrap=True),
        mo.hstack([sl_frame, mo.md(f"t = **{pid_data['t'][_i]:.1f} h** (frame {_i + 1} of {len(pid_data['frames'])})")],
                  justify="start", gap=1.5),
        mo.Html(_svg),
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
    pid_svg,
    run_tests,
    simulate_disturbance,
    stability_margins,
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
        ("Mixer-settler dynamics, efficiency sweep and start-up (implicit Euler)", assemble),
        ("Step test, FOPDT fit, SIMC tuning, P/PI/feedforward control and disturbances", simulate_disturbance),
        ("Frequency response, stability margins and PI tuning rules", stability_margins),
        ("Interactive P&ID drawing (SVG)", pid_svg),
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
| 12 | Skogestad S., *Simple analytic rules for model reduction and PID controller tuning*, Journal of Process Control 13 (2003) 291–309: [doi:10.1016/S0959-1524(02)00062-8](https://doi.org/10.1016/S0959-1524(02)00062-8) | SIMC PI tuning rule (B7, Lab 8) | Not consulted in this session (from the model's background knowledge) |
| 13 | Smith C.L., *Digital Computer Process Control*, Intext (1972); the two-point (28.3 % / 63.2 %) FOPDT fit as presented in standard process-control textbooks (e.g. Seborg, Edgar, Mellichamp & Doyle, *Process Dynamics and Control*, Wiley) | Step-test identification (B6, Lab 8) | Not consulted in this session (from the model's background knowledge) |
| 14 | General background on liquid–liquid extraction, extractant families, saponification, mixer-settlers and the lanthanide contraction (A1–A2, B2): standard hydrometallurgy texts such as Gupta C.K. & Krishnamurthy N., *Extractive Metallurgy of Rare Earths*, CRC Press, and ref. 2 | Concept text on page 1 | Not consulted in this session (from the model's background knowledge; ref. 2 seen as excerpt only) |
| 15 | Seborg D.E., Edgar T.F., Mellichamp D.A., Doyle F.J., *Process Dynamics and Control*, Wiley (4th ed. 2016) | Control terminology, transfer functions, P/PI/PID, frequency response, margins guideline, feedforward, cascade, RGA (B1–B16) | Not consulted in this session (from the model's background knowledge) |
| 16 | Ziegler J.G., Nichols N.B., *Optimum settings for automatic controllers*, Trans. ASME 64 (1942) 759–768; Cohen G.H., Coon G.A., *Theoretical consideration of retarded control*, Trans. ASME 75 (1953) 827–834; Rivera D.E., Morari M., Skogestad S., *Internal model control. 4. PID controller design*, Ind. Eng. Chem. Process Des. Dev. 25 (1986) 252–265 | PI tuning rules (B12, Lab 8) | Not consulted in this session (from the model's background knowledge) |
| 17 | Åström K.J., Hägglund T., *PID Controllers: Theory, Design, and Tuning*, ISA (1995) | Anti-windup, relay auto-tuning, derivative filtering (B9, B10) | Not consulted in this session (from the model's background knowledge) |
| 18 | Rydberg J., Cox M., Musikas C., Choppin G.R. (eds.), *Solvent Extraction Principles and Practice*, Marcel Dekker (2nd ed. 2004); Ritcey G.M., Ashbrook A.W., *Solvent Extraction: Principles and Applications to Process Metallurgy*, Elsevier (1984) | Extraction mechanisms, mass transfer, phase separation, equipment, circuits (A1) | Not consulted in this session (from the model's background knowledge) |

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
* **Process gain:** central difference over S ± 5 %. Kc = LG/|Kp|. The controller is direct acting (output rises with the CV) because the process gain is negative.
* **Phase volumes** in the dynamic model are fixed at the nominal split, even when the flows change.
* **Ce(IV) switch:** applies to sections 1 and 2 only.
* **Project balance and depth:** following the author's requests, the notebook is split 50/50 into Part A
  (Separations, lab sections 1–5) and Part B (Process dynamics and control, lab sections 6–10). Page 1 is written as a
  textbook chapter with derivations and worked examples, and has a matching chapter for every lab section.
* **Extra tools beyond the original specification:** the stage-efficiency sweep (Lab 6), start-up profile snapshots
  (Lab 7), the open-loop step test with Smith's FOPDT fit, SIMC suggestion and a tuning-rule table (Lab 8), the Bode
  plot with gain and phase margins (Lab 9), and the P-only and PI + S/F feedforward modes, adjustable solvent-flow limit,
  and IAE, ISE, extract purity and time-on-limit tracking (Lab 10). The reference controller (Kc = LG/|Kp| from ±5 %
  steady states) is unchanged.
* **Stability margins** are computed on the fitted FOPDT model plus the analyzer delay. The full plant was checked by
  simulation: its true P-only stability limit (LG between 60 and 65) is about twice the model's (about 28). Page 1, B11
  discusses this.
* **Interactive P&ID (Lab 11):** drawn as SVG in Python (no extra libraries). Tags follow ISA-5.1 style but the loop
  numbers are invented. Frames are stored every 30 min of simulated time (every 15 integration steps). Bars show the
  settler outlet concentrations, and ▲/▼ mark stages whose settler metal changes by more than 2 % per hour. Play uses
  marimo's refresh timer. The pH and level loops are drawn but treated as ideal.
* **Feedforward** is static (S = S0·F/F0 using the measured feed flow, with no lag) and acts only on the feed-flow disturbance.
* **SIMC closed-loop time constant:** τ_c is set to the total dead time (process θ + analyzer delay), the usual "tight
  control" choice.
* **Test tolerances:** set to the number of decimals quoted in the specification (for example ±0.0005 for pH50 and ±0.1 point for purities).
"""
    )
    return


if __name__ == "__main__":
    app.run()
