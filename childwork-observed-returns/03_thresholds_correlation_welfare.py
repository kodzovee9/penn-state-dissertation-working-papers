"""
Threshold verification, ability-wealth correlation, and welfare accounting.

Parts:
  A  numerical verification of the selection and reversal thresholds of
     Propositions 1-3 (finite differences against closed forms), including the
     kappa = 1 benchmark in which unconstrained positive selection is impossible
     (Proposition 1(iii))
  B  constrained-reversal search within the illustration parameter ranges
  C  population with corr(ln a, ln A) = rho: Figure 4 and the three-part
     decomposition of the naive coefficient
  D  consumption-equivalent welfare gains from the credit expansion (Table 3)
Generates: Figure 1 (regime_map_kappa.png), Figure 4 (rho_decomposition.png).
"""

import numpy as np

BASE = dict(alpha=0.5, w=0.3, w2=3.5, theta=0.25, R=1.4, beta=0.8)
LMAX = 0.92


def solve(a, A, sigma, phi, kappa, delta, dbar, p):
    """Vectorized solver, general kappa. Returns l, con mask."""
    alpha, w, w2, theta, R, beta = p['alpha'], p['w'], p['w2'], p['theta'], p['R'], p['beta']
    w1 = w * a**phi

    rhs = R * (w1 + theta) / w2 + delta
    s_u = (rhs / (a**kappa * alpha))**(1.0 / (alpha - 1.0))
    l_u = np.clip(1.0 - s_u, 0.0, LMAX)
    s_u = 1.0 - l_u
    H_u = a**kappa * s_u**alpha + delta * l_u
    kap = (beta * R)**(-1.0 / sigma)
    d_u = (kap * w2 * H_u - (A + (w1 + theta) * l_u - theta)) / (1.0 + kap * R)
    unc = d_u <= dbar

    l = np.where(unc, l_u, np.nan)
    con = ~unc
    if con.any():
        ac, Ac, w1c = a[con], A[con], w1[con]

        def G(lc):
            sc = 1.0 - lc
            c1 = Ac + (w1c + theta) * lc - theta + dbar
            c2 = w2 * (ac**kappa * sc**alpha + delta * lc) - R * dbar
            m = ac**kappa * alpha * sc**(alpha - 1.0) - delta
            with np.errstate(invalid='ignore'):
                return (w1c + theta) * np.where(c1 > 0, c1, np.nan)**(-sigma) \
                    - beta * w2 * m * np.where(c2 > 0, c2, np.nan)**(-sigma)

        lo = np.maximum(1e-9, (theta - Ac - dbar) / (w1c + theta) + 1e-9)
        hi = np.full_like(ac, LMAX)
        corner0 = (lo <= 1e-8) & (G(np.maximum(np.full_like(ac, 1e-9), lo)) <= 0)
        cornerT = G(hi) > 0
        lom, him = lo.copy(), hi.copy()
        for _ in range(80):
            mid = 0.5 * (lom + him)
            pos = G(mid) > 0
            lom = np.where(pos, mid, lom)
            him = np.where(pos, him, mid)
        sol = 0.5 * (lom + him)
        sol = np.where(corner0, 0.0, np.where(cornerT, hi, sol))
        l[con] = sol
    return l, con


def hfun(l, a, kappa, delta, alpha):
    return a**kappa * (1.0 - l)**alpha + delta * l


# ============ A: threshold verification ============

print("=" * 100)
print("A1/A2. Unconstrained thresholds (delta=0), finite differences on closed-form l_u")
print("=" * 100)
rng = np.random.default_rng(0)
mis_sel = mis_rev = tot = 0
for _ in range(6000):
    p = dict(BASE)
    p['w'] = float(rng.uniform(0.2, 1.5)); p['theta'] = float(rng.uniform(0.05, 0.5))
    p['w2'] = float(rng.uniform(1.5, 4.0)); p['R'] = float(rng.uniform(1.1, 1.9))
    kappa = float(rng.uniform(0.2, 1.0)); phi = float(rng.uniform(0.0, 1.0))
    a = float(rng.uniform(0.6, 1.8))
    alpha, w, w2, theta, R = p['alpha'], p['w'], p['w2'], p['theta'], p['R']

    def lu(aa):
        w1 = w * aa**phi
        s = (R * (w1 + theta) / (w2 * aa**kappa * alpha))**(1.0 / (alpha - 1.0))
        return 1.0 - s
    l0 = lu(a)
    if not (1e-4 < l0 < LMAX - 1e-4):
        continue
    ha_ = 1e-6
    dlda = (lu(a + ha_) - lu(a - ha_)) / (2 * ha_)
    dhda = (hfun(lu(a + ha_), a + ha_, kappa, 0.0, alpha) -
            hfun(lu(a - ha_), a - ha_, kappa, 0.0, alpha)) / (2 * ha_)
    w1 = w * a**phi
    gap_sel = phi * w1 - kappa * (w1 + theta)
    gap_rev = phi * w1 - (kappa / alpha) * (w1 + theta)
    tot += 1
    if (gap_sel > 0) != (dlda > 0) and abs(gap_sel) > 1e-4:
        mis_sel += 1
    if (gap_rev > 0) != (dhda < 0) and abs(gap_rev) > 1e-4:
        mis_rev += 1
print(f"draws: {tot} | selection-threshold mismatches: {mis_sel} | reversal-threshold mismatches: {mis_rev}")

print()
print("=" * 100)
print("A3. Constrained threshold sigma* = c2/(w2 h) with kappa != 1 (delta=0, phi=0, dbar=0)")
print("=" * 100)
ok = tot = 0
rng = np.random.default_rng(1)
for _ in range(600):
    sig = float(rng.uniform(0.4, 5.0)); kappa = float(rng.uniform(0.2, 1.0))
    a0 = float(rng.uniform(0.7, 1.5)); A0 = float(rng.uniform(0.5, 2.5))
    def single(aa):
        l, c = solve(np.array([aa]), np.array([A0]), sig, 0.0, kappa, 0.0, 0.0, BASE)
        return float(l[0]), bool(c[0])
    l0, c0 = single(a0)
    if not c0 or not (1e-5 < l0 < LMAX - 1e-5):
        continue
    lp, cp = single(a0 + 1e-5); lm, cm = single(a0 - 1e-5)
    if not (cp and cm):
        continue
    dlda = (lp - lm) / 2e-5
    h0 = hfun(l0, a0, kappa, 0.0, BASE['alpha'])
    sigstar = (BASE['w2'] * h0) / (BASE['w2'] * h0)  # dbar=0 -> c2 = w2 h -> sigma* = 1
    tot += 1
    if ((sig > 1.0) == (dlda > 0)) or abs(sig - 1.0) < 2e-2:
        ok += 1
print(f"constrained draws: {tot}, threshold (sigma*=1 at dbar=0) sign agreement: {ok}/{tot}")

print()
print("=" * 100)
print("A4. kappa=1 benchmark: unconstrained-positive cells (none expected, Prop 1(iii))")
print("=" * 100)
avals = np.linspace(0.6, 1.8, 40); Avals = np.linspace(0.02, 1.4, 40)
p_old = dict(alpha=0.5, w=1.0, w2=2.0, theta=0.2, R=1.5, beta=0.65)
count_up = count_u = 0
for A0 in Avals:
    for a0 in avals:
        def single(aa):
            l, c = solve(np.array([aa]), np.array([A0]), 2.0, 0.6, 1.0, 0.0, 0.0, p_old)
            return float(l[0]), bool(c[0])
        l0, c0 = single(a0)
        if c0 or not (1e-5 < l0 < LMAX - 1e-5):
            continue
        lp, _ = single(a0 + 1e-5); lm, _ = single(a0 - 1e-5)
        count_u += 1
        if (lp - lm) > 0:
            count_up += 1
print(f"kappa=1, phi=0.6 grid: interior unconstrained cells: {count_u}, with positive selection: {count_up}")
print("(none expected: with kappa=1 and phi<=1, unconstrained positive selection is impossible)")

# ============ B: constrained reversal search in the open band ============

print()
print("=" * 100)
print("B. Constrained reversal search: kappa in {0.3,0.5}, phi in (kappa(1+th/w1), 1]")
print("=" * 100)
rng = np.random.default_rng(2)
worst = None
found_rev = 0
checked = 0
for _ in range(8000):
    kappa = float(rng.choice([0.3, 0.5]))
    phi = float(rng.uniform(0.0, 1.0))
    sig = float(rng.uniform(0.5, 6.0))
    p = dict(BASE)
    p['w'] = float(rng.uniform(0.3, 1.5)); p['theta'] = float(rng.uniform(0.05, 0.4))
    a0 = float(rng.uniform(0.6, 1.8)); A0 = float(rng.uniform(0.05, 1.5))
    def single(aa):
        l, c = solve(np.array([aa]), np.array([A0]), sig, phi, kappa, 0.0, 0.0, p)
        return float(l[0]), bool(c[0])
    l0, c0 = single(a0)
    if not c0 or not (1e-5 < l0 < LMAX - 1e-5):
        continue
    lp, cp = single(a0 + 1e-5); lm, cm = single(a0 - 1e-5)
    if not (cp and cm):
        continue
    dhda = (hfun(lp, a0 + 1e-5, kappa, 0.0, p['alpha']) -
            hfun(lm, a0 - 1e-5, kappa, 0.0, p['alpha'])) / 2e-5
    checked += 1
    if dhda < 0:
        found_rev += 1
    if worst is None or dhda < worst[0]:
        w1 = p['w'] * a0**phi
        worst = (dhda, kappa, phi, sig, p['w'], p['theta'], a0, A0, phi * w1 / (kappa * (w1 + p['theta'])))
print(f"constrained interior draws checked: {checked}, reversal cases (dh/da<0): {found_rev}")
d = worst
print(f"min dh/da = {d[0]:+.4f} at kappa={d[1]}, phi={d[2]:.2f}, sigma={d[3]:.2f}, w={d[4]:.2f}, "
      f"theta={d[5]:.2f}, a={d[6]:.2f}, A={d[7]:.2f}  [phi*w1/(kappa(w1+th)) = {d[8]:.2f}]")

# ============ C: population with corr(ln a, ln A) = rho ============

print()
print("=" * 100)
print("C. Ability-wealth correlation: E3-style economy (sigma=3, kappa=1, phi=0, s_A=0.15)")
print("=" * 100)

def ols(y, X):
    X1 = np.column_stack([np.ones(len(y))] + [x for x in X])
    return np.linalg.lstsq(X1, y, rcond=None)[0][1:]

def run_rho(N, rho, sigma=3.0, mu_A=0.7, s_A=0.15, s_a=0.35, kappa=1.0, phi=0.0, seed=42, noise=0.3):
    rng = np.random.default_rng(seed)
    za, zA = rng.normal(size=N), rng.normal(size=N)
    lna = s_a * za
    lnA = mu_A + s_A * (rho * za + np.sqrt(1 - rho**2) * zA)
    a, A = np.exp(lna), np.exp(lnA)
    l, con = solve(a, A, sigma, phi, kappa, 0.0, 0.0, BASE)
    s = 1.0 - l
    h = a**kappa * s**BASE['alpha']
    lny2 = np.log(BASE['w2'] * h) + rng.normal(0.0, noise, N)
    vl = l.var()
    disp = BASE['alpha'] * np.cov(l, np.log(s))[0, 1] / vl
    # within/between split by 20 wealth bins
    bins = np.quantile(lnA, np.linspace(0, 1, 21))
    idx = np.clip(np.digitize(lnA, bins[1:-1]), 0, 19)
    within = 0.0
    mlv, mav, wt = [], [], []
    for b in range(20):
        mk = idx == b
        if mk.sum() > 2:
            within += mk.mean() * np.cov(l[mk], lna[mk])[0, 1]
            mlv.append(l[mk].mean()); mav.append(lna[mk].mean()); wt.append(mk.mean())
    between = np.cov(mlv, mav, aweights=wt)[0, 1] if len(mlv) > 2 else 0.0
    return dict(b_naive=ols(lny2, [l])[0], disp=disp,
                within=kappa * within / vl, between=kappa * between / vl,
                con=con.mean(), rho_hat=np.corrcoef(lna, lnA)[0, 1])

print(f"{'rho':>5}{'con%':>6}{'b_naive':>9}{'disp':>8}{'within':>8}{'between':>9}{'sum':>8}")
rows = []
for rho in [0.0, 0.2, 0.4, 0.6]:
    o = run_rho(20000, rho)
    rows.append((rho, o))
    print(f"{rho:>5.1f}{o['con']:>6.0%}{o['b_naive']:>9.3f}{o['disp']:>8.3f}"
          f"{o['within']:>8.3f}{o['between']:>9.3f}{o['disp']+o['within']+o['between']:>8.3f}")

# ============ D: welfare accounting of the credit expansion ============

print()
print("=" * 100)
print("D. Credit expansion welfare (E2 economy): consumption-equivalent gains by wealth quintile")
print("=" * 100)

def run_policy_welfare(N, dbar, sigma=3.0, mu_A=0.7, s_A=0.35, s_a=0.35, seed=42):
    p = BASE
    rng = np.random.default_rng(seed)
    lna = rng.normal(0.0, s_a, N)
    a = np.exp(lna)
    A = np.exp(rng.normal(mu_A, s_A, N))
    l, con = solve(a, A, sigma, 0.0, 1.0, 0.0, dbar, p)
    s = 1.0 - l
    h = a * s**p['alpha']
    kap = (p['beta'] * p['R'])**(-1.0 / sigma)
    # consumptions: unconstrained use Euler d; constrained d = dbar
    d_u = (kap * p['w2'] * h - (A + (p['w'] + p['theta']) * l - p['theta'])) / (1.0 + kap * p['R'])
    d = np.where(con, dbar, d_u)
    c1 = A + (p['w'] + p['theta']) * l - p['theta'] + d
    c2 = p['w2'] * h - p['R'] * d
    V = (c1**(1 - sigma)) / (1 - sigma) + p['beta'] * (c2**(1 - sigma)) / (1 - sigma)
    return dict(A=A, V=V, l=l, h=h)

base_ = run_policy_welfare(20000, 0.0)
print(f"{'dbar':>6}{'mean l':>8}{'mean h':>8}{'mean CE gain':>13}  CE gain by wealth quintile (poorest -> richest)")
for dbar in [0.1, 0.2, 0.4, 0.6]:
    new = run_policy_welfare(20000, dbar)
    sig = 3.0
    lam = (new['V'] / base_['V'])**(1.0 / (1.0 - sig)) - 1.0
    q = np.quantile(base_['A'], [0.2, 0.4, 0.6, 0.8])
    qi = np.digitize(base_['A'], q)
    qgains = [lam[qi == k].mean() for k in range(5)]
    print(f"{dbar:>6.1f}{new['l'].mean():>8.3f}{new['h'].mean():>8.3f}{lam.mean():>12.1%} "
          + "  ".join(f"{g:+.1%}" for g in qgains))

# ============ figures ============

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# regime map v2: kappa=1 vs kappa=0.4 (unconstrained positive selection appears)
def classify_grid(sigma, phi, kappa, p, avals, Avals):
    Z = np.full((len(Avals), len(avals)), np.nan)
    for i, A0 in enumerate(Avals):
        for j, a0 in enumerate(avals):
            def single(aa):
                l, c = solve(np.array([aa]), np.array([A0]), sigma, phi, kappa, 0.0, 0.0, p)
                return float(l[0]), bool(c[0])
            l0, c0 = single(a0)
            if not (1e-5 < l0 < LMAX - 1e-5):
                continue
            lp, cp = single(a0 + 1e-5); lm, cm = single(a0 - 1e-5)
            if cp != cm:
                continue
            dlda = (lp - lm) / 2e-5
            v = (1 if dlda > 0 else 0) + (2 if not c0 else 0)
            Z[i, j] = v
    return Z

p_e5 = dict(alpha=0.5, w=1.2, w2=3.5, theta=0.3, R=1.4, beta=0.8)
avals = np.linspace(0.6, 1.8, 70); Avals = np.linspace(0.05, 3.0, 70)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
Z1 = classify_grid(2.0, 0.8, 1.0, p_e5, avals, Avals)
Z2 = classify_grid(2.0, 0.8, 0.4, p_e5, avals, Avals)
cmap = ListedColormap(['#4472c4', '#e8b64c', '#9dc3e6', '#f4d999'])
norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
for ax, Z, ttl in [(axes[0], Z1, r'(a) $\kappa=1$, $\varphi=0.8$'),
                   (axes[1], Z2, r'(b) $\kappa=0.4$, $\varphi=0.8$')]:
    ax.pcolormesh(avals, Avals, Z, cmap=cmap, norm=norm, shading='auto')
    ax.set_title(ttl); ax.set_xlabel('ability $a$')
axes[0].set_ylabel('household resources $A$')
handles = [mpatches.Patch(color='#4472c4', label='constrained, negative selection'),
           mpatches.Patch(color='#e8b64c', label='constrained, positive selection'),
           mpatches.Patch(color='#9dc3e6', label='unconstrained, negative selection'),
           mpatches.Patch(color='#f4d999', label='unconstrained, positive selection')]
fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False)
fig.suptitle(r'Selection regimes, $\sigma=2$: the comparative-advantage channel requires $\varphi > \kappa(1+\theta/w_1)$', y=1.0)
fig.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig('regime_map_kappa.png', dpi=150, bbox_inches='tight')
print("\nfigure saved: regime_map_kappa.png")

# rho figure
fig2, ax = plt.subplots(figsize=(7.2, 4.4))
rhos = [r for r, _ in rows]
ax.plot(rhos, [o['b_naive'] for _, o in rows], 'o-', color='#c0504d', label=r'$\beta_{naive}$')
ax.plot(rhos, [o['within'] for _, o in rows], 's--', color='#e8862d', label='within-wealth selection')
ax.plot(rhos, [o['between'] for _, o in rows], 'd--', color='#4472c4', label='between-wealth term')
ax.plot(rhos, [o['disp'] for _, o in rows], 'v--', color='#7f7f7f', label='displacement')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel(r'corr$(\ln a, \ln A) = \rho$')
ax.set_ylabel('contribution to naive coefficient')
ax.set_title('Ability-wealth correlation and the naive coefficient (E3 economy)')
ax.legend(frameon=False, fontsize=9)
fig2.tight_layout()
fig2.savefig('rho_decomposition.png', dpi=150, bbox_inches='tight')
print("figure saved: rho_decomposition.png")
