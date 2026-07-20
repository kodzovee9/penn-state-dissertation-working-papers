"""
Population economies and the naive-regression decomposition.

Generates: Table 2 (economies E1-E4) and Figure 2 (naive-coefficient sweeps).
Also verifies the displacement + selection decomposition (eq. 3.2 of the paper).
Baseline technology: kappa = 1, delta = 0, dbar = 0; see the paper, Section 4.
Deterministic (fixed seeds). Requires numpy, matplotlib.
"""

import numpy as np

P = dict(alpha=0.5, w=0.3, w2=3.5, theta=0.25, R=1.4, beta=0.8)
LMAX = 0.92
S_A_ABILITY = 0.35     # sd of ln ability


def solve_economy(a, A, sigma, phi, p=P, lmax=LMAX):
    """Vectorized. Returns l, constrained mask."""
    alpha, w, w2, theta, R, beta = p['alpha'], p['w'], p['w2'], p['theta'], p['R'], p['beta']
    w1 = w * a**phi

    # unconstrained candidate (closed form), corner at l=0 if s_u >= 1
    s_u = (R * (w1 + theta) / (w2 * a * alpha))**(1.0 / (alpha - 1.0))
    l_u = np.clip(1.0 - s_u, 0.0, lmax)
    s_u = 1.0 - l_u
    kappa = (beta * R)**(-1.0 / sigma)
    d_u = (kappa * w2 * a * s_u**alpha - (A + (w1 + theta) * l_u - theta)) / (1.0 + kappa * R)
    unc = d_u <= 0.0

    l = np.where(unc, l_u, np.nan)
    con = ~unc
    if con.any():
        ac, Ac, w1c = a[con], A[con], w1[con]

        def G(lc):
            sc = 1.0 - lc
            c1 = Ac + (w1c + theta) * lc - theta
            with np.errstate(invalid='ignore'):
                return (w1c + theta) * np.where(c1 > 0, c1, np.nan)**(-sigma) \
                    - beta * w2 * ac * alpha * sc**(alpha - 1.0) * (w2 * ac * sc**alpha)**(-sigma)

        lo = np.maximum(1e-9, (theta - Ac) / (w1c + theta) + 1e-9)
        hi = np.full_like(ac, lmax)
        corner0 = (lo <= 1e-8) & (G(np.maximum(np.full_like(ac, 1e-9), lo)) <= 0)
        cornerT = G(hi) > 0
        lom, him = lo.copy(), hi.copy()
        for _ in range(80):
            mid = 0.5 * (lom + him)
            pos = G(mid) > 0
            lom = np.where(pos, mid, lom)
            him = np.where(pos, him, mid)
        sol = 0.5 * (lom + him)
        sol = np.where(corner0, 0.0, np.where(cornerT, lmax, sol))
        l[con] = sol
    return l, con


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y))] + [x for x in X])
    return np.linalg.lstsq(X1, y, rcond=None)[0][1:]


def run_economy(N, sigma, phi, mu_A, s_A=0.35, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    lna = rng.normal(0.0, S_A_ABILITY, N)
    a = np.exp(lna)
    A = np.exp(rng.normal(mu_A, s_A, N))
    l, con = solve_economy(a, A, sigma, phi)
    s = 1.0 - l
    h = a * s**P['alpha']
    lny2 = np.log(P['w2']) + np.log(h) + rng.normal(0.0, noise, N)
    lns = np.log(s)

    corner0 = l <= 1e-7
    cornerT = l >= LMAX - 1e-7
    cint = con & ~corner0 & ~cornerT

    def slope(mask):
        if mask.sum() < 50:
            return np.nan
        return np.cov(lna[mask], l[mask])[0, 1] / np.var(lna[mask])

    vl = l.var()
    out = dict(
        con_share=con.mean(), l0=corner0.mean(), lmean=l.mean(),
        lmean_w=l[l > 1e-7].mean() if (l > 1e-7).any() else 0.0,
        sel_con=slope(cint), sel_all=np.cov(l, lna)[0, 1] / np.var(lna),
        b_naive=ols(lny2, [l])[0],
        b_sel=ols(lny2, [l, lns])[0],
        b_ctrl=ols(lny2, [l, lna])[0],
        disp_term=P['alpha'] * np.cov(l, lns)[0, 1] / vl,
        sel_term=np.cov(l, lna)[0, 1] / vl,
    )
    bf = ols(lny2, [l, lns, lna])
    out['b_full_l'], out['b_full_s'], out['b_full_a'] = bf
    out['identity_gap'] = out['b_naive'] - (out['disp_term'] + out['sel_term'])
    return out


# ================= 1. showcase economies =================

print("=" * 112)
print("1. Showcase economies (N=20,000)")
print("=" * 112)
cases = [
    ("E1: sigma=0.7, phi=0, sA=.35 (R1, negative selection)", 0.7, 0.0, 0.7, 0.35),
    ("E2: sigma=3.0, phi=0, sA=.35 (R2, wealth-heterogeneous)", 3.0, 0.0, 0.7, 0.35),
    ("E3: sigma=3.0, phi=0, sA=.15 (R2, wealth-homogeneous)", 3.0, 0.0, 0.7, 0.15),
    ("E4: sigma=3.0, phi=0.5, sA=.15 (R2, both channels)", 3.0, 0.5, 0.7, 0.15),
]
print(f"{'economy':<58}{'con%':>5}{'l=0%':>6}{'l|work':>7}{'selcon':>8}{'naive':>8}"
      f"{'disp':>7}{'sel':>7}{'RS(l)':>7}{'RA(l)':>7}{'RF(l)':>7}{'RF(s)':>7}{'RF(a)':>7}")
show = {}
for name, sig, phi, muA, sA in cases:
    o = run_economy(20000, sig, phi, muA, s_A=sA, seed=42)
    show[name[:2]] = o
    print(f"{name:<58}{o['con_share']:>5.0%}{o['l0']:>6.0%}{o['lmean_w']:>7.2f}"
          f"{o['sel_con']:>8.3f}{o['b_naive']:>8.3f}{o['disp_term']:>7.2f}{o['sel_term']:>7.2f}"
          f"{o['b_sel']:>7.2f}{o['b_ctrl']:>7.2f}{o['b_full_l']:>7.2f}"
          f"{o['b_full_s']:>7.2f}{o['b_full_a']:>7.2f}")
print(f"\nOVB identity gaps: " + ", ".join(f"{k}: {v['identity_gap']:+.1e}" for k, v in show.items()))

# ================= 2. sweep figures =================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

# (a) beta_naive vs wealth heterogeneity sd(ln A)
sAs = np.linspace(0.10, 1.0, 10)
for sig, phi, col, lab in [(3.0, 0.0, '#4472c4', r'$\sigma=3,\ \varphi=0$'),
                           (3.0, 0.5, '#e8862d', r'$\sigma=3,\ \varphi=0.5$'),
                           (0.7, 0.0, '#7f7f7f', r'$\sigma=0.7,\ \varphi=0$')]:
    ys = [run_economy(8000, sig, phi, 0.7, s_A=sA, seed=11)['b_naive'] for sA in sAs]
    axes[0].plot(sAs, ys, 'o-' if sig > 1 else 's--', color=col, label=lab)
axes[0].axhline(0, color='k', lw=0.8)
axes[0].set_xlabel(r'wealth heterogeneity  sd$(\ln A)$')
axes[0].set_ylabel(r'$\beta_{naive}$')
axes[0].set_title('(a) Ability- vs wealth-driven variation in child work')
axes[0].legend(frameon=False, fontsize=9)

# (b) beta_naive vs average household resources mu_A
muAs = np.linspace(0.2, 1.6, 10)
for sig, phi, col, lab in [(3.0, 0.0, '#4472c4', r'$\sigma=3,\ \varphi=0$'),
                           (3.0, 0.5, '#e8862d', r'$\sigma=3,\ \varphi=0.5$'),
                           (0.7, 0.0, '#7f7f7f', r'$\sigma=0.7,\ \varphi=0$')]:
    ys = [run_economy(8000, sig, phi, muA, s_A=0.35, seed=11)['b_naive'] for muA in muAs]
    axes[1].plot(muAs, ys, 'o-' if sig > 1 else 's--', color=col, label=lab)
axes[1].axhline(0, color='k', lw=0.8)
axes[1].set_xlabel(r'average household resources  $\mu_{\ln A}$')
axes[1].set_title('(b) Same economies, richer populations')
axes[1].legend(frameon=False, fontsize=9)

fig.suptitle(r'Naive coefficient from regressing adult log earnings on child work', y=1.0)
fig.tight_layout()
fig.savefig('naive_coefficient_sweeps.png', dpi=150, bbox_inches='tight')
print("\nfigure saved: naive_coefficient_sweeps.png")

# ================= 3. specification ladder (economy E3) =================

o = show['E3']
fig2, ax = plt.subplots(figsize=(7.4, 4.5))
labels = ['naive:\nln y on l', '+ schooling:\ncoef on l', '+ ability:\ncoef on l', 'full model:\ncoef on l']
vals = [o['b_naive'], o['b_sel'], o['b_ctrl'], o['b_full_l']]
cols = ['#c0504d' if v > 0 else '#4472c4' for v in vals]
ax.bar(labels, vals, color=cols, width=0.55)
ax.axhline(0, color='k', lw=0.9)
for i, v in enumerate(vals):
    ax.text(i, v + (0.03 if v >= 0 else -0.07), f'{v:+.2f}', ha='center', fontsize=11)
ax.set_ylabel('coefficient on child work $l$')
ax.set_title('Specification ladder, wealth-homogeneous R2 economy (E3):\n'
             'selection masquerades as a return to child work')
fig2.tight_layout()
fig2.savefig('specification_ladder_noisy.png', dpi=150, bbox_inches='tight')
print("figure saved: specification_ladder_noisy.png")
print(f"\nE3 decomposition: beta_naive {o['b_naive']:+.3f} = displacement {o['disp_term']:+.3f}"
      f" + selection {o['sel_term']:+.3f}  (gap {o['identity_gap']:+.1e})")
