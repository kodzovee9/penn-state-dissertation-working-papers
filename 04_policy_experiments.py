"""
Policy experiments: credit expansion and schooling subsidy (Figures 5-6).

Tracks real aggregates (child work, human capital) and the naive regression
coefficient an outside evaluator would estimate in each policy state, in the
positive-selection economies E2 and E3 (sigma = 3). See the paper, Section 5.
"""

import numpy as np

BASE = dict(alpha=0.5, w=0.3, w2=3.5, theta=0.25, R=1.4, beta=0.8)
LMAX = 0.92
S_ABILITY = 0.35


def solve_economy(a, A, sigma, phi, delta, dbar, p):
    alpha, w, w2, theta, R, beta = p['alpha'], p['w'], p['w2'], p['theta'], p['R'], p['beta']
    w1 = w * a**phi

    rhs = R * (w1 + theta) / w2 + delta
    s_u = (rhs / (a * alpha))**(1.0 / (alpha - 1.0))
    l_u = np.clip(1.0 - s_u, 0.0, LMAX)
    s_u = 1.0 - l_u
    H_u = a * s_u**alpha + delta * l_u
    kappa = (beta * R)**(-1.0 / sigma)
    d_u = (kappa * w2 * H_u - (A + (w1 + theta) * l_u - theta)) / (1.0 + kappa * R)
    unc = d_u <= dbar

    l = np.where(unc, l_u, np.nan)
    con = ~unc
    if con.any():
        ac, Ac, w1c = a[con], A[con], w1[con]

        def hfun(lc):
            return ac * (1.0 - lc)**alpha + delta * lc

        def G(lc):
            sc = 1.0 - lc
            c1 = Ac + (w1c + theta) * lc - theta + dbar
            c2 = w2 * hfun(lc) - R * dbar
            m = ac * alpha * sc**(alpha - 1.0) - delta
            with np.errstate(invalid='ignore'):
                return (w1c + theta) * np.where(c1 > 0, c1, np.nan)**(-sigma) \
                    - beta * w2 * m * np.where(c2 > 0, c2, np.nan)**(-sigma)

        lo = np.maximum(1e-9, (theta - Ac - dbar) / (w1c + theta) + 1e-9)
        # upper bound: c2 > 0  =>  a s^alpha + delta l > R dbar / w2
        hi = np.full_like(ac, LMAX)
        if dbar > 0:
            s_min = (R * dbar / (BASE['w2'] * ac))**(1.0 / alpha)   # delta=0 bound; safe for small delta
            hi = np.minimum(hi, 1.0 - np.minimum(s_min * 1.02 + 1e-6, 0.999))
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


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y))] + [x for x in X])
    return np.linalg.lstsq(X1, y, rcond=None)[0][1:]


def run(N, sigma, phi, mu_A, s_A, delta, dbar, theta, seed=42, noise=0.3):
    p = dict(BASE, theta=theta)
    rng = np.random.default_rng(seed)
    lna = rng.normal(0.0, S_ABILITY, N)
    a = np.exp(lna)
    A = np.exp(rng.normal(mu_A, s_A, N))
    l, con = solve_economy(a, A, sigma, phi, delta, dbar, p)
    s = 1.0 - l
    h = a * s**p['alpha'] + delta * l
    m = a * p['alpha'] * s**(p['alpha'] - 1.0) - delta
    lny2 = np.log(p['w2'] * h - p['R'] * dbar * con) + rng.normal(0.0, noise, N)
    # note: adult consumption net of repayment only for constrained borrowers;
    # earnings regressions use gross earnings w2*h:
    lny2 = np.log(p['w2'] * h) + rng.normal(0.0, noise, N)

    work = l > 1e-7
    interior = work & (l < LMAX - 1e-7)
    c2 = p['w2'] * h - p['R'] * dbar
    sigstar = np.where(con & interior, c2 / (p['w2'] * h), np.nan)
    with np.errstate(invalid='ignore'):
        possel = np.nanmean((sigma > sigstar)[con & interior]) if (con & interior).any() else np.nan

    return dict(
        mean_l=l.mean(), part=work.mean(), mean_h=h.mean(), mean_s=s.mean(),
        con=con.mean(), possel=possel,
        b_naive=ols(lny2, [l])[0], b_ctrl=ols(lny2, [l, lna])[0],
        marg=(-m[work] / h[work]).mean() if work.any() else np.nan,
    )


ECONS = [("E2 (s_A=0.35)", 0.7, 0.35), ("E3 (s_A=0.15)", 0.7, 0.15)]
N = 20000

print("=" * 108)
print("P1. CREDIT EXPANSION (dbar up; sigma=3, phi=0, delta=0, theta=0.25)")
print("=" * 108)
print(f"{'economy':<16}{'dbar':>6}{'con%':>6}{'part%':>7}{'mean l':>8}{'mean h':>8}{'possel%':>8}{'b_naive':>9}{'b_ctrl':>8}{'E[-m/h]':>9}")
p1 = {}
for name, muA, sA in ECONS:
    p1[name] = []
    for dbar in [0.0, 0.1, 0.2, 0.4, 0.6]:
        o = run(N, 3.0, 0.0, muA, sA, 0.0, dbar, 0.25)
        p1[name].append((dbar, o))
        print(f"{name:<16}{dbar:>6.1f}{o['con']:>6.0%}{o['part']:>7.0%}{o['mean_l']:>8.3f}"
              f"{o['mean_h']:>8.3f}{o['possel']:>8.0%}{o['b_naive']:>9.3f}{o['b_ctrl']:>8.3f}{o['marg']:>9.3f}")

print()
print("=" * 108)
print("P2. SCHOOLING SUBSIDY (theta down; sigma=3, phi=0, delta=0, dbar=0)")
print("=" * 108)
print(f"{'economy':<16}{'theta':>6}{'con%':>6}{'part%':>7}{'mean l':>8}{'mean h':>8}{'possel%':>8}{'b_naive':>9}{'b_ctrl':>8}{'E[-m/h]':>9}")
p2 = {}
for name, muA, sA in ECONS:
    p2[name] = []
    for theta in [0.25, 0.20, 0.15, 0.10, 0.05]:
        o = run(N, 3.0, 0.0, muA, sA, 0.0, 0.0, theta)
        p2[name].append((theta, o))
        print(f"{name:<16}{theta:>6.2f}{o['con']:>6.0%}{o['part']:>7.0%}{o['mean_l']:>8.3f}"
              f"{o['mean_h']:>8.3f}{o['possel']:>8.0%}{o['b_naive']:>9.3f}{o['b_ctrl']:>8.3f}{o['marg']:>9.3f}")

# ---------------- figure ----------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

cols = {"E2 (s_A=0.35)": '#4472c4', "E3 (s_A=0.15)": '#c0504d'}
marks = {"E2 (s_A=0.35)": 'o-', "E3 (s_A=0.15)": 's--'}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
for name, _, _ in ECONS:
    x = [d for d, _ in p1[name]]
    axes[0].plot(x, [o['mean_l'] for _, o in p1[name]], marks[name], color=cols[name], label=name, lw=2)
    axes[1].plot(x, [o['b_naive'] for _, o in p1[name]], marks[name], color=cols[name], label=name, lw=2)
for ax, ttl, yl in [(axes[0], '(a) Real effect', 'mean child work $l$'),
                    (axes[1], '(b) Naive coefficient', r'$\beta_{naive}$')]:
    ax.set_title(ttl); ax.set_ylabel(yl); ax.set_xlabel(r'borrowing limit $\bar d$')
    ax.legend(frameon=False, fontsize=12)
axes[1].axhline(0, color='k', lw=0.8)
fig.tight_layout()
fig.savefig('policy_credit.png', dpi=150, bbox_inches='tight')
print("\nfigure saved: policy_credit.png")

fig2, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
for name, _, _ in ECONS:
    x = [t for t, _ in p2[name]]
    axes[0].plot(x, [o['mean_l'] for _, o in p2[name]], marks[name], color=cols[name], label=name, lw=2)
    axes[1].plot(x, [o['b_naive'] for _, o in p2[name]], marks[name], color=cols[name], label=name, lw=2)
for ax, ttl, yl in [(axes[0], '(a) Real effect', 'mean child work $l$'),
                    (axes[1], '(b) Naive coefficient', r'$\beta_{naive}$')]:
    ax.set_title(ttl); ax.set_ylabel(yl); ax.set_xlabel(r'schooling cost $\theta$ (subsidy $\rightarrow$ left)')
    ax.legend(frameon=False, fontsize=12)
    ax.invert_xaxis()
axes[1].axhline(0, color='k', lw=0.8)
fig2.tight_layout()
fig2.savefig('policy_subsidy.png', dpi=150, bbox_inches='tight')
print("figure saved: policy_subsidy.png")
