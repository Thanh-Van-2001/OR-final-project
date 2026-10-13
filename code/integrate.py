"""Integrated policy analysis: combine M4 pricing, M3 guidance and M2 investment on a common yardstick.

For growth g in {1.5, 2, 2.5} and policies {none, pricing, guidance (3 km), pricing + guidance}:
  unstable zones, piles needed for stability, and the minimum MILP budget (new = 1, relocation = 0.35)
  that brings the average wait below TARGET_MIN minutes per driver (bisection on B with the M2 model).
Also: LP-relaxation shadow price of the budget along the budget axis (baseline, g = 2).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load, load_profiles, OUT, FIG
import m2_ip_allocation as m2
import m3_network_flow as m3
import m4_pricing_nlp as m4

TARGET_MIN = 0.25          # average waiting minutes per driver
DMAX = 3


def guidance(A, c, dist, dmax=DMAX):
    newA = A.copy()
    for h in range(24):
        r = m3.redirect(A[h], c, dist, dmax)
        for i, j, fl in r['flows']:
            newA[h, i] -= fl; newA[h, j] += fl
    return newA


def min_budget(A, target_hours, lo=0, hi=6000):
    info = m2.build(A=A)
    need = sum(i['need'] for i in info)
    if need == 0 and sum(i['W_base'] for i in info) <= target_hours:
        return 0.0, info
    while solve_ok(info, hi, target_hours) is False:
        hi *= 2
    for _ in range(14):                                   # bisection to ~0.5 unit
        mid = (lo + hi) / 2
        if solve_ok(info, mid, target_hours):
            hi = mid
        else:
            lo = mid
    return hi, info


def solve_ok(info, B, target):
    s = m2.solve(info, B)
    return s is not None and s['total_wait'] <= target


def main():
    d = load(); zt = d['zt']; mean, _ = load_profiles()
    zones = list(mean.columns); c = zt.piles.reindex(zones).values.astype(float)
    dist = d['dist'].reindex(index=zones, columns=zones).values
    beta = float(m4.estimate_beta()['beta'])
    rows = []
    for g in [1.5, 2.0, 2.5]:
        A0 = g * mean.values
        mm = m4.Model(beta, g); pr = m4.optimise(mm, 'A'); Ap = mm.loads(pr['p'])
        arrivals = A0.sum()                                # per day (tau = 1 h)
        target_h = TARGET_MIN / 60 * arrivals
        for name, A in [('none', A0), ('pricing', Ap), ('guidance_3km', guidance(A0, c, dist)), ('pricing+guidance', guidance(Ap, c, dist))]:
            unst = int((A.max(axis=0) >= c).sum()); need = int(np.maximum(0, np.floor(A.max(axis=0)) + 1 - c).sum())
            B, _ = min_budget(A, target_h)
            rows.append(dict(growth=g, policy=name, unstable_zones=unst, piles_needed=need, min_budget_for_target=round(B, 1),
                             target_wait_hours_day=round(target_h, 1)))
            print(rows[-1])
    df = pd.DataFrame(rows); df.to_csv(OUT / 'integrated_policies.csv', index=False)

    # shadow price of budget (LP relaxation) along the budget axis, baseline g = 2
    info = m2.build()
    sp = []
    for B in [250, 300, 400, 500, 600, 800, 1000, 1500, 2000]:
        lp = m2.solve(info, B, relax=True)
        sp.append(dict(budget=B, shadow_hours_per_unit=float(-lp.ineqlin.marginals[0]),
                       lp_wait=float(lp.fun + sum(i['W_base'] for i in info))))
    pd.DataFrame(sp).to_csv(OUT / 'm2_shadow_prices.csv', index=False)
    print(pd.DataFrame(sp))
    json.dump(dict(policies=rows, shadow_prices=sp, target_min_per_driver=TARGET_MIN, dmax=DMAX), open(OUT / 'integrated_summary.json', 'w'), indent=2)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    pols = ['none', 'pricing', 'guidance_3km', 'pricing+guidance']; colors = ['#636363', '#fd8d3c', '#6baed6', '#31a354']
    w = 0.2
    for k, p in enumerate(pols):
        sub = df[df.policy == p]
        ax.bar(np.arange(3) + (k - 1.5) * w, sub.min_budget_for_target, w, label=p, color=colors[k])
    ax.set_xticks(range(3)); ax.set_xticklabels([f'demand x{g}' for g in [1.5, 2.0, 2.5]])
    ax.set_ylabel('min budget (new-pile units)'); ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')
    ax.set_title(f'Budget needed for average wait <= {TARGET_MIN*60:.0f} s per driver (MILP, relocation cost 0.35)', fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / 'integrated_budget.png', dpi=200); plt.close()


if __name__ == '__main__':
    main()
