"""M2 - Integer programming: budget-constrained expansion + relocation of charging piles.

Decision (per zone z): add u_z piles (new or relocated in) and remove v_z piles (relocated out).
    min  sum_z W_z(c_z + u_z - v_z)                       total expected waiting hours per day (M1)
    s.t. sum_z u_z = N_new + N_rel,   sum_z v_z = N_rel      (relocated piles leave donors, arrive at receivers)
         N_new + kappa * N_rel <= B                         budget (new pile = 1 unit, relocation = kappa)
         c_z + u_z - v_z >= s_z = floor(max_h a_zh) + 1     every zone stable
         v_z <= floor(0.5 c_z)                              at most half of a zone's piles can be moved
W_z is convex decreasing in the pile count, so the incremental (0-1) formulation
    u_z = sum_k x_zk,  v_z = sum_k y_zk,  W = W_z(c_z) - sum_k dplus_zk x_zk + sum_k dminus_zk y_zk
is exact: the marginal gains dplus_zk decrease in k and the marginal losses dminus_zk increase in k,
so an optimal solution always fills increments in order. Solved as a MILP with HiGHS (scipy.optimize.milp).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds, linprog
from common import load, load_profiles, wait_curve, OUT, FIG

G = 2.0          # planning scenario: charging demand x2 relative to 2022/23
KAPPA = 0.35     # relocation cost relative to a new pile
K_ADD = 120      # max piles added to one zone (beyond the stability requirement)


def build(g=G, A=None):
    """A: optional 24 x Z matrix of hourly loads (e.g. after pricing / guidance); default g * observed mean day."""
    d = load(); zt = d['zt']; mean, _ = load_profiles()
    zones = list(mean.columns)
    info = []
    for k, z in enumerate(zones):
        loads = g * mean[z].values if A is None else A[:, k]
        c0 = int(zt.piles[z]); s = int(np.floor(loads.max())) + 1
        need = max(0, s - c0)
        R = max(0, min(c0 - s, c0 // 2))
        W = wait_curve(loads, c0 + need + K_ADD)
        base_c = c0 + need
        dplus = np.array([W[base_c + k - 1] - W[base_c + k] for k in range(1, K_ADD + 1)])
        dminus = np.array([W[c0 - k] - W[c0 - k + 1] for k in range(1, R + 1)]) if need == 0 else np.array([])
        # keep only useful add increments (gain > 1e-4 h/day) to shrink the model
        dplus = dplus[dplus > 1e-4]
        info.append(dict(zone=z, c0=c0, stab=s, need=need, W_base=W[base_c], dplus=dplus, dminus=dminus, W=W))
    return info


def solve(info, B, kappa=KAPPA, allow_new=True, allow_rel=True, relax=False):
    nplus = [len(i['dplus']) for i in info]; nminus = [len(i['dminus']) for i in info]
    NX, NY = sum(nplus), sum(nminus)
    nv = NX + NY + 2                                   # ... + N_new, N_rel
    cvec = np.concatenate([-np.concatenate([i['dplus'] for i in info]) if NX else [],
                           np.concatenate([i['dminus'] for i in info]) if NY else [], [0.0, 0.0]])
    need_tot = sum(i['need'] for i in info)
    A = np.zeros((3, nv)); lb = np.zeros(3); ub = np.zeros(3)
    A[0, :NX] = 1; A[0, -2] = -1; A[0, -1] = -1; lb[0] = ub[0] = -need_tot   # sum adds (incl. forced) = N_new + N_rel
    A[1, NX:NX + NY] = 1; A[1, -1] = -1                                      # sum removals = N_rel
    A[2, -2] = 1; A[2, -1] = kappa; lb[2] = -np.inf; ub[2] = B               # budget
    hi = np.ones(nv); hi[-2] = np.inf if allow_new else 0; hi[-1] = np.inf if allow_rel else 0
    integrality = np.zeros(nv) if relax else np.ones(nv)
    if relax:
        res = linprog(cvec, A_ub=A[2:3], b_ub=[B], A_eq=A[:2], b_eq=[-need_tot, 0],
                      bounds=list(zip(np.zeros(nv), hi)), method='highs')
        return res
    res = milp(cvec, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=Bounds(np.zeros(nv), hi),
               options={'time_limit': 120, 'mip_rel_gap': 1e-6})
    if res.x is None:
        return None
    x = np.round(res.x).astype(int)
    out, pos, posy = [], 0, NX
    for i, n1, n2 in zip(info, nplus, nminus):
        add = i['need'] + x[pos:pos + n1].sum(); rem = x[posy:posy + n2].sum()
        pos += n1; posy += n2
        out.append(dict(zone=i['zone'], c0=i['c0'], add=int(add), remove=int(rem), c_new=int(i['c0'] + add - rem),
                        wait=float(i['W'][i['c0'] + add - rem])))
    df = pd.DataFrame(out).set_index('zone')
    return dict(plan=df, total_wait=float(df.wait.sum()), n_new=int(x[-2]), n_rel=int(x[-1]),
                cost=float(x[-2] + kappa * x[-1]), status=res.status, gap=getattr(res, 'mip_gap', None))


def proportional(info, B):
    """Naive practice: B new piles split in proportion to current pile counts (largest remainder)."""
    c0 = np.array([i['c0'] for i in info]); share = B * c0 / c0.sum()
    add = np.floor(share).astype(int); rem = B - add.sum()
    add[np.argsort(-(share - add))[:rem]] += 1
    W = [i['W'][min(i['c0'] + a, len(i['W']) - 1)] if i['c0'] + a < len(i['W']) else 0.0 for i, a in zip(info, add)]
    W = np.array(W)
    return dict(total_wait=float(W[np.isfinite(W)].sum()), unstable=int(np.isinf(W).sum()))


def main():
    info = build()
    need_tot = sum(i['need'] for i in info)
    removable = sum(len(i['dminus']) for i in info)
    unstable = sum(1 for i in info if i['need'] > 0)
    summary = dict(scenario_growth=G, kappa=KAPPA, zones_unstable_before=unstable, piles_needed_for_stability=need_tot,
                   piles_removable_from_donors=removable, min_cost_to_stabilise=KAPPA * min(need_tot, removable) + max(0, need_tot - removable),
                   wait_if_only_stabilised_minimally=None)
    budgets = [150, 250, 400, 600, 800, 1000, 1500, 2000, 3000]
    rows = []
    for B in budgets:
        r = {'budget': B}
        for name, kw in [('combined', {}), ('new_only', {'allow_rel': False}), ('relocation_only', {'allow_new': False})]:
            s = solve(info, B, **kw)
            r[name] = s['total_wait'] if s else np.nan
            if name == 'combined' and s:
                r['combined_new'] = s['n_new']; r['combined_rel'] = s['n_rel']
        p = proportional(info, B); r['proportional_wait_stable_zones'] = p['total_wait']; r['proportional_unstable_zones'] = p['unstable']
        rows.append(r); print(r)
    sweep = pd.DataFrame(rows); sweep.to_csv(OUT / 'm2_budget_sweep.csv', index=False)

    # base case B = 1000: plan, LP relaxation dual (shadow price of budget), kappa sensitivity
    B0 = 1000
    base = solve(info, B0); base['plan'].to_csv(OUT / 'm2_plan_B1000.csv')
    lp = solve(info, B0, relax=True)
    shadow = float(-lp.ineqlin.marginals[0])        # hours/day saved per extra budget unit
    summary.update(base_budget=B0, base_total_wait=base['total_wait'], base_new=base['n_new'], base_rel=base['n_rel'],
                   base_cost=base['cost'], lp_relaxation_obj_gap_hours=float(base['total_wait'] - (lp.fun + sum(i['W_base'] for i in info))),
                   budget_shadow_price_hours_per_unit=shadow,
                   zones_receiving=int((base['plan']['add'] > 0).sum()), zones_donating=int((base['plan']['remove'] > 0).sum()))
    kap = {}
    for k in [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
        s = solve(info, B0, kappa=k); kap[k] = dict(wait=s['total_wait'], new=s['n_new'], rel=s['n_rel'])
    summary['kappa_sensitivity_B1000'] = kap
    # growth sensitivity of the minimum stabilising effort
    gs = {}
    for g in [1.5, 2.0, 2.5]:
        inf2 = build(g); nd = sum(i['need'] for i in inf2); rm = sum(len(i['dminus']) for i in inf2)
        gs[g] = dict(piles_needed=nd, removable=rm, zones_unstable=sum(1 for i in inf2 if i['need'] > 0))
    summary['growth_sensitivity'] = gs
    json.dump(summary, open(OUT / 'm2_summary.json', 'w'), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))

    # ---- figures
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for col, lab, st in [('combined', 'MILP: new + relocation', '-o'), ('new_only', 'MILP: new piles only', '-s'),
                         ('relocation_only', 'MILP: relocation only', '-^')]:
        ax.plot(sweep.budget, sweep[col], st, ms=4, label=lab)
    ax.set_yscale('log'); ax.set_xlabel('budget (new-pile cost units)'); ax.set_ylabel('total waiting (driver-hours/day)')
    ax.set_title(f'Waiting vs budget, demand x{G:g}, relocation cost {KAPPA} (naive proportional rule leaves zones unstable)', fontsize=8.5)
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / 'm2_budget_curve.png', dpi=200); plt.close()

    d = load(); zt = d['zt']; pl = base['plan']
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(zt.lon, zt.lat, s=6, c='#cccccc')
    rec = pl[pl['add'] > 0]; don = pl[pl['remove'] > 0]
    ax.scatter(zt.lon[rec.index], zt.lat[rec.index], s=10 + rec['add'] * 2, c='#d95f0e', alpha=.8, label='receives piles')
    ax.scatter(zt.lon[don.index], zt.lat[don.index], s=10 + don['remove'] * 2, c='#2b8cbe', alpha=.8, label='donates piles')
    ax.set_aspect(1 / np.cos(np.radians(22.6))); ax.legend(fontsize=8)
    ax.set_title(f'Optimal plan, budget {B0}: {base["n_new"]} new + {base["n_rel"]} relocated piles', fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / 'm2_plan_map.png', dpi=200); plt.close()


if __name__ == '__main__':
    main()
