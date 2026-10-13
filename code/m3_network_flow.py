"""M3 - Network optimisation: redirecting charging demand between zones (navigation guidance, no new piles).

For each hour h (demand scenario x G, current piles):
  excess e_i = max(0, a_ih - rho* c_i)  at overloaded zones,  spare s_j = max(0, rho* c_j - a_jh) at other zones
Min-cost flow / transportation problem on the bipartite network (arc i->j only if distance d_ij <= Dmax):
  min  sum_ij d_ij f_ij + M sum_i r_i
  s.t. sum_j f_ij + r_i = e_i   (redirect or leave unresolved),   sum_i f_ij <= s_j,   f, r >= 0
The constraint matrix is totally unimodular, so integer data give integer optimal flows. Solved as an LP (HiGHS)
and cross-checked with the network simplex algorithm (networkx) on integer flows of 0.1 busy pile.
Loads are in Erlangs (busy piles); divided by tau they become drivers per hour.
"""
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from common import load, load_profiles, wait_curve, OUT, FIG, RHO_TARGET

G = 2.0
DMAX_LIST = [1, 2, 3, 5, 8]
BIG_M = 1000.0


def redirect(a, c, dist, dmax, rho=RHO_TARGET, compliance=1.0):
    """compliance = share of the excess whose drivers follow guidance (1 = everybody)."""
    e = compliance * np.maximum(0, a - rho * c); s = np.maximum(0, rho * c - a)
    I = np.where(e > 1e-9)[0]; J = np.where(s > 1e-9)[0]
    arcs = [(i, j, dist[i, j]) for i in I for j in J if i != j and dist[i, j] <= dmax]
    nI, nA = len(I), len(arcs)
    if nI == 0:
        return dict(excess=0.0, moved=0.0, cost_km=0.0, flows=[])
    ipos = {i: k for k, i in enumerate(I)}; jpos = {j: k for k, j in enumerate(J)}
    cvec = np.concatenate([[d for _, _, d in arcs], np.full(nI, BIG_M)])
    rows, cols, vals = [], [], []
    for k, (i, j, _) in enumerate(arcs):
        rows.append(ipos[i]); cols.append(k); vals.append(1.0)
    for k in range(nI):
        rows.append(k); cols.append(nA + k); vals.append(1.0)
    Aeq = coo_matrix((vals, (rows, cols)), shape=(nI, nA + nI))
    rows, cols, vals = [], [], []
    for k, (i, j, _) in enumerate(arcs):
        rows.append(jpos[j]); cols.append(k); vals.append(1.0)
    Aub = coo_matrix((vals, (rows, cols)), shape=(len(J), nA + nI))
    res = linprog(cvec, A_ub=Aub, b_ub=s[J], A_eq=Aeq, b_eq=e[I], bounds=(0, None), method='highs')
    f = res.x[:nA]
    flows = [(i, j, fl) for (i, j, _), fl in zip(arcs, f) if fl > 1e-9]
    moved = float(f.sum())
    return dict(excess=float(e.sum()), moved=moved, cost_km=float(sum(d * fl for (_, _, d), fl in zip(arcs, f))), flows=flows,
                duals_capacity=res.ineqlin.marginals)


def network_simplex_check(a, c, dist, dmax, rho=RHO_TARGET, unit=0.1):
    """Same problem as a min-cost flow graph solved by the network simplex (integer units of 0.1 busy pile)."""
    e = np.floor(np.maximum(0, a - rho * c) / unit).astype(int); s = np.floor(np.maximum(0, rho * c - a) / unit).astype(int)
    Gph = nx.DiGraph()
    tot = int(e.sum())
    Gph.add_node('S', demand=-tot); Gph.add_node('T', demand=tot)
    for i in np.where(e > 0)[0]:
        Gph.add_edge('S', ('o', i), capacity=int(e[i]), weight=0)
        Gph.add_edge(('o', i), 'T', capacity=int(e[i]), weight=int(BIG_M * 100))       # unresolved
        for j in np.where(s > 0)[0]:
            if i != j and dist[i, j] <= dmax:
                Gph.add_edge(('o', i), ('d', j), capacity=int(e[i]), weight=int(round(dist[i, j] * 100)))
    for j in np.where(s > 0)[0]:
        Gph.add_edge(('d', j), 'T', capacity=int(s[j]), weight=0)
    cost, flow = nx.network_simplex(Gph)
    moved = sum(v for u, dct in flow.items() if isinstance(u, tuple) and u[0] == 'o' for w, v in dct.items() if isinstance(w, tuple))
    return moved * unit


def max_flow_min_cut(a, c, dist, dmax, rho=RHO_TARGET, unit=0.01):
    """Max-flow / min-cut view of guidance: S -> overloaded i (cap e_i) -> reachable j (uncapacitated) -> T (cap s_j).
    The max flow is the largest redirectable excess (the min-cost-flow LP above reaches it, because M > any detour).
    The min cut certifies it: cut = sum of e_i over zones on the sink side (fully absorbed) + sum of s_j over
    receiving zones on the source side (saturated neighbourhoods). Source-side overloaded zones form the set K
    whose excess exceeds the spare capacity of all zones within dmax (Hall-type deficiency) -> where piles are needed."""
    e = np.floor(np.maximum(0, a - rho * c) / unit).astype(int); s = np.floor(np.maximum(0, rho * c - a) / unit).astype(int)
    Gph = nx.DiGraph()
    I = np.where(e > 0)[0]; J = np.where(s > 0)[0]
    for i in I:
        Gph.add_edge('S', ('o', i), capacity=int(e[i]))
        for j in J:
            if i != j and dist[i, j] <= dmax:
                Gph.add_edge(('o', i), ('d', j))                 # no capacity attribute = infinite
    for j in J:
        Gph.add_edge(('d', j), 'T', capacity=int(s[j]))
    cut, (S_side, T_side) = nx.minimum_cut(Gph, 'S', 'T')
    K = [n[1] for n in S_side if isinstance(n, tuple) and n[0] == 'o']
    NK = [n[1] for n in S_side if isinstance(n, tuple) and n[0] == 'd']
    isolated = [i for i in I if not any(dist[i, j] <= dmax and i != j for j in J)]
    return dict(max_flow=cut * unit, excess_total=float(e.sum() * unit),
                cut_from_absorbed_zones=float(sum(e[i] for i in I if i not in K) * unit),
                cut_from_saturated_receivers=float(sum(s[j] for j in NK) * unit),
                deficient_set_size=len(K), deficient_set_excess=float(sum(e[i] for i in K) * unit),
                deficient_set_neighbour_spare=float(sum(s[j] for j in NK) * unit),
                saturated_receivers=len(NK), isolated_overloaded_zones=len(isolated), overloaded_zones=len(I),
                deficient_zone_ids=[int(i) for i in K], saturated_receiver_ids=[int(j) for j in NK])


def rho_sensitivity(A, c, dist, dmax=3, rhos=(0.75, 0.80, 0.85, 0.90, 0.95), compliance=1.0, key='rho'):
    """Guidance results vs the planning target utilisation rho* (Dmax fixed), or vs compliance (key='compliance')."""
    rows = []
    for rho in (rhos if key == 'rho' else [RHO_TARGET] * len(compliance)):
        comp = 1.0 if key == 'rho' else compliance[len(rows)]
        tot_e = tot_m = tot_km = 0.0; newA = A.copy()
        for h in range(24):
            r = redirect(A[h], c, dist, dmax, rho, comp)
            tot_e += r['excess']; tot_m += r['moved']; tot_km += r['cost_km']
            for i, j, fl in r['flows']:
                newA[h, i] -= fl; newA[h, j] += fl
        rows.append(dict(rho=rho, compliance=comp, share_redirected=tot_m / tot_e, mean_detour_km=tot_km / max(tot_m, 1e-9),
                         unstable_zones_after=int((newA.max(axis=0) >= c).sum()),
                         piles_needed_after=int(np.maximum(0, np.floor(newA.max(axis=0)) + 1 - c).sum()),
                         max_util_after=float((newA / c).max())))
    return rows


def main():
    d = load(); zt = d['zt']; mean, _ = load_profiles()
    zones = list(mean.columns); c = zt.piles.reindex(zones).values.astype(float)
    dist = d['dist'].reindex(index=zones, columns=zones).values
    A = G * mean.values                                    # 24 x Z
    rows = []
    best = None
    for dmax in DMAX_LIST:
        tot_e = tot_m = tot_km = 0.0
        newA = A.copy()
        for h in range(24):
            r = redirect(A[h], c, dist, dmax)
            tot_e += r['excess']; tot_m += r['moved']; tot_km += r['cost_km']
            for i, j, fl in r['flows']:
                newA[h, i] -= fl; newA[h, j] += fl
        # stability after guidance: piles still needed so that every zone has c > max_h load (links to M2)
        need_b = int(np.maximum(0, np.floor(A.max(axis=0)) + 1 - c).sum())
        need_a = int(np.maximum(0, np.floor(newA.max(axis=0)) + 1 - c).sum())
        unst_b = int((A.max(axis=0) >= c).sum()); unst_a = int((newA.max(axis=0) >= c).sum())
        rows.append(dict(dmax_km=dmax, excess_erlang_hours=tot_e, redirected_erlang_hours=tot_m, share_redirected=tot_m / tot_e,
                         mean_detour_km=tot_km / max(tot_m, 1e-9), unstable_zones_before=unst_b, unstable_zones_after=unst_a,
                         piles_needed_before=need_b, piles_needed_after=need_a))
        print(rows[-1])
    df = pd.DataFrame(rows); df.to_csv(OUT / 'm3_guidance.csv', index=False)

    # peak hour detail and network-simplex cross-check (Dmax = 3 km)
    h_peak = int(np.argmax(A.sum(axis=1)))
    r = redirect(A[h_peak], c, dist, 3)
    ns = network_simplex_check(A[h_peak], c, dist, 3)
    # capacity duals: value (km saved) of one extra unit of spare capacity at each receiving zone
    duals = -r['duals_capacity']          # = value of one more unit of spare capacity at a receiving zone (km-equivalent)
    summary = dict(scenario_growth=G, rho_target=RHO_TARGET, peak_hour=h_peak,
                   peak_excess=r['excess'], peak_redirected_lp=r['moved'], peak_redirected_network_simplex=ns,
                   peak_flows=len(r['flows']), receiving_zones_capacity_binding=int((duals > 1e-6).sum()),
                   receiving_zones_total=int(len(duals)), results=rows)
    mfmc = max_flow_min_cut(A[h_peak], c, dist, 3)
    # do the saturated receivers of the min cut coincide with the receiving zones whose capacity dual is positive?
    J = np.where(np.maximum(0, RHO_TARGET * c - A[h_peak]) > 1e-9)[0]
    dual_pos = set(int(J[k]) for k in np.where(duals > 1e-6)[0])
    sat = set(mfmc.pop('saturated_receiver_ids'))
    summary['max_flow_min_cut_peak'] = {k: v for k, v in mfmc.items() if k != 'deficient_zone_ids'}
    summary['max_flow_min_cut_peak']['deficient_zones'] = [int(zones[i]) for i in mfmc['deficient_zone_ids']]
    summary['max_flow_min_cut_peak']['lp_redirected_same_hour'] = r['moved']
    summary['max_flow_min_cut_peak']['saturated_receivers_with_positive_dual'] = len(sat & dual_pos)
    summary['rho_sensitivity_dmax3'] = rho_sensitivity(A, c, dist)
    # share_redirected is relative to the complying excess; piles needed is the planning-relevant measure
    summary['compliance_sensitivity_dmax3'] = rho_sensitivity(A, c, dist, compliance=(0.5, 0.75, 1.0), key='compliance')
    json.dump(summary, open(OUT / 'm3_summary.json', 'w'), indent=2, default=float)
    print({k: v for k, v in summary.items() if k != 'results'})

    fig, ax1 = plt.subplots(figsize=(6.5, 3.4))
    ax1.plot(df.dmax_km, 100 * df.share_redirected, '-o', color='#08519c', label='excess load redirected (%)')
    ax1.set_xlabel('maximum detour distance Dmax (km)'); ax1.set_ylabel('% of excess redirected', color='#08519c'); ax1.set_ylim(0, 105)
    ax2 = ax1.twinx(); ax2.plot(df.dmax_km, df.unstable_zones_after, '-s', color='#e6550d', label='unstable zones after')
    ax2.set_ylabel('unstable zones after guidance', color='#e6550d')
    ax1.set_title(f'Guidance only (no new piles), demand x{G:g}, target utilisation {RHO_TARGET}', fontsize=9); ax1.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / 'm3_guidance_curve.png', dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(zt.lon, zt.lat, s=5, c='#cccccc')
    lon = zt.lon.reindex(zones).values; lat = zt.lat.reindex(zones).values
    for i, j, fl in r['flows']:
        ax.annotate('', xy=(lon[j], lat[j]), xytext=(lon[i], lat[i]),
                    arrowprops=dict(arrowstyle='->', color='#d95f0e', lw=0.4 + fl / 4, alpha=.8))
    ax.set_aspect(1 / np.cos(np.radians(22.6)))
    ax.set_title(f'Optimal redirection flows at peak hour {h_peak}:00 (Dmax = 3 km)', fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / 'm3_flows_map.png', dpi=200); plt.close()


if __name__ == '__main__':
    main()
