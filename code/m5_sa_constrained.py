"""M5 variant: simulated annealing with the stability constraint of M2 enforced.

The SA in m5_simulation_sa.py only requires x_i >= 0, so it may leave a zone with fewer piles than its peak load
(it relies on drivers diverting). Here a move i -> j is allowed only if zone i keeps c_i + x_i > max_h a_ih, so the
SA plan and the Erlang-greedy plan are compared on the same feasible set. Same seeds, CRN streams, cooling schedule
and iteration budget as the unconstrained run; then the same out-of-sample and robustness evaluations as
m5_robustness.py. Output: results/m5_sa_constrained.json (+ column sa_stable_add in results/m5_sa_constrained.csv).
"""
import json
import math
import time
import numpy as np
import pandas as pd
import m5_simulation_sa as m5
from m5_robustness import evaluate
from common import load, load_profiles, OUT


def main():
    t0 = time.time()
    d = load(); mean, _ = load_profiles()
    zones = list(mean.columns); A = m5.G * mean.values
    alloc = pd.read_csv(OUT / 'm5_allocations.csv')
    idx = [zones.index(z) for z in alloc.zone]
    c_c = alloc.piles.values; need = alloc.need.values; x_e = alloc.erlang_add.values; x_s = alloc.sa_add.values
    rates = A[:, idx] / m5.TAU
    dist = d['dist'].reindex(index=zones, columns=zones).values[np.ix_(idx, idx)]
    K = len(idx)
    crn = [m5.gen_streams(rates, m5.DAYS, s) for s in (11, 12)]

    def J(xv):
        return float(np.mean([m5.simulate(c_c + xv, st, dist, m5.DAYS)['J'] for st in crn]))

    rng = np.random.default_rng(7)
    cur = x_e.copy(); fcur = J(cur); best, fbest = cur.copy(), fcur
    T = 0.05 * fcur; hist = [(0, fcur, fbest)]; rejected_infeasible = 0
    for it in range(1, 251):
        i, j = rng.choice(K, 2, replace=False)
        if cur[i] - 1 < need[i]:                 # donor would fall to or below its peak load: infeasible
            rejected_infeasible += 1
            continue
        cand = cur.copy(); cand[i] -= 1; cand[j] += 1
        fc = J(cand)
        if fc < fcur or rng.random() < math.exp(-(fc - fcur) / max(T, 1e-9)):
            cur, fcur = cand, fc
            if fc < fbest:
                best, fbest = cand.copy(), fc
        T *= 0.985
        hist.append((it, fcur, fbest))
    peak = A[:, idx].max(axis=0)
    assert np.all(c_c + best > peak), 'constrained plan must be stable'
    out = dict(sa_iterations=250, infeasible_moves_rejected=rejected_infeasible, best_J_crn=fbest, start_J_crn=hist[0][1],
               piles_moved_vs_erlang=int(np.abs(best - x_e).sum() // 2),
               base=evaluate(c_c, x_e, best, rates, dist),
               no_diversion=evaluate(c_c, x_e, best, rates, dist, divert=False),
               q1=evaluate(c_c, x_e, best, rates, dist, q_div=1), q5=evaluate(c_c, x_e, best, rates, dist, q_div=5),
               r2=evaluate(c_c, x_e, best, rates, dist, r_div=2.0), r5=evaluate(c_c, x_e, best, rates, dist, r_div=5.0),
               cv05=evaluate(c_c, x_e, best, rates, dist, cv=0.5), cv10=evaluate(c_c, x_e, best, rates, dist, cv=1.0),
               unconstrained_plan_unstable_zones=[int(alloc.zone[k]) for k in range(K) if c_c[k] + x_s[k] <= peak[k]],
               history=hist)
    out['runtime_s'] = time.time() - t0
    json.dump(out, open(OUT / 'm5_sa_constrained.json', 'w'), indent=2, default=float)
    pd.DataFrame(dict(zone=alloc.zone, piles=c_c, need=need, erlang_add=x_e, sa_add=x_s, sa_stable_add=best)).to_csv(
        OUT / 'm5_sa_constrained.csv', index=False)
    print(json.dumps({k: v for k, v in out.items() if k != 'history'}, indent=1, default=float))


if __name__ == '__main__':
    main()
