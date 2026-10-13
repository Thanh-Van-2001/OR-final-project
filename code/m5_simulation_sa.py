"""M5 - Discrete-event simulation + simulated annealing (simulation-optimisation) with driver spill-over.

Why: M1/M2 treat zones as independent stationary M/M/c queues. Real drivers (i) arrive at a time-varying rate and
(ii) divert to a nearby zone when the local queue is long. Both effects are captured by a discrete-event simulation
of a 30-zone cluster around the most overloaded zone (demand x2).

Simulation (per replication, D days after one warm-up day):
  arrivals: non-homogeneous Poisson, rate a_zh / tau in hour h (thinning); session length lognormal(mean tau, CV)
  on arrival at zone i: free pile -> start; else if queue_i >= q_div -> divert to the nearest zone within 3 km that has a
  free pile (detour cost); else join the FIFO queue of i.
  objective J = waiting hours/day + ALPHA * detour km/day   (ALPHA = 0.05 h per km, i.e. 3 min per km)
Verification: stationary arrivals + exponential service + no diversion reproduce Erlang-C.
Optimisation: allocate B new piles in the cluster (no removals). Start = Erlang-greedy allocation (optimal for the
separable M/M/c model, i.e. what M2 would do); simulated annealing moves one pile between zones, evaluated with
common random numbers; both plans are then re-evaluated on fresh random numbers (out-of-sample, 95% CI).
"""
import heapq
import json
import math
import time
from collections import deque
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load, load_profiles, wait_curve, erlang_c, OUT, FIG

G = 2.0; TAU = 1.0; CV = 0.8; ALPHA = 0.05; Q_DIV = 3; R_DIV = 3.0; K = 30; DAYS = 4


def lognormal_params(mean, cv):
    s2 = math.log(1 + cv * cv); return math.log(mean) - s2 / 2, math.sqrt(s2)


def gen_streams(rates, days, seed, cv=CV, tau=TAU):
    """rates: 24 x n arrival rates per hour. Returns per-zone arrays of (arrival time, service time)."""
    rng = np.random.default_rng(seed)
    mu, sg = lognormal_params(tau, cv)
    H = 24 * (days + 1)
    out = []
    for z in range(rates.shape[1]):
        lam_max = rates[:, z].max()
        if lam_max <= 0:
            out.append((np.array([]), np.array([]))); continue
        n = rng.poisson(lam_max * H)
        t = np.sort(rng.uniform(0, H, n))
        keep = rng.uniform(0, 1, n) < rates[(t.astype(int)) % 24, z] / lam_max
        t = t[keep]
        s = rng.lognormal(mu, sg, len(t)) if cv > 0 else np.full(len(t), tau)
        out.append((t, s))
    return out


def simulate(c, streams, dist, days, divert=True, q_div=Q_DIV):
    n = len(c)
    ev = []
    for z, (t, s) in enumerate(streams):
        for k in range(len(t)):
            ev.append((t[k], 1, z, s[k]))
    heapq.heapify(ev)
    busy = np.zeros(n, int); queue = [deque() for _ in range(n)]
    warm = 24.0; T_end = 24.0 * (days + 1)
    wait = detour = 0.0; ndiv = 0; nserved = 0; narr = 0
    near = [[j for j in np.argsort(dist[i]) if j != i and dist[i, j] <= R_DIV] for i in range(n)]
    while ev:
        t, kind, z, s = heapq.heappop(ev)
        if t > T_end:
            break
        if kind == 1:                                   # arrival
            narr += t >= warm
            if busy[z] < c[z]:
                busy[z] += 1; heapq.heappush(ev, (t + s, 0, z, 0.0)); nserved += t >= warm
                continue
            if divert and len(queue[z]) >= q_div:
                for j in near[z]:
                    if busy[j] < c[j]:
                        busy[j] += 1; heapq.heappush(ev, (t + s, 0, j, 0.0))
                        if t >= warm: detour += dist[z, j]; ndiv += 1; nserved += 1
                        break
                else:
                    queue[z].append((t, s))
                continue
            queue[z].append((t, s))
        else:                                           # departure
            if queue[z]:
                ta, sa = queue[z].popleft()
                if ta >= warm: wait += t - ta; nserved += 1
                heapq.heappush(ev, (t + sa, 0, z, 0.0))
            else:
                busy[z] -= 1
    left = sum(max(0.0, T_end - ta) for q in queue for ta, _ in q if ta >= warm)   # still waiting at the end
    wait += left
    return dict(wait_h_day=wait / days, detour_km_day=detour / days, diversions_day=ndiv / days, arrivals=narr,
                J=(wait + ALPHA * detour) / days)


def verify_against_erlang():
    """Stationary Poisson arrivals, exponential service, no diversion: simulated Wq vs Erlang-C."""
    rows = []
    for c, a in [(5, 4.0), (20, 17.0), (50, 45.0)]:
        rates = np.full((24, 1), a / TAU)
        sims = []
        for rep in range(8):
            st = gen_streams(rates, 60, 1000 + rep, cv=1.0)
            st = [(t, np.random.default_rng(2000 + rep).exponential(TAU, len(t))) for t, _ in st]
            r = simulate(np.array([c]), st, np.zeros((1, 1)), 60, divert=False)
            sims.append(r['wait_h_day'] * 60 / (r['arrivals'] / 60))  # minutes per arriving driver
        th = erlang_c(c, a) * TAU / (c - a) * 60
        rows.append(dict(c=c, a=a, erlang_wq_min=th, sim_wq_min=float(np.mean(sims)), sim_se=float(np.std(sims, ddof=1) / np.sqrt(8))))
    return rows


def main():
    t0 = time.time()
    d = load(); zt = d['zt']; mean, _ = load_profiles()
    zones = list(mean.columns); A = G * mean.values; c0 = zt.piles.reindex(zones).values.astype(int)
    dist_all = d['dist'].reindex(index=zones, columns=zones).values
    overload = (A.max(axis=0) - c0)
    hub = int(np.argmax(overload))
    idx = np.argsort(dist_all[hub])[:K]
    A_c = A[:, idx]; c_c = c0[idx]; dist = dist_all[np.ix_(idx, idx)]; zc = [zones[i] for i in idx]
    rates = A_c / TAU
    need = np.maximum(0, np.floor(A_c.max(axis=0)).astype(int) + 1 - c_c)
    B = int(need.sum() + 40)                          # budget: stabilise + 40 extra piles
    # Erlang-greedy (separable, stationary per hour) allocation = M2 logic restricted to the cluster, new piles only
    curves = [wait_curve(A_c[:, k], c_c[k] + need[k] + B) for k in range(K)]
    x = need.copy()
    for _ in range(B - need.sum()):
        gains = [curves[k][c_c[k] + x[k]] - curves[k][c_c[k] + x[k] + 1] for k in range(K)]
        x[int(np.argmax(gains))] += 1
    x_erlang = x.copy()
    ver = verify_against_erlang(); print('verification', ver)

    crn = [gen_streams(rates, DAYS, s) for s in (11, 12)]           # common random numbers for the search
    def J(xv, streams=crn, divert=True):
        return float(np.mean([simulate(c_c + xv, st, dist, DAYS, divert)['J'] for st in streams]))

    # SIPP (Erlang per hour) vs simulation without diversion, for the Erlang plan
    sipp = float(sum(curves[k][c_c[k] + x_erlang[k]] for k in range(K)))
    sim_nodiv = J(x_erlang, divert=False)
    # simulated annealing
    rng = np.random.default_rng(7)
    cur = x_erlang.copy(); fcur = J(cur); best, fbest = cur.copy(), fcur
    T = 0.05 * fcur; hist = [(0, fcur, fbest)]
    iters = 250
    for it in range(1, iters + 1):
        i, j = rng.choice(K, 2, replace=False)
        if cur[i] <= 0:
            continue
        cand = cur.copy(); cand[i] -= 1; cand[j] += 1
        fc = J(cand)
        if fc < fcur or rng.random() < math.exp(-(fc - fcur) / max(T, 1e-9)):
            cur, fcur = cand, fc
            if fc < fbest:
                best, fbest = cand.copy(), fc
        T *= 0.985
        hist.append((it, fcur, fbest))
    # out-of-sample evaluation (fresh seeds)
    test = [gen_streams(rates, DAYS, s) for s in range(100, 110)]
    ev_e = [simulate(c_c + x_erlang, st, dist, DAYS) for st in test]
    ev_s = [simulate(c_c + best, st, dist, DAYS) for st in test]
    diff = np.array([e['J'] - s['J'] for e, s in zip(ev_e, ev_s)])
    ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    summ = lambda L, k: float(np.mean([r[k] for r in L]))
    out = dict(cluster_hub_zone=int(zones[hub]), cluster_zones=[int(z) for z in zc], budget_new_piles=B, piles_needed=int(need.sum()),
               verification_erlang=ver, sipp_wait_erlang_plan=sipp, sim_wait_no_diversion_erlang_plan=sim_nodiv,
               erlang_plan_oos=dict(J=summ(ev_e, 'J'), wait=summ(ev_e, 'wait_h_day'), detour=summ(ev_e, 'detour_km_day'), diversions=summ(ev_e, 'diversions_day')),
               sa_plan_oos=dict(J=summ(ev_s, 'J'), wait=summ(ev_s, 'wait_h_day'), detour=summ(ev_s, 'detour_km_day'), diversions=summ(ev_s, 'diversions_day')),
               improvement_J_mean=float(diff.mean()), improvement_J_ci95=float(ci),
               improvement_pct=float(100 * diff.mean() / summ(ev_e, 'J')),
               piles_moved_vs_erlang=int(np.abs(best - x_erlang).sum() // 2), sa_iterations=iters,
               params=dict(G=G, TAU=TAU, CV=CV, ALPHA=ALPHA, Q_DIV=Q_DIV, R_DIV=R_DIV, DAYS=DAYS), runtime_s=time.time() - t0)
    json.dump(out, open(OUT / 'm5_summary.json', 'w'), indent=2, default=float)
    pd.DataFrame(dict(zone=zc, piles=c_c, need=need, erlang_add=x_erlang, sa_add=best)).to_csv(OUT / 'm5_allocations.csv', index=False)
    print(json.dumps({k: v for k, v in out.items() if k != 'cluster_zones'}, indent=2, default=float))

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(h[:, 0], h[:, 1], color='#9ecae1', label='current solution')
    ax.plot(h[:, 0], h[:, 2], color='#08519c', lw=2, label='best so far')
    ax.axhline(h[0, 1], color='#e6550d', ls='--', lw=1, label='Erlang-greedy start (M2 logic)')
    ax.set_xlabel('SA iteration'); ax.set_ylabel('simulated J (h/day, CRN)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f'Simulated annealing on a {K}-zone cluster with driver spill-over', fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / 'm5_sa_convergence.png', dpi=200); plt.close()


if __name__ == '__main__':
    main()
