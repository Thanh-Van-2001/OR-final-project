"""M5 robustness: are the M5 conclusions sensitive to the behavioural parameters that the data cannot identify?

Uses the two plans saved by m5_simulation_sa.py (Erlang-greedy start and the simulated-annealing plan, both found with
the base parameters CV = 0.8, q_div = 3, R_div = 3 km) and re-evaluates them on the same 10 fresh seeds under:
  (1) session-length variability CV in {0.5, 0.8, 1.0, 1.2}, no diversion (Erlang plan): why is SIPP conservative?
  (2) diversion threshold q_div in {1, 3, 5} and no diversion, R_div = 3 km
  (3) diversion radius R_div in {2, 3, 5} km, q_div = 3
  (4) CV in {0.5, 1.0} with diversion (q_div = 3, R_div = 3 km)
  (5) time-scale experiment: the daily profile stretched k = 1, 4, 16 times (exponential sessions, no diversion)
The SA plan is NOT re-optimised: the question is whether a plan tuned under the base assumptions stays better.
Paired differences (Erlang - SA) on common random numbers give 95% confidence intervals.
"""
import json
import time
import numpy as np
import pandas as pd
import m5_simulation_sa as m5
from common import load, load_profiles, OUT

SEEDS = range(100, 110)          # the same fresh seeds as the out-of-sample test in m5_simulation_sa.py


def evaluate(c_c, x_e, x_s, rates, dist, cv=m5.CV, q_div=m5.Q_DIV, r_div=m5.R_DIV, divert=True):
    m5.R_DIV = r_div                                   # simulate() reads the module-level radius
    Je, Js, We, Ws, De, Ds = [], [], [], [], [], []
    for sd in SEEDS:
        st = m5.gen_streams(rates, m5.DAYS, sd, cv=cv)
        e = m5.simulate(c_c + x_e, st, dist, m5.DAYS, divert, q_div)
        s = m5.simulate(c_c + x_s, st, dist, m5.DAYS, divert, q_div)
        Je.append(e['J']); Js.append(s['J']); We.append(e['wait_h_day']); Ws.append(s['wait_h_day'])
        De.append(e['detour_km_day']); Ds.append(s['detour_km_day'])
    m5.R_DIV = 3.0
    diff = np.array(Je) - np.array(Js)
    return dict(J_erlang=float(np.mean(Je)), J_sa=float(np.mean(Js)), wait_erlang=float(np.mean(We)), wait_sa=float(np.mean(Ws)),
                detour_erlang=float(np.mean(De)), detour_sa=float(np.mean(Ds)),
                improvement=float(diff.mean()), ci95=float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
                improvement_pct=float(100 * diff.mean() / np.mean(Je)))


def stretched_streams(rates, k, cycles, seed):
    """Arrivals whose daily rate profile is stretched k times in time (each hourly rate lasts k hours),
    exponential sessions with mean TAU. One warm-up day (24 h) is added before the measured cycles."""
    rng = np.random.default_rng(seed)
    H = 24.0 + 24.0 * k * cycles
    out = []
    for z in range(rates.shape[1]):
        lam_max = rates[:, z].max()
        n = rng.poisson(lam_max * H)
        t = np.sort(rng.uniform(0, H, n))
        hour = (np.floor(t / k).astype(int)) % 24
        t = t[rng.uniform(0, 1, n) < rates[hour, z] / lam_max]
        out.append((t, rng.exponential(m5.TAU, len(t))))
    return out


def timescale_experiment(c_c, x_e, rates, dist, ks=(1, 4, 16), seeds=range(200, 205)):
    """Why is SIPP conservative? Stretch the day k times: as k grows each hour lasts long enough to reach
    steady state, so the simulated wait per (stretched) day / k should approach the SIPP value.
    M/M/c sessions (exponential) and no diversion, so the only difference from SIPP is non-stationarity."""
    rows = []
    for k in ks:
        cycles = 4 if k < 16 else 2
        w = []
        for sd in seeds:
            st = stretched_streams(rates, k, cycles, sd)
            r = m5.simulate(c_c + x_e, st, dist, k * cycles, divert=False)   # wait per 24 h of simulated time
            w.append(r['wait_h_day'])
        rows.append(dict(k=k, wait_per_day_equiv=float(np.mean(w)), se=float(np.std(w, ddof=1) / np.sqrt(len(w)))))
        print('timescale', rows[-1])
    return rows


def main():
    import sys
    if '--timescale' in sys.argv:                      # only (re)run the time-scale experiment on an existing result file
        d = load(); mean, _ = load_profiles(); zones = list(mean.columns); A = m5.G * mean.values
        alloc = pd.read_csv(OUT / 'm5_allocations.csv'); idx = [zones.index(z) for z in alloc.zone]
        dist = d['dist'].reindex(index=zones, columns=zones).values[np.ix_(idx, idx)]
        out = json.load(open(OUT / 'm5_robustness.json'))
        out['timescale_no_diversion_exponential'] = timescale_experiment(alloc.piles.values, alloc.erlang_add.values,
                                                                          A[:, idx] / m5.TAU, dist)
        json.dump(out, open(OUT / 'm5_robustness.json', 'w'), indent=2, default=float)
        return
    t0 = time.time()
    d = load(); zt = d['zt']; mean, _ = load_profiles()
    zones = list(mean.columns); A = m5.G * mean.values
    alloc = pd.read_csv(OUT / 'm5_allocations.csv')
    idx = [zones.index(z) for z in alloc.zone]
    c_c = alloc.piles.values; x_e = alloc.erlang_add.values; x_s = alloc.sa_add.values
    rates = A[:, idx] / m5.TAU
    dist = d['dist'].reindex(index=zones, columns=zones).values[np.ix_(idx, idx)]
    out = dict(sessions_cv_no_diversion=[], diversion_threshold=[], diversion_radius=[], cv_with_diversion=[])
    for cv in (0.5, 0.8, 1.0, 1.2):
        r = evaluate(c_c, x_e, x_s, rates, dist, cv=cv, divert=False); r['cv'] = cv
        out['sessions_cv_no_diversion'].append(r); print('cv no-div', r)
    for q in (1, 3, 5):
        r = evaluate(c_c, x_e, x_s, rates, dist, q_div=q); r['q_div'] = q
        out['diversion_threshold'].append(r); print('q', r)
    r = out['sessions_cv_no_diversion'][1].copy(); r['q_div'] = 'none'
    out['diversion_threshold'].append(r)
    for rd in (2.0, 5.0):
        r = evaluate(c_c, x_e, x_s, rates, dist, r_div=rd); r['r_div_km'] = rd
        out['diversion_radius'].append(r); print('R', r)
    base = [x for x in out['diversion_threshold'] if x['q_div'] == 3][0].copy(); base['r_div_km'] = 3.0
    out['diversion_radius'].insert(1, base)
    for cv in (0.5, 1.0):
        r = evaluate(c_c, x_e, x_s, rates, dist, cv=cv); r['cv'] = cv
        out['cv_with_diversion'].append(r); print('cv div', r)
    # analytical SIPP prediction for the Erlang plan (stationary per hour, exponential sessions, no diversion)
    from common import wait_curve
    out['sipp_wait_erlang_plan'] = float(sum(wait_curve(A[:, i], c + x + 1)[c + x] for i, c, x in zip(idx, c_c, x_e)))
    out['timescale_no_diversion_exponential'] = timescale_experiment(c_c, x_e, rates, dist)
    out['runtime_s'] = time.time() - t0
    json.dump(out, open(OUT / 'm5_robustness.json', 'w'), indent=2, default=float)
    print('done', round(out['runtime_s']), 's')


if __name__ == '__main__':
    main()
