"""M4 robustness: a moderate fee range [0.5, 1.0] yuan/kWh (close to the observed 5th-95th percentile) vs the wide range
[0.2, 1.5] (observed extremes), and its combination with 3-km guidance on the M2 budget yardstick (demand x2)."""
import json
import numpy as np
from common import load, load_profiles, OUT
import m4_pricing_nlp as m4
import integrate as ig

beta = float(m4.estimate_beta()['beta'])
d = load(); mean, _ = load_profiles(); zones = list(mean.columns)
c = d['zt'].piles.reindex(zones).values.astype(float); dist = d['dist'].reindex(index=zones, columns=zones).values
out = {}
for name, lo, hi in [('wide_0.2_1.5', 0.2, 1.5), ('moderate_0.5_1.0', 0.5, 1.0)]:
    m = m4.Model(beta, 2.0); r = m4.optimise(m, 'A', lo, hi); Ap = m.loads(r['p'])
    target = ig.TARGET_MIN / 60 * m.A0.sum()
    Bp, _ = ig.min_budget(Ap, target); Bpg, _ = ig.min_budget(ig.guidance(Ap, c, dist), target)
    out[name] = dict(metrics=m.metrics(r['p']), fee=[round(x, 3) for x in r['p']], mu=r['mu'], kkt=r['kkt_stationarity_resid'],
                     budget_pricing=Bp, budget_pricing_guidance=Bpg)
    print(name, out[name]['metrics'], Bp, Bpg)
json.dump(out, open(OUT / 'm4_moderate.json', 'w'), indent=2, default=float)
