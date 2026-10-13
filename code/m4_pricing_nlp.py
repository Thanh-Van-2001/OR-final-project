"""M4 - Nonlinear programming: time-of-use service-fee design that shifts charging away from congested hours.

Drivers choose the charging hour by a multinomial logit on the total price (grid electricity e_h + service fee p_h):
    s_h(p) = w_h exp(-beta (e_h + p_h)) / sum_k w_k exp(-beta (e_k + p_k)),  w_h calibrated so s(p0) = observed shares.
Daily demand is conserved (pure shifting); zone loads scale as a_zh(p) = a_zh0 * s_h(p) / s_h(p0).
beta is estimated from the data: TOU-priced zones vs flat-priced zones, zone fixed effects (see estimate_beta).

NLP (24 decision variables p_h, operator's service fee, yuan/kWh):
  (A) congestion-aware:  min  sum_z sum_h max(0, a_zh(p) - rho* c_z)^2
  (B) load flattening:    min  sum_h L_h(p)^2           (L_h = network load)
  s.t. revenue neutrality   sum_h p_h s_h(p) = sum_h p0_h s_h(p0)      (same average fee, only re-timed; because demand
       is conserved, an inequality would let the fee level rise without limit - a degenerate optimum)
       0.2 <= p_h <= 1.5
Solved with trust-constr (returns Lagrange multipliers; KKT stationarity checked numerically) and SLSQP (cross-check).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from common import load, load_profiles, OUT, FIG, RHO_TARGET

G = 2.0
P_LO, P_HI = 0.2, 1.5


def estimate_beta():
    d = load(); occ = d['occ']; tot = d['s_price'] + d['e_price']
    nun = tot.nunique(); tou = nun[nun > 1].index; flat = nun[nun == 1].index
    hod = occ.index.hour
    prof = occ.groupby(hod).mean(); share = prof / prof.sum(); ptot = tot.groupby(hod).mean()
    sflat = share[flat].mean(axis=1)
    Y = np.log(share[tou].div(sflat, axis=0)); X = ptot[tou]
    Yd = (Y - Y.mean()).values.ravel(order='F'); Xd = (X - X.mean()).values.ravel(order='F')
    grp = np.repeat(np.arange(len(tou)), 24)
    b = (Xd @ Yd) / (Xd @ Xd); e = Yd - b * Xd
    se = np.sqrt(sum((Xd[grp == k] @ e[grp == k]) ** 2 for k in range(len(tou)))) / (Xd @ Xd)
    return dict(beta=-b, se=se, t=b / se, tou_zones=len(tou), flat_zones=len(flat),
                mean_total_price=float(ptot[tou].values.mean()), tou_spread_median=float((ptot[tou].max() - ptot[tou].min()).median()))


class Model:
    def __init__(self, beta, g=G):
        d = load(); mean, _ = load_profiles()
        self.zones = list(mean.columns)
        self.c = d['zt'].piles.reindex(self.zones).values.astype(float)
        self.A0 = g * mean.values                                             # 24 x Z
        hod = d['occ'].index.hour
        self.e = d['e_price'].groupby(hod).mean().mean(axis=1).values          # grid price by hour (city average)
        self.p0 = d['s_price'].groupby(hod).mean().mean(axis=1).values         # current service fee by hour
        self.s0 = self.A0.sum(axis=1) / self.A0.sum()
        self.beta = beta
        self.w = self.s0 * np.exp(beta * (self.e + self.p0))
        self.R0 = float(self.p0 @ self.s0)

    def shares(self, p):
        u = self.w * np.exp(-self.beta * (self.e + p)); return u / u.sum()

    def loads(self, p):
        return self.A0 * (self.shares(p) / self.s0)[:, None]

    def congestion(self, p):
        return float(np.sum(np.maximum(0, self.loads(p) - RHO_TARGET * self.c) ** 2))

    def flatness(self, p):
        return float(np.sum(self.loads(p).sum(axis=1) ** 2)) / 1e6

    def revenue(self, p):
        return float(p @ self.shares(p))

    def metrics(self, p):
        A = self.loads(p)
        return dict(congestion=self.congestion(p), network_peak=float(A.sum(axis=1).max()),
                    peak_to_mean=float(A.sum(axis=1).max() / A.sum(axis=1).mean()),
                    unstable_zones=int((A.max(axis=0) >= self.c).sum()),
                    piles_needed_for_stability=int(np.maximum(0, np.floor(A.max(axis=0)) + 1 - self.c).sum()),
                    revenue_per_kwh=self.revenue(p), mean_total_price_paid=float((self.e + p) @ self.shares(p)))


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h; g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def optimise(m, obj, lo=None, hi=None):
    lo = P_LO if lo is None else lo; hi = P_HI if hi is None else hi
    f = m.congestion if obj == 'A' else m.flatness
    scale = f(m.p0) or 1.0
    fs = lambda p: f(p) / scale
    con = NonlinearConstraint(m.revenue, m.R0, m.R0)
    bnd = Bounds(np.full(24, lo), np.full(24, hi))
    best = None
    for start in [np.clip(m.p0, lo, hi), np.full(24, m.R0)]:
        r = minimize(fs, start, method='trust-constr', constraints=[con], bounds=bnd,
                     options={'maxiter': 3000, 'gtol': 1e-9, 'xtol': 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    r2 = minimize(fs, np.clip(m.p0, lo, hi), method='SLSQP', bounds=list(zip(np.full(24, lo), np.full(24, hi))),
                  constraints=[{'type': 'eq', 'fun': lambda p: m.revenue(p) - m.R0}], options={'maxiter': 1000, 'ftol': 1e-12})
    # KKT check: grad f - mu grad g - (bound multipliers) = 0 on the free variables
    p = best.x; mu = float(best.v[0][0]) if best.v and len(best.v[0]) else 0.0
    gf = num_grad(fs, p); gg = num_grad(m.revenue, p)
    free = (p > lo + 1e-4) & (p < hi - 1e-4)
    resid = gf + mu * gg                            # scipy sign convention: grad L = grad f + v * grad c
    kkt = float(np.max(np.abs(resid[free]))) if free.any() else 0.0
    return dict(p=p, obj_rel=float(best.fun), slsqp_obj_rel=float(r2.fun), slsqp_p=r2.x, mu=mu,
                kkt_stationarity_resid=kkt, free_vars=int(free.sum()), revenue_slack=m.revenue(p) - m.R0)


def main():
    est = estimate_beta(); print('beta estimate', est)
    beta0 = float(est['beta'])
    out = dict(beta_estimate=est)
    m = Model(beta0)
    out['baseline'] = m.metrics(m.p0)
    res = {}
    for obj in ['A', 'B']:
        r = optimise(m, obj)
        res[obj] = dict(metrics=m.metrics(r['p']), fee=r['p'].tolist(), mu=r['mu'], kkt_resid=r['kkt_stationarity_resid'],
                        free_vars=r['free_vars'], obj_rel=r['obj_rel'], slsqp_obj_rel=r['slsqp_obj_rel'],
                        revenue_slack=r['revenue_slack'])
        print(obj, {k: v for k, v in res[obj].items() if k != 'fee'})
    out['optimised'] = res
    sens = {}
    for b in [0.1, beta0, 0.6, 1.0]:
        mb = Model(b); r = optimise(mb, 'A')
        sens[round(b, 3)] = dict(metrics=mb.metrics(r['p']), baseline=mb.metrics(mb.p0))
        print('beta', round(b, 3), sens[round(b, 3)]['metrics'])
    out['beta_sensitivity_objA'] = sens
    json.dump(out, open(OUT / 'm4_summary.json', 'w'), indent=2, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ax = axes[0]
    ax.step(range(24), m.p0, where='mid', label='current service fee', color='#636363')
    ax.step(range(24), res['A']['fee'], where='mid', label='optimal (A) congestion-aware', color='#d95f0e')
    ax.step(range(24), res['B']['fee'], where='mid', label='optimal (B) load flattening', color='#3182bd', ls='--')
    ax.set_xlabel('hour'); ax.set_ylabel('service fee (yuan/kWh)'); ax.legend(fontsize=7.5); ax.grid(alpha=.3); ax.set_xticks(range(0, 24, 3))
    ax = axes[1]
    for p, lab, col in [(m.p0, 'current', '#636363'), (np.array(res['A']['fee']), 'optimal (A)', '#d95f0e')]:
        ax.plot(range(24), m.loads(p).sum(axis=1) / m.c.sum(), label=lab, color=col)
    ax.set_xlabel('hour'); ax.set_ylabel('network utilisation (demand x2)'); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_xticks(range(0, 24, 3))
    fig.suptitle(f'Time-of-use service fee (beta = {beta0:.2f} per yuan/kWh, revenue-neutral)', fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / 'm4_pricing.png', dpi=200); plt.close()


if __name__ == '__main__':
    main()
