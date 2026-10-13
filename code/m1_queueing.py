"""M1 - Queueing theory: M/M/c (Erlang-C) service-level diagnosis of every zone, hour and growth scenario.

For zone z with c_z piles and hourly offered load a_zh = g * (mean busy piles in hour h of the observed period):
  P(wait) = ErlangC(c, a),  Wq = P(wait) * tau / (c - a),  daily wait = sum_h (a_zh / tau) * Wq_zh.
P(wait) depends only on (c, a); the TOTAL daily waiting hours sum_h lambda*Wq = sum_h a*C/(c-a) do not depend on tau,
so every planning decision below is tau-free; only the wait per driver (minutes) scales with tau.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load, load_profiles, erlang_c, wq, daily_wait_hours, OUT, FIG, TAU

GROWTH = [1.0, 1.5, 2.0, 2.5]


def zone_table(c, prof, g, tau=TAU):
    rows = []
    for z in prof.columns:
        loads = g * prof[z].values
        h = int(np.argmax(loads)); a = loads[h]
        rows.append(dict(zone=z, piles=int(c[z]), peak_hour=h, peak_load=a, peak_util=a / c[z],
                         p_wait_peak=erlang_c(c[z], a), wq_peak_min=60 * wq(c[z], a, tau),
                         daily_wait_h=daily_wait_hours(c[z], loads, tau),
                         arrivals_day=loads.sum() / tau))
    return pd.DataFrame(rows).set_index('zone')


def main():
    d = load(); zt = d['zt']; c = zt.piles
    mean, design = load_profiles(0.9)
    summary = {}
    tables = {}
    for g in GROWTH:
        t = zone_table(c, mean, g); tables[g] = t
        fin = t.daily_wait_h[np.isfinite(t.daily_wait_h)]
        summary[str(g)] = dict(
            unstable_zones=int(np.isinf(t.daily_wait_h).sum()),
            zones_pwait_gt_0_5=int((t.p_wait_peak > 0.5).sum()),
            zones_peak_util_gt_0_85=int((t.peak_util > 0.85).sum()),
            wait_hours_day_stable_zones=float(fin.sum()),
            arrivals_day=float(t.arrivals_day.sum()),
            network_util_mean=float((g * mean).values.sum() / (24 * c.sum())),
            piles_total=int(c.sum()))
        t.to_csv(OUT / f'm1_zones_g{g}.csv')
    # tau check: total waiting hours are tau-invariant (numerical confirmation, g=1.5)
    tau_chk = {tau: float(zone_table(c, mean, 1.5, tau).daily_wait_h.replace(np.inf, np.nan).sum()) for tau in (0.5, 1.0, 2.0)}
    summary['tau_sensitivity_g1.5_wait_hours'] = tau_chk
    # design-day (p90) censoring diagnostic
    summary['design_day_p90_zones_saturated'] = int(((design.max() / c) >= 0.999).sum())
    summary['mean_day_zones_saturated'] = int(((mean.max() / c) >= 0.999).sum())
    json.dump(summary, open(OUT / 'm1_summary.json', 'w'), indent=2)
    print(json.dumps(summary, indent=2))

    # ---- figures
    util = (mean / c)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(range(24), util.quantile(0.1, axis=1), util.quantile(0.9, axis=1), color='#9ecae1', alpha=.5, label='10-90% of zones')
    ax.plot(range(24), (mean.sum(axis=1) / c.sum()), color='#08519c', lw=2, label='network average')
    ax.plot(range(24), util.max(axis=1), color='#e6550d', lw=1.5, ls='--', label='most loaded zone')
    ax.set_xlabel('hour of day'); ax.set_ylabel('pile utilisation (busy / piles)'); ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=.3)
    ax.set_title('Observed utilisation, Shenzhen public charging (Sep 2022-Feb 2023)', fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / 'm1_hourly_util.png', dpi=200); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, g in zip(axes, [1.0, 2.0]):
        t = tables[g]
        col = np.clip(t.p_wait_peak.values, 0, 1)
        sc = ax.scatter(zt.lon, zt.lat, c=col, s=8 + zt.piles / 3, cmap='RdYlGn_r', vmin=0, vmax=1, edgecolor='k', lw=.2)
        ax.set_title(f'P(wait) at zone peak hour, demand x{g:g}\n{(t.p_wait_peak > 0.5).sum()} zones > 0.5, '
                     f'{int(np.isinf(t.daily_wait_h).sum())} unstable', fontsize=9)
        ax.set_xlabel('longitude'); ax.set_aspect(1 / np.cos(np.radians(22.6)))
    axes[0].set_ylabel('latitude')
    fig.colorbar(sc, ax=axes, shrink=.8, label='Erlang-C P(wait)')
    plt.savefig(FIG / 'm1_map_pwait.png', dpi=200, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    main()
