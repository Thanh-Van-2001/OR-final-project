"""Shared data loading and queueing formulas for the Shenzhen EV-charging OR project.

Data: UrbanEV (Li et al., Scientific Data 2025), zone level, hourly, 2022-09-01 .. 2023-02-28, CC0.
Conventions
  zone  = traffic analysis zone (TAZ), 275 zones
  c     = number of public charging piles in the zone (servers)
  a     = offered load in Erlangs = average number of busy piles (from occupancy.csv, a count)
  tau   = mean charging session length (hours), lambda = a / tau
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'UrbanEV' / 'data'
OUT = ROOT / 'results'
FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

TAU = 1.0            # base-case mean session length (h); sensitivity 0.5-2 h
RHO_TARGET = 0.85    # planning target utilisation used by the network-flow model


@lru_cache(maxsize=1)
def load():
    inf = pd.read_csv(DATA / 'inf.csv')
    occ = pd.read_csv(DATA / 'occupancy.csv', index_col=0, parse_dates=True)
    zones = occ.columns.astype(int)
    occ.columns = zones
    piles = inf.groupby('TAZID').charge_count.sum().reindex(zones).astype(int)
    lon = inf.groupby('TAZID').longitude.mean().reindex(zones)
    lat = inf.groupby('TAZID').latitude.mean().reindex(zones)
    nst = inf.groupby('TAZID').station_id.count().reindex(zones)
    dist = pd.read_csv(DATA / 'distance.csv')
    dist.index = dist.columns.astype(int)
    dist.columns = dist.columns.astype(int)
    dist = dist.reindex(index=zones, columns=zones) / 1000.0          # km
    sp = pd.read_csv(DATA / 's_price.csv', index_col=0, parse_dates=True)
    ep = pd.read_csv(DATA / 'e_price.csv', index_col=0, parse_dates=True)
    sp.columns = ep.columns = zones
    vol = pd.read_csv(DATA / 'volume-11kW.csv', index_col=0, parse_dates=True)
    vol.columns = zones
    zt = pd.DataFrame({'piles': piles, 'stations': nst, 'lon': lon, 'lat': lat})
    return dict(occ=occ, zones=zones, zt=zt, dist=dist, s_price=sp, e_price=ep, volume=vol)


def load_profiles(quantile=0.9):
    """Hour-of-day offered load per zone: mean day and a busy ('design') day = given quantile over days."""
    d = load()
    occ = d['occ']
    hod = occ.index.hour
    mean = occ.groupby(hod).mean()                     # 24 x Z
    design = occ.groupby(hod).quantile(quantile)       # 24 x Z
    return mean, design


# ------------------------------------------------------------------ queueing (M/M/c)
def erlang_b(c, a):
    """Erlang-B blocking probability, stable recursion; c int >= 0, a >= 0 (scalar)."""
    b = 1.0
    for k in range(1, int(c) + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c, a):
    """Probability an arrival waits in M/M/c with offered load a (Erlangs). Returns 1 if a >= c."""
    c = int(c)
    if c <= 0 or a >= c:
        return 1.0
    if a <= 0:
        return 0.0
    b = erlang_b(c, a)
    return c * b / (c - a * (1.0 - b))


def wq(c, a, tau=TAU):
    """Mean waiting time in queue (hours) for M/M/c; inf if unstable."""
    c = int(c)
    if a <= 0:
        return 0.0
    if a >= c:
        return np.inf
    return erlang_c(c, a) * tau / (c - a)


def daily_wait_hours(c, loads, tau=TAU):
    """Expected total driver waiting hours per day in a zone: sum_h lambda_h * Wq_h (loads = 24 hourly Erlangs).
    Note lambda*Wq = (a/tau) * C*tau/(c-a) = a*C/(c-a): the TOTAL waiting time does not depend on tau
    (only the wait per driver does)."""
    tot = 0.0
    for a in loads:
        w = wq(c, a, tau)
        if not np.isfinite(w):
            return np.inf
        tot += (a / tau) * w
    return tot


def wait_curve(loads, cmax):
    """Daily waiting hours W[c] for c = 0..cmax in one pass (vectorised Erlang-B recursion over the 24 loads).
    W[c] = inf where c <= max load (unstable). Uses sum_h a_h * C(c, a_h) / (c - a_h)  (tau-free)."""
    a = np.asarray(loads, dtype=float)
    W = np.full(cmax + 1, np.inf)
    b = np.ones_like(a)
    amax = a.max()
    for k in range(1, cmax + 1):
        b = a * b / (k + a * b)
        if k > amax:
            C = k * b / (k - a * (1.0 - b))
            W[k] = float(np.sum(np.where(a > 0, a * C / (k - a), 0.0)))
    if amax <= 0:
        W[:] = 0.0
    return W
