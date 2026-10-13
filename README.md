# Right Place, Right Time — Capacity, Guidance and Pricing for Public EV Charging in Shenzhen

Operations Research 76000093 — Final Project. Team: Nguyen Van Thanh, [Member 2], [Member 3], [Member 4], [Member 5].

## Deliverables
| File | Content |
|---|---|
| `reports/proposal/proposal.pdf` | Project proposal (due Oct 26) |
| `reports/progress/progress_report.pdf` | Progress report (due Nov 30, hardcopy) |
| `reports/final/final_report.pdf` | Final report (17 pages): literature review, 5 models, results, analysis, recommendations, difficulties |
| `reports/slides/OR_Final_Presentation.pdf` | 15-slide presentation (10–12 min, one section per member) |
| `code/` | All models (Python) |
| `results/` | All numerical outputs (CSV/JSON) |
| `figures/` | All figures used in the reports and slides |
| `data/UrbanEV/data/` | The UrbanEV files used (CC0), Li et al., *Scientific Data* 2025 |

## Methods (one per member)
| Method | Script | Question |
|---|---|---|
| M1 Queueing (M/M/c, Erlang-C) | `code/m1_queueing.py` | Waiting per zone/hour; which zones fail under growth |
| M2 Integer programming (MILP) | `code/m2_ip_allocation.py` | New vs relocated piles under a budget; duality and sensitivity |
| M3 Network optimisation (min-cost flow) | `code/m3_network_flow.py` | How much overload driver guidance absorbs |
| M4 Nonlinear programming (logit + KKT) | `code/m4_pricing_nlp.py`, `code/m4_moderate.py` | Revenue-neutral time-of-use service fees |
| M5 Simulation + simulated annealing | `code/m5_simulation_sa.py` | Validity of M1/M2; spill-over-aware optimisation |
| Integrated analysis | `code/integrate.py` | All levers on one budget yardstick |

## How to reproduce
Requirements: Python 3.10+, numpy, pandas, scipy (≥1.9, HiGHS), networkx, matplotlib.
```
cd code
python m1_queueing.py        # ~10 s
python m2_ip_allocation.py   # ~1.5 min
python m3_network_flow.py    # ~40 s
python m4_pricing_nlp.py     # ~40 s
python m4_moderate.py        # ~1 min
python m5_simulation_sa.py   # ~8 min
python integrate.py          # ~12 min
```
Outputs are written to `results/` and `figures/`.

## Key results (demand ×2)
- 48 of 275 zones become unstable, while network-average utilisation is only 56%.
- MILP at a budget of 1,000 new-pile units: 5.0 driver-h/day of waiting with new + relocated piles, vs 952 with new piles only.
- Guidance within 3 km: 70% of the excess redirected; piles needed for stability fall from 598 to 185.
- Revenue-neutral time-of-use fee: congestion −49%, piles needed 598 → 403, but unstable zones only 48 → 45.
- Simulated annealing with driver spill-over: −34% (±8%) cost vs the Erlang-optimal plan, out of sample.
- Pricing + guidance cut the budget needed for a ≤15 s average wait by 77% (348 → 81).
