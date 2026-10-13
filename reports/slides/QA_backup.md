# Q&A backup — likely questions and short answers

For the presenters. Every number is in the final report (section in brackets). The owner of each method takes the questions on it.

## Data and problem (all)
**Why double the demand?** China's NEV stock grew 49% in one year to June 2025, and the data end in Feb 2023. We plan for ×2 (about 2026–27) and report ×1.5 and ×2.5 for every model (§3.2).

**Occupancy is not demand. Isn't it censored?** Yes, on busy days 10 zones are full. We use the mean day, where no zone is saturated, and treat variability by simulation. Censoring means today's shortage is, if anything, underestimated (§3.2, §11).

**Is the network really short of piles?** Not in total. At ×2, a waiting probability ≤ 20% in every zone needs 14,367 piles, fewer than the 17,532 that exist. The problem is distribution; a real total shortage starts only around ×2.5 (§4.3).

## M1 Queueing
**Why M/M/c?** Piles are parallel servers; many independent drivers give Poisson-like arrivals. M5 relaxes both assumptions (time-varying arrivals, log-normal sessions) and, in the simulated cluster, finds M/M/c conservative (§8.1).

**You don't know the session length.** Total waiting does not depend on it: λ·Wq = a·C/(c−a). Only the wait per driver scales with τ (Proposition 1).

**How did you size each zone?** Smallest c with Erlang-C P(wait) ≤ α at the peak hour, an exact monotone search. Square-root staffing (Halfin–Whitt) is within one pile in all 275 zones (§4.3).

## M2 Integer programming
**The waiting function is nonlinear. How is this a MILP?** It is convex in the number of piles (checked in all 275 zones), so each extra pile is a 0-1 increment with a smaller gain than the last, and the incremental model is exact (Proposition 2). About 8,400 binaries, solved in seconds; LP gap 0.003 h/day.

**Where does the relocation cost 0.35 come from?** It is an assumption. Relocation stays optimal for any κ below 1, i.e. unless moving a pile costs as much as a new one (§5.3).

**When should the city stop investing?** When the budget's shadow price (105 → 0.01 h/day per unit from B = 250 to 1,000) falls below the value of drivers' time per unit of cost (§5.3).

## M3 Network optimisation
**Isn't min-cost flow just LP, i.e. the same method as M2?** We use network structure that a generic LP does not: total unimodularity (integer flows), the network simplex method, and the max-flow min-cut theorem, which certifies where guidance runs out (§3.5, §6.3).

**Will drivers really detour 3 km?** The mean detour is 1.7 km. Kuang et al. find that price-induced demand changes spill over to stations within about 3.45 km in Shenzhen. Results are reported for 1–8 km (§6.2).

**What if drivers ignore the guidance?** With only 50% compliance the piles needed fall from 598 to 245; with 75% they are 184, the same as with full compliance (§6.4).

**What does the min cut tell the planner?** 23 overloaded zones whose total excess (481) exceeds all spare capacity within 3 km (165). Those zones and their 18 saturated neighbours are where new or relocated piles should go (§6.3).

## M4 Nonlinear programming
**Your β has |t| = 2.2. Can we trust the pricing results?** We report β from 0.1 to 1. Piles needed are 472 at β = 0.1 and 403 for β ≥ 0.28. Kuang et al.'s elasticity of −0.76 would mean β ≈ 0.49 if it were all a shift between hours; since most of it is a shift in space, that is an upper bound, and our estimate lies below it (§7.1, §7.3).

**Why revenue-neutral?** With "revenue ≥ today" the multiplier was zero and all fees went to the upper bound, because demand is conserved: a degenerate optimum. Revenue neutrality re-times the fee without raising it, which is also what a regulator would accept (§7.2).

**Why does pricing barely change the unstable zones (48 → 45)?** Price shifts demand in time; Shenzhen's main problem is in space. That is why pricing and guidance are complements (§7.3, §9).

## M5 Simulation and simulated annealing
**How do you know the simulation is correct?** With textbook assumptions it reproduces Erlang-C in all test cases, within 0.35 standard errors (§8.1).

**Why is the Erlang model conservative?** Not because of session variability: CV 1.2 still gives only 119 of the 203 predicted hours. When we stretch the day k times, the simulation rises to 150 (k = 4) and 188 (k = 16): the peaks are too short for queues to reach steady state. This holds for near-critical zones like those in our cluster; with more slack, SIPP can also under-estimate (§8.1).

**Is the 34% improvement robust?** It holds out of sample (15.8 ± 3.6 h/day). The first SA run left one zone (TAZ 592) below its stability limit; with stability enforced the gain is still 32% (14.8 ± 3.3). Both plans win under every diversion setting tested (10–43%). But if drivers never divert, both are worse, so they should be deployed with guidance (§8.3).

**How does the SA plan beat the Erlang plan?** It takes piles from busy zones whose overflow can divert to idle piles next door (TAZ 588 and 592, next to 587 and 589) and gives them to zones whose drivers have nowhere to divert (§8.2).

**Why simulated annealing?** The simulated objective is noisy and not separable. SA needs only objective evaluations, and common random numbers make the comparisons fair. It was still improving at 250 iterations, so 34% is not the limit (§8.2).

## Integration and extensions
**You combined the levers one after another. Is that optimal?** No; a joint bilevel model (operator prices, drivers respond, planner invests) is our main extension (§11). Even the sequential combination cuts the budget by 77% at ×2.

**What would you do with more time?** Censored-demand estimation from the 5-minute station data, zone-level growth forecasts, a grid-capacity model, and stochastic programming over growth scenarios (§11).
