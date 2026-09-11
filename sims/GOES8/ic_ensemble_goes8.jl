#=
ic_ensemble_goes8.jl — RQ1: IS THE INITIAL CONDITION FORGOTTEN, AND HOW FAST?

WHY THIS EXISTS RATHER THAN A RE-READ OF exp9_goes8_sweep.csv.
The brief asked me to check whether an existing sweep varies initial conditions
broadly "not just J".  It does — partly.  exp9 varies β₀ (6), I_d0 (18) and P_e
(3), so the premise "it only varies J at one fixed IC" is WRONG and is recorded
as such.  But three things in exp9 make it unable to answer RQ1 as posed:

  1. α₀ IS NEVER VARIED.  Every exp9 row starts at α₀ = 0.  α is one of the two
     coordinates of Ĥ in the O frame, so an ensemble that holds it fixed cannot
     show that the ensemble forgets its initial ORIENTATION — only its initial
     energy partition and spin rate.  (The one script in this repo that does
     draw α₀ is sims/Mass Sensitivity/inertia_sweep.jl, which has never been
     run — see [FLAG-EXP10-UNRUNNABLE] in this file's ledger.)
  2. exp9's HORIZON IS 20 yr, not 100.  Its header justifies that from GOES-8's
     ~4 yr YORP settling, which is a fair argument about the ATTRACTOR but not
     about the SPREAD: "forgotten" is a statement about the ensemble's dispersion
     as a function of time, and it needs the full RQ1 horizon to be stated.
  3. exp9's GRID IS A LATTICE, not a sample.  A full factorial over 6×18×3 makes
     every marginal distribution an artefact of the axis spacing.  A distribution
     plot (Figure 4) needs draws, not lattice points.

So this is a new ensemble with the SAME physics configuration as exp9 (GOES-8,
θ_sa = 17°, :bs optics, GEO graveyard, full perturbation set) and a genuinely
sampled initial condition, over the full RQ1 horizon.

WHAT THIS SCRIPT PRODUCES.  Four figures' worth of data, in three blocks:

  block "main"     Figures 4, 5, 7 — 96 LHS initial conditions × 6 J × 2 μ/J,
                   σ = −1.  The dispersion-vs-time of ω̄_e within this block IS
                   the RQ1 measurement.
  block "sigma"    Figure 6 — the σ question, run at ICs drawn specifically
                   NEAR the separatrix, plus a far-from-separatrix control.
  block "backend"  a replicate of the main ICs on the :numeric SRP backend.
                   [FLAG-BACKEND-SPLIT] (below) makes this mandatory, not
                   optional.

TWO TIME SERIES PER CASE, NOT JUST AN ENDPOINT.  "Forgotten" is a claim about
when the ensemble collapses, so ω̄_e, I_d and β are written on a shared
log-spaced time grid to ic_ensemble_goes8_series.csv.  The endpoint-only CSV
cannot answer "how fast".

────────────────────────────────────────────────────────────────────────────────
[FLAG-BACKEND-SPLIT] — READ BEFORE TRUSTING ANY NUMBER OUT OF THIS FILE.

sims/Figures/ (Phase 2) established that the two averaged-SRP backends are not
interchangeable: against the full Euler truth model for GOES-8, :numeric tracks
it and :analytic does not, yet :analytic is what exp9, telstar_sweep.jl,
inertia_sweep.jl and the Skynet scripts all use.

This script does NOT resolve that.  It measures how much it matters FOR THE RQ1
CONCLUSION, which is a different and cheaper question than "which backend is
right".  The main block stays on :analytic so its results sit alongside the rest
of the repo; the "backend" block re-runs the same 96 ICs on :numeric at the
reference dissipation.  If both blocks forget the initial condition on the same
timescale, the RQ1 answer is backend-robust even though the attractor VALUE is
not — and that is the claim this ensemble is entitled to make.

The Phase-2 comparison was run WITHOUT dissipation, so it is not evidence about
this configuration.  That is exactly why the replicate is run rather than argued.
────────────────────────────────────────────────────────────────────────────────

Run:  julia -t auto --project=. "sims/GOES8/ic_ensemble_goes8.jl"
Writes sims/GOES8/ic_ensemble_goes8.csv        (one row per case)
       sims/GOES8/ic_ensemble_goes8_series.csv (time series, shared grid)
=#

include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))
using .master
using StaticArrays, LinearAlgebra, Printf, Random
# Explicit, because several loaded packages export `DiscreteCallback`/`CallbackSet`
# and a bare `using DifferentialEquations` makes the name ambiguous in Main.
# inertia_sweep.jl:63 imports CallbackSet the same way.
using DifferentialEquations: DiscreteCallback, CallbackSet, terminate!

# NOT `const`: `I` is also exported by LinearAlgebra, and a const binding would
# collide.  exp9 (GOES8_sim) and exp11 (telstar_sweep.jl) both use this plain
# form, so this matches the siblings rather than inventing a third convention.
I  = goes8_inertia()
SH = goes8_shape_full(; θ_sa = deg2rad(17), optical = :bs)

# ══════════════════════════════════════════════════════════════════════════════
# GRID — edit here
# ══════════════════════════════════════════════════════════════════════════════
# RQ1_SMOKE=1 shrinks every axis so the whole script — including the CSV writers
# and the summary tables — can be exercised end to end in under a minute.  It is
# a plumbing test, NOT a result: an ensemble this small cannot support a
# dispersion statistic, and the run banner says so.
const SMOKE = get(ENV, "RQ1_SMOKE", "0") == "1"

const YEARS   = SMOKE ? 2.0 : 100.0        # RQ1 horizon (NOT exp9's 20 — see header)
const N_IC    = SMOKE ?  8 : 96            # main LHS initial conditions
const N_SEP   = SMOKE ?  4 : 64            # extra ICs drawn near the separatrix
const IC_SEED = 20260831

const Js    = SMOKE ? [1.8] : [0.1, 0.5, 1.0, 1.8, 5.0, 10.0]   # exp9's slug inertias [kg m²]
const MUoJs = SMOKE ? [1.0e-3] : [1.0e-3, 1.0e-4]                  # BOTH B&S values — exp9 used only 1e-3.
                                                # 1e-3 is the model-validity ceiling.
const J_REF, MUoJ_REF = 1.8, 1.0e-3             # B&S 2022 Fig. 9 reference point
const JS_BACKEND = SMOKE ? [1.8] : [0.1, 1.8, 10.0]             # J values re-run on :numeric — the
                                                # ends of exp9's range plus the middle

# IC ranges.  β₀ and P_e reuse inertia_sweep.jl's exp6-derived ranges verbatim so
# the two ensembles are comparable; α₀ is the axis exp9 never had.
const ALPHA_RANGE_DEG = (0.0, 360.0)
const BETA_RANGE_DEG  = (15.0, 165.0)
const PE_RANGE_MIN    = (30.0, 480.0)      # log-uniform [FLAG-EXP10-PE]

# I_d0 covers each band by FRACTION, exp9's spacing rationale: for GOES-8 LAM is
# 94.7% of the I_d axis and SAM 5.3%, so a uniform-in-I_d draw would barely
# sample SAM — where the attractor lives.
const BAND_FRAC_RANGE = (0.05, 0.95)       # kept off 0 (uniform spin) and 1 (separatrix)

# "NEAR THE SEPARATRIX" — the operational definition Figure 6 reports against.
# Distance is measured as a fraction of the FULL I_d axis, (I_s − I_l), not of
# I_i: the axis is what a trajectory has to traverse, and it is the only
# normalisation that means the same thing for GOES-8 (narrow SAM) and Telstar
# (wide SAM).  0.02 is 2% of that axis = 72 kg m² for GOES-8, ~1/3 of its whole
# SAM band — wide enough to hold a sample, narrow enough that every draw is
# inside the region where P_ψ diverges and the σ branch is undetermined.
const D_SEP_NEAR = 0.02
d_sep(Id) = abs(Id - I.Ii) / (I.Is - I.Il)

const SIGMAS_SEP = (-1, +1)                # Figure 6 runs BOTH branches
const SIGMA_MAIN = -1                      # exp9's value, for continuity

const R_ORB, I_ORB, OMEGA_ORB = 4.2575e7, 0.0, 0.0
const WE_FLOOR, WE_CEILING = 1e-8, 1.0
const EPS_BETA_MAX = 1.0                   # ε_β above this ⇒ outside the domain
const SEP_TOL      = 1e-4                  # relative |I_d − I_i| counted as "on it"
const N_SCAN       = 400                   # samples/trajectory for the ε_β scan
const N_PHI, N_TAU = 30, 60                # :numeric quadrature (exp9's setting)

# Wall-clock caps per propagation — see budget_callback's docstring for why a
# time cap and not `maxiters`.  Sized off measurement, not taste: a completed
# 100 yr GOES-8 run is 0.43 s on :analytic and 197 s on :numeric at (30,60), so
# these are ~100× and ~5× the honest cost.  A run that hits them was not going
# to finish.
const BUDGET_ANALYTIC_S = SMOKE ? 20.0 :  45.0
const BUDGET_NUMERIC_S  = SMOKE ? 60.0 : 900.0

# Shared time grid for the series CSV: log-spaced, because the whole question is
# a collapse that happens early.  t = 0 prepended explicitly.
const N_SERIES = 120
const T_GRID_YR = [0.0; exp.(range(log(1e-3), log(YEARS), length = N_SERIES - 1))]

const OUT_MAIN   = joinpath(@__DIR__, SMOKE ? "smoke_ic_ensemble_goes8.csv" : "ic_ensemble_goes8.csv")
const OUT_SERIES = joinpath(@__DIR__, SMOKE ? "smoke_ic_ensemble_goes8_series.csv" : "ic_ensemble_goes8_series.csv")

lam_frac(f) = I.Il + f * (I.Ii - I.Il)
sam_frac(f) = I.Ii + f * (I.Is - I.Ii)

# ══════════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════════
"""
    maximin_lhs(n, d; ncand, rng) → (X::Matrix (d×n), min_dist)

Latin hypercube design selected on the maximin criterion.

WRITTEN OUT RATHER THAN CALLED.  inertia_sweep.jl builds the same design from
QuasiMonteCarlo.jl, and its own docstring records why that is only a starting
point: `LatinHypercubeSample` there is a plain random LHS with no maximin
option, so the maximin selection is applied on top by hand either way.  Since
the library call contributes nothing but a dependency — and QuasiMonteCarlo is
NOT in this repo's Project.toml, which is one of the two reasons inertia_sweep.jl
cannot currently run — the plain-LHS construction is inlined here from `Random`
(stdlib).  The construction is identical: stratify each dimension into n bins,
permute independently, jitter within the bin.       [FLAG-RQ1-LHS-INLINED]
"""
function maximin_lhs(n::Int, d::Int; ncand::Int = 400,
                     rng::AbstractRNG = MersenneTwister(IC_SEED))
    best = zeros(d, n); bestsep = -Inf
    for _ in 1:ncand
        X = Matrix{Float64}(undef, d, n)
        for i in 1:d
            X[i, :] = (randperm(rng, n) .- rand(rng, n)) ./ n
        end
        sep = Inf
        for i in 1:n-1, j in i+1:n
            sep = min(sep, sum(abs2, view(X, :, i) - view(X, :, j)))
        end
        if sep > bestsep
            bestsep = sep; best = X
        end
    end
    return best, sqrt(bestsep)
end

"""
n initial conditions from a 5-D maximin LHS over
(α₀, β₀, P_e, band fraction, band selector).

The 5th coordinate splits the ensemble between LAM and SAM; because an LHS
stratifies EVERY dimension, that split is exactly even rather than binomial.
"""
function ensemble_draws(n::Int; seed::Int = IC_SEED)
    U, _ = maximin_lhs(n, 5; ncand = 60, rng = MersenneTwister(seed))
    map(1:n) do k
        uα, uβ, uP, uf, ub = U[1, k], U[2, k], U[3, k], U[4, k], U[5, k]
        α0 = deg2rad(ALPHA_RANGE_DEG[1] + uα * (ALPHA_RANGE_DEG[2] - ALPHA_RANGE_DEG[1]))
        β0 = deg2rad(BETA_RANGE_DEG[1]  + uβ * (BETA_RANGE_DEG[2]  - BETA_RANGE_DEG[1]))
        Pe = exp(log(PE_RANGE_MIN[1]) + uP * (log(PE_RANGE_MIN[2]) - log(PE_RANGE_MIN[1]))) * 60
        f  = BAND_FRAC_RANGE[1] + uf * (BAND_FRAC_RANGE[2] - BAND_FRAC_RANGE[1])
        Id0, band = ub < 0.5 ? (lam_frac(f), "LAM") : (sam_frac(f), "SAM")
        (α0 = α0, β0 = β0, Pe = Pe, Id0 = Id0, band = band, band_frac = f)
    end
end

"""
n initial conditions drawn NEAR the separatrix, half on each side.

The band fraction is mapped so that |I_d0 − I_i|/(I_s − I_l) lands uniformly in
(0, D_SEP_NEAR).  The two bands have very different widths for GOES-8, so this
is done in absolute I_d and then converted, rather than by shrinking each band's
fraction range — otherwise the LAM side would sit ~18× further from I_i than the
SAM side at the same nominal "fraction".
"""
function separatrix_draws(n::Int; seed::Int = IC_SEED + 1)
    U, _ = maximin_lhs(n, 4; ncand = 60, rng = MersenneTwister(seed))
    span = D_SEP_NEAR * (I.Is - I.Il)
    map(1:n) do k
        uα, uβ, uP, ud = U[1, k], U[2, k], U[3, k], U[4, k]
        α0 = deg2rad(ALPHA_RANGE_DEG[1] + uα * (ALPHA_RANGE_DEG[2] - ALPHA_RANGE_DEG[1]))
        β0 = deg2rad(BETA_RANGE_DEG[1]  + uβ * (BETA_RANGE_DEG[2]  - BETA_RANGE_DEG[1]))
        Pe = exp(log(PE_RANGE_MIN[1]) + uP * (log(PE_RANGE_MIN[2]) - log(PE_RANGE_MIN[1]))) * 60
        # ud ∈ (0,1): map to a signed offset, sign by parity so the split is exact
        off = span * (0.05 + 0.9 * ud)
        Id0 = isodd(k) ? I.Ii - off : I.Ii + off
        Id0 = clamp(Id0, I.Il + 1e-6 * (I.Ii - I.Il), I.Is - 1e-9)
        band = Id0 < I.Ii ? "LAM" : "SAM"
        f = band == "LAM" ? (Id0 - I.Il) / (I.Ii - I.Il) : (Id0 - I.Ii) / (I.Is - I.Ii)
        (α0 = α0, β0 = β0, Pe = Pe, Id0 = Id0, band = band, band_frac = f)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# One propagation
# ══════════════════════════════════════════════════════════════════════════════
make_cfg(J, μoJ, σ, backend) =
    PerturbationConfig(srp = true, dissipation = true, gravity_gradient = true,
                       μ = μoJ * J, J = J,
                       orbit_i = I_ORB, orbit_Ω = OMEGA_ORB, orbit_R = R_ORB,
                       srp_backend = backend, σ_branch = σ)

"ε_β with the separatrix guarded — P_ψ diverges there, so Inf is the right answer."
function eps_beta_at(st::OsculatingState, cfg)
    abs(st.Id - I.Ii) / I.Ii < SEP_TOL && return Inf
    return epsilon_beta(st, I, SH, cfg)
end

"Peak ε_β, first domain exit, closest separatrix approach, crossing count."
function trajectory_scan(sol, cfg)
    te = sol.t[end]
    εmax = 0.0; t_exit = NaN; dmin = Inf; ncross = 0; prev = NaN
    for t in range(0, te, length = N_SCAN)
        u = sol(t)
        d = u[4] - I.Ii
        dmin = min(dmin, abs(d) / (I.Is - I.Il))
        isnan(prev) || (d * prev < 0 && (ncross += 1))
        prev = d
        e = eps_beta_at(OsculatingState(u[1], u[2], u[3], u[4]), cfg)
        isfinite(e) || continue
        e > εmax && (εmax = e)
        isnan(t_exit) && e > EPS_BETA_MAX && (t_exit = t)
    end
    return εmax, t_exit, dmin, ncross
end

"""
Sample the solution on the SHARED log time grid.

Points beyond the solution's own end (a terminated run) are written as NaN
rather than held at the last value: a despun run has no ω̄_e at 100 yr, and
carrying the final value forward would silently pull the ensemble's dispersion
down at exactly the times the dispersion is the measurement.
"""
function series_sample(sol)
    te = sol.t[end]
    out = Vector{NTuple{5,Float64}}(undef, length(T_GRID_YR))
    for (i, tyr) in enumerate(T_GRID_YR)
        t = tyr * SECONDS_PER_YEAR
        if t > te + 1.0
            out[i] = (tyr, NaN, NaN, NaN, NaN)
        else
            u = sol(min(t, te))
            out[i] = (tyr, u[3] / u[4], u[4] / I.Is, rad2deg(u[2]), rad2deg(mod(u[1], 2π)))
        end
    end
    return out
end

"""
Terminate a propagation that has spent more than `budget` seconds of WALL CLOCK.

WHY THIS IS NEEDED, MEASURED NOT GUESSED.  A trajectory that grazes I_d = I_i
drives P_ψ → ∞ and the averaged RHS with it; Tsit5 responds by shrinking the
step without bound, so `maxiters = 1e7` is not a usable bound — the run neither
finishes nor errors.  A smoke run of this script sat at 24/32 cases for 11
minutes on exactly the near-separatrix block.  inertia_sweep.jl's header records
the same behaviour from the other direction ([FLAG-EXP10-CALLBACKS]: "both
ContinuousCallbacks hang on a grazing trajectory; 0.12 yr run did not finish in
9 min"), and its author's response was to remove the callbacks — which does not
help here, because the stall is in the stepper, not the callbacks.

A wall-clock cap is an honest instrument as long as it is REPORTED: a capped run
is written with `timed_out = 1`, its own `t_end_yr`, and NaN in the series beyond
that time.  It is not silently treated as a completed 100-year run, and it is not
dropped.                                                 [FLAG-RQ1-WALLCAP]
"""
function budget_callback(budget::Real)
    t0 = time()
    return DiscreteCallback((u, t, integ) -> time() - t0 > budget,
                            integ -> terminate!(integ);
                            save_positions = (false, false))
end

function run_one(ic, J, μoJ, σ, backend)
    cfg = make_cfg(J, μoJ, σ, backend)
    kw  = backend === :numeric ? (N_φ = N_PHI, N_τ = N_TAU) : NamedTuple()
    ωe0 = 2π / ic.Pe
    H0  = ic.Id0 * ωe0
    st0 = OsculatingState(ic.α0, ic.β0, H0, ic.Id0)
    tf  = YEARS * SECONDS_PER_YEAR
    budget = backend === :numeric ? BUDGET_NUMERIC_S : BUDGET_ANALYTIC_S
    cb = CallbackSet(spin_termination_callbacks(ωe_floor = WE_FLOOR,
                                                ωe_ceiling = WE_CEILING),
                     budget_callback(budget))
    sol = propagate_averaged(I, st0, (0.0, tf); shape = SH, cfg = cfg,
              reltol = 1e-8, abstol = 1e-10, maxiters = Int(1e7),
              callback = cb, kw...)
    out = classify_spin_outcome(sol, tf; ωe_floor = WE_FLOOR, ωe_ceiling = WE_CEILING)
    # ε_β is scanned on :analytic regardless of the propagation backend, so the
    # validity column means the same thing in every row (exp11 does the same).
    cfg_scan = make_cfg(J, μoJ, σ, :analytic)
    εmax, t_exit, dmin, ncross = trajectory_scan(sol, cfg_scan)
    return out, ωe0, εmax, t_exit, dmin, ncross, series_sample(sol)
end

final_regime(Id) = abs(Id - I.Ii) / I.Ii < SEP_TOL ? "near-separatrix" :
                   (classify_regime(Id, I) isa LAM ? "LAM" : "SAM")

"""
Did this run stop because it ran out of WALL CLOCK rather than for a physical
reason?  Inferred rather than plumbed out of the callback: the run failed to
reach the horizon AND spent essentially its whole budget.  The 0.95 slack covers
the trailing `trajectory_scan`/`series_sample` work that `wall` also includes.
A `:despin` or `:spinup` run terminates early too, but in milliseconds of budget,
so it is not caught by this test.
"""
function timed_out(r)
    r.out === nothing && return 0
    r.out.reached_horizon && return 0
    budget = r.c.backend === :numeric ? BUDGET_NUMERIC_S : BUDGET_ANALYTIC_S
    return r.wall >= 0.95 * budget ? 1 : 0
end

# ══════════════════════════════════════════════════════════════════════════════
# Case list
# ══════════════════════════════════════════════════════════════════════════════
const ICS_MAIN = ensemble_draws(N_IC)
const ICS_SEP  = separatrix_draws(N_SEP)

struct Case
    block::String
    ic_id::Int
    ic::NamedTuple{(:α0,:β0,:Pe,:Id0,:band,:band_frac),
                   Tuple{Float64,Float64,Float64,Float64,String,Float64}}
    J::Float64
    μoJ::Float64
    σ::Int
    backend::Symbol
end

cases = Case[]
# main — Figures 4, 5, 7
for (k, ic) in enumerate(ICS_MAIN), J in Js, m in MUoJs
    push!(cases, Case("main", k, ic, J, m, SIGMA_MAIN, :analytic))
end
# sigma — Figure 6.  Near-separatrix ICs on BOTH branches, plus the main ICs on
# both branches as the far-from-separatrix control.
for (k, ic) in enumerate(ICS_SEP), σ in SIGMAS_SEP
    push!(cases, Case("sigma_near", 1000 + k, ic, J_REF, MUoJ_REF, σ, :analytic))
end
for (k, ic) in enumerate(ICS_MAIN), σ in SIGMAS_SEP
    σ == SIGMA_MAIN && continue          # already in "main" at (J_REF, MUoJ_REF)
    push!(cases, Case("sigma_far", k, ic, J_REF, MUoJ_REF, σ, :analytic))
end
# backend — the [FLAG-BACKEND-SPLIT] replicate.
# N_IC_BACKEND < N_IC because :numeric costs ~460x what :analytic does per run
# (0.43 s vs 197 s for 100 yr, measured).  48 of the 96 ICs across all three J
# values is 144 runs ~= 8 CPU-hours; the full 96 would be 16.  The subset is the
# first 48 LHS draws, which an LHS stratifies in every dimension, so it is a
# coverage-preserving thinning rather than an arbitrary truncation.
const N_IC_BACKEND = parse(Int, get(ENV, "RQ1_NIC_BACKEND", "48"))
for (k, ic) in enumerate(ICS_MAIN), J in JS_BACKEND
    k <= N_IC_BACKEND || continue
    push!(cases, Case("backend", k, ic, J, MUoJ_REF, SIGMA_MAIN, :numeric))
end

# ── BLOCK / SHARD SELECTION ────────────────────────────────────────────────────
# The :numeric block CANNOT share a process with the rest.  [FLAG-RQ1-GC-THREADS]
#
# Measured, on 8 identical 0.5 yr GOES-8 propagations: 3.2 s each run serially,
# >25 min each run under `Threads.@threads` — a >400× per-case slowdown that
# grows with thread count, and under a step cap the threaded runs blow `maxiters`
# while the serial ones converge in ~1800 steps.  It is not a bad initial
# condition (all eight were checked individually) and it is not shared mutable
# state (`averaged_srp_torques` is pure, returning SVectors).  What it is: the
# :numeric backend evaluates an N_φ×N_τ = 1800-point quadrature per RHS call,
# each point calling Elliptic.jl's Jacobi functions, which allocate.  Julia's GC
# is stop-the-world, so eight threads allocating at that rate spend essentially
# all their time synchronising.  :analytic has a closed form and does not.
#
# The fix is process-level parallelism, not thread-level: separate `julia -t 1`
# processes have independent heaps and independent GCs.  Hence these two knobs.
#
#   RQ1_BLOCKS=main,sigma_near,sigma_far   which blocks to run (default: all)
#   RQ1_SHARD=k RQ1_NSHARD=n              run every n-th case, offset k (0-based)
#
# A sharded run writes to a suffixed CSV; the shards are concatenated afterwards.
#
# NOTE FOR THE REPO, NOT JUST FOR THIS SCRIPT: exp9 (GOES8_sim) runs its
# `NUMERIC_EVERY = 25` cross-check inside its own `Threads.@threads` loop, which
# is exactly this configuration.  Its `omega_e_tailmean_numeric` column was
# produced under the same pathology and should not be trusted until re-run this
# way.  Not touched here — reporting, not silently fixing.
const BLOCKS = let s = get(ENV, "RQ1_BLOCKS", "")
    isempty(s) ? nothing : Set(strip.(split(s, ',')))
end
const SHARD  = parse(Int, get(ENV, "RQ1_SHARD",  "0"))
const NSHARD = parse(Int, get(ENV, "RQ1_NSHARD", "1"))

#   RQ1_CASES=33,35,45,...                rerun exactly these case_ids
#
# RQ1_CASES was added for the I_d → I_s clamp re-run (src/torque_free.jl).  70 of
# the 1477 rows in ic_ensemble_goes8.csv died on that DomainError; with the clamp
# in place they can complete, and ONLY they need redoing — the other 1407 are
# unaffected, because the clamp is unreachable unless I_d overshoots I_s.
#
# case_id IS NOW THE INDEX INTO THE UNFILTERED CASE LIST, not the position within
# the filtered one.  For a default full run those are the same number, so existing
# output is unchanged; but it is what makes a subset re-run mergeable back into
# the parent CSV by id.  `sel` is carried to the writers as ORIG_ID.
#
# Filter ORDER is preserved: BLOCKS first, then the shard stride over the
# survivors, exactly as before — so RQ1_SHARD keeps the meaning the already-
# written backend shards were produced with.
const CASE_SEL = let s = get(ENV, "RQ1_CASES", "")
    isempty(s) ? nothing : Set(parse(Int, strip(x)) for x in split(s, ','))
end

sel = collect(1:length(cases))
BLOCKS   === nothing || (sel = [i for i in sel if cases[i].block in BLOCKS])
NSHARD   == 1        || (sel = [s for (j, s) in enumerate(sel) if (j - 1) % NSHARD == SHARD])
CASE_SEL === nothing || (sel = [i for i in sel if i in CASE_SEL])
const ORIG_ID = sel
cases = cases[sel]
isempty(cases) && error("no cases selected: RQ1_BLOCKS=$(get(ENV,"RQ1_BLOCKS","")) " *
                        "RQ1_SHARD=$SHARD/$NSHARD RQ1_CASES=$(get(ENV,"RQ1_CASES",""))")
if CASE_SEL !== nothing && length(cases) != length(CASE_SEL)
    @warn "RQ1_CASES asked for $(length(CASE_SEL)) ids but $(length(cases)) survive the " *
          "other filters" missing = sort(collect(setdiff(CASE_SEL, Set(sel))))
end

const TAG = (BLOCKS === nothing ? "" : "_" * join(sort(collect(BLOCKS)), "+")) *
            (NSHARD == 1 ? "" : @sprintf("_shard%02dof%02d", SHARD, NSHARD)) *
            (CASE_SEL === nothing ? "" : "_cases")

# TAG is empty for a full single-process run, so the default filenames are
# exactly the ones the header advertises.
const OUT_MAIN_T   = replace(OUT_MAIN,   ".csv" => TAG * ".csv")
const OUT_SERIES_T = replace(OUT_SERIES, ".csv" => TAG * ".csv")

# ══════════════════════════════════════════════════════════════════════════════
# Progress (same one-line format as exp9/exp11)
# ══════════════════════════════════════════════════════════════════════════════
const _done  = Threads.Atomic{Int}(0)
const _lock  = ReentrantLock()
const _t0    = Ref(0.0)
const _fates = Dict{Symbol,Int}()

_hms(s) = (s = max(s, 0.0); h = floor(Int, s/3600); m = floor(Int, (s%3600)/60);
           sec = floor(Int, s%60);
           h > 0 ? (@sprintf("%dh%02dm", h, m)) : (@sprintf("%2dm%02ds", m, sec)))

function progress!(ntot, fate)
    d = Threads.atomic_add!(_done, 1) + 1
    lock(_lock) do
        _fates[fate] = get(_fates, fate, 0) + 1
        el  = time() - _t0[]
        eta = d < ntot ? el * (ntot - d) / d : 0.0
        W = 28; nf = round(Int, W * d / ntot)
        tally = join([@sprintf("%s:%d", k, v)
                      for (k, v) in sort(collect(_fates), by = x -> -x[2])], " ")
        @printf("\r  [%s%s] %5d/%-5d %5.1f%%  up %s  eta %s │ %-42s",
                "█"^nf, "░"^(W-nf), d, ntot, 100d/ntot, _hms(el), _hms(eta), tally)
        flush(stdout)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT
# ══════════════════════════════════════════════════════════════════════════════
println("="^80)
println("RQ1 IC ENSEMBLE — GOES-8, ", YEARS, " yr, ", length(cases), " cases on ",
        Threads.nthreads(), " threads")
println("="^80)
@printf("GOES-8  I_l=%.1f  I_i=%.1f  I_s=%.1f kg m²   (%d facets, %.2f m²)\n",
        I.Il, I.Ii, I.Is, n_facets(SH), total_area(SH))
@printf("  LAM band I_d/I_s ∈ [%.4f, %.4f]  (%.1f%% of the axis)\n",
        I.Il/I.Is, I.Ii/I.Is, 100*(I.Ii - I.Il)/(I.Is - I.Il))
@printf("  SAM band I_d/I_s ∈ [%.4f, %.4f]  (%.1f%% of the axis)\n",
        I.Ii/I.Is, 1.0, 100*(I.Is - I.Ii)/(I.Is - I.Il))
@printf("  \"near the separatrix\" ⇔ |I_d − I_i|/(I_s − I_l) < %.3f  = %.1f kg m²\n",
        D_SEP_NEAR, D_SEP_NEAR*(I.Is - I.Il))

println("\nBLOCKS")
for b in unique(c.block for c in cases)
    n = count(c -> c.block == b, cases)
    bk = unique(c.backend for c in cases if c.block == b)
    σs = sort(unique(c.σ for c in cases if c.block == b))
    @printf("  %-11s %5d cases   backend %-20s σ %s\n", b, n, string(bk), string(σs))
end

# α₀ IS the new axis; show that it is actually spread, and that exp9's is not.
αs = sort(rad2deg.([ic.α0 for ic in ICS_MAIN]))
@printf("\nα₀ coverage (the axis exp9 lacks): n=%d  min %.1f°  median %.1f°  max %.1f°\n",
        length(αs), αs[1], αs[cld(end,2)], αs[end])
println("  exp9_goes8_sweep.csv: every row has α₀ = 0 (no α column at all).")

nlam = count(ic -> ic.band == "LAM", ICS_MAIN)
@printf("band split, main ensemble: LAM %d / SAM %d\n", nlam, N_IC - nlam)
@printf("near-separatrix draws: LAM side %d / SAM side %d, |I_d0−I_i|/(I_s−I_l) ∈ [%.4f, %.4f]\n",
        count(ic -> ic.band == "LAM", ICS_SEP), count(ic -> ic.band == "SAM", ICS_SEP),
        minimum(d_sep(ic.Id0) for ic in ICS_SEP), maximum(d_sep(ic.Id0) for ic in ICS_SEP))

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
println("\nrunning…")
_t0[] = time()
results = Vector{Any}(undef, length(cases))

Threads.@threads for idx in eachindex(cases)
    c = cases[idx]
    t0 = time()
    r = try
        out, ωe0, εmax, t_exit, dmin, ncross, ser = run_one(c.ic, c.J, c.μoJ, c.σ, c.backend)
        (c = c, out = out, ωe0 = ωe0, εmax = εmax, t_exit = t_exit, dmin = dmin,
         ncross = ncross, ser = ser, wall = time() - t0, err = "")
    catch e
        # The failure mode this used to catch — a DomainError from
        # sqrt(I_s − I_d) inside torquefree_params_SAM when dissipation drives
        # I_d onto I_s — is FIXED (src/torque_free.jl clamps I_d to I_s;
        # test/test_Id_clamp.jl).  It cost this ensemble 70 of 1477 rows, all of
        # which now complete; they were re-run via RQ1_CASES and merged back by
        # merge_case_rerun.py.  The catch stays because a 1477-case batch should
        # not lose the other 1476 runs to one bad one, whatever the next bad one
        # turns out to be.
        (c = c, out = nothing, ωe0 = 2π/c.ic.Pe, εmax = NaN, t_exit = NaN,
         dmin = NaN, ncross = 0, ser = NTuple{5,Float64}[], wall = time() - t0,
         err = first(replace(sprint(showerror, e), ',' => ';', '\n' => ' '), 90))
    end
    results[idx] = r
    progress!(length(cases), r.out === nothing ? :error : r.out.fate)
end
println()

# ══════════════════════════════════════════════════════════════════════════════
# WRITE
# ══════════════════════════════════════════════════════════════════════════════
open(OUT_MAIN_T, "w") do io
    println(io, "case_id,block,backend,ic_id,alpha0_deg,beta0_deg,Pe_min,",
                "Id0_over_Is,band0,band_frac0,d_sep0,J,mu_over_J,sigma,omega_e0,",
                "fate,omega_e_final,omega_e_tailmean,omega_e_peak,omega_e_min,ripple,",
                "t_end_yr,reached_horizon,alpha_final_deg,beta_final_deg,",
                "Id_final_over_Is,regime_final,n_sep_crossings,min_d_separatrix,",
                "eps_beta_max,in_domain,t_eps_exit_yr,timed_out,wall_s,error")
    for (idx, r) in enumerate(results)
        c = r.c; ic = c.ic
        @printf(io, "%d,%s,%s,%d,%.3f,%.3f,%.3f,%.6f,%s,%.4f,%.6f,%.4f,%.1e,%+d,%.8e,",
                ORIG_ID[idx], c.block, c.backend, c.ic_id, rad2deg(ic.α0), rad2deg(ic.β0),
                ic.Pe/60, ic.Id0/I.Is, ic.band, ic.band_frac, d_sep(ic.Id0),
                c.J, c.μoJ, c.σ, 2π/ic.Pe)
        if r.out === nothing
            @printf(io, "error,,,,,,,0,,,,,%d,%.6e,,,,,%.2f,%s\n",
                    r.ncross, NaN, r.wall, r.err)
        else
            o = r.out
            @printf(io, "%s,%.8e,%.8e,%.8e,%.8e,%.6e,%.4f,%d,%.3f,%.4f,%.6f,%s,%d,%.6e,%.6e,%d,%.4f,%d,%.2f,\n",
                    o.fate, o.ωe_final, o.ωe_mean_tail, o.ωe_peak, o.ωe_min, o.ripple,
                    o.t_end/SECONDS_PER_YEAR, o.reached_horizon,
                    rad2deg(mod(o.α_final, 2π)), rad2deg(o.β_final), o.Id_final/I.Is,
                    final_regime(o.Id_final), r.ncross, r.dmin, r.εmax,
                    isnan(r.t_exit) ? 1 : 0, r.t_exit/SECONDS_PER_YEAR,
                    timed_out(r), r.wall)
        end
    end
end
println("wrote ", OUT_MAIN_T)

open(OUT_SERIES_T, "w") do io
    println(io, "case_id,block,backend,ic_id,J,mu_over_J,sigma,band0,band_frac0,",
                "d_sep0,t_yr,omega_e,Id_over_Is,beta_deg,alpha_deg")
    for (idx, r) in enumerate(results)
        isempty(r.ser) && continue
        c = r.c
        for (tyr, we, idis, bdeg, adeg) in r.ser
            @printf(io, "%d,%s,%s,%d,%.4f,%.1e,%+d,%s,%.4f,%.6f,%.6e,%.8e,%.6f,%.4f,%.4f\n",
                    ORIG_ID[idx], c.block, c.backend, c.ic_id, c.J, c.μoJ, c.σ, c.ic.band,
                    c.ic.band_frac, d_sep(c.ic.Id0), tyr, we, idis, bdeg, adeg)
        end
    end
end
println("wrote ", OUT_SERIES_T)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY — the RQ1 numbers, printed so a failed plot still leaves the finding
# ══════════════════════════════════════════════════════════════════════════════
ok(rs) = [r for r in rs if r.out !== nothing && r.out.fate !== :error]

println("\n== outcome census ==")
for b in unique(c.block for c in cases)
    rs = [r for r in results if r.c.block == b]
    tally = Dict{Symbol,Int}()
    for r in rs
        f = r.out === nothing ? :error : r.out.fate
        tally[f] = get(tally, f, 0) + 1
    end
    @printf("  %-11s n=%4d  %s\n", b, length(rs),
            join([@sprintf("%s:%d", k, v) for (k,v) in sort(collect(tally), by=x->-x[2])], " "))
end

nto = count(timed_out(r) == 1 for r in results)
@printf("\nwall-clock cap [FLAG-RQ1-WALLCAP]: %d/%d runs (%.1f%%) hit it\n",
        nto, length(results), 100nto/length(results))
if nto > 0
    tos = [r for r in results if timed_out(r) == 1]
    @printf("  they stopped at t = %.3f–%.3f yr; %d/%d were drawn near the separatrix\n",
            minimum(r.out.t_end for r in tos)/SECONDS_PER_YEAR,
            maximum(r.out.t_end for r in tos)/SECONDS_PER_YEAR,
            count(r -> d_sep(r.c.ic.Id0) < D_SEP_NEAR, tos), length(tos))
    println("  their series carry NaN beyond that time, so they leave the dispersion")
    println("  statistic rather than biasing it downward — see the n column below.")
end

"""
Relative dispersion of ω̄_e across an ensemble at one time: (p90 − p10)/median.

WHY THIS AND NOT THE STANDARD DEVIATION.  ω̄_e spans two decades across the
ensemble because P_e0 does, and a handful of runs approach the despin floor.  A
moment-based spread is then dominated by the tail rather than by the bulk, which
is the opposite of what "the ensemble has collapsed onto one distribution"
means.  p90−p10 over the median is scale-free and insensitive to both tails.
"""
function rel_dispersion(vals)
    v = sort([x for x in vals if isfinite(x) && x > 0])
    length(v) < 8 && return NaN
    q(p) = v[clamp(ceil(Int, p*length(v)), 1, length(v))]
    m = v[cld(length(v), 2)]
    return (q(0.90) - q(0.10)) / m
end

"""
FORGETTING TIME — the operational definition this ensemble reports.

T_forget(J, μ/J) is the earliest grid time t such that the ensemble's relative
dispersion of ω̄_e stays below FORGET_TOL for every LATER grid time as well.
The "and stays" clause matters: dispersion is not monotone (a group can pass
through a common value on its way somewhere else), and without it the first
crossing can be a coincidence rather than a collapse.

NaN means the ensemble never collapsed inside the horizon — reported as such,
not silently replaced by the horizon.
"""
const FORGET_TOL = 0.10

function forget_time(rs)
    isempty(rs) && return (NaN, fill(NaN, length(T_GRID_YR)))
    disp = [rel_dispersion([r.ser[i][2] for r in rs if !isempty(r.ser)])
            for i in eachindex(T_GRID_YR)]
    tf = NaN
    for i in length(T_GRID_YR):-1:1
        d = disp[i]
        (isnan(d) || d > FORGET_TOL) && break
        tf = T_GRID_YR[i]
    end
    return tf, disp
end

println("\n== RQ1: relative dispersion of ω̄_e, (p90−p10)/median, main block ==")
println("  \"forgotten\" ⇔ dispersion < ", FORGET_TOL, " and stays below for the rest of the run")
println("  n0 = runs in the group; n100 = of those, how many still have a ω̄_e at 100 yr")
@printf("%8s %10s %5s %6s %9s %9s %9s %9s %12s\n",
        "J", "μ/J", "n0", "n100", "t=0", "1 yr", "10 yr", "100 yr", "T_forget[yr]")
_at(disp, tyr) = disp[argmin(abs.(T_GRID_YR .- tyr))]
for J in Js, m in MUoJs
    rs = ok([r for r in results if r.c.block == "main" && r.c.J == J && r.c.μoJ == m])
    isempty(rs) && continue
    tf, disp = forget_time(rs)
    n100 = count(r -> !isempty(r.ser) && isfinite(r.ser[end][2]), rs)
    @printf("%8.2f %10.0e %5d %6d %9.3f %9.3f %9.3f %9.3f %12s\n",
            J, m, length(rs), n100, _at(disp,0.0), _at(disp,1.0), _at(disp,10.0),
            _at(disp,100.0), isnan(tf) ? "never" : @sprintf("%.3f", tf))
end

println("\n== [FLAG-BACKEND-SPLIT] replicate: :analytic vs :numeric, same ICs, μ/J = 1e-3 ==")
@printf("%8s %-10s %5s %12s %16s %26s\n",
        "J", "backend", "n", "T_forget[yr]", "settled ω̄_e med", "[min, max]")
for J in JS_BACKEND, bk in (:analytic, :numeric)
    rs = ok([r for r in results
             if r.c.backend == bk && r.c.J == J && r.c.μoJ == MUoJ_REF &&
                r.c.σ == SIGMA_MAIN && r.c.block in ("main", "backend")])
    isempty(rs) && continue
    tf, disp = forget_time(rs)
    w = sort([r.out.ωe_mean_tail for r in rs if isfinite(r.out.ωe_mean_tail)])
    @printf("%8.2f %-10s %5d %12s %16.4e   [%.4e, %.4e]\n",
            J, bk, length(rs), isnan(tf) ? "never" : @sprintf("%.3f", tf),
            w[cld(end,2)], w[1], w[end])
end
println("  If T_forget agrees and the settled value does not, the RQ1 CONCLUSION is")
println("  backend-robust while the attractor VALUE is not.  That is the claim.")

println("\n== Figure 6: σ near the separatrix ==")
for b in ("sigma_near", "sigma_far")
    for σ in SIGMAS_SEP
        rs = ok([r for r in results if r.c.block == b && r.c.σ == σ])
        b == "sigma_far" && σ == SIGMA_MAIN &&
            (rs = ok([r for r in results if r.c.block == "main" && r.c.σ == σ &&
                      r.c.J == J_REF && r.c.μoJ == MUoJ_REF]))
        isempty(rs) && continue
        nsam = count(r -> final_regime(r.out.Id_final) == "SAM", rs)
        w = sort([r.out.ωe_mean_tail for r in rs if isfinite(r.out.ωe_mean_tail)])
        @printf("  %-11s σ=%+d  n=%3d   final SAM %3d (%.1f%%)   settled ω̄_e median %.4e\n",
                b, σ, length(rs), nsam, 100nsam/length(rs), w[cld(end,2)])
    end
end

@printf("\ntotal wall %.1f min (sum over cases), elapsed %.1f min\n",
        sum(r.wall for r in results)/60, (time() - _t0[])/60)
