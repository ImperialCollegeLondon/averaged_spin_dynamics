

include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))
using .master
using StaticArrays, LinearAlgebra, Printf, Random

using DifferentialEquations: DiscreteCallback, CallbackSet, terminate!


I  = goes8_inertia()
SH = goes8_shape_full(; θ_sa = deg2rad(17), optical = :bs)

# GRID — edit here

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


const ALPHA_RANGE_DEG = (0.0, 360.0)
const BETA_RANGE_DEG  = (15.0, 165.0)
const PE_RANGE_MIN    = (30.0, 480.0)      # log-uniform [

const BAND_FRAC_RANGE = (0.05, 0.95)       # kept off 0 (uniform spin) and 1 (separatrix)

const D_SEP_NEAR = 0.02
d_sep(Id) = abs(Id - I.Ii) / (I.Is - I.Il)

const SIGMAS_SEP = (-1, +1)                # Figure 6 runs BOTH branches
const SIGMA_MAIN = -1                      # exp9's value, for continuity

const R_ORB, I_ORB, OMEGA_ORB = 4.2575e7, 0.0, 0.0
const WE_FLOOR, WE_CEILING = 1e-8, 1.0
const EPS_BETA_MAX = 1.0                   # ε_β above this ⇒ outside the domain
const SEP_TOL      = 1e-4                  # relative |I_d − I_i| 
const N_SCAN       = 400                   # samples/trajectory for the ε_β scan
const N_PHI, N_TAU = 30, 60                # :numeric quadrature 


const BUDGET_ANALYTIC_S = SMOKE ? 20.0 :  45.0
const BUDGET_NUMERIC_S  = SMOKE ? 60.0 : 900.0


const N_SERIES = 120
const T_GRID_YR = [0.0; exp.(range(log(1e-3), log(YEARS), length = N_SERIES - 1))]

const OUT_MAIN   = joinpath(@__DIR__, SMOKE ? "smoke_ic_ensemble_goes8.csv" : "ic_ensemble_goes8.csv")
const OUT_SERIES = joinpath(@__DIR__, SMOKE ? "smoke_ic_ensemble_goes8_series.csv" : "ic_ensemble_goes8_series.csv")

lam_frac(f) = I.Il + f * (I.Ii - I.Il)
sam_frac(f) = I.Ii + f * (I.Is - I.Ii)

# Sampling

"""
    maximin_lhs(n, d; ncand, rng) → (X::Matrix (d×n), min_dist)

Latin hypercube design selected on the maximin criterion.

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
fraction range.
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


# One propagation

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


function timed_out(r)
    r.out === nothing && return 0
    r.out.reached_horizon && return 0
    budget = r.c.backend === :numeric ? BUDGET_NUMERIC_S : BUDGET_ANALYTIC_S
    return r.wall >= 0.95 * budget ? 1 : 0
end


# Case list

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

const N_IC_BACKEND = parse(Int, get(ENV, "RQ1_NIC_BACKEND", "48"))
for (k, ic) in enumerate(ICS_MAIN), J in JS_BACKEND
    k <= N_IC_BACKEND || continue
    push!(cases, Case("backend", k, ic, J, MUoJ_REF, SIGMA_MAIN, :numeric))
end

# BLOCK SELECTION 

const BLOCKS = let s = get(ENV, "RQ1_BLOCKS", "")
    isempty(s) ? nothing : Set(strip.(split(s, ',')))
end
const SHARD  = parse(Int, get(ENV, "RQ1_SHARD",  "0"))
const NSHARD = parse(Int, get(ENV, "RQ1_NSHARD", "1"))

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



const OUT_MAIN_T   = replace(OUT_MAIN,   ".csv" => TAG * ".csv")
const OUT_SERIES_T = replace(OUT_SERIES, ".csv" => TAG * ".csv")

# Progress

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


# PRE-FLIGHT CHECKS

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


# RUN

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
        (c = c, out = nothing, ωe0 = 2π/c.ic.Pe, εmax = NaN, t_exit = NaN,
         dmin = NaN, ncross = 0, ser = NTuple{5,Float64}[], wall = time() - t0,
         err = first(replace(sprint(showerror, e), ',' => ';', '\n' => ' '), 90))
    end
    results[idx] = r
    progress!(length(cases), r.out === nothing ? :error : r.out.fate)
end
println()

# WRITE

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

# SUMMARY

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


function rel_dispersion(vals)
    v = sort([x for x in vals if isfinite(x) && x > 0])
    length(v) < 8 && return NaN
    q(p) = v[clamp(ceil(Int, p*length(v)), 1, length(v))]
    m = v[cld(length(v), 2)]
    return (q(0.90) - q(0.10)) / m
end


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
