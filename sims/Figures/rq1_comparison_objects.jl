#=
rq1_comparison_objects.jl — FIGURES 8, 9, 10: does the RQ1 answer generalise?

Phase 3 measured, for GOES-8, whether an ensemble of initial conditions forgets
where it started.  Phases 4's question is whether that answer is a property of
GOES-8 or of the framework.  Answering it needs the OTHER objects run through
the SAME ensemble, not a re-read of their existing sweeps — because those sweeps
disagree with Phase 3's design in three ways that all matter:

  * they never vary α₀ (every existing row starts at α₀ = 0);
  * they are full-factorial lattices, so their marginal distributions are
    artefacts of the axis spacing rather than samples; and
  * they store ENDPOINTS only.  "Forgotten by when" is a statement about the
    dispersion as a function of time, and no endpoint CSV can answer it.

So all four comparison configurations are re-run here on the identical 96-point
LHS initial-condition ensemble that Phase 3 used, at the same reference
dissipation, with the same log time grid.  That is the whole point: the only
thing that differs between the groups is the OBJECT.

CONFIGURATIONS

  goes8_sa17     GOES-8, θ_sa = 17°, :bs optics — the Phase 3 baseline, re-run
                 here so the comparison is like-for-like rather than copied
  goes8_sa00     GOES-8, θ_sa =  0°, :bs optics
  goes8_albuja   GOES-8, θ_sa = 17°, :albuja optics
  telstar401     Telstar 401 (dissipation + GG only — see below)
  skynet1a       Skynet 1A, the asymmetric/triaxial variant from Phase 1

────────────────────────────────────────────────────────────────────────────────
FIGURE 8 IS NOT GOES-10 AND GOES-12, AND CANNOT BE.       [FLAG-RQ1-NO-GOES1012]

The brief asks for GOES 10/12 "only if θ_sa/reflectivity variants actually exist
in this repo — report as a gap if not."  Checked, and the answer is split:

  the KNOBS exist.  goes8_shape_full takes θ_sa and optical ∈ (:bs, :albuja).
  the OBJECTS do not.  There is no goes10_inertia, no goes12_inertia, no
  goes10_shape, no published θ_sa or optical set for either spacecraft anywhere
  in this repo.  A repo-wide grep for "GOES-10"/"GOES-12" returns three hits,
  all of them prose in a comment about where exp9's J range came from.

Inventing a θ_sa and calling the result "GOES-10" would be inventing a number.
So Figure 8 is reported as a GAP for GOES 10/12, and what is run instead is the
sensitivity that the existing knobs genuinely support: the same body with its
solar array at a different angle and with a different published optical set.
That answers "does the RQ1 result survive a change of torque geometry", which is
the useful half of the question, and it is labelled as that and not as a second
spacecraft.
────────────────────────────────────────────────────────────────────────────────

TELSTAR 401 IS A NULL CONTROL, NOT A YORP CASE.  Its assumed geometry has
IDENTICALLY ZERO net SRP torque ([FLAG-TELSTAR-INERT] in telstar_shape), so the
averaged model reduces to internal dissipation + gravity gradient, both of which
conserve H.  Its outcome is available in closed form, ω̄_e(∞) = ω_e0·(I_d0/I_s),
which means its ensemble CANNOT forget its initial condition — the endpoint is a
fixed function of it.  That is exactly what makes it worth running: it is the
limiting case that shows the forgetting in Figures 4 and 8 is DRIVEN BY THE
TORQUE and is not an artefact of the dissipation model or of the metric.

SKYNET 1A rests on invented antenna geometry ([FLAG-SKYNET-INVENTED-HORN]) and a
rigid-body assumption defended only for its defunct state
([FLAG-SKYNET-TWOBODY]).  Its RQ1 answer is a statement about that assumed
shape, not about the real spacecraft.

Run:  julia -t auto --project=. "sims/Figures/rq1_comparison_objects.jl"
Writes sims/Figures/rq1_comparison_objects.csv
       sims/Figures/rq1_comparison_objects_series.csv
=#

include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))
using .master
using StaticArrays, LinearAlgebra, Printf, Random
using DifferentialEquations: DiscreteCallback, CallbackSet, terminate!

# ── ONE OBJECT FAMILY PER PROCESS.                       [FLAG-RQ1-MP-COLLISION]
# telstar_inertia:105 defines a plain global `mp`, and skynet_inertia:429 defines
# `const mp`.  Both are included into Main by their shape scripts, so loading both
# in one process fails with "cannot declare Main.mp constant; it was already
# declared global".  Neither file is wrong on its own and both follow the same
# local convention; the collision is a consequence of this script being the first
# thing to want two objects at once.
#
# Deliberately NOT fixed by editing either sibling: telstar_inertia is not mine to
# rename, and wrapping them in modules would break their own `using .master`,
# which resolves against Main.  Instead RQ1_CMP selects one family per process and
# the three CSVs are concatenated afterwards — which costs nothing, since the
# ensembles are independent by construction.
const WHICH = get(ENV, "RQ1_CMP", "goes8")
WHICH in ("goes8", "telstar", "skynet") ||
    error("RQ1_CMP must be goes8 | telstar | skynet; got $WHICH")
WHICH == "telstar" && include(joinpath(@__DIR__, "..", "Telstar 402", "telstar_shape"))
WHICH == "skynet"  && include(joinpath(@__DIR__, "..", "Skynet 1A", "skynet_shape"))

# ══════════════════════════════════════════════════════════════════════════════
# GRID — matched to sims/GOES8/ic_ensemble_goes8.jl, deliberately
# ══════════════════════════════════════════════════════════════════════════════
const SMOKE   = get(ENV, "RQ1_SMOKE", "0") == "1"
const YEARS   = SMOKE ? 2.0 : 100.0
const N_IC    = SMOKE ? 6 : 96
const IC_SEED = 20260831                    # SAME seed as Phase 3
const J_REF, MUoJ_REF, SIGMA = 1.8, 1.0e-3, -1
const ALPHA_RANGE_DEG, BETA_RANGE_DEG, PE_RANGE_MIN = (0.0, 360.0), (15.0, 165.0), (30.0, 480.0)
const BAND_FRAC_RANGE = (0.05, 0.95)
const R_ORB, I_ORB, OMEGA_ORB = 4.2575e7, 0.0, 0.0
const WE_FLOOR, WE_CEILING = 1e-8, 1.0
const SEP_TOL, N_SCAN, EPS_BETA_MAX = 1e-4, 400, 1.0
const BUDGET_S = SMOKE ? 20.0 : 45.0
const N_SERIES = 120
const T_GRID_YR = [0.0; exp.(range(log(1e-3), log(YEARS), length = N_SERIES - 1))]
const OUT  = joinpath(@__DIR__, "rq1_comparison_objects_" * WHICH * ".csv")
const OUTS = joinpath(@__DIR__, "rq1_comparison_objects_" * WHICH * "_series.csv")

configs =
    WHICH == "goes8" ? [
        (name = "goes8_sa17",   I = goes8_inertia(),
         sh = goes8_shape_full(; θ_sa = deg2rad(17), optical = :bs),
         note = "Phase 3 baseline"),
        (name = "goes8_sa00",   I = goes8_inertia(),
         sh = goes8_shape_full(; θ_sa = deg2rad(0),  optical = :bs),
         note = "array angle 0 deg — NOT GOES-10"),
        (name = "goes8_albuja", I = goes8_inertia(),
         sh = goes8_shape_full(; θ_sa = deg2rad(17), optical = :albuja),
         note = "Albuja optical set — NOT GOES-12"),
    ] :
    WHICH == "telstar" ? [
        (name = "telstar401",   I = telstar401_inertia(),
         sh = telstar401_shape(; warn_inert = false),
         note = "SRP torque identically zero — null control"),
    ] : [
        (name = "skynet1a",     I = skynet1a_inertia(),
         sh = skynet1a_shape(),
         note = "invented antenna geometry"),
    ]

# ══════════════════════════════════════════════════════════════════════════════
function maximin_lhs(n::Int, d::Int; ncand::Int = 60,
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
        sep > bestsep && (bestsep = sep; best = X)
    end
    return best
end

"""
The SAME LHS draw for every object, mapped onto each object's own I_d bands.

The unit-hypercube draw is shared so the objects are compared at matched
initial conditions; the band fraction is then mapped through each object's own
(I_l, I_i, I_s), because I_d/I_s means something different for a long thin body
than for a blunt one.  GOES-8's SAM band is 5.3% of its I_d axis and Telstar's
is 15.6%, so a shared ABSOLUTE I_d0/I_s would put the two ensembles in
different regimes and the comparison would be meaningless.
"""
const U = maximin_lhs(N_IC, 5)

function draws_for(P::PrincipalInertias)
    map(1:N_IC) do k
        uα, uβ, uP, uf, ub = U[1, k], U[2, k], U[3, k], U[4, k], U[5, k]
        α0 = deg2rad(ALPHA_RANGE_DEG[1] + uα * (ALPHA_RANGE_DEG[2] - ALPHA_RANGE_DEG[1]))
        β0 = deg2rad(BETA_RANGE_DEG[1]  + uβ * (BETA_RANGE_DEG[2]  - BETA_RANGE_DEG[1]))
        Pe = exp(log(PE_RANGE_MIN[1]) + uP * (log(PE_RANGE_MIN[2]) - log(PE_RANGE_MIN[1]))) * 60
        f  = BAND_FRAC_RANGE[1] + uf * (BAND_FRAC_RANGE[2] - BAND_FRAC_RANGE[1])
        Id0, band = ub < 0.5 ? (P.Il + f*(P.Ii - P.Il), "LAM") : (P.Ii + f*(P.Is - P.Ii), "SAM")
        (α0 = α0, β0 = β0, Pe = Pe, Id0 = Id0, band = band, band_frac = f)
    end
end

budget_cb(b) = (t0 = time();
    DiscreteCallback((u, t, integ) -> time() - t0 > b, integ -> terminate!(integ);
                     save_positions = (false, false)))

function run_one(cfgobj, ic)
    P, sh = cfgobj.I, cfgobj.sh
    cfg = PerturbationConfig(srp = true, dissipation = true, gravity_gradient = true,
                             μ = MUoJ_REF * J_REF, J = J_REF,
                             orbit_i = I_ORB, orbit_Ω = OMEGA_ORB, orbit_R = R_ORB,
                             srp_backend = :analytic, σ_branch = SIGMA)
    ωe0 = 2π / ic.Pe
    tf  = YEARS * SECONDS_PER_YEAR
    sol = propagate_averaged(P, OsculatingState(ic.α0, ic.β0, ic.Id0 * ωe0, ic.Id0),
              (0.0, tf); shape = sh, cfg = cfg, reltol = 1e-8, abstol = 1e-10,
              maxiters = Int(1e7),
              callback = CallbackSet(
                  spin_termination_callbacks(ωe_floor = WE_FLOOR, ωe_ceiling = WE_CEILING),
                  budget_cb(BUDGET_S)))
    out = classify_spin_outcome(sol, tf; ωe_floor = WE_FLOOR, ωe_ceiling = WE_CEILING)

    te = sol.t[end]
    εmax = 0.0; ncross = 0; prev = NaN
    for t in range(0, te, length = N_SCAN)
        u = sol(t); d = u[4] - P.Ii
        isnan(prev) || (d * prev < 0 && (ncross += 1)); prev = d
        if abs(d) / P.Ii >= SEP_TOL
            e = epsilon_beta(OsculatingState(u[1], u[2], u[3], u[4]), P, sh, cfg)
            isfinite(e) && e > εmax && (εmax = e)
        end
    end
    ser = map(T_GRID_YR) do tyr
        t = tyr * SECONDS_PER_YEAR
        t > te + 1.0 ? (tyr, NaN, NaN, NaN) :
            (u = sol(min(t, te)); (tyr, u[3]/u[4], u[4]/P.Is, rad2deg(u[2])))
    end
    return out, ωe0, εmax, ncross, ser
end

# ══════════════════════════════════════════════════════════════════════════════
println("="^80)
println("RQ1 COMPARISON OBJECTS — ", YEARS, " yr, ", N_IC, " shared LHS ICs x ",
        length(configs), " configurations on ", Threads.nthreads(), " threads")
println("="^80)
for c in configs
    P = c.I
    @printf("  %-13s I_l=%8.1f  I_i=%8.1f  I_s=%8.1f   LAM %4.1f%% / SAM %4.1f%% of the I_d axis   %s\n",
            c.name, P.Il, P.Ii, P.Is,
            100*(P.Ii - P.Il)/(P.Is - P.Il), 100*(P.Is - P.Ii)/(P.Is - P.Il), c.note)
end

jobs = [(c = c, k = k, ic = ic) for c in configs for (k, ic) in enumerate(draws_for(c.I))]
results = Vector{Any}(undef, length(jobs))
done = Threads.Atomic{Int}(0)
t0 = time()

Threads.@threads for idx in eachindex(jobs)
    j = jobs[idx]
    w0 = time()
    results[idx] = try
        out, ωe0, εmax, ncross, ser = run_one(j.c, j.ic)
        (j = j, out = out, ωe0 = ωe0, εmax = εmax, ncross = ncross, ser = ser,
         wall = time() - w0, err = "")
    catch e
        (j = j, out = nothing, ωe0 = 2π/j.ic.Pe, εmax = NaN, ncross = 0,
         ser = NTuple{4,Float64}[], wall = time() - w0,
         err = first(replace(sprint(showerror, e), ',' => ';', '\n' => ' '), 80))
    end
    d = Threads.atomic_add!(done, 1) + 1
    d % 25 == 0 && (@printf("\r  %4d/%-4d  %5.1f%%  %.0fs", d, length(jobs),
                            100d/length(jobs), time() - t0); flush(stdout))
end
println()

open(OUT, "w") do io
    println(io, "config,ic_id,alpha0_deg,beta0_deg,Pe_min,Id0_over_Is,band0,band_frac0,",
                "omega_e0,fate,omega_e_final,omega_e_tailmean,ripple,t_end_yr,",
                "reached_horizon,beta_final_deg,Id_final_over_Is,regime_final,",
                "n_sep_crossings,eps_beta_max,in_domain,wall_s,error")
    for r in results
        j = r.j; P = j.c.I
        @printf(io, "%s,%d,%.3f,%.3f,%.3f,%.6f,%s,%.4f,%.8e,",
                j.c.name, j.k, rad2deg(j.ic.α0), rad2deg(j.ic.β0), j.ic.Pe/60,
                j.ic.Id0/P.Is, j.ic.band, j.ic.band_frac, 2π/j.ic.Pe)
        if r.out === nothing
            @printf(io, "error,,,,,0,,,,%.6e,,%.2f,%s\n", NaN, r.wall, r.err)
        else
            o = r.out
            reg = abs(o.Id_final - P.Ii)/P.Ii < SEP_TOL ? "near-separatrix" :
                  (classify_regime(o.Id_final, P) isa LAM ? "LAM" : "SAM")
            @printf(io, "%s,%.8e,%.8e,%.6e,%.4f,%d,%.4f,%.6f,%s,%d,%.6e,%d,%.2f,\n",
                    o.fate, o.ωe_final, o.ωe_mean_tail, o.ripple,
                    o.t_end/SECONDS_PER_YEAR, o.reached_horizon,
                    rad2deg(o.β_final), o.Id_final/P.Is, reg, r.ncross, r.εmax,
                    r.εmax <= EPS_BETA_MAX ? 1 : 0, r.wall)
        end
    end
end
println("wrote ", OUT)

open(OUTS, "w") do io
    println(io, "config,ic_id,band0,t_yr,omega_e,Id_over_Is,beta_deg")
    for r in results
        isempty(r.ser) && continue
        for (tyr, we, idis, bdeg) in r.ser
            @printf(io, "%s,%d,%s,%.6f,%.8e,%.6f,%.4f\n",
                    r.j.c.name, r.j.k, r.j.ic.band, tyr, we, idis, bdeg)
        end
    end
end
println("wrote ", OUTS)

# ── summary: the RQ1 comparison, printed so a failed plot still leaves it ────
function rel_disp(vals)
    v = sort([x for x in vals if isfinite(x) && x > 0])
    length(v) < 8 && return NaN
    q(p) = v[clamp(ceil(Int, p*length(v)), 1, length(v))]
    return (q(0.90) - q(0.10)) / v[cld(length(v), 2)]
end

println("\n== RQ1 across objects:  relative dispersion of ω̄_e ==")
@printf("%-14s %5s %7s %10s %10s %10s %12s\n",
        "config", "n", "err", "D(t=0)", "D(1 yr)", "D(100 yr)", "T_forget[yr]")
for c in configs
    rs = [r for r in results if r.j.c.name == c.name && r.out !== nothing]
    ne = count(r -> r.j.c.name == c.name && r.out === nothing, results)
    isempty(rs) && continue
    disp = [rel_disp([r.ser[i][2] for r in rs if !isempty(r.ser)]) for i in eachindex(T_GRID_YR)]
    tf = NaN
    for i in length(T_GRID_YR):-1:1
        (isnan(disp[i]) || disp[i] > 0.10) && break
        tf = T_GRID_YR[i]
    end
    at(x) = disp[argmin(abs.(T_GRID_YR .- x))]
    @printf("%-14s %5d %7d %10.3f %10.3f %10.3f %12s\n",
            c.name, length(rs), ne, at(0.0), at(1.0), at(100.0),
            isnan(tf) ? "never" : @sprintf("%.2f", tf))
end
@printf("\nelapsed %.1f min\n", (time() - t0)/60)
