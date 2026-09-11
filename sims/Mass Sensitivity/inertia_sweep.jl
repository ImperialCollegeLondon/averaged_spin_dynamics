
include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))
using .master
using QuasiMonteCarlo
using DifferentialEquations: CallbackSet
using StaticArrays, LinearAlgebra, Printf, Random



# CONFIGURATION — everything held fixed, and everything swept


#HELD FIXED
const SHAPE      = goes8_shape_full(; θ_sa = deg2rad(17), optical = :bs)
const I_GOES8    = goes8_inertia()
const IS_FIXED   = I_GOES8.Is        # 3570.0 kg m² — the pinned scale
const J_SLUG     = 1.8               # kg m²  — B&S 2022 Fig. 9 value 
const MU_OVER_J  = 1.0e-3            # s⁻¹    — the model-validity ceiling
const R_ORB      = 4.2575e7          # m      — GEO graveyard radius
const I_ORB      = 0.0
const OMEGA_ORB  = 0.0
const SIGMA      = -1                
const BACKEND    = :analytic         # regular through the separatrix (M6)

#  SWEPT: ratio space
const N_RATIO            = 20        # pilot size, per the experiment plan
const ENFORCE_REALIZABLE = true      
const RATIO_MARGIN       = 0.02      # keep r₁, r₂ off 0 and 1: at r₂ → 1 the LAM
                                     # band collapses (I_l → I_i) and at r₁ → 1
                                     # the SAM band does, and the tumbling period
                                     # diverges on both.  Not physics — a guard.
const LHS_CANDIDATES     = 400       # maximin: best of this many LHS designs
const RATIO_SEED         = 20260824

# SWEPT: initial-condition ensemble, per ratio point
const N_ENSEMBLE_DEFAULT = 50        # overridable — `run_sweep(; n_ensemble=…)`
const YEARS              = 100.0


const BETA_RANGE_DEG = (15.0, 165.0)      #  swept 15:15:165
const PE_RANGE_MIN   = (30.0, 480.0)      #  swept 30, 60, 120, 240, 480

const EXP6_IDIS_LAM = [0.30, 0.40, 0.50, 0.60]
const EXP6_IDIS_SAM = [0.965, 0.975, 0.98, 0.99, 0.999]

const ALPHA_RANGE_DEG = (0.0, 360.0)
const ENSEMBLE_SEED   = 8401

# Diagnostics 
const EPS_BETA_MAX   = 1.0     # ε_β above this ⇒ outside the averaging domain
const EPS_BETA_WARN  = 0.1     # practical warning level (dynamics_averaged.jl)
const SEP_TOL        = 1e-4    # relative distance to I_d = I_i counted as "on it"
const N_SCAN         = 400     # samples per trajectory for the ε_β / regime scan
const WE_FLOOR       = 1e-8    # rad/s — despin threshold (pole guard)
const WE_CEILING     = 1.0     # rad/s — numerical guard ONLY, not break-up

# LIVE CALLBACKS vs POST-HOC SCAN 

const USE_LIVE_CALLBACKS = false

const OUTFILE = joinpath(@__DIR__, "exp10_inertia_ratio_sweep.csv")


# Sampling


"""
    maximin_lhs(n, d; ncand, rng) → (X::Matrix (d×n), min_dist)

Latin hypercube design chosen for the maximin criterion: the candidate with the
largest minimum pairwise Euclidean distance.
"""
function maximin_lhs(n::Int, d::Int; ncand::Int = LHS_CANDIDATES,
                     rng::AbstractRNG = MersenneTwister(RATIO_SEED))
    best = zeros(d, n); bestsep = -Inf
    for _ in 1:ncand
        X = QuasiMonteCarlo.sample(n, d, LatinHypercubeSample(rng))
        sep = Inf
        for i in 1:n-1, j in i+1:n
            sep = min(sep, sum(abs2, view(X, :, i) - view(X, :, j)))
        end
        if sep > bestsep
            bestsep = sep; best = Matrix(X)
        end
    end
    return best, sqrt(bestsep)
end

"Map a unit-square draw (u₁, u₂) to a physically realizable inertia triple."
function ratios_to_inertias(u1, u2; enforce = ENFORCE_REALIZABLE,
                            margin = RATIO_MARGIN, Is = IS_FIXED)
    r2 = margin + (1 - 2margin) * u2                    # I_l / I_i
    lo = enforce ? max(1 / (1 + r2), margin) : margin   # triangle inequality
    hi = 1 - margin
    r1 = lo + u1 * (hi - lo)                            # I_i / I_s
    Ii = r1 * Is
    Il = r2 * Ii
    return (r1 = r1, r2 = r2, P = PrincipalInertias(Il, Ii, Is))
end

"Would the LITERAL unit-square draw (r₁ = u₁) have been unphysical?"
naive_unphysical(u1, u2; margin = RATIO_MARGIN) = begin
    r2 = margin + (1 - 2margin) * u2
    r1 = margin + (1 - 2margin) * u1
    r1 * (1 + r2) < 1
end

# I_d0/I_s values expressed as fractions across GOES-8's own two bands.
_lam_f(r, I) = (r * I.Is - I.Il) / (I.Ii - I.Il)
_sam_f(r, I) = (r * I.Is - I.Ii) / (I.Is - I.Ii)
const LAM_FRAC_RANGE = (minimum(_lam_f.(EXP6_IDIS_LAM, Ref(I_GOES8))),
                        maximum(_lam_f.(EXP6_IDIS_LAM, Ref(I_GOES8))))
const SAM_FRAC_RANGE = (minimum(_sam_f.(EXP6_IDIS_SAM, Ref(I_GOES8))),
                        maximum(_sam_f.(EXP6_IDIS_SAM, Ref(I_GOES8))))


function ensemble_draws(P::PrincipalInertias, n::Int; seed::Int = ENSEMBLE_SEED)
    U, _ = maximin_lhs(n, 5; ncand = 60, rng = MersenneTwister(seed))
    map(1:n) do k
        uα, uβ, uP, uf, ub = U[1, k], U[2, k], U[3, k], U[4, k], U[5, k]
        α0 = deg2rad(ALPHA_RANGE_DEG[1] + uα * (ALPHA_RANGE_DEG[2] - ALPHA_RANGE_DEG[1]))
        β0 = deg2rad(BETA_RANGE_DEG[1]  + uβ * (BETA_RANGE_DEG[2]  - BETA_RANGE_DEG[1]))
        Pe = exp(log(PE_RANGE_MIN[1]) + uP * (log(PE_RANGE_MIN[2]) - log(PE_RANGE_MIN[1]))) * 60
        if ub < 0.5
            f   = LAM_FRAC_RANGE[1] + uf * (LAM_FRAC_RANGE[2] - LAM_FRAC_RANGE[1])
            Id0 = P.Il + f * (P.Ii - P.Il); band = "LAM"
        else
            f   = SAM_FRAC_RANGE[1] + uf * (SAM_FRAC_RANGE[2] - SAM_FRAC_RANGE[1])
            Id0 = P.Ii + f * (P.Is - P.Ii); band = "SAM"
        end
        (α0 = α0, β0 = β0, Pe = Pe, Id0 = Id0, band = band, band_frac = f)
    end
end

# One propagation


make_cfg(σ) = PerturbationConfig(srp = true, dissipation = true,
                                 gravity_gradient = true,
                                 μ = MU_OVER_J * J_SLUG, J = J_SLUG,
                                 orbit_i = I_ORB, orbit_Ω = OMEGA_ORB,
                                 orbit_R = R_ORB,
                                 srp_backend = BACKEND, σ_branch = σ)

"ε_β with the separatrix guarded — P_ψ diverges there, so Inf is the right answer."
function eps_beta_at(st::OsculatingState, P::PrincipalInertias, cfg)
    abs(st.Id - P.Ii) / P.Ii < SEP_TOL && return Inf
    return epsilon_beta(st, P, SHAPE, cfg)
end


function trajectory_scan(sol, P::PrincipalInertias, cfg)
    te = sol.t[end]
    εmax = 0.0; t_exit = NaN; dsep = Inf; ncross = 0
    prev = NaN
    for t in range(0, te, length = N_SCAN)
        u  = sol(t)
        st = OsculatingState(u[1], u[2], u[3], u[4])
        d  = u[4] - P.Ii
        dsep = min(dsep, abs(d) / P.Ii)
        isnan(prev) || (d * prev < 0 && (ncross += 1))
        prev = d
        e = eps_beta_at(st, P, cfg)
        isfinite(e) || continue
        e > εmax && (εmax = e)
        isnan(t_exit) && e > EPS_BETA_MAX && (t_exit = t)
    end
    return εmax, t_exit, dsep, ncross
end

function final_regime(Id, P::PrincipalInertias)
    abs(Id - P.Ii) / P.Ii < SEP_TOL && return "near-separatrix"
    return classify_regime(Id, P) isa LAM ? "LAM" : "SAM"
end

"Run one (ratio sample, ensemble draw) pair."
function run_one(P::PrincipalInertias, ic; years = YEARS)
    tf  = years * SECONDS_PER_YEAR
    cfg = make_cfg(SIGMA)
    ωe0 = 2π / ic.Pe
    H0  = ic.Id0 * ωe0
    st0 = OsculatingState(ic.α0, ic.β0, H0, ic.Id0)
    cb  = USE_LIVE_CALLBACKS ?
        CallbackSet(spin_termination_callbacks(ωe_floor = WE_FLOOR,
                                               ωe_ceiling = WE_CEILING),
                    averaging_validity_callback(P, SHAPE, cfg; ε_max = EPS_BETA_WARN),
                    separatrix_crossing_callback(P)) :
        CallbackSet(spin_termination_callbacks(ωe_floor = WE_FLOOR,
                                               ωe_ceiling = WE_CEILING))
    sol = propagate_averaged(P, st0, (0.0, tf); shape = SHAPE, cfg = cfg,
                             reltol = 1e-8, abstol = 1e-10, maxiters = Int(1e7),
                             callback = cb)
    out = classify_spin_outcome(sol, tf; ωe_floor = WE_FLOOR, ωe_ceiling = WE_CEILING)
    εmax, t_exit, dsep, ncross = trajectory_scan(sol, P, cfg)
    return out, ωe0, εmax, t_exit, dsep, ncross
end


# Progress (one line, rewritten in place — see exp9 for the rationale)

const _done  = Threads.Atomic{Int}(0)
const _lock  = ReentrantLock()
const _t0    = Ref(0.0)
const _fates = Dict{Symbol,Int}()

_hms(s) = (s = max(s, 0.0); h = floor(Int, s/3600); m = floor(Int, (s%3600)/60);
           sec = floor(Int, s%60);
           h > 0 ? (@sprintf("%dh%02dm", h, m)) : (@sprintf("%2dm%02ds", m, sec)))

function progress!(ntot, fate)
    d = Threads.atomic_add!(_done, 1) + 1
    _fates[fate] = get(_fates, fate, 0) + 1
    el  = time() - _t0[]
    eta = d < ntot ? el * (ntot - d) / d : 0.0
    W = 28; nf = round(Int, W * d / ntot)
    tally = join([@sprintf("%s:%d", k, v)
                  for (k, v) in sort(collect(_fates), by = x -> -x[2])], " ")
    @printf("\r  [%s%s] %5d/%-5d %5.1f%%  up %s  eta %s │ %-40s",
            "█"^nf, "░"^(W-nf), d, ntot, 100d/ntot, _hms(el), _hms(eta), tally)
    flush(stdout)
end

# Sweep driver

const CSV_HEADER =
    "ratio_id,r1_Ii_over_Is,r2_Il_over_Ii,Il_over_Is,Il,Ii,Is," *
    "draw_id,alpha0_deg,beta0_deg,Pe_min,Id0_over_Is,band0,band_frac0,omega_e0," *
    "sigma,fate,omega_e_final,omega_e_tailmean,omega_e_peak,omega_e_min,ripple," *
    "t_end_yr,reached_horizon,alpha_final_deg,beta_final_deg,Id_final_over_Is," *
    "regime_final,ever_near_separatrix,n_sep_crossings,min_d_separatrix," *
    "eps_beta_max,eps_beta_exceeded,t_eps_exit_yr,wall_s"

function csv_row(io, rs, ic, d, r)
    P = rs.P; o = r.out
    # closest approach over the WHOLE trajectory, not the final state - the
    # final state is reported separately by `regime_final`.
    nearsep = r.dsep < SEP_TOL
    @printf(io, "%d,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,",
            rs.id, rs.r1, rs.r2, rs.r1*rs.r2, P.Il, P.Ii, P.Is)
    @printf(io, "%d,%.3f,%.3f,%.3f,%.6f,%s,%.6f,%.8e,%+d,",
            d, rad2deg(ic.α0), rad2deg(ic.β0), ic.Pe/60, ic.Id0/P.Is,
            ic.band, ic.band_frac, r.ωe0, SIGMA)
    @printf(io, "%s,%.8e,%.8e,%.8e,%.8e,%.6e,%.4f,%d,",
            o.fate, o.ωe_final, o.ωe_mean_tail, o.ωe_peak, o.ωe_min,
            o.ripple, o.t_end/SECONDS_PER_YEAR, o.reached_horizon)
    @printf(io, "%.3f,%.3f,%.6f,%s,%d,%d,%.6e,",
            rad2deg(o.α_final), rad2deg(o.β_final), o.Id_final/P.Is,
            final_regime(o.Id_final, P), nearsep, r.ncross, r.dsep)
    @printf(io, "%.6e,%d,%.4f,%.2f\n",
            r.εmax, r.εmax > EPS_BETA_MAX,
            isnan(r.t_exit) ? NaN : r.t_exit/SECONDS_PER_YEAR, r.wall)
end

"""
    run_sweep(; n_ratio, n_ensemble, years, outfile) → Vector of result rows

`n_ensemble` is a keyword so scaling the pilot up is a one-word edit.  Rows are
written to the CSV as they finish (never only at the end), so a sweep killed
part-way still leaves usable data.
"""
function run_sweep(; n_ratio::Int = N_RATIO,
                     n_ensemble::Int = N_ENSEMBLE_DEFAULT,
                     years::Real = YEARS,
                     outfile::AbstractString = OUTFILE)

    U, minsep = maximin_lhs(n_ratio, 2)
    n_naive_bad = count(k -> naive_unphysical(U[1, k], U[2, k]), 1:n_ratio)

    ratio_samples = [ (id = k, ratios_to_inertias(U[1, k], U[2, k])...) for k in 1:n_ratio ]

    @printf("exp10: %d ratio samples × %d IC draws = %d runs × %g yr\n",
            n_ratio, n_ensemble, n_ratio*n_ensemble, years)
    @printf("       maximin LHS min pairwise separation %.4f (best of %d designs)\n",
            minsep, LHS_CANDIDATES)
    @printf("       triangle inequality: %d/%d unit-square draws would have been unphysical; %s\n", n_naive_bad, n_ratio,
            ENFORCE_REALIZABLE ? "r₁ remapped onto [1/(1+r₂), 1)" :
                                 "NOT enforced — UNPHYSICAL SAMPLES INCLUDED")
    @printf("       held fixed: GOES-8 facets θ_sa=17° :bs, J=%.1f, μ/J=%.0e, σ=%+d, backend=:%s\n",
            J_SLUG, MU_OVER_J, SIGMA, BACKEND)
    @printf("       I_s pinned at %.1f kg m² (GOES-8) — ratios only, not scale\n", IS_FIXED)
    @printf("       IC ranges (exp6): β₀ %.0f–%.0f°, P_e %.0f–%.0f min, LAM frac %.3f–%.3f, SAM frac %.3f–%.3f; α₀ %.0f–%.0f° (new here)\n",
            BETA_RANGE_DEG..., PE_RANGE_MIN..., LAM_FRAC_RANGE..., SAM_FRAC_RANGE...,
            ALPHA_RANGE_DEG...)
    @printf("       %d threads\n\n", Threads.nthreads())

    println("ratio samples (I_s = $(IS_FIXED) kg m² throughout):")
    @printf("  %3s %10s %10s %10s %12s %12s %10s\n",
            "id", "Ii/Is", "Il/Ii", "Il/Is", "Ii", "Il", "LAM width")
    for rs in ratio_samples
        @printf("  %3d %10.4f %10.4f %10.4f %12.1f %12.1f %10.4f\n",
                rs.id, rs.r1, rs.r2, rs.r1*rs.r2, rs.P.Ii, rs.P.Il,
                (rs.P.Ii - rs.P.Il)/rs.P.Is)
    end
    @printf("  (GOES-8 itself: Ii/Is = %.4f, Il/Ii = %.4f, Il/Is = %.4f)\n\n",
            I_GOES8.Ii/I_GOES8.Is, I_GOES8.Il/I_GOES8.Ii, I_GOES8.Il/I_GOES8.Is)

    jobs = [(rs = rs, d = d, ic = ic)
            for rs in ratio_samples
            for (d, ic) in enumerate(ensemble_draws(rs.P, n_ensemble;
                                                    seed = ENSEMBLE_SEED + rs.id))]
    ntot = length(jobs)
    results = Vector{Any}(undef, ntot)

    io = open(outfile, "w")
    println(io, CSV_HEADER); flush(io)
    _t0[] = time()

    Threads.@threads for k in eachindex(jobs)
        j = jobs[k]
        local r
        try
            local out, ωe0, εmax, t_exit, dsep, ncross
            wall = @elapsed ((out, ωe0, εmax, t_exit, dsep, ncross) =
                run_one(j.rs.P, j.ic; years = years))
            r = (out = out, ωe0 = ωe0, εmax = εmax, t_exit = t_exit,
                 dsep = dsep, ncross = ncross, wall = wall)
        catch e
            r = (out = SpinOutcome(:error, NaN, NaN, NaN, NaN, NaN, NaN,
                                   false, NaN, NaN, NaN),
                 ωe0 = 2π/j.ic.Pe, εmax = NaN, t_exit = NaN,
                 dsep = NaN, ncross = -1, wall = NaN)
        end
        results[k] = (j = j, r = r)
        lock(_lock) do
            csv_row(io, j.rs, j.ic, j.d, r); flush(io)   # incremental, per handoff
            progress!(ntot, r.out.fate)
        end
    end

    close(io)
    println("\n\nwrote ", outfile)
    return ratio_samples, results
end


# Summary


function summarise(ratio_samples, results)
    fates = [r.r.out.fate for r in results]
    println("\n== outcome census (all runs) ==")
    for f in sort(unique(fates), by = string)
        n = count(==(f), fates)
        @printf("  %-12s %5d  (%.1f%%)\n", f, n, 100n/length(fates))
    end

    fin = [r.r.out.ωe_mean_tail for r in results if isfinite(r.r.out.ωe_mean_tail)]
    if !isempty(fin)
        s = sort(fin)
        @printf("\nsettled ω̄_e (tail mean): min %.4e  median %.4e  max %.4e rad/s  (%.1f decades)\n",
                s[1], s[cld(length(s),2)], s[end], log10(s[end]/max(s[1], eps())))
    end

    nεx = count(r -> isfinite(r.r.εmax) && r.r.εmax > EPS_BETA_MAX, results)
    nεw = count(r -> isfinite(r.r.εmax) && r.r.εmax > EPS_BETA_WARN, results)
    @printf("\nε_β validity: %d/%d runs (%.1f%%) exceeded ε_β = %.1f at some point; %d/%d (%.1f%%) exceeded the %.1f warning level\n",
            nεx, length(results), 100nεx/length(results), EPS_BETA_MAX,
            nεw, length(results), 100nεw/length(results), EPS_BETA_WARN)
    println("  (exp6 found this across most of its parameter space; it is tracked, not assumed away)")

    ncr = count(r -> r.r.ncross > 0, results)
    nns = count(r -> isfinite(r.r.dsep) && r.r.dsep < SEP_TOL, results)
    @printf("separatrix: %d/%d runs (%.1f%%) crossed I_d = I_i at least once with σ held fixed; %d came within %.0e of it at some point\n",
            ncr, length(results), 100ncr/length(results), nns, SEP_TOL)

    println("\n== per ratio sample ==")
    @printf("%4s %8s %8s %8s │ %6s %12s %12s %8s │ %8s %8s\n",
            "id", "Ii/Is", "Il/Is", "n", "LAM%", "median ω̄_e", "IQR/median",
            "spread", "ε_β>1 %", "SAMend%")
    for rs in ratio_samples
        sub = [r for r in results if r.j.rs.id == rs.id]
        isempty(sub) && continue
        v = sort([r.r.out.ωe_mean_tail for r in sub if isfinite(r.r.out.ωe_mean_tail)])
        med = isempty(v) ? NaN : v[cld(length(v),2)]
        q1  = isempty(v) ? NaN : v[max(1, cld(length(v),4))]
        q3  = isempty(v) ? NaN : v[max(1, cld(3*length(v),4))]
        spread = isempty(v) ? NaN : log10(v[end]/max(v[1], eps()))
        nlam = count(r -> r.j.ic.band == "LAM", sub)
        nεx  = count(r -> isfinite(r.r.εmax) && r.r.εmax > EPS_BETA_MAX, sub)
        nsam = count(r -> final_regime(r.r.out.Id_final, rs.P) == "SAM", sub)
        @printf("%4d %8.4f %8.4f %8d │ %6.0f %12.4e %12.3f %8.2f │ %8.0f %8.0f\n",
                rs.id, rs.r1, rs.r1*rs.r2, length(sub), 100nlam/length(sub),
                med, isempty(v) ? NaN : (q3-q1)/max(abs(med), eps()), spread,
                100nεx/length(sub), 100nsam/length(sub))
    end
    println("  spread = decades between the smallest and largest settled ω̄_e in the sample")

    # Is the mass axis doing anything?  Compare the spread of the per-sample
    # MEDIANS (across ratio space) with the median WITHIN-sample spread.  If mass
    # distribution did not matter these would be the same size.
    meds = Float64[]
    withins = Float64[]
    for rs in ratio_samples
        v = sort([r.r.out.ωe_mean_tail for r in results
                  if r.j.rs.id == rs.id && isfinite(r.r.out.ωe_mean_tail)])
        isempty(v) && continue
        push!(meds, v[cld(length(v),2)])
        push!(withins, log10(v[end]/max(v[1], eps())))
    end
    if length(meds) > 1
        sm = sort(meds)
        @printf("\nacross ratio space : %.2f decades between ratio-sample medians\n",
                log10(sm[end]/max(sm[1], eps())))
        @printf("within a ratio point: %.2f decades (median over samples)\n",
                sort(withins)[cld(length(withins),2)])
        println("  → if the first is the larger, mass distribution matters more than the IC.")
    end
end


function main()
    ratio_samples, results = run_sweep()
    summarise(ratio_samples, results)

    println("\n" * "="^78)
    println("PLACEHOLDERS AND INHERITED CHOICES IN THIS SWEEP")
    println("="^78)
    @printf("  σ branch                 %+d, held fixed for whole propagations\n", SIGMA)
    println("                             ↳ INHERITED from exp9 / B&S Fig. 4/9 (SAM−).")
    println("                               No σ_policy field exists; the fixed-σ")
    println("                               assumption is open (docs/sigma_separatrix.md)")
    println("                               and was NOT decided here. [FLAG-EXP10-SIGMA]")
    @printf("  I_s                      %.1f kg m², pinned (GOES-8)\n", IS_FIXED)
    println("                             ↳ ratio sweep, NOT a scale sweep [FLAG-EXP10-SCALE]")
    println("  facets / optics          GOES-8 26-facet, θ_sa=17°, :bs — fixed for every sample")
    println("                             ↳ inertia varied under a fixed shape; far-from-GOES-8")
    println("                               samples are not constructible bodies [FLAG-EXP10-DECOUPLED]")
    @printf("  triangle inequality      %s\n",
            ENFORCE_REALIZABLE ? "ENFORCED by remapping r₁ ∈ [1/(1+r₂), 1)" : "NOT enforced")
    println("                             ↳ the unit-square scheme alone guarantees ordering,")
    println("                               not realizability [FLAG-EXP10-TRIANGLE]")
    @printf("  ratio margin             %.2f off 0 and 1 on both r₁ and r₂\n", RATIO_MARGIN)
    println("                             ↳ numerical guard (band collapse), not physics")
    @printf("  maximin LHS              best of %d QuasiMonteCarlo LHS designs\n", LHS_CANDIDATES)
    println("                             ↳ QuasiMonteCarlo v0.4.0 has NO maximin option;")
    println("                               applied on top of its plain LHS [FLAG-EXP10-MAXIMIN]")
    @printf("  ensemble size            %d draws per ratio point (keyword `n_ensemble`)\n",
            N_ENSEMBLE_DEFAULT)
    @printf("  α₀ range                 %.0f–%.0f°, uniform\n", ALPHA_RANGE_DEG...)
    println("                             ↳ exp6/7/8 all FIX α₀ = 0°, so this range is new")
    println("                               here, not inherited [FLAG-EXP10-ALPHA]")
    @printf("  P_e range                %.0f–%.0f min, LOG-uniform\n", PE_RANGE_MIN...)
    println("                             ↳ exp6's five values are geometrically spaced [FLAG-EXP10-PE]")
    println("  I_d0 sampling            exp6's I_d0/I_s list converted to band FRACTIONS")
    println("                             ↳ absolute I_d/I_s is meaningless at a different I_i")
    @printf("  J, μ/J                   %.1f kg m², %.0e s⁻¹ — exp6/7/8 values, held fixed\n",
            J_SLUG, MU_OVER_J)
    @printf("  live validity callbacks  %s
",
            USE_LIVE_CALLBACKS ? "ATTACHED" : "OFF — ε_β and separatrix crossings post-scanned")
    println("                             ↳ both ContinuousCallbacks hang on a grazing")
    println("                               trajectory (0.12 yr run did not finish in 9 min);")
    println("                               same diagnostics recovered by trajectory_scan,")
    println("                               same epsilon_beta source [FLAG-EXP10-CALLBACKS]")
    println("  ω_e ceiling              1.0 rad/s — numerical guard, NOT a break-up criterion")
end

# NB: `@__FILE__ && main()` would parse as `@__FILE__(&& main())` — the macro
# swallows the rest of the line — so this guard needs the `if` form.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
