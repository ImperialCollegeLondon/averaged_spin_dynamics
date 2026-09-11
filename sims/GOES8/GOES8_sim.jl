
include("C:/Users/noahl/Desktop/ICL/masters project/code/src/master.jl")  
using .master
using StaticArrays, LinearAlgebra, Printf


I  = goes8_inertia()
sh = goes8_shape_full(; θ_sa=deg2rad(17), optical=:bs)

# GRID — edit here

lam_frac(f) = I.Il/I.Is + f * (I.Ii - I.Il) / I.Is      # f=0 at I_l, 1 at I_i
sam_frac(f) = I.Ii/I.Is + f * (I.Is - I.Ii) / I.Is      # f=0 at I_i, 1 at I_s

β0s   = deg2rad.(15.0:30.0:165.0)                 # 6  coning angles
IdIs  = [lam_frac.(0.1:0.1:0.9) ;                 # 9  even across the LAM band
         sam_frac.(0.1:0.1:0.9)]                  # 9  even across the SAM band
Pes   = [20.0, 60.0, 120.0] .* 60.0               # 3  initial spin periods [s]
Js    = [0.1, 0.5, 1.0, 1.8, 5.0, 10.0]           # 6  slug inertia [kg m²]
μoJs  = [1.0e-3]                                  # 1  μ/J [s⁻¹]; 1e-3 is the
                                                  #    model-validity ceiling
SIGMA        = -1        # spin branch, held fixed (see header)
YEARS        = 20.0      # attractor settles by ~4 yr, so 100 yr buys little
EPS_BETA_MAX = 1.0       # ε_β above this ⇒ outside the averaging domain
NUMERIC_EVERY = 25       # re-run every Nth case on :numeric (0 disables)
N_PHI, N_TAU  = 30, 60   # :numeric quadrature — converged to ~0.03%
VERBOSE_CASES = false    # true ⇒ also print one detail line per finished case
NUMERIC_SEP_MIN = 5e-3   # skip the :numeric cross-check if the trajectory came
                         # closer than this (relative) to I_d = I_i — see below

R_orb, i_orb, Ω_orb = 4.2575e7, 0.0, 0.0
ωe_floor, ωe_ceiling = 1e-8, 1.0     # pole guard / numerical guard, NOT breakup




tf = YEARS * SECONDS_PER_YEAR


const SEP_TOL = 1e-4     # relative distance to I_d = I_i inside which we do not evaluate

function eps_beta(state::OsculatingState, cfg::PerturbationConfig)
    abs(state.Id - I.Ii) / I.Ii < SEP_TOL && return Inf
    return epsilon_beta(state, I, sh, cfg)
end

# charting where the resonances are
function resonance_catalog(I; maxord::Int = 5)
    cat = NamedTuple[]
    for reg in (LAM(), SAM()), m in 1:maxord, n in 1:maxord
        gcd(m, n) == 1 || continue
        Id = resonance_Id(m, n, I, reg)
        isnan(Id) && continue
        push!(cat, (tag = reg isa LAM ? "LAM" : "SAM", m = m, n = n, r = Id / I.Is))
    end
    sort!(cat, by = x -> x.r)
    return cat
end

"""
Resonance bookkeeping for one trajectory: how many times I_d crossed a
resonance, and which one it ended nearest.
"""
function resonance_scan(sol, CAT; N = 600)
    te = sol.t[end]
    rs = [sol(t)[4] / I.Is for t in range(0, te, length = N)]
    ncross = 0
    for res in CAT
        for j in 2:length(rs)
            (rs[j] - res.r) * (rs[j-1] - res.r) < 0 && (ncross += 1)
        end
    end
    rend = rs[end]
    k = argmin([abs(res.r - rend) for res in CAT])
    return ncross, CAT[k], rend - CAT[k].r
end

"Walk the solution and find where (if ever) ε_β first exceeds the limit."
function domain_scan(sol, cfg::PerturbationConfig; N = 400)
    te = sol.t[end]
    εmax = 0.0; t_exit = NaN; dsep = Inf
    for t in range(0, te, length=N)
        u = sol(t)
        dsep = min(dsep, abs(u[4] - I.Ii) / I.Ii)
        state = OsculatingState(u[1], u[2], u[3], u[4])
        e = eps_beta(state, cfg)
        isfinite(e) || continue
        e > εmax && (εmax = e)
        isnan(t_exit) && e > EPS_BETA_MAX && (t_exit = t)
    end
    return εmax, t_exit, dsep
end

"Run one case; returns the outcome plus domain diagnostics."
function run_case(β0, r, Pe, J, μoJ, σ, backend)
    Id0 = r * I.Is; ωe0 = 2π / Pe; H0 = Id0 * ωe0
    cfg = PerturbationConfig(srp=true, dissipation=true, gravity_gradient=true,
                             μ = μoJ * J, J = J,
                             orbit_i=i_orb, orbit_Ω=Ω_orb, orbit_R=R_orb,
                             srp_backend = backend, σ_branch = σ, resonant=false)
    kw = backend === :numeric ? (N_φ=N_PHI, N_τ=N_TAU) : NamedTuple()
    sol = propagate_averaged(I, OsculatingState(0.0, β0, H0, Id0), (0.0, tf);
              shape=sh, cfg=cfg, reltol=1e-8, abstol=1e-10, maxiters=Int(1e7),
              callback=spin_termination_callbacks(ωe_floor=ωe_floor,
                                                  ωe_ceiling=ωe_ceiling), kw...)
    out = classify_spin_outcome(sol, tf; ωe_floor=ωe_floor, ωe_ceiling=ωe_ceiling)
    εmax, t_exit, dsep = domain_scan(sol, cfg)
    ncross, nearest, dres = resonance_scan(sol, CAT)
    return out, εmax, t_exit, ωe0, ncross, nearest, dres, dsep
end
# live terminal progress 
# One line, rewritten in place with \r, updated as each case finishes.  Per-YEAR
# output is not useful here: with ~2000 cases on N threads you would get tens of
# thousands of interleaved lines from different runs.  Per-case is the right
# granularity — and if the bar stops advancing, something is stuck, which is
# exactly the signal you want.
const _prog_done  = Threads.Atomic{Int}(0)
const _prog_lock  = ReentrantLock()
const _prog_t0    = Ref(0.0)
const _prog_fates = Dict{Symbol,Int}()

function _hms(s)
    s = max(s, 0.0)
    h = floor(Int, s/3600); m = floor(Int, (s%3600)/60); sec = floor(Int, s%60)
    h > 0 ? (@sprintf("%dh%02dm", h, m)) : (@sprintf("%2dm%02ds", m, sec))
end

function progress!(ntot, fate)
    d = Threads.atomic_add!(_prog_done, 1) + 1
    lock(_prog_lock) do
        _prog_fates[fate] = get(_prog_fates, fate, 0) + 1
        el  = time() - _prog_t0[]
        eta = d < ntot ? el * (ntot - d) / d : 0.0
        W   = 30; nf = round(Int, W * d / ntot)
        tally = join([@sprintf("%s:%d", k, v)
                      for (k, v) in sort(collect(_prog_fates), by = x -> -x[2])], " ")
        @printf("\r  [%s%s] %5d/%-5d %5.1f%%  up %s  eta %s │ %-46s",
                "█"^nf, "░"^(W-nf), d, ntot, 100d/ntot, _hms(el), _hms(eta), tally)
        flush(stdout)
    end
end

const CAT = resonance_catalog(I; maxord = 5)
@printf("resonance catalogue: %d resonances (order ≤ 5) — %d LAM, %d SAM\n",
        length(CAT), count(c -> c.tag == "LAM", CAT), count(c -> c.tag == "SAM", CAT))
for g in IdIs
    k = argmin([abs(c.r - g) for c in CAT])
    d = CAT[k].r - g
    # NB: @printf needs a LITERAL format string — no `*` concatenation here.
    abs(d) < 0.01 && @printf("  ! grid point I_d0/I_s = %.3f is %+.4f from %s %d:%d — essentially ON a resonance\n",
                             g, d, CAT[k].tag, CAT[k].m, CAT[k].n)
end

cases = [(β0=β, r=r, Pe=P, J=J, μoJ=m)
         for β in β0s, r in IdIs, P in Pes, J in Js, m in μoJs] |> vec

@printf("exp9: %d cases × %.0f yr, averaged C4 (YORP+diss+GG), :analytic, σ=%+d, %d threads\n",
        length(cases), YEARS, SIGMA, Threads.nthreads())
@printf("      grid: %d β₀ × %d I_d0 × %d P_e × %d J × %d μ/J\n",
        length(β0s), length(IdIs), length(Pes), length(Js), length(μoJs))
NUMERIC_EVERY > 0 && @printf("      + every %dth case re-run on :numeric (%d×%d) to quantify the backend offset\n",
                             NUMERIC_EVERY, N_PHI, N_TAU)
println()

results = Vector{Any}(undef, length(cases))
_prog_t0[] = time()
Threads.@threads for k in eachindex(cases)
    c = cases[k]
    local out, εmax, t_exit, ωe0, wall, num, ncross, nearest, dres, dsep
    try
        wall = @elapsed ((out, εmax, t_exit, ωe0, ncross, nearest, dres, dsep) =
            run_case(c.β0, c.r, c.Pe, c.J, c.μoJ, SIGMA, :analytic))
    catch e
        @warn "case failed" c=c exception=(e, catch_backtrace())
        wall = NaN; ωe0 = 2π/c.Pe; εmax = NaN; t_exit = NaN
        ncross = -1; nearest = (tag="-", m=0, n=0, r=NaN); dres = NaN; dsep = NaN
        out = SpinOutcome(:error, NaN, NaN, NaN, NaN, NaN, NaN, false, NaN, NaN, NaN)
    end
    num = NaN
    # The :numeric backend evaluates Jacobi elliptic functions directly, so it
    # becomes pathologically slow as k → 1, i.e. whenever the trajectory
    # approaches the separatrix.  The :analytic backend carries the k → 1 limits
    # and stays regular there (M6).  So only cross-check runs that kept clear.
    if NUMERIC_EVERY > 0 && k % NUMERIC_EVERY == 0 && out.fate !== :error &&
       isfinite(dsep) && dsep > NUMERIC_SEP_MIN
        try
            num = run_case(c.β0, c.r, c.Pe, c.J, c.μoJ, SIGMA, :numeric)[1].ωe_mean_tail
        catch e
        end
    end
    results[k] = (c=c, out=out, εmax=εmax, t_exit=t_exit, ωe0=ωe0, wall=wall, num=num,
                  ncross=ncross, nearest=nearest, dres=dres, dsep=dsep)
     progress!(length(cases), out.fate)
    if VERBOSE_CASES
        lock(_prog_lock) do
            @printf("\n    β₀=%5.1f° I_d0/I_s=%.4f P_e=%5.1f min J=%.2f → %-11s ω̄_e=%.4e  β_f=%6.1f°  I_df/I_s=%.4f  ε_β^max=%.2e  %s\n",
                    rad2deg(c.β0), c.r, c.Pe/60, c.J, out.fate, out.ωe_mean_tail,
                    rad2deg(out.β_final), out.Id_final/I.Is, εmax,
                    isnan(t_exit) ? "in-domain" :
                        @sprintf("LEFT DOMAIN at %.2f yr", t_exit/SECONDS_PER_YEAR))
            flush(stdout)
        end
    end
end

println("
")   # step off the progress-bar line
open(joinpath(@__DIR__, "exp9_goes8_sweep.csv"), "w") do io
    println(io, "beta0_deg,Id0_over_Is,Pe_min,J,mu_over_J,sigma,omega_e0,",
                "fate,omega_e_final,omega_e_peak,omega_e_min,omega_e_tailmean,ripple,",
                "t_end_yr,reached_horizon,beta_final_deg,Id_final_over_Is,",
                "eps_beta_max,t_domain_exit_yr,in_domain,omega_e_tailmean_numeric,",
                "n_resonance_crossings,nearest_resonance,d_to_resonance,min_d_separatrix,wall_s")
    for r in results
        o = r.out
        @printf(io, "%.2f,%.4f,%.1f,%.4f,%.1e,%+d,%.8e,%s,%.8e,%.8e,%.8e,%.8e,%.6e,",
                rad2deg(r.c.β0), r.c.r, r.c.Pe/60, r.c.J, r.c.μoJ, SIGMA, r.ωe0, o.fate,
                o.ωe_final, o.ωe_peak, o.ωe_min, o.ωe_mean_tail, o.ripple)
        @printf(io, "%.4f,%d,%.4f,%.6f,%.6e,%.4f,%d,%.8e,",
                o.t_end/SECONDS_PER_YEAR, o.reached_horizon,
                rad2deg(o.β_final), o.Id_final/I.Is,
                r.εmax, r.t_exit/SECONDS_PER_YEAR, isnan(r.t_exit) ? 1 : 0, r.num)
        @printf(io, "%d,%s %d:%d,%.6f,%.6e,%.2f\n",
                r.ncross, r.nearest.tag, r.nearest.m, r.nearest.n,
                r.dres, r.dsep, r.wall)
    end
end
println("\nwrote scripts/experiments/exp9_goes8_sweep.csv")

#  summary 
ok = [r for r in results if r.out.fate !== :error]
println("\n== outcome census (fate as the averaged model reports it) ==")
for f in unique(r.out.fate for r in results)
    n = count(r -> r.out.fate === f, results)
    ind = count(r -> r.out.fate === f && isnan(r.t_exit), results)
    @printf("  %-12s %5d (%.1f%%)   of which in-domain: %d (%.0f%%)\n",
            f, n, 100n/length(results), ind, 100ind/max(n,1))
end

nd = count(r -> !isnan(r.t_exit), ok)
@printf("\n== averaging domain ==\n  %d/%d runs (%.1f%%) left the domain (ε_β > %.1f) before the horizon\n",
        nd, length(ok), 100nd/length(ok), EPS_BETA_MAX)
@printf("  their fates: %s\n", join(unique(string(r.out.fate) for r in ok if !isnan(r.t_exit)), ", "))

println("\n== settled ω̄_e vs dissipation (in-domain runs only) ==")
@printf("%8s %10s %7s %14s %14s %14s\n", "J", "μ/J", "n", "median", "min", "max")
for J in Js, m in μoJs
    g = [r.out.ωe_mean_tail for r in ok
         if r.c.J == J && r.c.μoJ == m && isnan(r.t_exit) && isfinite(r.out.ωe_mean_tail)]
    isempty(g) && continue
    s = sort(g)
    @printf("%8.2f %10.0e %7d %14.4e %14.4e %14.4e\n",
            J, m, length(g), s[cld(length(s),2)], s[1], s[end])
end

# backend offset — the characterised error, not a correction
cmp = [(r.out.ωe_mean_tail, r.num) for r in ok
       if isfinite(r.num) && isfinite(r.out.ωe_mean_tail) && r.num > 0 && isnan(r.t_exit)]
if !isempty(cmp)
    d = [100*(a-b)/b for (a,b) in cmp]
    s = sort(d)
    println("\n== backend offset: :analytic settled ω̄_e relative to :numeric ==")
    @printf("  n=%d   median %+.1f%%   range [%+.1f%%, %+.1f%%]\n",
            length(d), s[cld(length(s),2)], s[1], s[end])
    println("  (the analytic Fourier average is the B&S production engine; this is the")
    println("   price of the closed form, quantified — not a defect to be corrected)")
end
@printf("\ntotal wall: %.1f min\n", sum(r.wall for r in results if isfinite(r.wall))/60)
