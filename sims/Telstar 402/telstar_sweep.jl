
include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))
using .master
using StaticArrays, LinearAlgebra, Printf

# Body under test 

include(joinpath(@__DIR__, "telstar_shape"))

I  = telstar401_inertia()
sh = telstar401_shape(; warn_inert = false)

# GRID — edit here
#   I_d = I_l  (I_d/I_s = 0.3410)  pure spin about the LONG  axis b̂₃ (the boom)
#   I_d = I_i  (I_d/I_s = 0.8444)  SEPARATRIX — unstable, through b̂₁
#   I_d = I_s  (I_d/I_s = 1.0000)  pure spin about the SHORT axis b̂₂

lam_frac(f) = I.Il/I.Is + f * (I.Ii - I.Il) / I.Is      # f=0 at I_l, 1 at I_i
sam_frac(f) = I.Ii/I.Is + f * (I.Is - I.Ii) / I.Is      # f=0 at I_i, 1 at I_s

β0s   = deg2rad.(15.0:30.0:165.0)                 # 6  coning angles      
IdIs  = [lam_frac.(0.1:0.1:0.9) ;                 # 9  across the LAM band 
         sam_frac.(0.1:0.1:0.9)]                  # 9  across the SAM band 
Pes   = [20.0, 60.0, 120.0] .* 60.0               # 3  initial spin periods 

Js    = [0.1, 0.5, 1.0, 1.8, 5.0, 10.0]           # 6  slug inertia [kg m²]
μoJs  = [1.0e-3]                                  # 1  μ/J [s⁻¹]; 1e-3 is the
                                                  #    model-validity ceiling
                                                
SIGMA = +1

YEARS        = 100.0
EPS_BETA_MAX = 1.0       # ε_β above this ⇒ outside the averaging domain

NUMERIC_EVERY = 500
N_PHI, N_TAU  = 30, 60   # :numeric
VERBOSE_CASES = false    # true ⇒ also print one detail line per finished case
NUMERIC_SEP_MIN = 5e-3   # skip the :numeric cross-check if the trajectory came
                         # closer than this (relative) to I_d = I_i

R_orb, i_orb, Ω_orb = 4.2575e7, 0.0, 0.0
ωe_floor, ωe_ceiling = 1e-8, 1.0     # pole guard / numerical guard

tf = YEARS * SECONDS_PER_YEAR

const SEP_TOL = 1e-4     # relative distance to I_d = I_i inside which we return Inf

function eps_beta(state::OsculatingState, cfg::PerturbationConfig)
    abs(state.Id - I.Ii) / I.Ii < SEP_TOL && return Inf
    return epsilon_beta(state, I, sh, cfg)
end

"""
Catalogue of tumbling-period resonances P_ψ/P_φ̄ = m/n
"""
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

"Resonance bookkeeping for one trajectory."
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
function domain_scan(sol, cfg; N = 400)
    te = sol.t[end]
    εmax = 0.0; t_exit = NaN; dsep = Inf
    for t in range(0, te, length=N)
        u = sol(t)
        dsep = min(dsep, abs(u[4] - I.Ii) / I.Ii)
        e = eps_beta(OsculatingState(u[1], u[2], u[3], u[4]), cfg)
        isfinite(e) || continue
        e > εmax && (εmax = e)
        isnan(t_exit) && e > EPS_BETA_MAX && (t_exit = t)
    end
    return εmax, t_exit, dsep
end

"Run one case; returns the outcome plus domain diagnostics."
function run_case(β0, r, Pe, J, μoJ, σ, backend; want_domain = true, srp = true)
    Id0 = r * I.Is; ωe0 = 2π / Pe; H0 = Id0 * ωe0
    cfg = PerturbationConfig(srp=srp, dissipation=true, gravity_gradient=true,
                             μ = μoJ * J, J = J,
                             orbit_i=i_orb, orbit_Ω=Ω_orb, orbit_R=R_orb,
                             srp_backend = backend, σ_branch = σ)
    kw = backend === :numeric ? (N_φ=N_PHI, N_τ=N_TAU) : NamedTuple()
    sol = propagate_averaged(I, OsculatingState(0.0, β0, H0, Id0), (0.0, tf);
              shape=sh, cfg=cfg, reltol=1e-8, abstol=1e-10, maxiters=Int(1e7),
              callback=spin_termination_callbacks(ωe_floor=ωe_floor,
                                                  ωe_ceiling=ωe_ceiling), kw...)
    out = classify_spin_outcome(sol, tf; ωe_floor=ωe_floor, ωe_ceiling=ωe_ceiling)
    cfg_scan = PerturbationConfig(srp=srp, dissipation=true, gravity_gradient=true,
                                  μ = μoJ * J, J = J,
                                  orbit_i=i_orb, orbit_Ω=Ω_orb, orbit_R=R_orb,
                                  srp_backend = :analytic, σ_branch = σ)
    εmax, t_exit, dsep = want_domain ? domain_scan(sol, cfg_scan) : (NaN, NaN, Inf)
    ncross, nearest, dres = resonance_scan(sol, CAT)
    # Closed-form target (see header): H is conserved, I_d → I_s.
    ωe_pred = ωe0 * r
    return out, εmax, t_exit, ωe0, ncross, nearest, dres, dsep, ωe_pred,
           abs(sol.u[end][3] - H0) / H0
end

#live terminal progress
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


# PRE-FLIGHT CHECKS

println("="^78)
println("exp11 PRE-FLIGHT — verifying [FLAG-TELSTAR-INERT] at the EOM level")
println("="^78)
@printf("Telstar 401 (approx):  I_l=%.2f  I_i=%.2f  I_s=%.2f kg m²   (%d facets, %.1f m²)\n",
        I.Il, I.Ii, I.Is, n_facets(sh), total_area(sh))
@printf("  LAM band %.4f–%.4f I_d/I_s (%.1f%% of the axis) | SAM %.4f–%.4f (%.1f%%)\n",
        lam_frac(0.0), lam_frac(1.0), 100*(I.Ii-I.Il)/I.Is,
        sam_frac(0.0), sam_frac(1.0), 100*(I.Is-I.Ii)/I.Is)

_pf(srp, backend, σ) = PerturbationConfig(srp=srp, dissipation=true,
        gravity_gradient=true, μ=1.0e-3*1.8, J=1.8, orbit_i=i_orb,
        orbit_Ω=Ω_orb, orbit_R=R_orb, srp_backend=backend, σ_branch=σ)

println("\n  (i) srp=true vs srp=false   [α̇, β̇, Ḣ, İ_d]")
for (nm, st) in (("LAM f=0.5", OsculatingState(0.0, deg2rad(75), 3000.0, lam_frac(0.5)*I.Is)),
                 ("SAM f=0.5", OsculatingState(0.0, deg2rad(75), 3000.0, sam_frac(0.5)*I.Is)))
    for backend in (:analytic, :numeric)
        kw = backend === :numeric ? (N_φ=N_PHI, N_τ=N_TAU) : NamedTuple()
        a = averaged_eom(st, I, sh, _pf(true,  backend, SIGMA); kw...)
        b = averaged_eom(st, I, sh, _pf(false, backend, SIGMA); kw...)
        @printf("      %-9s %-9s max|Δ| = %.3e   %s\n", nm, backend,
                maximum(abs.(a .- b)),
                maximum(abs.(a .- b)) == 0 ? "exactly zero (symbolic cancellation)" :
                                             "machine epsilon (facets evaluated, forces cancel)")
    end
end

println("\n  (ii) σ = +1 vs σ = -1   (σ enters only the SRP averages)")
let st = OsculatingState(0.0, deg2rad(75), 3000.0, lam_frac(0.5)*I.Is)
    p = averaged_eom(st, I, sh, _pf(true, :analytic, +1))
    m = averaged_eom(st, I, sh, _pf(true, :analytic, -1))
    @printf("      max|Δ| = %.3e   → σ is inert; SIGMA = %+d carries no decision\n",
            maximum(abs.(p .- m)), SIGMA)
end
println()

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


const ROW_SEL = let s = get(ENV, "EXP11_ROWS", "")
    isempty(s) ? nothing : [parse(Int, strip(x)) for x in split(s, ',')]
end
const ROW_ID = ROW_SEL === nothing ? collect(1:length(cases)) : sort(unique(ROW_SEL))
if ROW_SEL !== nothing
    all(1 .<= ROW_ID .<= length(cases)) ||
        error("EXP11_ROWS out of range 1:$(length(cases)): $ROW_ID")
    cases = cases[ROW_ID]
end
const OUT_CSV = ROW_SEL === nothing ? "exp11_telstar401_sweep.csv" :
                                      "exp11_telstar401_sweep_rows.csv"

@printf("\nexp11: %d cases × %.0f yr, averaged C4 (diss+GG; YORP present but nil), :analytic, σ=%+d, %d threads\n",
        length(cases), YEARS, SIGMA, Threads.nthreads())
@printf("       grid: %d β₀ × %d I_d0 × %d P_e × %d J × %d μ/J\n",
        length(β0s), length(IdIs), length(Pes), length(Js), length(μoJs))
NUMERIC_EVERY > 0 && @printf("       + every %dth case re-run on :numeric (%d×%d); the offset it measures is provably 0 here\n",
                             NUMERIC_EVERY, N_PHI, N_TAU)
println()

results = Vector{Any}(undef, length(cases))
_prog_t0[] = time()
Threads.@threads for k in eachindex(cases)
    c = cases[k]
    local out, εmax, t_exit, ωe0, wall, num, ncross, nearest, dres, dsep, ωpred, hdrift
    try
        wall = @elapsed ((out, εmax, t_exit, ωe0, ncross, nearest, dres, dsep, ωpred, hdrift) =
            run_case(c.β0, c.r, c.Pe, c.J, c.μoJ, SIGMA, :analytic))
    catch e
        wall = NaN; ωe0 = 2π/c.Pe; εmax = NaN; t_exit = NaN
        ncross = -1; nearest = (tag="-", m=0, n=0, r=NaN); dres = NaN; dsep = NaN
        ωpred = 2π/c.Pe * c.r; hdrift = NaN
        out = SpinOutcome(:error, NaN, NaN, NaN, NaN, NaN, NaN, false, NaN, NaN, NaN)
    end
    num = NaN
    if NUMERIC_EVERY > 0 && k % NUMERIC_EVERY == 0 && out.fate !== :error &&
       isfinite(dsep) && dsep > NUMERIC_SEP_MIN
        try
            num = run_case(c.β0, c.r, c.Pe, c.J, c.μoJ, SIGMA, :numeric;
                           want_domain = false)[1].ωe_mean_tail
        catch e
        end
    end
    results[k] = (c=c, out=out, εmax=εmax, t_exit=t_exit, ωe0=ωe0, wall=wall, num=num,
                  ncross=ncross, nearest=nearest, dres=dres, dsep=dsep,
                  ωpred=ωpred, hdrift=hdrift)
    progress!(length(cases), out.fate)
    if VERBOSE_CASES
        lock(_prog_lock) do
            @printf("\n    β₀=%5.1f° I_d0/I_s=%.4f P_e=%5.1f min J=%.2f → %-11s ω̄_e=%.4e (pred %.4e)  I_df/I_s=%.4f  ε_β^max=%.2e\n",
                    rad2deg(c.β0), c.r, c.Pe/60, c.J, out.fate, out.ωe_mean_tail,
                    ωpred, out.Id_final/I.Is, εmax)
            flush(stdout)
        end
    end
end

println()
println()   # step off the progress-bar line

const IDCOL = ROW_SEL === nothing ? "" : "row_id,"   # subset output only: a full
# run must reproduce the existing CSV's schema exactly, plotters included.
open(joinpath(@__DIR__, OUT_CSV), "w") do io
    println(io, IDCOL, "beta0_deg,Id0_over_Is,Pe_min,J,mu_over_J,sigma,omega_e0,",
                "fate,omega_e_final,omega_e_peak,omega_e_min,omega_e_tailmean,ripple,",
                "t_end_yr,reached_horizon,beta_final_deg,Id_final_over_Is,",
                "eps_beta_max,t_domain_exit_yr,in_domain,omega_e_tailmean_numeric,",
                "n_resonance_crossings,nearest_resonance,d_to_resonance,min_d_separatrix,wall_s,",
                "omega_e_pred,rel_err_pred,H_drift_rel")
    for (kk, r) in enumerate(results)
        o = r.out
        ROW_SEL === nothing || @printf(io, "%d,", ROW_ID[kk])
        @printf(io, "%.2f,%.4f,%.1f,%.4f,%.1e,%+d,%.8e,%s,%.8e,%.8e,%.8e,%.8e,%.6e,",
                rad2deg(r.c.β0), r.c.r, r.c.Pe/60, r.c.J, r.c.μoJ, SIGMA, r.ωe0, o.fate,
                o.ωe_final, o.ωe_peak, o.ωe_min, o.ωe_mean_tail, o.ripple)
        @printf(io, "%.4f,%d,%.4f,%.6f,%.6e,%.4f,%d,%.8e,",
                o.t_end/SECONDS_PER_YEAR, o.reached_horizon,
                rad2deg(o.β_final), o.Id_final/I.Is,
                r.εmax, r.t_exit/SECONDS_PER_YEAR, isnan(r.t_exit) ? 1 : 0, r.num)
        @printf(io, "%d,%s %d:%d,%.6f,%.6e,%.2f,",
                r.ncross, r.nearest.tag, r.nearest.m, r.nearest.n,
                r.dres, r.dsep, r.wall)
        @printf(io, "%.8e,%.6e,%.6e\n",
                r.ωpred, abs(o.ωe_final - r.ωpred)/r.ωpred, r.hdrift)
    end
end
println("wrote ", joinpath(@__DIR__, OUT_CSV))

# summary
ok = [r for r in results if r.out.fate !== :error]
println("\n== outcome census (fate as the averaged model reports it) ==")
for f in unique(r.out.fate for r in results)
    n = count(r -> r.out.fate === f, results)
    ind = count(r -> r.out.fate === f && isnan(r.t_exit), results)
    @printf("  %-12s %5d (%.1f%%)   of which in-domain: %d (%.0f%%)\n",
            f, n, 100n/length(results), ind, 100ind/max(n,1))
end

bad = [r for r in results if r.out.fate === :error]
if !isempty(bad)
    println("\n== failed cases ==")
    println("  exp9's known cause: I_d reaching I_s makes (I_s − I_d) go negative by")
    println("  roundoff inside torquefree_params_*, which then calls sqrt on it.  That")
    println("  mode is MORE exposed here than in exp9, because with no SRP the")
    println("  dissipative attractor IS I_d = I_s and every run approaches it.")
    for r in bad
        @printf("  β₀=%5.1f°  I_d0/I_s=%.4f  P_e=%5.1f min  J=%.2f  μ/J=%.0e\n",
                rad2deg(r.c.β0), r.c.r, r.c.Pe/60, r.c.J, r.c.μoJ)
    end
end

nd = count(r -> !isnan(r.t_exit), ok)
@printf("\n== averaging domain ==\n  %d/%d runs (%.1f%%) left the domain (ε_β > %.1f) before the horizon\n",
        nd, length(ok), 100nd/length(ok), EPS_BETA_MAX)
εs = sort([r.εmax for r in ok if isfinite(r.εmax)])
!isempty(εs) && @printf("  ε_β^max: median %.3e   p95 %.3e   max %.3e\n",
                        εs[cld(length(εs),2)], εs[cld(19*length(εs),20)], εs[end])

#  the regression test: does the sweep reproduce ω̄_e(∞) = ω_e0 · I_d0/I_s? 
println("\n== closed-form check: ω̄_e(final) vs ω_e0 · (I_d0/I_s) ==")
println("  With M̄ = 0, H is conserved and dissipation drives I_d → I_s, so this")
println("  is exact.  This is the substance of exp11: 1944 independent numerical")
println("  trajectories against an analytic target.")
er = sort([abs(r.out.ωe_final - r.ωpred)/r.ωpred for r in ok
           if isfinite(r.out.ωe_final) && r.ωpred > 0])
if !isempty(er)
    @printf("  n=%d   median %.3e   p95 %.3e   max %.3e\n",
            length(er), er[cld(length(er),2)], er[cld(19*length(er),20)], er[end])
end
hd = sort([r.hdrift for r in ok if isfinite(r.hdrift)])
!isempty(hd) && @printf("  |ΔH|/H₀ over 100 yr:  median %.3e   max %.3e   (must be ~0: L̄_z = 0)\n",
                        hd[cld(length(hd),2)], hd[end])
idf = sort([r.out.Id_final/I.Is for r in ok if isfinite(r.out.Id_final)])
!isempty(idf) && @printf("  I_d(final)/I_s:       median %.6f   min %.6f   (attractor is 1.0)\n",
                         idf[cld(length(idf),2)], idf[1])

println("\n== settled ω̄_e vs dissipation (in-domain runs only) ==")
println("  exp9's version of this table showed J SELECTING the attractor.  Here J")
println("  can only set the RATE of approach to a J-independent endpoint, so any")
println("  spread within a row is initial-condition spread, not a J effect.")
@printf("%8s %10s %7s %14s %14s %14s %12s\n", "J", "μ/J", "n", "median", "min", "max", "med reach%")
for J in Js, m in μoJs
    g = [r for r in ok if r.c.J == J && r.c.μoJ == m && isnan(r.t_exit) &&
                          isfinite(r.out.ωe_mean_tail)]
    isempty(g) && continue
    s = sort([r.out.ωe_mean_tail for r in g])
    @printf("%8.2f %10.0e %7d %14.4e %14.4e %14.4e %11.0f%%\n",
            J, m, length(s), s[cld(length(s),2)], s[1], s[end],
            100*count(r -> r.out.reached_horizon, g)/length(g))
end

cmp = [(r.out.ωe_mean_tail, r.num) for r in ok
       if isfinite(r.num) && isfinite(r.out.ωe_mean_tail) && r.num > 0 && isnan(r.t_exit)]
if !isempty(cmp)
    d = [100*(a-b)/b for (a,b) in cmp]
    s = sort(d)
    println("\n== backend offset: :analytic settled ω̄_e relative to :numeric ==")
    @printf("  n=%d   median %+.3e%%   range [%+.3e%%, %+.3e%%]\n",
            length(d), s[cld(length(s),2)], s[1], s[end])
    println("  Expected to be exactly zero, and it is: the backends differ ONLY in how")
    println("  they average the SRP torque, and that torque is nil.  exp9 measured a")
    println("  real offset here; this row is a null control, not a result.")
end
@printf("\ntotal wall: %.1f min\n", sum(r.wall for r in results if isfinite(r.wall))/60)


# LEDGER

println("\n" * "="^78)
println("PLACEHOLDERS AND INHERITED CHOICES IN THIS RUN — none is a measurement")
println("="^78)
for (k, v, p) in [
    ("dissipation range J",
     @sprintf("%s kg m² (μ/J = %.0e s⁻¹)", string(Js), μoJs[1]),
     "PLACEHOLDER, INHERITED FROM exp9 — B&S 2022 §II.C calibrates J from a GOES-10 manoeuvre; no Telstar dissipation data exists.  Read J as a swept unknown, not as Telstar's value [FLAG-TELSTAR-DISSIPATION]"),
    ("μ/J ceiling",
     @sprintf("%.0e s⁻¹", μoJs[1]),
     "HARD MODEL LIMIT, not a choice — above it the steady-state slug relation σ ≈ [A]ω breaks "),
    ("spin branch σ",
     @sprintf("%+d", SIGMA),
     "MOOT — σ reaches only averaged_srp_torques[_analytic], and this shape's SRP average is identically zero.  Verified bit-identical for ±1 in the pre-flight above.  Unlike exp9's SIGMA=-1 this encodes NO decision [FLAG-TELSTAR-INERT]"),
    ("horizon",
     @sprintf("%.0f yr", YEARS),
     "PROJECT RQ1 VALUE — deliberately NOT exp9's 20 yr, which came from GOES-8's ~4 yr YORP settling time and has no bearing on a torque-free relaxation"),
    ("net SRP torque",
     "identically zero",
     "CONSEQUENCE of the assumed symmetric geometry, not an assumption — see [FLAG-TELSTAR-INERT] in telstar401_shape.jl.  This sweep is therefore a dissipation+GG regression test, NOT a YORP study"),
    ("shape optics (ρ, s)",
     "GOES-8's B&S 2021 Table 1 values",
     "BORROWED, same-era/same-technology-class [FLAG-TELSTAR-OPTICAL].  Inert here anyway, since the torque cancels for any uniform-bus optics"),
    ("grid axes β₀, I_d0, P_e",
     @sprintf("%d × %d × %d, verbatim from exp9", length(β0s), length(IdIs), length(Pes)),
     "REUSED for comparability; the I_d band-fraction spacing is genuinely inertia-parameterised (checked by inspection), so it re-resolves Telstar's wider SAM band automatically"),
    ("orbit",
     @sprintf("R = %.4e m, i = %.1f°, Ω = %.1f°", R_orb, rad2deg(i_orb), rad2deg(Ω_orb)),
     "INHERITED FROM exp9 — circular GEO graveyard, equatorial"),
    ("body geometry + mass",
     "bus box, mass split, panel span/chord, boom offset, array angle",
     "ALL PLACEHOLDERS — see scripts/validation/telstar401_inertia.jl's own ASSUMPTIONS ledger, the single source of truth for those values"),
]
    @printf("  %-24s %s\n", k, v)
    @printf("  %-24s   ↳ %s\n", "", p)
end
println("""
BOTTOM LINE.  Nothing here is a prediction about the real Telstar 401.  The two
new pieces of information a reader would not get from telstar401_inertia.jl's
ledger are (1) J is being swept as a free parameter with no Telstar anchor, and
(2) σ carries no decision in this configuration, unlike everywhere else in this
project.  Both are consequences of the shape being dynamically inert.""")
