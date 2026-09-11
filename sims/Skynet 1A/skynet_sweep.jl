
include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))
using .master
using Printf, LinearAlgebra

include(joinpath(@__DIR__, "skynet_shape"))

I  = skynet1a_inertia()
sh = skynet1a_shape()

# GRID — edit here.  Values reused from telstar_sweep.jl for comparability.
lam_frac(f) = I.Il/I.Is + f * (I.Ii - I.Il) / I.Is
sam_frac(f) = I.Ii/I.Is + f * (I.Is - I.Ii) / I.Is

β0s  = deg2rad.(15.0:30.0:165.0)                  # 6  coning angles
IdIs = [lam_frac.(0.1:0.1:0.9) ;                  # 9  across the LAM band
        sam_frac.(0.1:0.1:0.9)]                   # 9  across the SAM band
Pes  = [20.0, 60.0, 120.0] .* 60.0                # 3  initial spin periods
Js   = [0.1, 0.5, 1.0, 1.8, 5.0, 10.0]            # 6  slug inertia [kg m²]
μoJs = [1.0e-3]                                   # 1  μ/J, the validity ceiling

SIGMA = -1

YEARS        = 100.0
EPS_BETA_MAX = 1.0
N_DIAG       = 400                # samples per solution for ε_β (the expensive one)

N_ALPHA      = 20000

R_orb = 4.2575e7
ωe_floor, ωe_ceiling = 1e-8, 1.0

tf = YEARS * SECONDS_PER_YEAR
const SEP_TOL = 1e-4              # relative distance to I_d = I_i inside which
                                  # ε_β is returned as Inf (P_ψ diverges there)

#  one case 
function run_case(β0, r, Pe, J, μoJ, σ)
    Id0 = r * I.Is
    ωe0 = 2π / Pe
    st0 = OsculatingState(0.0, β0, Id0 * ωe0, Id0)
    cfg = PerturbationConfig(srp = true, dissipation = true, gravity_gradient = true,
                             μ = μoJ * J, J = J, orbit_R = R_orb,
                             srp_backend = :analytic, σ_branch = σ, resonant = false)
    t0 = time()
    try
        sol = propagate_averaged(I, st0, (0.0, tf); shape = sh, cfg = cfg,
                  reltol = 1e-8, abstol = 1e-10, maxiters = Int(1e7),
                  callback = spin_termination_callbacks(ωe_floor = ωe_floor,
                                                        ωe_ceiling = ωe_ceiling))
        te = sol.t[end]
        o  = classify_spin_outcome(sol, tf; ωe_floor = ωe_floor, ωe_ceiling = ωe_ceiling)

        ts  = range(0.0, te, length = N_DIAG)
        us  = [sol(t) for t in ts]
        εs  = [abs(u[4] - I.Ii)/I.Ii < SEP_TOL ? Inf :
               epsilon_beta(OsculatingState(u[1], u[2], u[3], u[4]), I, sh, cfg)
               for u in us]
        εf  = filter(isfinite, εs)
        εmax = isempty(εf) ? NaN : maximum(εf)

       
        αs  = [sol(t)[1] for t in range(0.0, te, length = N_ALPHA)]
        ua  = unwrap_angles(αs)
        rev = te > 0 ? (ua[end] - ua[1]) / (2π) / (te / SECONDS_PER_YEAR) : NaN

        
        βmin_pole = minimum(min(abs(u[2]), abs(π - u[2])) for u in us)

        return (ok = true, fate = o.fate, ωe0 = ωe0,
                ωe_final = o.ωe_final, ωe_tail = o.ωe_mean_tail, ripple = o.ripple,
                β_final = us[end][2], Id_final = us[end][4],
                te = te, reached = te >= tf - 1.0,
                εmax = εmax, in_domain = (isnan(εmax) || εmax <= EPS_BETA_MAX),
                ωe_pred = ωe0 * r, rev = rev, βpole = βmin_pole,
                wall = time() - t0, err = "")
    catch e
        return (ok = false, fate = :error, ωe0 = ωe0,
                ωe_final = NaN, ωe_tail = NaN, ripple = NaN,
                β_final = NaN, Id_final = NaN, te = NaN, reached = false,
                εmax = NaN, in_domain = false, ωe_pred = ωe0 * r, rev = NaN,
                βpole = NaN, wall = time() - t0,
                err = first(replace(sprint(showerror, e), "\n" => " "), 60))
    end
end


function unwrap_angles(a::AbstractVector)
    out = collect(float(a)); off = 0.0
    for k in 2:length(out)
        d = out[k] + off - out[k-1]
        while d >  π; off -= 2π; d -= 2π; end
        while d < -π; off += 2π; d += 2π; end
        out[k] += off
    end
    return out
end

# run
cases = [(β0=β, r=r, Pe=P, J=J, μoJ=m)
         for β in β0s, r in IdIs, P in Pes, J in Js, m in μoJs] |> vec

@printf("skynet_sweep: %d cases × %.0f yr, averaged (SRP transverse-only + diss + GG), :analytic, σ=%+d, %d threads\n",
        length(cases), YEARS, SIGMA, Threads.nthreads())
@printf("  grid: %d β₀ × %d I_d0 × %d P_e × %d J × %d μ/J\n",
        length(β0s), length(IdIs), length(Pes), length(Js), length(μoJs))
@printf("  bands: LAM %.4f–%.4f (%.1f%% of axis), SAM %.4f–%.4f (%.1f%%), separatrix %.4f\n",
        lam_frac(0.0), lam_frac(1.0), 100*(I.Ii-I.Il)/I.Is,
        sam_frac(0.0), sam_frac(1.0), 100*(I.Is-I.Ii)/I.Is, I.Ii/I.Is)
println("  EXPECT MANY :error ROWS — see [FLAG-SKYNET-IDIS] in the header.")

results = Vector{Any}(undef, length(cases))
t_start = time()
Threads.@threads for k in eachindex(cases)
    c = cases[k]
    results[k] = (c = c, out = run_case(c.β0, c.r, c.Pe, c.J, c.μoJ, SIGMA))
    if k % 200 == 0
        @printf("  ... %d/%d done  [%.1f min]\n", k, length(cases), (time()-t_start)/60)
    end
end

open(joinpath(@__DIR__, "skynet_sweep.csv"), "w") do io
    println(io, "beta0_deg,Id0_over_Is,Pe_min,J,mu_over_J,sigma,omega_e0,fate," *
                "omega_e_final,omega_e_tailmean,ripple,t_end_yr,reached_horizon," *
                "beta_final_deg,Id_final_over_Is,eps_beta_max,in_domain," *
                "omega_e_pred,rel_err_pred,alpha_rev_per_yr,beta_pole_min_deg,wall_s,error")
    for r in results
        c, o = r.c, r.out
        rel = isfinite(o.ωe_final) ? (o.ωe_final - o.ωe_pred)/o.ωe_pred : NaN
        @printf(io, "%.2f,%.4f,%.1f,%.4f,%.1e,%+d,%.8e,%s,%.8e,%.8e,%.6e,%.4f,%d,%.4f,%.6f,%.6e,%d,%.8e,%.6e,%.6f,%.4f,%.2f,%s\n",
                rad2deg(c.β0), c.r, c.Pe/60, c.J, c.μoJ, SIGMA, o.ωe0, o.fate,
                o.ωe_final, o.ωe_tail, o.ripple,
                isfinite(o.te) ? o.te/SECONDS_PER_YEAR : NaN, o.reached ? 1 : 0,
                isfinite(o.β_final) ? rad2deg(o.β_final) : NaN,
                isfinite(o.Id_final) ? o.Id_final/I.Is : NaN,
                o.εmax, o.in_domain ? 1 : 0, o.ωe_pred, rel, o.rev,
                isfinite(o.βpole) ? rad2deg(o.βpole) : NaN, o.wall, o.err)
    end
end
println("wrote skynet_sweep.csv")

# summary
println("\n" * "="^78)
println("SUMMARY")
println("="^78)
fates = Dict{Symbol,Int}()
for r in results; fates[r.out.fate] = get(fates, r.out.fate, 0) + 1; end
@printf("\nfates over %d cases:\n", length(results))
for (f, n) in sort(collect(fates), by = x -> -x[2])
    @printf("   %-12s %5d  (%.1f%%)\n", f, n, 100n/length(results))
end

# The measurement the header promises: where the bug bites, by J.
println("\n:error rate by J  — [FLAG-SKYNET-IDIS], a property of the solver, not the body:")
for J in Js
    g = [r for r in results if r.c.J == J]
    e = count(r -> r.out.fate === :error, g)
    @printf("   J = %-5g %4d/%4d  (%5.1f%%)\n", J, e, length(g), 100e/length(g))
end
println("\n:error rate by starting regime:")
for (nm, sel) in (("LAM start", r -> r.c.r < I.Ii/I.Is), ("SAM start", r -> r.c.r >= I.Ii/I.Is))
    g = [r for r in results if sel(r)]
    e = count(r -> r.out.fate === :error, g)
    @printf("   %-10s %4d/%4d  (%5.1f%%)\n", nm, e, length(g), 100e/length(g))
end

ok = [r for r in results if r.out.fate !== :error]
if !isempty(ok)
    # With M̄_z = 0 the closed form ω_e(∞) = ω_e0·(I_d0/I_s) should hold exactly.
    er = sort([abs(r.out.ωe_final - r.out.ωe_pred)/r.out.ωe_pred for r in ok
               if isfinite(r.out.ωe_final)])
    if !isempty(er)
        @printf("\nclosed-form check |ω_e_final/ω_e_pred − 1| over %d completed runs:\n", length(er))
        @printf("   median %.3e   p90 %.3e   max %.3e\n",
                er[max(1,end÷2)], er[max(1,Int(floor(0.9*end)))], er[end])
        println("   (M̄_z ≈ 0 ⇒ |H| conserved ⇒ ω_e(∞) = ω_e0·I_d0/I_s exactly)")
    end

    εs = sort([r.out.εmax for r in ok if isfinite(r.out.εmax)])
    if !isempty(εs)
        @printf("\naveraging validity ε_β,max over %d runs: median %.3e  p90 %.3e  max %.3e\n",
                length(εs), εs[max(1,end÷2)], εs[max(1,Int(floor(0.9*end)))], εs[end])
        @printf("   runs with ε_β > %.1f: %d/%d\n", EPS_BETA_MAX,
                count(>(EPS_BETA_MAX), εs), length(εs))
    end

    # α is only meaningful away from its coordinate pole (β = 0 or π), where
    # α̇ ∝ 1/sinβ is singular and `averaged_eom` itself warns.  Split, not mixed.
    POLE_DEG = 5.0
    clean = [r for r in ok if isfinite(r.out.rev) && isfinite(r.out.βpole) &&
                              rad2deg(r.out.βpole) > POLE_DEG]
    pole  = [r for r in ok if isfinite(r.out.rev) && isfinite(r.out.βpole) &&
                              rad2deg(r.out.βpole) <= POLE_DEG]
    rv = sort([r.out.rev for r in clean])
    if !isempty(rv)
        @printf("\nα circulation rate [rev/yr], %d runs that stay >%.0f° from the β pole:\n",
                length(rv), POLE_DEG)
        @printf("   min %+.3f   p25 %+.3f   median %+.3f   p75 %+.3f   max %+.3f\n",
                rv[1], rv[max(1,end÷4)], rv[max(1,end÷2)], rv[max(1,3end÷4)], rv[end])
        @printf("   |rate| > 1 rev/yr (circulating, not librating): %d/%d  (%.1f%%)\n",
                count(x -> abs(x) > 1, rv), length(rv), 100*count(x -> abs(x) > 1, rv)/length(rv))
        @printf("   |rate| < 0.1 rev/yr (effectively librating):    %d/%d  (%.1f%%)\n",
                count(x -> abs(x) < 0.1, rv), length(rv), 100*count(x -> abs(x) < 0.1, rv)/length(rv))
        println("   Telstar's α librates (≈0 rev/yr); a non-zero rate here is M̄_⊥ ≠ 0 at work.")
        println("   Sampled at N_ALPHA — see [FLAG-SKYNET-ALPHA-ALIAS]; at N_DIAG this")
        println("   whole distribution aliases into ±2 rev/yr and its median flips sign.")
    end
    if !isempty(pole)
        @printf("   EXCLUDED: %d runs came within %.0f° of β = 0 or π, where α is not\n",
                length(pole), POLE_DEG)
        println("   trustworthy (α̇ ∝ 1/sinβ; averaged_dynamics.jl:75 warns).  Not mixed in above.")
    end
end
@printf("\ntotal wall: %.1f min\n", (time()-t_start)/60)
