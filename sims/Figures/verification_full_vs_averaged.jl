
include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))
using .master
using Printf, LinearAlgebra, StaticArrays

const a_b_H        = master.a_b_H
const H_alpha_beta = master.H_alpha_beta

I  = goes8_inertia()
sh = goes8_shape_full(; θ_sa = deg2rad(17), optical = :bs)
const IDIAG = inertia_body(I)          # (Ii, Is, Il) — the body-frame ordering

# configuration
# The representative IC is traj_goes8.jl's, so this figure verifies the model at
# exactly the state the trajectory figures are drawn from.
β0      = deg2rad(75.0)
α0      = 0.0
Id0_rel = 0.3433
Pe0     = 20.0 * 60.0
SIGMA   = -1                    # traj_goes8.jl's choice, carried over

YEARS   = 2.0                   
SAVE_DT = 300.0                 # [s]

# O frame 
#  Columns are X̂_O, Ŷ_O, Ẑ_O in inertial components, so
# `ON(t) * v_O = v_N` and `ON(t)' * v_N = v_O`.
function ON(t::Real; t0::Real = 0.0)
    λ = 2π * (t - t0) / SECONDS_PER_YEAR
    x̂O = SVector(0.0, 0.0, 1.0)                # ecliptic pole = the rotation axis
    ẑO = SVector(cos(λ), sin(λ), 0.0)          # Sun direction
    ŷO = SVector(sin(λ), -cos(λ), 0.0)         # = ẑ_O × x̂_O
    return hcat(x̂O, ŷO, ẑO)
end

#  forward map: full state → osculating elements
"""
    osculating_from_full(q, ω, t) → (α, β, H, Id)

H and I_d are frame-independent; β needs only Ẑ_O (the Sun direction).
"""
function osculating_from_full(q, ω, t::Real)
    Hb  = SVector(IDIAG[1]*ω[1], IDIAG[2]*ω[2], IDIAG[3]*ω[3])   # body frame
    Hn  = norm(Hb)
    T   = kinetic_energy_full(I, ω)
    Id  = Id_from_energy_momentum(T, Hn)
    Ĥ_N = quat_to_dcm(q)' * (Hb / Hn)          # DCM maps N→B, so transpose B→N
    Ĥ_O = ON(t)' * Ĥ_N
    α, β = a_b_H(SVector{3,Float64}(Ĥ_O))
    return (α = α, β = β, H = Hn, Id = Id)
end

# inverse map: osculating elements → full state
"""
    full_from_osculating(α, β, H, Id; σ) → (q, ω)

Body rates come from the torque-free solution at τ = 0, which is EXACTLY
consistent with (ω_e, I_d) — verified to 0.0 relative error below.  The attitude
is then the minimal rotation carrying Ĥ_N onto Ĥ_B.
"""
function full_from_osculating(α::Real, β::Real, H::Real, Id::Real;
                              σ::Integer = 1, t::Real = 0.0)
    ωe  = omega_e(H, Id)
    reg = classify_regime(Id, I)
    ω   = torque_free_body_rates(0.0, ωe, Id, I, reg; σ = σ)
    Hb  = SVector(IDIAG[1]*ω[1], IDIAG[2]*ω[2], IDIAG[3]*ω[3])
    Ĥ_B = Hb / norm(Hb)
    Ĥ_N = SVector{3,Float64}(ON(t) * H_alpha_beta(α, β))

    # Minimal rotation taking Ĥ_N to Ĥ_B, as a quaternion in `quat_to_dcm`'s
    # convention (which maps N → B).  Degenerate cases handled explicitly.
    c = clamp(dot(Ĥ_N, Ĥ_B), -1.0, 1.0)
    v = cross(Ĥ_N, Ĥ_B)
    nv = norm(v)
    q = if nv < 1e-14
        c > 0 ? SVector(1.0, 0.0, 0.0, 0.0) :          # already aligned
                (a = abs(Ĥ_N[1]) < 0.9 ? SVector(1.0,0.0,0.0) : SVector(0.0,1.0,0.0);
                 p = normalize(cross(Ĥ_N, a));
                 SVector(0.0, p[1], p[2], p[3]))       # 180°, any perpendicular axis
    else
        θ = atan(nv, c); n̂ = v / nv
        SVector(cos(θ/2), -sin(θ/2)*n̂[1], -sin(θ/2)*n̂[2], -sin(θ/2)*n̂[3])
    end
    q = quat_normalize(q)
    # The sign convention above is ASSERTED, not assumed:
    dev = norm(quat_to_dcm(q) * Ĥ_N - Ĥ_B)
    dev < 1e-12 || error("full_from_osculating: quaternion convention wrong — " *
                         "|[BN]Ĥ_N − Ĥ_B| = $dev.  Flip the vector-part sign.")
    return q, ω
end

# round-trip check: the inverse map must invert the forward map
println("="^78)
println("FIGURE 1 — full Euler vs tumbling-averaged, GOES-8, SRP ONLY")
println("="^78)

Id0 = Id0_rel * I.Is
ωe0 = 2π / Pe0
H0  = Id0 * ωe0
q0, ω0 = full_from_osculating(α0, β0, H0, Id0; σ = SIGMA, t = 0.0)
rt = osculating_from_full(q0, ω0, 0.0)

println("\nround-trip check  (osculating → full → osculating):")
@printf("   %-10s %18s %18s %12s\n", "", "target", "recovered", "error")

for (nm, a, b, rel) in (("α [deg]", rad2deg(α0), rad2deg(rt.α), false),
                        ("β [deg]", rad2deg(β0), rad2deg(rt.β), false),
                        ("H",       H0,          rt.H,           true),
                        ("I_d",     Id0,         rt.Id,          true))
    e = rel ? abs(b - a)/max(abs(a), eps()) : abs(b - a)
    @printf("   %-10s %18.10f %18.10f %12.2e %-6s %s\n", nm, a, b, e,
            rel ? "(rel)" : "(abs)", e < 1e-9 ? "" : "  <== CHECK")
end
@printf("   ω_body = (%.6e, %.6e, %.6e)   regime = %s\n",
        ω0[1], ω0[2], ω0[3], typeof(classify_regime(Id0, I)))

# TORQUE-FREE PRE-FLIGHT: the test validates the O frame
println("\n" * "-"^78)
println("TORQUE-FREE PRE-FLIGHT — validates the O frame against a known answer")
println("-"^78)
let tfree = 0.5 * SECONDS_PER_YEAR, npts = 9
    tsf = collect(range(0.0, tfree, length = npts))
    sf  = propagate_full(I, q0, ω0, (0.0, tfree); torque = zero_torque,
                         reltol = 1e-12, abstol = 1e-12, saveat = tsf,
                         save_everystep = false, maxiters = Int(1e9))
    cf0 = PerturbationConfig(srp = false, dissipation = false,
                             gravity_gradient = false, srp_backend = :analytic,
                             σ_branch = SIGMA, resonant = false)
    sa  = propagate_averaged(I, OsculatingState(α0, β0, H0, Id0), (0.0, tfree);
                             shape = sh, cfg = cf0, reltol = 1e-12, abstol = 1e-14,
                             maxiters = Int(1e8))
    u00 = sf(0.0)
    Ĥfix = let Hb = SVector(IDIAG[1]*u00[5], IDIAG[2]*u00[6], IDIAG[3]*u00[7])
        quat_to_dcm(quat_normalize(SVector(u00[1],u00[2],u00[3],u00[4])))' * (Hb/norm(Hb))
    end
    dβ_fe, dβ_fa, dα_fa, dH, dId = 0.0, 0.0, 0.0, 0.0, 0.0
    for t in tsf
        u = sf(t); ua = sa(t)
        f = osculating_from_full(SVector(u[1],u[2],u[3],u[4]), SVector(u[5],u[6],u[7]), t)
        αx, βx = a_b_H(SVector{3,Float64}(ON(t)' * Ĥfix))
        dβ_fe = max(dβ_fe, abs(f.β - βx))
        dβ_fa = max(dβ_fa, abs(f.β - ua[2]))
        dα_fa = max(dα_fa, abs(rem2pi(f.α - ua[1], RoundNearest)))
        dH    = max(dH,  abs(f.H - H0)/H0)
        dId   = max(dId, abs(f.Id - Id0)/Id0)
    end
    @printf("   max |β_full − β_closedform|  = %.3e rad   (validates the extraction + frame)\n", dβ_fe)
    @printf("   max |β_full − β_averaged|    = %.3e rad   (validates the averaged frame terms)\n", dβ_fa)
    @printf("   max |α_full − α_averaged|    = %.3e rad\n", dα_fa)
    @printf("   full-model conservation:  max|ΔH|/H = %.3e   max|ΔI_d|/I_d = %.3e\n", dH, dId)
    ok = dβ_fe < 1e-6 && dβ_fa < 1e-6 && dα_fa < 1e-6 && dH < 1e-8 && dId < 1e-8
    println(ok ? "   → PASS: all three routes agree; the O frame is correct." :
                 "   → FAIL: frame or extraction is wrong — the SRP result below is meaningless.")
    ok || error("torque-free pre-flight failed; refusing to report Figure 1.")
end

# propagate both models
tf  = YEARS * SECONDS_PER_YEAR
ts  = collect(0.0:SAVE_DT:tf)

@printf("\npropagating %.1f yr, %d output samples at %.0f s cadence...\n",
        YEARS, length(ts), SAVE_DT)

t0 = time()

sol_f = propagate_full(I, q0, ω0, (0.0, tf); torque = srp_torque_fn(sh),
                       reltol = 1e-12, abstol = 1e-12, saveat = ts,
                       save_everystep = false, maxiters = Int(1e9))
sol_f.t[end] >= tf - SAVE_DT || error(
    "full Euler solve stopped at $(sol_f.t[end]/SECONDS_PER_YEAR) yr of $YEARS — " *
    "raise maxiters.  Refusing to report an RMS over a truncated window.")
@printf("   full Euler:  %6.1f s wall,  %d accepted steps\n",
        time()-t0, sol_f.stats.naccept)

# BOTH backends are propagated

function run_avg(backend::Symbol)
    cfg = PerturbationConfig(srp = true, dissipation = false, gravity_gradient = false,
                             srp_backend = backend, σ_branch = SIGMA, resonant = false)
    t0 = time()
    s = propagate_averaged(I, OsculatingState(α0, β0, H0, Id0), (0.0, tf);
                           shape = sh, cfg = cfg, reltol = 1e-10, abstol = 1e-12,
                           maxiters = Int(1e7), N_φ = 90, N_τ = 180)
    @printf("   averaged %-9s %6.1f s wall,  ran %.4f/%.1f yr\n",
            String(backend) * ":", time()-t0, s.t[end]/SECONDS_PER_YEAR, YEARS)
    return s
end
sol_num = run_avg(:numeric)
sol_ana = run_avg(:analytic)
sol_a   = sol_num          # `sol_a` is the REFERENCE averaged run used below.
                           # :numeric, not the repo's :analytic default, because
                           # Figure 1 itself shows :numeric is the one that
                           # tracks the full Euler truth model.

# extract and compare 

tt = [t for t in ts if t <= sol_f.t[end] && t <= sol_a.t[end]]
full = [osculating_from_full(SVector(u[1],u[2],u[3],u[4]),
                             SVector(u[5],u[6],u[7]), t)
        for (t, u) in zip(tt, (sol_f(t) for t in tt))]
avg  = [(u = sol_a(t);   (α = u[1], β = u[2], H = u[3], Id = u[4])) for t in tt]
ana  = [(u = sol_ana(t); (α = u[1], β = u[2], H = u[3], Id = u[4])) for t in tt]

ωe_f = [x.H/x.Id for x in full];  ωe_a = [x.H/x.Id for x in avg]
β_f  = [x.β      for x in full];  β_a  = [x.β      for x in avg]
Id_f = [x.Id     for x in full];  Id_a = [x.Id     for x in avg]
ωe_n = [x.H/x.Id for x in ana];   β_n = [x.β for x in ana];  Id_n = [x.Id for x in ana]
H_f  = [x.H      for x in full];  H_a  = [x.H      for x in avg]
α_f  = [x.α      for x in full];  α_a  = [x.α      for x in avg]

# Cycle-mean of the full model
"""
    boxcar(y, t, W) → smoothed y

Centred moving average of half-width `W` seconds.  Endpoints use the largest
symmetric window that fits, so no wrap-around or edge padding is invented.
"""
function boxcar(y::Vector{Float64}, t::Vector{Float64}, W::Real)
    n = length(y); out = similar(y)
    for k in 1:n
        lo = searchsortedfirst(t, t[k] - W)
        hi = searchsortedlast(t, t[k] + W)
        h  = min(k - lo, hi - k)                      # symmetric half-width
        out[k] = sum(@view y[k-h:k+h]) / (2h + 1)
    end
    return out
end

# One tumbling period at the initial state, from the torque-free solution.
Pψ = tumbling_periods(ωe0, Id0, I)[1]
@printf("\ntumbling period P_ψ at the initial state = %.2f s (%.2f min)\n", Pψ, Pψ/60)

ωe_fm = boxcar(ωe_f, tt, Pψ/2)
β_fm  = boxcar(β_f,  tt, Pψ/2)
Id_fm = boxcar(Id_f, tt, Pψ/2)

# the metric
rms(x) = sqrt(sum(abs2, x) / length(x))

println("\n" * "="^78)
println("RMS DEVIATION vs the full Euler model, BY WINDOW AND BY BACKEND")
println("="^78)
println("""
Reported against the CYCLE-MEAN of the full model, since the averaged model
predicts the mean over one tumbling period and not the instantaneous value; the
raw residual is dominated by ripple that is not error.  ω_e and I_d are
normalised by their initial values, β is in radians (absolute).""")

@printf("\n%-10s %-10s | %12s %12s %12s | %12s\n", "window", "backend",
        "ω_e (rel)", "β [rad]", "I_d (rel)", "n samples")
for W in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00, YEARS)
    m = tt .<= W * SECONDS_PER_YEAR
    count(m) < 2 && continue
    for (bk, wa, ba, da) in (("numeric", ωe_a, β_a, Id_a), ("analytic", ωe_n, β_n, Id_n))
        @printf("%-10.2f %-10s | %12.4e %12.4e %12.4e | %12d\n", W, bk,
                rms(wa[m] .- ωe_fm[m])/ωe0, rms(ba[m] .- β_fm[m]),
                rms(da[m] .- Id_fm[m])/Id0, count(m))
    end
end

# Secular drift.
println("\nsecular drift over the full window:")
@printf("   %-22s %14s %14s %14s\n", "", "ω_e final", "Δω_e/ω_e0", "I_d/I_s final")
@printf("   %-22s %14.6e %+13.2f%% %14.6f\n", "full Euler (cyc-mean)",
        ωe_fm[end], 100*(ωe_fm[end]-ωe_fm[1])/ωe_fm[1], Id_fm[end]/I.Is)
@printf("   %-22s %14.6e %+13.2f%% %14.6f\n", "averaged :numeric",
        ωe_a[end], 100*(ωe_a[end]-ωe_a[1])/ωe_a[1], Id_a[end]/I.Is)
@printf("   %-22s %14.6e %+13.2f%% %14.6f\n", "averaged :analytic",
        ωe_n[end], 100*(ωe_n[end]-ωe_n[1])/ωe_n[1], Id_n[end]/I.Is)

# β vs the α coordinate pole — the mechanism behind the long-window divergence.
println("\nβ excursion toward the α coordinate pole (β → 0 or π), where α̇ ∝ 1/sinβ:")
@printf("   %-22s range %6.2f°–%6.2f°   fraction of samples with β > 175°: %.3f\n",
        "full Euler (cyc-mean)", rad2deg(minimum(β_fm)), rad2deg(maximum(β_fm)),
        count(>(deg2rad(175)), β_fm)/length(β_fm))
@printf("   %-22s range %6.2f°–%6.2f°   fraction of samples with β > 175°: %.3f\n",
        "averaged :numeric", rad2deg(minimum(β_a)), rad2deg(maximum(β_a)),
        count(>(deg2rad(175)), β_a)/length(β_a))
@printf("   %-22s range %6.2f°–%6.2f°   fraction of samples with β > 175°: %.3f\n",
        "averaged :analytic", rad2deg(minimum(β_n)), rad2deg(maximum(β_n)),
        count(>(deg2rad(175)), β_n)/length(β_n))

# Ripple, for context on why the cycle-mean is used at all.
mr = tt .<= 10Pψ
@printf("\ntumbling ripple in the FULL model (peak-to-peak, first 10 P_ψ):\n")
@printf("   ω_e  %.4e  (%.3f%% of ω_e0)   I_d/I_s  %.6f\n",
        maximum(ωe_f[mr])-minimum(ωe_f[mr]),
        100*(maximum(ωe_f[mr])-minimum(ωe_f[mr]))/ωe0,
        (maximum(Id_f[mr])-minimum(Id_f[mr]))/I.Is)

# write 
open(joinpath(@__DIR__, "verification_full_vs_averaged.csv"), "w") do io
    println(io, "t_yr,omega_e_full,omega_e_full_cycmean,omega_e_num,omega_e_ana," *
                "beta_full_deg,beta_full_cycmean_deg,beta_num_deg,beta_ana_deg," *
                "Id_full_over_Is,Id_full_cycmean_over_Is,Id_num_over_Is,Id_ana_over_Is," *
                "H_full,H_num,alpha_full_deg,alpha_num_deg")
    for k in eachindex(tt)
        @printf(io, "%.8f,%.10e,%.10e,%.10e,%.10e,%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.10e,%.10e,%.4f,%.4f\n",
                tt[k]/SECONDS_PER_YEAR, ωe_f[k], ωe_fm[k], ωe_a[k], ωe_n[k],
                rad2deg(β_f[k]), rad2deg(β_fm[k]), rad2deg(β_a[k]), rad2deg(β_n[k]),
                Id_f[k]/I.Is, Id_fm[k]/I.Is, Id_a[k]/I.Is, Id_n[k]/I.Is,
                H_f[k], H_a[k], rad2deg(mod2pi(α_f[k])), rad2deg(mod2pi(α_a[k])))
    end
end
println("\nwrote verification_full_vs_averaged.csv")
