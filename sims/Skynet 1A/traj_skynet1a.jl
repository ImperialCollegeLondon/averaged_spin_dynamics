
include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))  # sims/Skynet 1A/ is TWO levels down
using .master
using Printf

# skynet_shape brings in skynet_inertia (guarded, output suppressed) and exposes
# both skynet1a_shape() and skynet1a_inertia().
include(joinpath(@__DIR__, "skynet_shape"))

#  configuration — edit here 
I  = skynet1a_inertia()
sh = skynet1a_shape()

YEARS   = parse(Float64, get(ENV, "TRAJ_YEARS", "20.0"))
const TSUF = YEARS == 20.0 ? "" : "_" * string(round(Int, YEARS)) * "yr"

N_SAVE  = round(Int, 3000 * YEARS / 20.0)
μoJ     = 1.0e-3                  # model-validity ceiling
SIGMA   = -1

# Initial condition 

lam_frac(f) = I.Il/I.Is + f * (I.Ii - I.Il) / I.Is      # f=0 at I_l, 1 at I_i
sam_frac(f) = I.Ii/I.Is + f * (I.Is - I.Ii) / I.Is      # f=0 at I_i, 1 at I_s

LAM_F   = 0.1
β0      = deg2rad(75.0)                 # same as both siblings
Id0_rel = lam_frac(LAM_F)
Pe0     = 20.0 * 60.0                   # same as both siblings

CASES = [(label = @sprintf("J = %g", J), J = J) for J in (0.1, 0.5, 1.0, 1.8, 5.0, 10.0)]

# ALL SIX J VALUES RUN.


# run 
Id0 = Id0_rel * I.Is
ωe0 = 2π / Pe0
H0  = Id0 * ωe0
st0 = OsculatingState(0.0, β0, H0, Id0)
tf  = YEARS * SECONDS_PER_YEAR

# The closed-form endpoint.  Valid here for the SAME reason as Telstar's even
# though the SRP torque is not zero: what it needs is M̄_z = 0, not M̄ = 0.
ωe_pred = ωe0 * Id0_rel

@printf("traj_skynet1a: %d cases × %.0f yr from β₀=%.0f°, I_d0/I_s=%.4f (lam_frac %.2f), P_e=%.0f min, σ=%+d\n",
        length(CASES), YEARS, rad2deg(β0), Id0_rel, LAM_F, Pe0/60, SIGMA)
@printf("  LAM band %.4f–%.4f, SAM %.4f–%.4f  (separatrix I_i/I_s = %.4f)\n",
        lam_frac(0.0), lam_frac(1.0), sam_frac(0.0), sam_frac(1.0), I.Ii/I.Is)

# PRE-FLIGHT: does σ actually matter here?
let st = OsculatingState(0.0, deg2rad(75), 3000.0, lam_frac(0.5)*I.Is)
    _pf(s) = PerturbationConfig(srp = true, dissipation = true, gravity_gradient = true,
                                μ = μoJ, J = 1.0, orbit_R = 4.2575e7,
                                srp_backend = :analytic, σ_branch = s, resonant = false)
    p = averaged_eom(st, I, sh, _pf(+1))
    m = averaged_eom(st, I, sh, _pf(-1))
    @printf("  pre-flight σ: max|Δ(dstate/dt)| between σ=+1 and σ=-1 = %.3e  → σ is %s\n",
            maximum(abs.(p .- m)),
            maximum(abs.(p .- m)) > 0 ? "LIVE (a real choice, unlike Telstar's)" : "inert")

    # And confirm the M̄_z = 0 claim at the EOM level, not just from the survey.
    T = averaged_srp_torques_analytic(sh, st, I; σ = SIGMA)
    @printf("  pre-flight M̄: M̄_z = %.3e,  |M̄_⊥| = %.3e  N m   (ratio %.3e)\n",
            T.M_H[3], hypot(T.M_H[1], T.M_H[2]),
            abs(T.M_H[3])/max(hypot(T.M_H[1], T.M_H[2]), eps()))
end

@printf("  M̄_z ≈ 0 ⇒ |H| conserved ⇒ expect monotonic relaxation ω_e %.4e → %.4e for EVERY J;\n",
        ωe0, ωe_pred)
println("  J sets the rate, not the destination.  The physics to look at is α.  See the header.")

open(joinpath(@__DIR__, "traj_skynet1a" * TSUF * ".csv"), "w") do io
    # Schema identical to traj_goes8.csv / traj_telstar401.csv.
    println(io, "label,J,mu_over_J,t_yr,omega_e,beta_deg,alpha_deg,Id_over_Is,eps_beta")
    for c in CASES
        cfg = PerturbationConfig(srp = true, dissipation = true, gravity_gradient = true,
                                 μ = μoJ * c.J, J = c.J, orbit_R = 4.2575e7,
                                 srp_backend = :analytic, σ_branch = SIGMA,
                                 resonant = false)
        t0 = time()
        # One bad case must not kill the batch: the I_d → I_s sqrt failure is
        # still open, and this configuration is as exposed to it as Telstar's
        # with M̄_z = 0 the dissipative attractor IS I_d = I_s, so every run walks
        # straight at it.
        try
            sol = propagate_averaged(I, st0, (0.0, tf); shape = sh, cfg = cfg,
                      reltol = 1e-8, abstol = 1e-10, maxiters = Int(1e7),
                      callback = spin_termination_callbacks(ωe_floor = 1e-8,
                                                            ωe_ceiling = 1.0))
            te = sol.t[end]
            for t in range(0.0, te, length = N_SAVE)
                u  = sol(t)
                α, β, H, Id = u[1], u[2], u[3], u[4]
                εβ = abs(Id - I.Ii)/I.Ii < 1e-4 ? Inf :
                     epsilon_beta(OsculatingState(α, β, H, Id), I, sh, cfg)
                @printf(io, "%s,%.4f,%.1e,%.6f,%.8e,%.4f,%.4f,%.6f,%.6e\n",
                        c.label, c.J, μoJ, t/SECONDS_PER_YEAR, H/Id,
                        rad2deg(β), rad2deg(mod2pi(α)), Id/I.Is, εβ)
            end
            o = classify_spin_outcome(sol, tf; ωe_floor = 1e-8, ωe_ceiling = 1.0)
            uend  = sol(te)
            state = te < tf - 1.0 ?
                    @sprintf("TRUNCATED at I_d/I_s=%.9f [FLAG-SKYNET-IDIS]", uend[4]/I.Is) :
                    "full horizon"
            @printf("  %-10s → %-11s  ω_e %.3e → %.3e  (pred %.3e, %+.2e rel)  ran %.4f/%.0f yr  %s   [%.1f s]\n",
                    c.label, o.fate, ωe0, o.ωe_final, ωe_pred,
                    (o.ωe_final - ωe_pred)/ωe_pred,
                    te/SECONDS_PER_YEAR, YEARS, state, time()-t0)
        catch e
            @printf("  %-10s → FAILED after %.1f s: %s\n", c.label, time()-t0,
                    first(sprint(showerror, e), 90))
        end
    end
end
println("wrote ", "traj_skynet1a" * TSUF * ".csv")
