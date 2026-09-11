
include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))  # sims/GOES8/ is TWO levels down
using .master
using Printf

# configuration — edit here 
I   = goes8_inertia()
sh  = goes8_shape_full(; θ_sa = deg2rad(17.0), optical = :bs)


YEARS   = parse(Float64, get(ENV, "TRAJ_YEARS", "20.0"))
const TSUF = YEARS == 20.0 ? "" : "_" * string(round(Int, YEARS)) * "yr"
t.
N_SAVE  = round(Int, 3000 * YEARS / 20.0)
SIGMA   = -1
μoJ     = 1.0e-3                  # model-validity ceiling

β0      = deg2rad(75.0)
Id0_rel = 0.3433        # LAM band (I_l/I_s = 0.2746, separatrix 0.9614)
Pe0     = 20.0 * 60.0

# one case per J
CASES = [(label = @sprintf("J = %g", J), J = J) for J in (0.1, 0.5, 1.0, 1.8, 5.0, 10.0)]

#  run
Id0 = Id0_rel * I.Is
H0  = Id0 * (2π / Pe0)
st0 = OsculatingState(0.0, β0, H0, Id0)
tf  = YEARS * SECONDS_PER_YEAR

@printf("traj_goes8: %d cases × %.0f yr from β₀=%.0f°, I_d0/I_s=%.3f, P_e=%.0f min, σ=%+d\n",
        length(CASES), YEARS, rad2deg(β0), Id0_rel, Pe0/60, SIGMA)

open(joinpath(@__DIR__, "traj_goes8" * TSUF * ".csv"), "w") do io
    println(io, "label,J,mu_over_J,t_yr,omega_e,beta_deg,alpha_deg,Id_over_Is,eps_beta")
    for c in CASES
        # NB: `resonant = false` was once REQUIRED here — PerturbationConfig used
        # to default it to true and the averaged propagator errors out on it.
        # The default is false now, so this is belt-and-braces rather than a fix.
        cfg = PerturbationConfig(srp = true, dissipation = true, gravity_gradient = true,
                                 μ = μoJ * c.J, J = c.J, orbit_R = 4.2575e7,
                                 srp_backend = :analytic, σ_branch = SIGMA,
                                 resonant = false)
        t0 = time()
        # One bad case must not kill the batch: the I_d → I_s sqrt failure is
        # still open, and it takes out ~15% of the sweep.
        try
            sol = propagate_averaged(I, st0, (0.0, tf); shape = sh, cfg = cfg,
                      reltol = 1e-8, abstol = 1e-10, maxiters = Int(1e7),
                      callback = spin_termination_callbacks(ωe_floor = 1e-8,
                                                            ωe_ceiling = 1.0))
            te = sol.t[end]
            for t in range(0.0, te, length = N_SAVE)
                u  = sol(t)
                α, β, H, Id = u[1], u[2], u[3], u[4]
                # ε_β: guard the separatrix, where P_ψ diverges and the averaging
                # premise fails outright (classify_regime's own tolerance is
                # ~1e-12, far too tight to protect the elliptic evaluation)
                εβ = abs(Id - I.Ii)/I.Ii < 1e-4 ? Inf :
                     epsilon_beta(OsculatingState(α, β, H, Id), I, sh, cfg)
                @printf(io, "%s,%.4f,%.1e,%.6f,%.8e,%.4f,%.4f,%.6f,%.6e\n",
                        c.label, c.J, μoJ, t/SECONDS_PER_YEAR, H/Id,
                        rad2deg(β), rad2deg(mod2pi(α)), Id/I.Is, εβ)
            end
            o = classify_spin_outcome(sol, tf; ωe_floor = 1e-8, ωe_ceiling = 1.0)
            @printf("  %-10s → %-11s  ω_e %.3e → %.3e   ran %.2f/%.0f yr   [%.1f s]\n",
                    c.label, o.fate, H0/Id0, o.ωe_final,
                    te/SECONDS_PER_YEAR, YEARS, time()-t0)
        catch e
            @printf("  %-10s → FAILED after %.1f s: %s\n", c.label, time()-t0,
                    first(sprint(showerror, e), 90))
        end
    end
end
println("wrote ", "traj_goes8" * TSUF * ".csv")
