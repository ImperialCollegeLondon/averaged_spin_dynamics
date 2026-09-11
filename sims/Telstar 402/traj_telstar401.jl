
include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))  # sims/Telstar 402/ is TWO levels down
using .master
using Printf

# telstar_shape brings in telstar_inertia (guarded, output suppressed) and
# exposes both telstar401_shape() and telstar401_inertia().
include(joinpath(@__DIR__, "telstar_shape"))

# configuration — edit here 
I   = telstar401_inertia()
sh  = telstar401_shape(; warn_inert = false)   # inertness is the subject of the
                                               # header, not something skipped

YEARS   = parse(Float64, get(ENV, "TRAJ_YEARS", "20.0"))
const TSUF = YEARS == 20.0 ? "" : "_" * string(round(Int, YEARS)) * "yr"
N_SAVE  = round(Int, 3000 * YEARS / 20.0)
μoJ     = 1.0e-3                  # model-validity ceiling
SIGMA   = +1

# Initial condition 
# Taken as a FRACTION ACROSS THE LAM BAND rather than an absolute I_d/I_s, using
# the same helpers as telstar_sweep.jl.  This matters: Telstar's bands are
# nothing like GOES-8's (LAM spans 0.341–0.844 of the I_d axis here, versus
# 0.275–0.961 there), so a hand-copied 0.3433 would sit at a completely
# different place in the band and the two trajectories would not be comparable.
#
# LAM_F = 0.1 is chosen because traj_goes8.jl's I_d0/I_s = 0.3433 is itself
# lam_frac(0.1) for GOES-8 — so both bodies start one tenth of the way up their
# own LAM band, which is the like-for-like comparison.
lam_frac(f) = I.Il/I.Is + f * (I.Ii - I.Il) / I.Is      # f=0 at I_l, 1 at I_i
sam_frac(f) = I.Ii/I.Is + f * (I.Is - I.Ii) / I.Is      # f=0 at I_i, 1 at I_s

LAM_F   = 0.1
β0      = deg2rad(75.0)                 # same as traj_goes8.jl
Id0_rel = lam_frac(LAM_F)
Pe0     = 20.0 * 60.0                   # same as traj_goes8.jl

# one case per J — the SAME six values traj_goes8.jl uses
CASES = [(label = @sprintf("J = %g", J), J = J) for J in (0.1, 0.5, 1.0, 1.8, 5.0, 10.0)]

# run
Id0 = Id0_rel * I.Is
ωe0 = 2π / Pe0
H0  = Id0 * ωe0
st0 = OsculatingState(0.0, β0, H0, Id0)
tf  = YEARS * SECONDS_PER_YEAR

# The closed-form endpoint, valid because H is conserved and I_d → I_s.
ωe_pred = ωe0 * Id0_rel

@printf("traj_telstar401: %d cases × %.0f yr from β₀=%.0f°, I_d0/I_s=%.4f (lam_frac %.2f), P_e=%.0f min, σ=%+d\n",
        length(CASES), YEARS, rad2deg(β0), Id0_rel, LAM_F, Pe0/60, SIGMA)
@printf("  LAM band %.4f–%.4f, SAM %.4f–%.4f  (separatrix I_i/I_s = %.4f)\n",
        lam_frac(0.0), lam_frac(1.0), sam_frac(0.0), sam_frac(1.0), I.Ii/I.Is)
@printf("  NO YORP (M̄ = 0): expect monotonic relaxation ω_e %.4e → %.4e for EVERY J;\n",
        ωe0, ωe_pred)
println("  J sets the rate of approach, not the destination.  See the header.")

open(joinpath(@__DIR__, "traj_telstar401" * TSUF * ".csv"), "w") do io
    # Schema identical to traj_goes8.csv so one plotting script reads both.
    println(io, "label,J,mu_over_J,t_yr,omega_e,beta_deg,alpha_deg,Id_over_Is,eps_beta")
    for c in CASES
        cfg = PerturbationConfig(srp = true, dissipation = true, gravity_gradient = true,
                                 μ = μoJ * c.J, J = c.J, orbit_R = 4.2575e7,
                                 srp_backend = :analytic, σ_branch = SIGMA,
                                 resonant = false)
        t0 = time()
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
            # Relative miss against the closed form is the honest quality metric
            # here — there is no attractor to compare against, only the exact
            # H₀/I_s endpoint.
            @printf("  %-10s → %-11s  ω_e %.3e → %.3e  (pred %.3e, %+.2e rel)  ran %.2f/%.0f yr   [%.1f s]\n",
                    c.label, o.fate, ωe0, o.ωe_final, ωe_pred,
                    (o.ωe_final - ωe_pred)/ωe_pred,
                    te/SECONDS_PER_YEAR, YEARS, time()-t0)
        catch e
            @printf("  %-10s → FAILED after %.1f s: %s\n", c.label, time()-t0,
                    first(sprint(showerror, e), 90))
        end
    end
end
println("wrote ", "traj_telstar401" * TSUF * ".csv")
