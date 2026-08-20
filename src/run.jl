include("master.jl")     # run from the repo root, or use the full path
using .master
using Printf

# --- one satellite: GOES-8 ---
I  = goes8_inertia()                          # B&S θ_sa=17° inertias
sh = goes8_shape_full(; θ_sa = deg2rad(17.0), optical = :bs)

# --- initial osculating state ---
β0  = deg2rad(45.0)
Pe0 = 120.0 * 60.0            # spin period [s]
Id0 = 0.98 * I.Is             # SAM band
H0  = Id0 * (2π / Pe0)
st0 = OsculatingState(0.0, β0, H0, Id0)

cfg = PerturbationConfig(srp = true, dissipation = true, gravity_gradient = true,
                          srp_backend = :analytic, σ_branch = +1, resonant = false)

sol = propagate_averaged(I, st0, (0.0, 10 * SECONDS_PER_YEAR);
                          shape = sh, cfg = cfg, reltol = 1e-8, abstol = 1e-10)

αf, βf, Hf, Idf = sol.u[end]
println("final ω_e = ", Hf/Idf, " rad/s   β = ", rad2deg(βf), "°")

# --- write full trajectory to CSV ---
csv_path = joinpath(@__DIR__, "run_output.csv")
open(csv_path, "w") do io
    println(io, "t_yr,alpha_deg,beta_deg,H,Id,omega_e,Pe_min,regime")
    for (t, u) in zip(sol.t, sol.u)
        α, β, H, Id = u
        ωe     = H / Id
        Pe_min = 2π / ωe / 60
        regime = classify_regime(Id, I)
        rstr   = regime isa LAM ? "LAM" : regime isa SAM ? "SAM" : "Separatrix"
        @printf(io, "%.6f,%.6f,%.6f,%.6e,%.6e,%.6e,%.4f,%s\n",
                t / SECONDS_PER_YEAR, rad2deg(α), rad2deg(β), H, Id, ωe, Pe_min, rstr)
    end
end
println("wrote ", csv_path, "  (", length(sol.t), " rows)")