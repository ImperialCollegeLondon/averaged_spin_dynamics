
include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))
using .master
using StaticArrays, LinearAlgebra, Printf, Random

I  = goes8_inertia()
SH = goes8_shape_full(; θ_sa = deg2rad(17), optical = :bs)

const PORTRAIT_YEARS = 20.0
const N_DENSE        = 8000          # 400 samples/yr — resolves the limit cycle
const N_TRAJ         = 12
const J_REF, MUoJ_REF, SIGMA = 1.8, 1.0e-3, -1
const R_ORB, I_ORB, OMEGA_ORB = 4.2575e7, 0.0, 0.0
const OUT = joinpath(@__DIR__, "phase_portrait_goes8.csv")

# The same IC construction as the ensemble, but a deliberately SPREAD dozen
# rather than a random subset: 6 across the LAM band and 6 across SAM, at three
# coning angles, so the portrait shows where trajectories come FROM.
lam(f) = I.Il + f * (I.Ii - I.Il)
sam(f) = I.Ii + f * (I.Is - I.Ii)

ics = NamedTuple[]
for (bandf, band) in ((lam, "LAM"), (sam, "SAM"))
    for (i, f) in enumerate((0.10, 0.45, 0.85))
        for (j, β0) in enumerate((30.0, 90.0, 150.0))
            length(ics) >= 2N_TRAJ && continue
            (i + j) % 2 == 0 || continue          # thin the 3x3 to a spread 5
            push!(ics, (α0 = 0.0, β0 = deg2rad(β0), Pe = 120 * 60.0,
                        Id0 = bandf(f), band = band, f = f))
        end
    end
end

cfg = PerturbationConfig(srp = true, dissipation = true, gravity_gradient = true,
                         μ = MUoJ_REF * J_REF, J = J_REF,
                         orbit_i = I_ORB, orbit_Ω = OMEGA_ORB, orbit_R = R_ORB,
                         srp_backend = :analytic, σ_branch = SIGMA)

tf = PORTRAIT_YEARS * SECONDS_PER_YEAR
ts = range(0.0, tf, length = N_DENSE)

println("phase portrait: ", length(ics), " trajectories x ", N_DENSE,
        " samples over ", PORTRAIT_YEARS, " yr")

open(OUT, "w") do io
    println(io, "traj_id,band0,band_frac0,beta0_deg,Id0_over_Is,t_yr,Id_over_Is,beta_deg,omega_e")
    for (k, ic) in enumerate(ics)
        ωe0 = 2π / ic.Pe
        st0 = OsculatingState(ic.α0, ic.β0, ic.Id0 * ωe0, ic.Id0)
        local sol
        try
            sol = propagate_averaged(I, st0, (0.0, tf); shape = SH, cfg = cfg,
                      reltol = 1e-8, abstol = 1e-10, maxiters = Int(1e7),
                      callback = spin_termination_callbacks(ωe_floor = 1e-8,
                                                            ωe_ceiling = 1.0))
        catch e
            @printf("  traj %2d  ERROR %s\n", k, first(sprint(showerror, e), 70))
            continue
        end
        te = sol.t[end]
        n = 0
        for t in ts
            t > te && break
            u = sol(t)
            @printf(io, "%d,%s,%.3f,%.1f,%.6f,%.6f,%.8f,%.5f,%.8e\n",
                    k, ic.band, ic.f, rad2deg(ic.β0), ic.Id0 / I.Is,
                    t / SECONDS_PER_YEAR, u[4] / I.Is, rad2deg(u[2]), u[3] / u[4])
            n += 1
        end
        @printf("  traj %2d  %s f=%.2f  β₀=%5.1f°  ran to %6.2f yr  (%d samples)\n",
                k, ic.band, ic.f, rad2deg(ic.β0), te / SECONDS_PER_YEAR, n)
    end
end
println("wrote ", OUT)
