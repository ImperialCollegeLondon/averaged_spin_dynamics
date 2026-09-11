
include(joinpath(@__DIR__, "..", "..", "src", "master.jl"))
using .master
using Printf, LinearAlgebra, StaticArrays

I  = goes8_inertia()
sh = goes8_shape_full(; θ_sa = deg2rad(17), optical = :bs)

# grid

N_BAND  = 40                     # per regime
N_BETA  = 48
SIGMA   = -1                     # GOES-8's choice throughout this repo
ωe_ref  = 2π / 1200.0            # M̄ is ω_e-independent; used only to build states

lam(f) = I.Il/I.Is + f*(I.Ii - I.Il)/I.Is
sam(f) = I.Ii/I.Is + f*(I.Is - I.Ii)/I.Is
# Endpoints are trimmed: I_d = I_i IS the separatrix, where the torque-free
# solution degenerates and `averaged_srp_torques` throws by design.
rs = [lam.(range(0.01, 0.985, length = N_BAND)) ; sam.(range(0.015, 0.99, length = N_BAND))]
βs = range(deg2rad(1.0), deg2rad(179.0), length = N_BETA)

# numeric-backend convergence
println("="^78)
println("FIGURE 2 — analytic vs numeric averaged-SRP backends, GOES-8")
println("="^78)
println("\nnumeric-backend convergence (must converge before it can be a reference):")
@printf("   %-16s %16s %16s %16s\n", "(N_φ, N_τ)", "M̄_z", "|M̄_⊥|", "rel. Δ vs finest")
let β = deg2rad(75.0), Id = 0.3433*I.Is
    st = OsculatingState(0.0, β, Id*ωe_ref, Id)
    ref = averaged_srp_torques(sh, st, I; σ = SIGMA, N_φ = 360, N_τ = 720)
    for (nφ, nτ) in ((45, 90), (90, 180), (180, 360), (360, 720))
        t = averaged_srp_torques(sh, st, I; σ = SIGMA, N_φ = nφ, N_τ = nτ)
        @printf("   %-16s %16.6e %16.6e %16.3e\n", "($nφ, $nτ)", t.M_H[3],
                hypot(t.M_H[1], t.M_H[2]),
                abs(t.M_H[3] - ref.M_H[3])/max(abs(ref.M_H[3]), eps()))
    end
end
const NPHI, NTAU = 180, 360      # 2× the default, checked above

# sweep 
@printf("\ngrid: %d I_d (%d LAM + %d SAM) × %d β = %d states, σ=%+d, numeric at (%d,%d)\n",
        length(rs), N_BAND, N_BAND, N_BETA, length(rs)*N_BETA, SIGMA, NPHI, NTAU)

# İ_d's YORP part (Eq. 39), formed from azM — the quantity that decides the
# rotation mode.  H cancels out of the comparison, so a reference H is fine.
function iddot_yorp(azM, Id, H)
    -(2Id/H) * ((Id - I.Ii)/I.Ii * azM[1] +
                (Id - I.Is)/I.Is * azM[2] +
                (Id - I.Il)/I.Il * azM[3])
end

rows = NamedTuple[]
t0 = time()
for r in rs, β in βs
    Id = r * I.Is; H = Id * ωe_ref
    st = OsculatingState(0.0, β, H, Id)
    local tn, ta
    try
        tn = averaged_srp_torques(sh, st, I; σ = SIGMA, N_φ = NPHI, N_τ = NTAU)
        ta = averaged_srp_torques_analytic(sh, st, I; σ = SIGMA)
    catch e
        continue
    end
    push!(rows, (r = r, β = β,
                 mz_n = tn.M_H[3], mz_a = ta.M_H[3],
                 mp_n = hypot(tn.M_H[1], tn.M_H[2]), mp_a = hypot(ta.M_H[1], ta.M_H[2]),
                 mx_n = tn.M_H[1], mx_a = ta.M_H[1],
                 my_n = tn.M_H[2], my_a = ta.M_H[2],
                 id_n = iddot_yorp(tn.azM, Id, H), id_a = iddot_yorp(ta.azM, Id, H)))
end
@printf("evaluated %d states in %.1f s\n", length(rows), time()-t0)

# report
scale_mz = maximum(abs(x.mz_n) for x in rows)
scale_mp = maximum(abs(x.mp_n) for x in rows)
scale_id = maximum(abs(x.id_n) for x in rows)

dmz = [abs(x.mz_a - x.mz_n)/scale_mz for x in rows]
dmp = [abs(x.mp_a - x.mp_n)/scale_mp for x in rows]
did = [abs(x.id_a - x.id_n)/scale_id for x in rows]
sgn = [sign(x.mz_a) != sign(x.mz_n) && abs(x.mz_n) > 0.01*scale_mz for x in rows]

pct(v, q) = sort(v)[clamp(ceil(Int, q*length(v)), 1, length(v))]
println("\n" * "="^78)
println("DEVIATION, analytic vs numeric  (normalised by max|numeric| on the grid)")
println("="^78)
@printf("\n%-22s %12s %12s %12s %12s\n", "component", "median", "p90", "p99", "max")
for (nm, v) in (("M̄_z", dmz), ("|M̄_⊥|", dmp), ("İ_d (YORP part)", did))
    @printf("%-22s %12.4e %12.4e %12.4e %12.4e\n", nm,
            pct(v,0.5), pct(v,0.9), pct(v,0.99), maximum(v))
end
@printf("\nSIGN DISAGREEMENTS in M̄_z (where |M̄_z^num| > 1%% of its grid max): %d/%d (%.1f%%)\n",
        count(sgn), length(rows), 100*count(sgn)/length(rows))
println("   M̄_z is the ONLY driver of Ḣ, so a sign flip there is spin-up vs spin-down.")

# Where is it worst, relative to the separatrix?
sep = I.Ii/I.Is
println("\nWHERE THE DEVIATION LIVES, relative to the separatrix I_d = I_i:")
@printf("   separatrix at I_d/I_s = %.4f;  LAM band %.4f–%.4f, SAM %.4f–%.4f\n",
        sep, lam(0.0), lam(1.0), sam(0.0), sam(1.0))
for (nm, v) in (("M̄_z", dmz), ("|M̄_⊥|", dmp), ("İ_d", did))
    k = argmax(v)
    x = rows[k]
    d = (x.r - sep)/sep
    @printf("   %-8s worst at I_d/I_s = %.4f (%+.1f%% from separatrix, %s), β = %5.1f°  → %.3e\n",
            nm, x.r, 100d, x.r < sep ? "LAM" : "SAM", rad2deg(x.β), v[k])
end
# Binned by distance from the separatrix — the brief asks specifically for this.
println("\n   median deviation binned by |I_d − I_i|/I_s:")
@printf("   %-16s %8s %12s %12s %12s\n", "band", "n", "M̄_z", "|M̄_⊥|", "İ_d")
edges = [0.0, 0.02, 0.05, 0.10, 0.20, 1.0]
for b in 1:length(edges)-1
    idx = [k for k in eachindex(rows) if edges[b] <= abs(rows[k].r - sep) < edges[b+1]]
    isempty(idx) && continue
    @printf("   %-16s %8d %12.4e %12.4e %12.4e\n",
            @sprintf("[%.2f, %.2f)", edges[b], edges[b+1]), length(idx),
            pct(dmz[idx],0.5), pct(dmp[idx],0.5), pct(did[idx],0.5))
end

open(joinpath(@__DIR__, "verification_analytic_vs_numeric.csv"), "w") do io
    println(io, "Id_over_Is,beta_deg,Mz_num,Mz_ana,Mperp_num,Mperp_ana," *
                "Mx_num,Mx_ana,My_num,My_ana,Iddot_num,Iddot_ana," *
                "dev_Mz,dev_Mperp,dev_Iddot,sign_flip_Mz")
    for (k, x) in enumerate(rows)
        @printf(io, "%.6f,%.4f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%d\n",
                x.r, rad2deg(x.β), x.mz_n, x.mz_a, x.mp_n, x.mp_a,
                x.mx_n, x.mx_a, x.my_n, x.my_a, x.id_n, x.id_a,
                dmz[k], dmp[k], did[k], sgn[k] ? 1 : 0)
    end
end
println("\nwrote verification_analytic_vs_numeric.csv")
@printf("normalisation scales: max|M̄_z| = %.4e, max|M̄_⊥| = %.4e, max|İ_d| = %.4e\n",
        scale_mz, scale_mp, scale_id)
