
using DifferentialEquations

const MEAN_MOTION = 2π / SECONDS_PER_YEAR   # [rad s⁻¹]  (= 0.9856°/day)
#slug dissipation parameter defined in B&S 2022
const MAX_MU_OVER_J = 1e-3   # [s⁻¹]

function averaged_eom(state::OsculatingState, I::PrincipalInertias,
                      shape::ShapeModel, cfg::PerturbationConfig;
                      t::Real = 0.0, n::Real = MEAN_MOTION,
                      N_φ::Int = 90, N_τ::Int = 180, P_SRP::Real = P_SRP_1AU)
    α, β, H, Id = state.α, state.β, state.H, state.Id
    ωe     = omega_e(H, Id)
    regime = classify_regime(Id, I)

    # ── YORP: averaged SRP torque (B&S 2021 Eqs. 16–19) ──
    at = if cfg.srp
        if cfg.srp_backend === :numeric
            averaged_srp_torques(shape, β, ωe, Id, I, regime;
                                 N_φ = N_φ, N_τ = N_τ, P_SRP = P_SRP,
                                 σ = cfg.σ_branch)
        elseif cfg.srp_backend === :analytic
            averaged_srp_torques_analytic(shape, β, Id, I, regime;
                                          P_SRP = P_SRP, σ = cfg.σ_branch)
        else
            error("averaged_eom: unknown srp_backend :$(cfg.srp_backend) " *
                  "(available: :numeric, :analytic; :fourier is M7)")
        end
    else
        AveragedTorques(SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0, 0, 0))
    end
    M̄x, M̄y, M̄z          = at.M_H[1], at.M_H[2], at.M_H[3]
    az1M1, az2M2, az3M3 = at.azM[1], at.azM[2], at.azM[3]

    # ── Gravity gradient: adds L̄_x to β̇, L̄_y to α̇; L̄_z=0 (B&S 2022 Eqs. 32–36) ──
    L̄x = 0.0; L̄y = 0.0
    if cfg.gravity_gradient
        n_g   = gg_mean_motion(cfg.orbit_R)
        λ_o, δ = orbit_angles(cfg.orbit_i, cfg.orbit_Ω)
        λ     = -n * t + λ_o                                          # O frame rotates (Eq. 23)
        Īx, _, Īz = averaged_inertia(ωe, Id, I, regime)
        L̄x, L̄y, _ = averaged_gg_torque(α, β, λ, δ, Īx, Īz, n_g)      # Eqs. 32–34
    end

    # ── Dissipation: adds h_d to İ_d (B&S 2022 Eqs. 18, 38); h_d ≥ 0 ──
    hd = cfg.dissipation ? h_d(ωe, Id, I, regime, cfg.μ, cfg.J) : 0.0

    # h_d ≥ 0 is a physical requirement, not a convention: internal dissipation
    # burns rotational kinetic energy at constant H, and I_d = H²/2T, so T↓ forces
    # I_d↑ (relaxation toward uniform rotation about the max-inertia axis).
    # h_d < 0 would be dissipation *creating* kinetic energy — thermodynamically
    # impossible, and in practice a symptom of μ/J past MAX_MU_OVER_J or a sign
    # error in the t_ij coefficients.  Tiny negatives are roundoff and clamped.
    if hd < 0
        if hd > -1e-12 * Id
            hd = 0.0
        else
            error("averaged_eom: h_d = $hd < 0 violates the second law " *
                  "(dissipation cannot decrease I_d).  State: ωe=$ωe, Id=$Id, " *
                  "regime=$(typeof(regime)), μ=$(cfg.μ), J=$(cfg.J), " *
                  "μ/J=$(cfg.μ/cfg.J) s⁻¹ (bound $MAX_MU_OVER_J).")
        end
    end

    Il, Ii, Is = I.Il, I.Ii, I.Is
    sβ = sin(β)

    # Coordinate pole, not a physical singularity: (α,β) are spherical coordinates
    # of Ĥ in the O frame, so α is undefined when Ĥ lies on the sun/antisun line
    # (β = 0, π) exactly as longitude is undefined at a pole.  H itself is fine.
    # B&S 2021 §V note the same and suggest the (v,w) coordinates that `to_vw` /
    # `from_vw` implement.  Warned, not errored: B&S report never encountering it
    # in practice, and a transient near-pole pass is harmless.
    if sinβ_prev(β)
        @warn "averaged_eom: β = $β rad is within sinβ_guard tolerance of the " *
              "α coordinate pole (β = 0 or π); α̇ ∝ 1/sinβ is numerically " *
              "singular there.  α output is unreliable for this stretch — " *
              "consider the (v,w) coordinates (`to_vw`/`from_vw`)." maxlog = 1
    end

    α̇  = (M̄y + L̄y + H * n * cos(α) * cos(β)) / (H * sβ)             # Eq. 35
    β̇  = (M̄x + L̄x + H * n * sin(α)) / H                             # Eq. 36
    Ḣ  = M̄z                                                          # Eq. 37 (L̄_z = 0)
    İd_yorp = -(2Id / H) * ((Id - Ii) / Ii * az1M1 +                 # Eq. 39
                            (Id - Is) / Is * az2M2 +
                            (Id - Il) / Il * az3M3)
    İd = İd_yorp + hd                                                # Eq. 38
    return SVector{4,Float64}(α̇, β̇, Ḣ, İd)
end

function _averaged_rhs(u, p, t)
    I, shape, cfg, n, N_φ, N_τ, P_SRP = p
    state = OsculatingState(u[1], u[2], u[3], u[4])
    return averaged_eom(state, I, shape, cfg;
                        t = t, n = n, N_φ = N_φ, N_τ = N_τ, P_SRP = P_SRP)
end

function propagate_averaged(I::PrincipalInertias, state0::OsculatingState, tspan;
                            shape::ShapeModel,
                            cfg::PerturbationConfig = PerturbationConfig(),
                            n::Real = MEAN_MOTION,
                            N_φ::Int = 90, N_τ::Int = 180,
                            P_SRP::Real = P_SRP_1AU,
                            solver = Tsit5(), reltol = 1e-8, abstol = 1e-10,
                            kwargs...)
    cfg.srp_backend in (:numeric, :analytic) || error(
        "dynamics_averaged: srp_backend must be :numeric (M3) or :analytic (M6); " *
        "got :$(cfg.srp_backend).  The :fourier backend is M7.")
    cfg.resonant && error(
        "resonant averaged dynamics are gated (FLAG-RESONANCE); set cfg.resonant=false.")

    # Averaged-dissipation validity bound (CLAUDE.md; B&S 2022 §II.C).  Checked
    # only when dissipation is actually on.  Relative tolerance so the documented
    # figure configurations sitting exactly at μ/J = 1e-3 are not rejected.
    if cfg.dissipation
        μJ = cfg.μ / cfg.J
        μJ <= MAX_MU_OVER_J * (1 + 1e-9) || error(
            "dynamics_averaged: μ/J = $μJ s⁻¹ exceeds the averaged-dissipation " *
            "validity bound of $MAX_MU_OVER_J s⁻¹ (μ=$(cfg.μ), J=$(cfg.J)).  " *
            "Above it the steady-state slug relation σ ≈ [A]ω breaks and h_d is " *
            "invalid.  Reduce μ or raise J; do not sweep past this bound.")
    end

    u0   = to_svector(state0)
    p    = (I, shape, cfg, n, N_φ, N_τ, P_SRP)
    prob = ODEProblem(_averaged_rhs, u0, tspan, p)
    return solve(prob, solver; reltol = reltol, abstol = abstol, kwargs...)
end
