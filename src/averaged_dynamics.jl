"""
used equations
  ᾱ̇  = (M̄_y + H̄ n cos ᾱ cos β̄) / (H̄ sin β̄)                              (Eq. 16)
  β̄̇  = (M̄ₓ + H̄ n sin ᾱ) / H̄                                            (Eq. 17)
  H̄̇  = M̄_z                                                               (Eq. 18)
  Ī̇_d = −(2 Ī_d/H̄)[...]                                                   (Eq. 19)
  ω̄̇_e = (1/Ī_d)[ M̄_z − (H̄/Ī_d) Ī̇_d ]                                    (Eq. 20)
"""
using DifferentialEquations

const MEAN_MOTION = 2π / SECONDS_PER_YEAR   # [rad s⁻¹]  (= 0.9856°/day)

function averaged_eom(state::OsculatingState, I::PrincipalInertias,
                      shape::ShapeModel, cfg::PerturbationConfig;
                      t::Real = 0.0, n::Real = MEAN_MOTION,
                      N_φ::Int = 90, N_τ::Int = 180, P_SRP::Real = P_SRP_1AU)
    α, β, H, Id = state.α, state.β, state.H, state.Id
    ωe     = omega_e(H, Id)
    regime = classify_regime(Id, I)

    # ── YORP: averaged SRP torque (B&S 2021 Eqs. 16–19) ──
    # Backend per cfg.srp_backend: :numeric (M3, true-illumination quadrature)
    # or :analytic (M6, closed-form App. B averages; regular at the separatrix).
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

    Il, Ii, Is = I.Il, I.Ii, I.Is
    sβ = sin(β)

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

    u0   = to_svector(state0)
    p    = (I, shape, cfg, n, N_φ, N_τ, P_SRP)
    prob = ODEProblem(_averaged_rhs, u0, tspan, p)
    return solve(prob, solver; reltol = reltol, abstol = abstol, kwargs...)
end
