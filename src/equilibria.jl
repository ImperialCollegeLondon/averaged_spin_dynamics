"Locates fixed points of the tumbling-averaged EOM (Eqs. 35–39) and classifies 
their local stability, following B&S 2022 §V.B (pp. 1839–1840, Eqs. 42–47"
struct AveragedEquilibrium
    α::Float64
    β::Float64
    H::Float64
    Id::Float64
    J::Float64
    μ_over_J::Float64
    regime::Regime
    ωe::Float64
    residual::SVector{4,Float64}
end

function Base.show(io::IO, e::AveragedEquilibrium)
    print(io, "AveragedEquilibrium(α=$(round(rad2deg(e.α),digits=2))°, ",
              "β=$(round(rad2deg(e.β),digits=2))°, H=$(round(e.H,digits=4)), ",
              "Id=$(round(e.Id,digits=2)), J=$(round(e.J,digits=4)), ",
              "$(e.regime), |res|=$(round(norm(e.residual),sigdigits=3)))")
end

# Simple bisection on a bracketed sign change.
function _bisect(f, a::Real, b::Real; tol::Real = 1e-10, maxit::Int = 200)
    fa, fb = f(a), f(b)
    (isfinite(fa) && isfinite(fb)) || return nothing
    fa * fb > 0 && return nothing                     # no bracketed root
    for _ in 1:maxit
        m = 0.5 * (a + b)
        fm = f(m)
        (b - a) < tol && return m
        if fa * fm <= 0
            b, fb = m, fm
        else
            a, fa = m, fm
        end
    end
    return 0.5 * (a + b)
end

function _srp_torques(shape::ShapeModel, β::Real, ωe::Real, Id::Real,
                      I::PrincipalInertias, regime::Regime, cfg::PerturbationConfig;
                      N_φ::Int = 90, N_τ::Int = 180, P_SRP::Real = P_SRP_1AU)
    if cfg.srp_backend === :analytic
        return averaged_srp_torques_analytic(shape, β, Id, I, regime;
                                             P_SRP = P_SRP, σ = cfg.σ_branch)
    else
        return averaged_srp_torques(shape, β, ωe, Id, I, regime;
                                    N_φ = N_φ, N_τ = N_τ, P_SRP = P_SRP,
                                    σ = cfg.σ_branch)
    end
end
# Essentially another validity checker to check if there is an equilirium and will come back as nothing if there is no solution
function find_equilibrium(shape::ShapeModel, I::PrincipalInertias, β::Real;
                          μ_over_J::Real = 1e-3,
                          regime::Regime = SAM(),
                          cfg::PerturbationConfig = PerturbationConfig(
                              srp = true, dissipation = true, gravity_gradient = false,
                              srp_backend = :analytic),
                          n::Real = MEAN_MOTION,
                          J_ref::Real = 1.0,
                          Id_pad::Real = 1e-6,
                          N_φ::Int = 90, N_τ::Int = 180,
                          P_SRP::Real = P_SRP_1AU)
    μ_over_J <= 1e-3 || @warn "μ/J = $μ_over_J exceeds the averaged-dissipation validity ceiling 1e-3 s⁻¹  result is outside the model's stated range."

    # Band for the Eq. (44) bisection.  ωe is irrelevant to the torques, so any
    # positive placeholder works while solving for Ī_d.
    lo, hi = regime isa SAM ? (I.Ii, I.Is) : (I.Il, I.Ii)
    pad = Id_pad * (hi - lo)
    ωe_probe = 1.0

    Mz_of(Id) = _srp_torques(shape, β, ωe_probe, Id, I, regime, cfg;
                             N_φ = N_φ, N_τ = N_τ, P_SRP = P_SRP).M_H[3]

    Id_eq = _bisect(Mz_of, lo + pad, hi - pad)
    Id_eq === nothing && return nothing                            # Eq. (44) has no root

    at = _srp_torques(shape, β, ωe_probe, Id_eq, I, regime, cfg;
                      N_φ = N_φ, N_τ = N_τ, P_SRP = P_SRP)
    Mx, My = at.M_H[1], at.M_H[2]

    # Eq. (46) — negative signs are load-bearing (paper p.1839).
    α = atan(-Mx, -My / cos(β))

    # Eq. (43), falling back to Eq. (42) where sin α is ill-conditioned.
    H = abs(sin(α)) > 1e-8 ? -Mx / (n * sin(α)) : -My / (n * cos(α) * cos(β))
    (isfinite(H) && H > 0) || return nothing                       # no physical H̄

    # Eq. (45): İ_d,yorp must be negative for a J > 0 to balance it (h_d ≥ 0).
    Id_yorp = -(2Id_eq / H) * ((Id_eq - I.Ii) / I.Ii * at.azM[1] +
                               (Id_eq - I.Is) / I.Is * at.azM[2] +
                               (Id_eq - I.Il) / I.Il * at.azM[3])
    Id_yorp < 0 || return nothing

    ωe = H / Id_eq
    hd_ref = h_d(ωe, Id_eq, I, regime, μ_over_J * J_ref, J_ref)
    hd_ref > 0 || return nothing
    J = J_ref * (-Id_yorp) / hd_ref                                # exact: h_d ∝ J

    cfg_eq = PerturbationConfig(srp = true, dissipation = true,
                                gravity_gradient = false,
                                μ = μ_over_J * J, J = J,
                                srp_backend = cfg.srp_backend,
                                σ_branch = cfg.σ_branch)
    res = averaged_eom(OsculatingState(α, β, H, Id_eq), I, shape, cfg_eq;
                       n = n, N_φ = N_φ, N_τ = N_τ, P_SRP = P_SRP)
    return AveragedEquilibrium(α, β, H, Id_eq, J, μ_over_J, regime, ωe, res)
end