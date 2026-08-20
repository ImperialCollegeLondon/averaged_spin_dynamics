
# Full-model slug EOM (truth)

function slug_rates(β, ω, σ, Idiag, μ::Real, J::Real, M)
    I1, I2, I3 = Idiag
    Iω = SVector{3,Float64}(I1*ω[1], I2*ω[2], I3*ω[3])
    wIw = cross(ω, Iω)                                   # ω×(Iω) = [ω̃][I]ω
    ω̇ = SVector{3,Float64}(
        (-wIw[1] + μ*σ[1] + M[1]) / I1,
        (-wIw[2] + μ*σ[2] + M[2]) / I2,
        (-wIw[3] + μ*σ[3] + M[3]) / I3,
    )
    σ̇ = -ω̇ - cross(ω, σ) - (μ/J)*σ
    return quat_kinematics(β, ω), ω̇, σ̇
end

# 10-vector state u = (β₀..β₃, ω₁..ω₃, σ₁..σ₃) for DifferentialEquations.
function _slug_rhs(u, p, t)
    Idiag, μ, J, torque_fn = p
    β = SVector{4,Float64}(u[1], u[2], u[3], u[4])
    ω = SVector{3,Float64}(u[5], u[6], u[7])
    σ = SVector{3,Float64}(u[8], u[9], u[10])
    M = torque_fn(β, t)
    dβ, dω, dσ = slug_rates(β, ω, σ, Idiag, μ, J, M)
    return SVector{10,Float64}(dβ[1],dβ[2],dβ[3],dβ[4], dω[1],dω[2],dω[3], dσ[1],dσ[2],dσ[3])
end

# eq 3-4
function propagate_full_slug(I::PrincipalInertias, β0, ω0, σ0, tspan;
                             μ::Real, J::Real, torque = zero_torque,
                             solver = Vern9(), reltol = 1e-11, abstol = 1e-11, kwargs...)
    Idiag = inertia_body(I)
    β = quat_normalize(β0)
    u0 = SVector{10,Float64}(β[1],β[2],β[3],β[4], ω0[1],ω0[2],ω0[3], σ0[1],σ0[2],σ0[3])
    prob = ODEProblem(_slug_rhs, u0, tspan, (Idiag, μ, J, torque))
    return solve(prob, solver; reltol = reltol, abstol = abstol, kwargs...)
end

# Torque-free ω-averages needed by the t_ij (Eqs. A1–A9, 17)
# e13 B&S 2022
struct OmegaAverages
    w1::Float64; w2::Float64; w3::Float64          # ⟨ω₁²⟩, ⟨ω₂²⟩, ⟨ω₃²⟩
    w12::Float64; w13::Float64; w23::Float64       # ⟨ω₁²ω₂²⟩, ⟨ω₁²ω₃²⟩, ⟨ω₂²ω₃²⟩
end


function omega_averages(ωe::Real, Id::Real, I::PrincipalInertias, regime::Regime;
                        N_τ::Int = 1024)
    # Cap the modulus just below elliptic_function_averages' k→1 limit branch.
    # The exact limit zeroes ⟨ω₂²⟩, ⟨ω₃²⟩ and every cross moment, making the
    # t_ij system (Eqs. 14–16) rank-deficient (T\rhs → NaN) even though h_d has
    # a finite, side-independent limit at the separatrix (verified: h_d varies
    # by <1e-6 relative across Id = Ii ± 1e-6 kg·m²). The cap keeps the solve
    # full-rank for Id within ~1e-7 kg·m² of Ii, including Separatrix() exactly,
    # at an error far below that flatness.
    k_cap = sqrt(1 - 2e-9)
    if regime isa LAM
        k, _, _, B₁, B₂, B₃ = torquefree_params_LAM(ωe, Id, I)
        e = elliptic_function_averages(min(k, k_cap))
        # LAM roles: ω₁ ∝ sn, ω₂ ∝ cn, ω₃ ∝ dn (Eq. A2)
        return OmegaAverages(B₁^2*e.sn2, B₂^2*e.cn2, B₃^2*e.dn2,
                             B₁^2*B₂^2*e.sn2cn2, B₁^2*B₃^2*e.sn2dn2,
                             B₂^2*B₃^2*e.cn2dn2)
    else   # SAM and Separatrix (the SAM parameters are regular at Id = Ii, k = 1)
        k, _, _, B₁, B₂, B₃ = torquefree_params_SAM(ωe, Id, I)
        e = elliptic_function_averages(min(k, k_cap))
        # SAM roles: ω₁ ∝ sn, ω₂ ∝ dn, ω₃ ∝ cn (Eq. A11)
        return OmegaAverages(B₁^2*e.sn2, B₂^2*e.dn2, B₃^2*e.cn2,
                             B₁^2*B₂^2*e.sn2dn2, B₁^2*B₃^2*e.sn2cn2,
                             B₂^2*B₃^2*e.cn2dn2)
    end
end

function omega_averages_numeric(ωe::Real, Id::Real, I::PrincipalInertias, regime::Regime;
                                N_τ::Int = 1024)
    k = regime isa LAM ? torquefree_params_LAM(ωe, Id, I)[1] :
                         torquefree_params_SAM(ωe, Id, I)[1]
    Kval = elliptic_K(k)
    Δτ = 4Kval / N_τ
    w1 = w2 = w3 = w12 = w13 = w23 = 0.0
    for j in 0:(N_τ-1)
        τ = (j + 0.5) * Δτ
        ω = torque_free_body_rates(τ, ωe, Id, I, regime)
        s1 = ω[1]^2; s2 = ω[2]^2; s3 = ω[3]^2
        w1 += s1; w2 += s2; w3 += s3
        w12 += s1*s2; w13 += s1*s3; w23 += s2*s3
    end
    return OmegaAverages(w1/N_τ, w2/N_τ, w3/N_τ, w12/N_τ, w13/N_τ, w23/N_τ)
end


# app A coefficients

function t_coefficients(avg::OmegaAverages, I::PrincipalInertias, μ::Real, J::Real)
    Ii, Is, Il = I.Ii, I.Is, I.Il
    w1, w2, w3 = avg.w1, avg.w2, avg.w3
    w12, w13, w23 = avg.w12, avg.w13, avg.w23
    r = (μ/J)^2

    # NOTE — CORRECTED PAPER TYPO in the diagonal coefficients (A1, A5, A8).
    # The paper prints the ⟨ω_j²ω_k²⟩ factor as (I_j²+I_k²−I_jI_k)/I_i², but t_ii ≡
    # ∂²⟨D²⟩/∂x_i² requires the PERFECT SQUARE (I_j²+I_k²−2I_jI_k)/I_i² = (I_j−I_k)²/I_i²
    # (matching the same factor in A4/A7/A9). Verified to machine precision against the
    # numeric ⟨D²⟩ Hessian (`dbar2`); the paper form is off by up to ~45×. See
    # docs/BS2022_equations.md [[FLAG-TYPO A1/A5/A8]].
    t11 = 2*( w12 + w13 + w23*(Il - Is)^2/Ii^2 + w1*r )                           # A1 (corrected)
    t12 = 2*( w23*(Il - Is)/Ii - w12 - w13*(Ii - Il)/Is )                         # A2
    t13 = 2*( -w13 - w23*(Il - Is)/Ii - w12*(Ii - Is)/Il )                        # A3
    t14 = 2*( w23*(Il - Is)^2/Ii^2 - w12*(Ii - Is)/Il - w13*(Ii - Il)/Is )        # A4
    t22 = 2*( w12 + w23 + w13*(Ii - Il)^2/Is^2 + w2*r )                           # A5 (corrected)
    t23 = 2*( w12*(Ii - Is)/Il - w23 + w13*(Ii - Il)/Is )                         # A6
    t24 = 2*( w13*(Ii - Il)^2/Is^2 + w23*(Il - Is)/Ii + w12*(Ii - Is)/Il )        # A7
    t33 = 2*( w13 + w23 + w12*(Ii - Is)^2/Il^2 + w3*r )                           # A8 (corrected)
    t34 = 2*( w12*(Ii - Is)^2/Il^2 - w23*(Il - Is)/Ii + w13*(Ii - Il)/Is )        # A9

    T = @SMatrix [t11 t12 t13; t12 t22 t23; t13 t23 t33]
    rhs = SVector{3,Float64}(-t14, -t24, -t34)
    return T, rhs
end

# eqs 14-16
function solve_A(ωe::Real, Id::Real, I::PrincipalInertias, regime::Regime,
                 μ::Real, J::Real; N_τ::Int = 1024)
    avg = omega_averages(ωe, Id, I, regime; N_τ = N_τ)
    T, rhs = t_coefficients(avg, I, μ, J)
    return T \ rhs, avg
end

# Averaged dissipation rate and h_d (Eqs. 17–18)

averaged_dissipation_rate(abc, avg::OmegaAverages, μ::Real) =
    -μ * (abc[1]^2*avg.w1 + abc[2]^2*avg.w2 + abc[3]^2*avg.w3)

"""
    h_d(ωe, Id, I, regime, μ, J; N_τ=1024) → Float64   (B&S 2022 Eq. 18)

Tumbling-averaged dissipative İ_d contribution:  h_d = −2⟨Ṫ_d⟩/ω_e² ≥ 0.
Dissipation can only increase Ī_d (push toward uniform rotation about the max-inertia axis).
"""
function h_d(ωe::Real, Id::Real, I::PrincipalInertias, regime::Regime,
             μ::Real, J::Real; N_τ::Int = 1024)
    abc, avg = solve_A(ωe, Id, I, regime, μ, J; N_τ = N_τ)
    Ṫd = averaged_dissipation_rate(abc, avg, μ)
    return -2Ṫd / ωe^2
end