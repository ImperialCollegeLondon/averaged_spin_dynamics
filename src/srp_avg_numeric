
# call vectors for averaged torques
struct AveragedTorques
    M_H::SVector{3,Float64}   # (M̄ₓ, M̄_y, M̄_z), H frame
    azM::SVector{3,Float64}   # (⟨a_{z1}M₁⟩, ⟨a_{z2}M₂⟩, ⟨a_{z3}M₃⟩), body frame
end

# calculate averaged srp torque numerically rather than using appendix B
function averaged_srp_torques(shape::ShapeModel, β::Real, ωe::Real, Id::Real,
                              I::PrincipalInertias, regime::Regime;
                              N_φ::Int = 90, N_τ::Int = 180,
                              P_SRP::Real = P_SRP_1AU, σ::Integer = 1)
    regime isa Separatrix &&
        throw(ArgumentError("averaged_srp_torques undefined on the separatrix (Id = Ii); " *
                            "the torque-free solution degenerates there."))

    # Sun direction in the H frame (B&S 2021 p.754): û_H = [HO] Ẑ_O = (−sinβ, 0, cosβ).
    û_H = SVector{3,Float64}(-sin(β), 0.0, cos(β))

    # Elliptic modulus → averaging interval 4K in τ (one P_ψ), per Eq. (15).
    k = regime isa LAM ? torquefree_params_LAM(ωe, Id, I)[1] :
                         torquefree_params_SAM(ωe, Id, I)[1]
    Kval = elliptic_K(k)
    Δτ = 4Kval / N_τ
    Δφ = 2π / N_φ

    sumM_H  = SVector{3,Float64}(0.0, 0.0, 0.0)
    sum_azM = SVector{3,Float64}(0.0, 0.0, 0.0)

    for kτ in 0:(N_τ - 1)
        τ  = (kτ + 0.5) * Δτ                                   # midpoint in τ
        az = direction_cosines_az(τ, ωe, Id, I, regime; σ = σ) # (a_{z1},a_{z2},a_{z3}), unit
        # Recover nutation θ, spin ψ from Eq. (A1): cosθ = a_{z3}, (sinθsinψ,sinθcosψ)=(a_{z1},a_{z2}).
        θ = acos(clamp(az[3], -1.0, 1.0))
        ψ = atan(az[1], az[2])
        A_τ = R3(ψ) * R1(θ)                                    # [BH] = A_τ R3(φ), constant over φ (Eq. 1)
        for jφ in 0:(N_φ - 1)
            φ   = (jφ + 0.5) * Δφ                              # midpoint in φ
            BH  = A_τ * R3(φ)                                  # H→B attitude
            û_B = BH * û_H                                     # satellite→Sun in body frame
            M_B = srp_torque(shape, û_B; P_SRP = P_SRP)        # (M₁,M₂,M₃) body (Eqs. 11–12)
            sumM_H  += BH' * M_B                               # (M̄ₓ,M̄_y,M̄_z) in H frame
            sum_azM += az .* M_B                               # (a_{z1}M₁, a_{z2}M₂, a_{z3}M₃)
        end
    end
    invN = 1.0 / (N_φ * N_τ)
    return AveragedTorques(sumM_H * invN, sum_azM * invN)
end
# return state
function averaged_srp_torques(shape::ShapeModel, state::OsculatingState,
                              I::PrincipalInertias; kwargs...)
    ωe = omega_e(state.H, state.Id)
    regime = classify_regime(state.Id, I)
    return averaged_srp_torques(shape, state.β, ωe, state.Id, I, regime; kwargs...)
end
