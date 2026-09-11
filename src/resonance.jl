
using DifferentialEquations

# Elliptic pieces in the paper's convention

function _res_Pi_incomplete(τ::Real, n_c::Real, k::Real; N::Int = 2048)
    τ <= 0 && return 0.0
    # 4-point Gauss–Legendre on each of N subintervals
    gx = (-0.8611363115940526, -0.3399810435848563,
           0.3399810435848563,  0.8611363115940526)
    gw = ( 0.3478548451374538,  0.6521451548625461,
           0.6521451548625461,  0.3478548451374538)
    h = τ / N
    acc = 0.0
    for j in 0:(N - 1)
        c = (j + 0.5) * h
        for (x, w) in zip(gx, gw)
            s = _sn(c + 0.5h * x, k)
            acc += w * h / (2 * (1 + n_c * s^2))
        end
    end
    return acc
end


_res_Pi_complete(n_c::Real, k::Real) = _res_Pi_incomplete(elliptic_K(k), n_c, k)
# eqs A9-A10
function _res_Pi(τ::Real, n_c::Real, k::Real)
    Kv   = elliptic_K(k)
    ΠK   = _res_Pi_complete(n_c, k)
    m_K  = floor(Int, τ / Kv)
    τrem = τ - m_K * Kv
    return iseven(m_K) ? m_K * ΠK + _res_Pi_incomplete(τrem, n_c, k) :
                         (m_K + 1) * ΠK - _res_Pi_incomplete(Kv - τrem, n_c, k)
end

# Fundamental periods

_res_params(ωe, Id, I, regime::Regime) =
    regime isa LAM ? torquefree_params_LAM(ωe, Id, I) :
                     torquefree_params_SAM(ωe, Id, I)

"""
    _res_Pphi(ωe, Id, I, regime) → Float64

Average precession period of b̂₃ about H, B&S Dec-2021 Eq. (A11):

    P_φ̄ = (2π I_l)/(ω_e I_d) · [1 − ((I_s − I_l)/I_s)·Π(K,n_c)/K]⁻¹

Equivalently φ̄̇ = (H/I_l)[1 − ((I_s−I_l)/I_s)Π/K]  (Eq. B1), with H = I_d ω_e.

"""
function _res_Pphi(ωe::Real, Id::Real, I::PrincipalInertias, regime::Regime)
    k, n_c, _, _, _, _ = _res_params(ωe, Id, I, regime)
    Kv = elliptic_K(k)
    Π  = _res_Pi_complete(n_c, k)
    return (2π * I.Il) / (ωe * Id) / (1 - (I.Is - I.Il) / I.Is * Π / Kv)
end


function _res_Ppsi(ωe::Real, Id::Real, I::PrincipalInertias, regime::Regime)
    k, _, τ_rate, _, _, _ = _res_params(ωe, Id, I, regime)
    return 4 * elliptic_K(k) / τ_rate
end


function period_ratio(Id::Real, I::PrincipalInertias, regime::Regime = classify_regime(Id, I))
    ωe = 1.0
    return _res_Ppsi(ωe, Id, I, regime) / _res_Pphi(ωe, Id, I, regime)
end

function resonance_Id(m::Integer, n::Integer, I::PrincipalInertias,
                      regime::Regime; tol::Real = 1e-10, maxit::Int = 200)
    target = m / n
    lo, hi = regime isa LAM ? (I.Il, I.Ii) : (I.Ii, I.Is)
    ε  = 1e-9 * (hi - lo)
    a, b = lo + ε, hi - ε
    fa = period_ratio(a, I, regime) - target
    fb = period_ratio(b, I, regime) - target
    sign(fa) == sign(fb) && return NaN
    local mid = 0.5 * (a + b)
    for _ in 1:maxit
        mid = 0.5 * (a + b)
        fm  = period_ratio(mid, I, regime) - target
        (abs(fm) < tol || (b - a) < tol * (hi - lo)) && return mid
        sign(fm) == sign(fa) ? (a = mid; fa = fm) : (b = mid)
    end
    return mid
end

# Appendix B: I_d-derivatives (derived, not glyph-copied) - check this
# (A6)/(A15)
function _dk_dId(Id::Real, I::PrincipalInertias, regime::Regime)
    Il, Ii, Is = I.Il, I.Ii, I.Is
    if regime isa LAM
        return (Is - Il) / (2 * (Is - Id)) *
               sqrt((Is - Ii) / ((Ii - Il) * (Id - Il) * (Is - Id)))
    else
        return -(Is - Il) / (2 * (Id - Il)) *
               sqrt((Ii - Il) / ((Is - Ii) * (Id - Il) * (Is - Id)))
    end
end

_dn_dId(Id::Real, I::PrincipalInertias, regime::Regime) =
    regime isa LAM ? 0.0 :
    -(I.Il / I.Is) * (I.Is - I.Il) / (Id - I.Il)^2

"""
    _dK_dk(k) → Float64

∂K/∂k = (E − k′²K)/(k k′²),  k′² = 1 − k²   (Eq. B8; standard identity).
"""
function _dK_dk(k::Real)
    m  = k^2
    kp2 = 1 - m
    return (E(m) - kp2 * K(m)) / (k * kp2)
end
# B5
function _dPi_dk(n_c::Real, k::Real)
    m   = k^2
    kp2 = 1 - m
    Π   = _res_Pi_complete(n_c, k)
    return k * (E(m) - kp2 * Π) / (kp2 * (m + n_c))
end
# B6
function _dPi_dn(n_c::Real, k::Real)
    m = k^2
    Π = _res_Pi_complete(n_c, k)
    return (-(m + n_c) * K(m) + n_c * E(m) + (m - n_c^2) * Π) /
           (2 * n_c * (n_c + 1) * (n_c + m))
end
# A12/A17
function _c_factor(Id::Real, I::PrincipalInertias, regime::Regime)
    Il, Ii, Is = I.Il, I.Ii, I.Is
    num = regime isa LAM ? (Ii - Il) * (Is - Id) : (Is - Ii) * (Id - Il)
    return sqrt(num / (Id * Il * Ii * Is))
end
# B20
function _cdot(Id::Real, I::PrincipalInertias, regime::Regime, İd::Real)
    Il, Ii, Is = I.Il, I.Ii, I.Is
    c = _c_factor(Id, I, regime)
    d = regime isa LAM ? -(Ii - Il) / (Id^2 * Il * Ii) :
                          (Is - Ii) / (Id^2 * Ii * Is)
    return d / (2c) * İd
end

# Appendix B: φ̄̈ and τ̈_r

# (Eq. B1)

function phibar_dot(H::Real, Id::Real, I::PrincipalInertias, regime::Regime)
    k, n_c, _, _, _, _ = _res_params(H / Id, Id, I, regime)
    Kv = elliptic_K(k)
    return (H / I.Il) * (1 - (I.Is - I.Il) / I.Is * _res_Pi_complete(n_c, k) / Kv)
end
# Eq B2, B3-B8
function phibar_ddot(H::Real, Id::Real, I::PrincipalInertias, regime::Regime,
                     Ḣ::Real, İd::Real)
    k, n_c, _, _, _, _ = _res_params(H / Id, Id, I, regime)
    Kv  = elliptic_K(k)
    Π   = _res_Pi_complete(n_c, k)
    dk  = _dk_dId(Id, I, regime)
    dn  = _dn_dId(Id, I, regime)
    Π̇   = (_dPi_dk(n_c, k) * dk + _dPi_dn(n_c, k) * dn) * İd      # B4
    K̇   = _dK_dk(k) * dk * İd                                     # B7
    dΠK = (Kv * Π̇ - Π * K̇) / Kv^2                                 # B3
    return (Ḣ / I.Il) * (1 - (I.Is - I.Il) / I.Is * Π / Kv) -
           (H * (I.Is - I.Il) / (I.Il * I.Is)) * dΠK               # B2
end
# B11/B17
function taur_dot(H::Real, Id::Real, I::PrincipalInertias, regime::Regime)
    k, _, _, _, _, _ = _res_params(H / Id, Id, I, regime)
    return π * H / (2 * elliptic_K(k)) * _c_factor(Id, I, regime)
end
# B13/B19
function taur_ddot(H::Real, Id::Real, I::PrincipalInertias, regime::Regime,
                   Ḣ::Real, İd::Real)
    k, _, _, _, _, _ = _res_params(H / Id, Id, I, regime)
    Kv = elliptic_K(k)
    c  = _c_factor(Id, I, regime)
    ċ  = _cdot(Id, I, regime, İd)
    K̇  = _dK_dk(k) * _dk_dId(Id, I, regime) * İd
    return π / (2Kv) * (Ḣ * c + H * (ċ - c * K̇ / Kv))
end
# Eqs 11, 16
gamma_ddot(m::Integer, n::Integer, H::Real, Id::Real, I::PrincipalInertias,
           regime::Regime, Ḣ::Real, İd::Real) =
    n * phibar_ddot(H, Id, I, regime, Ḣ, İd) -
    m * taur_ddot(H, Id, I, regime, Ḣ, İd)

# Torque-free attitude over the resonant cycle

function _res_phi(τ::Real, τ_o::Real, t::Real, t_o::Real, φ_o::Real,
                  H::Real, Id::Real, I::PrincipalInertias, regime::Regime)
    Il, Ii, Is = I.Il, I.Ii, I.Is
    k, n_c, _, _, _, _ = _res_params(H / Id, Id, I, regime)
    den = regime isa LAM ? (Ii - Il) * (Is - Id) : (Is - Ii) * (Id - Il)
    S   = sqrt(Ii * Id / (Il * Is * den))
    return φ_o + (H / Il) * (t - t_o) -
           (Is - Il) * S * (_res_Pi(τ, n_c, k) - _res_Pi(τ_o, n_c, k))
end


function _res_attitude(t::Real, γ::Real, m::Integer, n::Integer,
                       H::Real, Id::Real, I::PrincipalInertias, regime::Regime;
                       σ::Integer = 1)
    ωe = H / Id
    _, _, τ_rate, _, _, _ = _res_params(ωe, Id, I, regime)
    τ  = τ_rate * t                                    # τ_o = 0 (Eq. 18)
    az = direction_cosines_az(τ, ωe, Id, I, regime; σ = σ)
    θ  = acos(clamp(az[3], -1.0, 1.0))
    ψ  = atan(az[1], az[2])
    φ  = _res_phi(τ, 0.0, t, 0.0, γ / n, H, Id, I, regime)
    return R3(ψ) * R1(θ) * R3(φ)
end

# Resonance-averaged torques (Eq. 12)

function resonance_averaged_torques(shape::ShapeModel, β::Real, γ::Real,
                                    Id::Real, I::PrincipalInertias, regime::Regime,
                                    m::Integer, n::Integer;
                                    N_t::Int = 720, P_SRP::Real = P_SRP_1AU,
                                    σ::Integer = 1)
    regime isa Separatrix &&
        throw(ArgumentError("resonance_averaged_torques undefined on the separatrix"))
    ωe  = 1.0                       # torque average is ω_e-independent; H scales out
    H   = Id * ωe
    P_r = n * _res_Ppsi(ωe, Id, I, regime)
    û_H = SVector{3,Float64}(-sin(β), 0.0, cos(β))
    sumM  = SVector{3,Float64}(0, 0, 0)
    sumaz = SVector{3,Float64}(0, 0, 0)
    Δt = P_r / N_t
    for j in 0:(N_t - 1)
        t  = (j + 0.5) * Δt
        BH = _res_attitude(t, γ, m, n, H, Id, I, regime; σ = σ)
        τ  = _res_params(ωe, Id, I, regime)[3] * t
        az = direction_cosines_az(τ, ωe, Id, I, regime; σ = σ)
        M_B = srp_torque(shape, BH * û_H; P_SRP = P_SRP)
        sumM  += BH' * M_B
        sumaz += az .* M_B
    end
    return AveragedTorques(sumM / N_t, sumaz / N_t)
end

# (β, γ) lookup table


struct ResonanceTable
    m::Int
    n::Int
    Id::Float64
    βs::Vector{Float64}
    γs::Vector{Float64}
    Mx::Matrix{Float64}
    My::Matrix{Float64}
    Mz::Matrix{Float64}
end

function resonance_table(shape::ShapeModel, I::PrincipalInertias,
                         m::Integer, n::Integer, regime::Regime;
                         Δβ::Real = deg2rad(1.0), Δγ::Real = deg2rad(1.0),
                         N_t::Int = 720, P_SRP::Real = P_SRP_1AU, σ::Integer = 1)
    Id = resonance_Id(m, n, I, regime)
    isnan(Id) && error("resonance_table: P_ψ/P_φ̄ = $m/$n is not attainable in " *
                       "$(regime isa LAM ? "LAM" : "SAM") for these inertias")
    βs = collect(Δβ:Δβ:(π - Δβ))
    γs = collect(0.0:Δγ:(2π - Δγ))
    Mx = zeros(length(βs), length(γs))
    My = similar(Mx); Mz = similar(Mx)
    for (i, β) in enumerate(βs), (j, γ) in enumerate(γs)
        at = resonance_averaged_torques(shape, β, γ, Id, I, regime, m, n;
                                        N_t = N_t, P_SRP = P_SRP, σ = σ)
        Mx[i, j] = at.M_H[1]; My[i, j] = at.M_H[2]; Mz[i, j] = at.M_H[3]
    end
    return ResonanceTable(Int(m), Int(n), Id, βs, γs, Mx, My, Mz)
end


# interpolation

function interp_table(T::ResonanceTable, β::Real, γ::Real)
    βc = clamp(β, T.βs[1], T.βs[end])
    Δβ = T.βs[2] - T.βs[1]; Δγ = T.γs[2] - T.γs[1]
    i  = clamp(floor(Int, (βc - T.βs[1]) / Δβ) + 1, 1, length(T.βs) - 1)
    γw = mod(γ, 2π)
    j  = clamp(floor(Int, (γw - T.γs[1]) / Δγ) + 1, 1, length(T.γs))
    j2 = j == length(T.γs) ? 1 : j + 1
    tβ = (βc - T.βs[i]) / Δβ
    tγ = (γw - T.γs[j]) / Δγ
    bil(A) = (1 - tβ) * ((1 - tγ) * A[i, j] + tγ * A[i, j2]) +
                   tβ * ((1 - tγ) * A[i+1, j] + tγ * A[i+1, j2])
    return SVector{3,Float64}(bil(T.Mx), bil(T.My), bil(T.Mz))
end


# Resonance-averaged EOM (Eqs. 13–17)

function resonance_eom(u, T::ResonanceTable, I::PrincipalInertias, regime::Regime;
                       n_s::Real = MEAN_MOTION)
    α, β, H, γ, γ̇ = u[1], u[2], u[3], u[4], u[5]
    Id = T.Id
    M  = interp_table(T, β, γ)
    M̄x, M̄y, M̄z = M[1], M[2], M[3]

    α̇ = (M̄y + H * n_s * cos(α) * cos(β)) / (H * sin(β))          # Eq. 13
    β̇ = (M̄x + H * n_s * sin(α)) / H                              # Eq. 14
    Ḣ = M̄z                                                        # Eq. 15

    # İ_d — Eq. (17): (2I_d/H) M̄ · (𝕀 − I_d[I]⁻¹) Ĥ, with Ĥ = ẑ in the H frame
    d  = inertia_body(I)
    Ĥ  = SVector{3,Float64}(0.0, 0.0, 1.0)
    v  = SVector{3,Float64}((1 - Id / d[1]) * Ĥ[1],
                            (1 - Id / d[2]) * Ĥ[2],
                            (1 - Id / d[3]) * Ĥ[3])
    İd = (2Id / H) * dot(M, v)

    γ̈ = gamma_ddot(T.m, T.n, H, Id, I, regime, Ḣ, İd)             # Eq. 16
    return SVector{5,Float64}(α̇, β̇, Ḣ, γ̇, γ̈)
end

_resonance_rhs(u, p, t) = resonance_eom(u, p[1], p[2], p[3]; n_s = p[4])

function propagate_resonance(I::PrincipalInertias, T::ResonanceTable, u0, tspan;
                             regime::Regime = classify_regime(T.Id, I),
                             n_s::Real = MEAN_MOTION,
                             solver = Tsit5(), reltol = 1e-8, abstol = 1e-10,
                             kwargs...)
    prob = ODEProblem(_resonance_rhs, SVector{5,Float64}(u0...), tspan,
                      (T, I, regime, n_s))
    return solve(prob, solver; reltol = reltol, abstol = abstol, kwargs...)
end