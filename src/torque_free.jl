using Elliptic: K, E
using Elliptic.Jacobi: sn, cn, dn

_K(k::Real)     = K(k^2)
_sn(u::Real, k::Real) = sn(u, k^2)
_cn(u::Real, k::Real) = cn(u, k^2)
_dn(u::Real, k::Real) = dn(u, k^2)

elliptic_K(k::Real) = _K(k)

#eq A7 and A8 from B&S 2021
function elliptic_Pi_branchtracked(τ::Real, n_param::Real, k::Real)
    Kval  = elliptic_K(k)
    Pi_K  = elliptic_Pi_complete(n_param, k)          # Π(K; n) — complete
    m     = floor(Int, τ / Kval)
    τ_rem = τ - m * Kval
    if iseven(m)
        # Eq. (A7)
        return m * Pi_K + elliptic_Pi_incomplete(τ_rem, n_param, k)
    else
        # Eq. (A8):  (m+1)K − τ = K − τ_rem
        return (m + 1) * Pi_K - elliptic_Pi_incomplete(Kval - τ_rem, n_param, k)
    end
end

# Define elliptic integrals using Carlson algorithm as seen in B&S ref [22]
function carlson_RF(x::Real, y::Real, z::Real)
    x, y, z = Float64(x), Float64(y), Float64(z)
    tol = 3e-8
    for _ in 1:50
        lam = sqrt(x*y) + sqrt(y*z) + sqrt(x*z)
        x = (x + lam) / 4; y = (y + lam) / 4; z = (z + lam) / 4
        avg = (x + y + z) / 3
        dx, dy, dz = 1 - x/avg, 1 - y/avg, 1 - z/avg
        max(abs(dx), abs(dy), abs(dz)) < tol && begin
            e2 = dx*dy - dz^2
            e3 = dx*dy*dz
            return (1 + e2*(-1/10 + 3e3/44 - e2^2/24) + e3/14) / sqrt(avg)
        end
    end
    error("carlson_RF failed to converge for x=$x, y=$y, z=$z")
end

function carlson_RJ(x::Real, y::Real, z::Real, p::Real)
    x, y, z, p = Float64(x), Float64(y), Float64(z), Float64(p)
    tol = 3e-8; sum = 0.0; fac = 1.0
    for _ in 1:50
        lam = sqrt(x*y) + sqrt(y*z) + sqrt(x*z)
        α_c = (p*(sqrt(x)+sqrt(y)+sqrt(z)) + sqrt(x*y*z))^2
        β_c = p*(p+lam)^2
        sum += fac * carlson_RC(α_c, β_c)
        fac /= 4
        x = (x + lam) / 4; y = (y + lam) / 4; z = (z + lam) / 4; p = (p + lam) / 4
        avg = (x + y + z + 2p) / 5
        dx = 1 - x/avg; dy = 1 - y/avg; dz = 1 - z/avg; dp = 1 - p/avg
        max(abs(dx), abs(dy), abs(dz), abs(dp)) < tol && begin
            xyz = dx*dy + dy*dz + dz*dx
            p2  = dp^2
            e2  = xyz - 3p2
            e3  = dx*dy*dz + 2*xyz*dp - 3p2*dp
            e4  = (2*dx*dy*dz + xyz*dp + 3p2*dp)*dp
            e5  = dx*dy*dz*dp^2
            val = 1 + e2*(-3/14) + e3*(1/6) + e2^2*(9/88) +
                  e4*(-3/22) + e2*e3*(-9/52) + e5*(3/26)
            return (3*sum + fac*val/avg^(3/2)) * 3/2 * fac
        end
    end
    error("carlson_RJ failed to converge for x=$x, y=$y, z=$z, p=$p")
end

function carlson_RC(x::Real, y::Real)
    x, y = Float64(x), Float64(y)
    if y > 0
        if x < y
            return acos(sqrt(x / y)) / sqrt(y - x)
        elseif x == y
            return 1.0 / sqrt(x)
        else
            return acosh(sqrt(x / y)) / sqrt(x - y)
        end
    else
        return sqrt(x / (x - y)) * carlson_RC(x - y, -y)
    end
end

#upper bound of Pi to gain an exact number for K to feed into P_ϕ, so is the period-averaged value

function elliptic_Pi_complete(n::Real, k::Real)
    # Use Carlson R_F and R_J
    # Π(n, m) = R_F(0,1-m,1) + (n/3) R_J(0,1-m,1,1-n)
    m  = k^2
    x0 = 0.0
    y0 = 1.0 - m
    z0 = 1.0
    p0 = 1.0 - n
    RF = carlson_RF(x0, y0, z0)
    RJ = carlson_RJ(x0, y0, z0, p0)
    return RF + (n / 3.0) * RJ
end
# trajectory following value
function elliptic_Pi_incomplete(u::Real, n::Real, k::Real)
    abs(u) < 1e-15 && return 0.0
    m = k^2
    sn_u = _sn(u, k)
    abs(sn_u) < 1e-15 && return elliptic_Pi_complete(n, k)
    c  = 1.0 / sn_u^2
    RF = carlson_RF(c - 1.0, c - m, c)
    RJ = carlson_RJ(c - 1.0, c - m, c, c - n)
    return RF + (n / 3.0) * RJ
end


# torque free solutions
#Long axis mode
function torquefree_params_LAM(ωe::Real, Id::Real, I::PrincipalInertias)
    Il, Ii, Is = I.Il, I.Ii, I.Is

    # Body-rate amplitudes (Eq. A2): ω₁=B₁ sn, ω₂=B₂ cn, ω₃=B₃ dn
    B₁ = ωe * sqrt(Id * (Id - Il) / (Ii * (Ii - Il)))   # b̂₁ → Ii
    B₂ = ωe * sqrt(Id * (Id - Il) / (Is * (Is - Il)))   # b̂₂ → Is
    B₃ = ωe * sqrt(Id * (Is - Id) / (Il * (Is - Il)))   # b̂₃ → Il

    # Scaled-time rate dτ/dt (Eq. A3)
    τ_rate = ωe * sqrt(Id * (Ii - Il) * (Is - Id) / (Il * Ii * Is))

    # Elliptic modulus and third characteristic (Eq. A5)
    k²      = (Is - Ii) * (Id - Il) / ((Ii - Il) * (Is - Id))
    k       = sqrt(clamp(k², 0.0, 1.0))
    n_param = (Il / Is) * (Is - Ii) / (Ii - Il)

    return k, n_param, τ_rate, B₁, B₂, B₃
end
#short axis mode
function torquefree_params_SAM(ωe::Real, Id::Real, I::PrincipalInertias)
    Il, Ii, Is = I.Il, I.Ii, I.Is

    # Body-rate amplitudes (Eq. A11): ω₁=B₁ sn, ω₂=B₂ dn, ω₃=B₃ cn
    B₁ = ωe * sqrt(Id * (Is - Id) / (Ii * (Is - Ii)))   # b̂₁ → Ii
    B₂ = ωe * sqrt(Id * (Id - Il) / (Is * (Is - Il)))   # b̂₂ → Is
    B₃ = ωe * sqrt(Id * (Is - Id) / (Il * (Is - Il)))   # b̂₃ → Il

    # Scaled-time rate dτ/dt (Eq. A12)
    τ_rate = ωe * sqrt(Id * (Is - Ii) * (Id - Il) / (Il * Ii * Is))

    # Elliptic modulus and third characteristic (Eq. A13)
    k²      = (Ii - Il) * (Is - Id) / ((Is - Ii) * (Id - Il))
    k       = sqrt(clamp(k², 0.0, 1.0))
    n_param = (Il / Is) * (Is - Id) / (Id - Il)

    return k, n_param, τ_rate, B₁, B₂, B₃
end

#body rates/ ang vel components seen in A2 and A11
#LAM
function torque_free_body_rates(τ::Real, ωe::Real, Id::Real,
                                 I::PrincipalInertias, ::LAM; σ::Integer = 1)
    abs(σ) == 1 || throw(ArgumentError("branch σ must be +1 or -1"))
    k, _, _, B₁, B₂, B₃ = torquefree_params_LAM(ωe, Id, I)
    ω₁ = B₁ * _sn(τ, k)
    ω₂ = B₂ * _cn(τ, k)
    ω₃ = σ * B₃ * _dn(τ, k)
    SVector{3,Float64}(ω₁, ω₂, ω₃)
end
#SAM
function torque_free_body_rates(τ::Real, ωe::Real, Id::Real,
                                 I::PrincipalInertias, ::SAM; σ::Integer = 1)
    abs(σ) == 1 || throw(ArgumentError("branch σ must be +1 or -1"))
    k, _, _, B₁, B₂, B₃ = torquefree_params_SAM(ωe, Id, I)
    ω₁ = B₁ * _sn(τ, k)
    ω₂ = σ * B₂ * _dn(τ, k)
    ω₃ = B₃ * _cn(τ, k)
    SVector{3,Float64}(ω₁, ω₂, ω₃)
end
# regime classification from rates
function torque_free_rates(τ::Real, ωe::Real, Id::Real, I::PrincipalInertias;
                           σ::Integer = 1)
    regime = classify_regime(Id, I)
    torque_free_body_rates(τ, ωe, Id, I, regime; σ = σ)
end

# tumbling periods i.e. length for full tumble
#LAM
function tumbling_periods(ωe::Real, Id::Real, I::PrincipalInertias, ::LAM)
    Il, Ii, Is = I.Il, I.Ii, I.Is
    k, n, τ_rate, _, _, _ = torquefree_params_LAM(ωe, Id, I)
    Kval = elliptic_K(k)
    Πc   = elliptic_Pi_complete(n, k)                       # Π(K; n)
    P_φ  = (2π / ωe) * (Il / Id) * (1 - (Is - Il) / Is * Πc / Kval)  # A9
    P_ψ  = 4Kval / τ_rate                                   # A10
    return P_φ, P_ψ
end
#SAM
function tumbling_periods(ωe::Real, Id::Real, I::PrincipalInertias, ::SAM)
    Il, Ii, Is = I.Il, I.Ii, I.Is
    k, n, τ_rate, _, _, _ = torquefree_params_SAM(ωe, Id, I)
    Kval = elliptic_K(k)
    Πc   = elliptic_Pi_complete(n, k)
    P_φ  = (2π / ωe)*(Il / Id) * (1 - (Is - Il) / Is * Πc / Kval)  # A9 with A13 n
    P_ψ  = 4Kval / τ_rate                                   # A15
    return P_φ, P_ψ
end

function tumbling_periods(ωe::Real, Id::Real, I::PrincipalInertias)
    regime = classify_regime(Id, I)
    tumbling_periods(ωe, Id, I, regime)
end

# Direction cosines

function direction_cosines_az(τ::Real, ωe::Real, Id::Real,
                               I::PrincipalInertias, regime::Regime;
                               σ::Integer = 1)
    p = az_amplitudes(ωe, Id, I, regime; σ = σ)
    _az_at(τ, p.k, p.A, p.mode)
end
# amplitudes found one time to reduce need for calculations each iteration
function az_amplitudes(ωe::Real, Id::Real, I::PrincipalInertias, ::LAM;
                       σ::Integer = 1)
    abs(σ) == 1 || throw(ArgumentError("branch σ must be +1 or -1"))
    k, _, _, B₁, B₂, B₃ = torquefree_params_LAM(ωe, Id, I)
    H = Id * ωe
    (k = k, mode = :LAM,
     A = SVector{3,Float64}(I.Ii * B₁ / H, I.Is * B₂ / H, σ * I.Il * B₃ / H))
end
function az_amplitudes(ωe::Real, Id::Real, I::PrincipalInertias, ::SAM;
                       σ::Integer = 1)
    abs(σ) == 1 || throw(ArgumentError("branch σ must be +1 or -1"))
    k, _, _, B₁, B₂, B₃ = torquefree_params_SAM(ωe, Id, I)
    H = Id * ωe
    (k = k, mode = :SAM,
     A = SVector{3,Float64}(I.Ii * B₁ / H, σ * I.Is * B₂ / H, I.Il * B₃ / H))
end
# ω1 then calculated for each az 
@inline function _az_at(τ::Real, k::Real, A::SVector{3,Float64}, mode::Symbol)
    if mode === :LAM
        SVector{3,Float64}(A[1]*_sn(τ,k), A[2]*_cn(τ,k), A[3]*_dn(τ,k))
    else
        SVector{3,Float64}(A[1]*_sn(τ,k), A[2]*_dn(τ,k), A[3]*_cn(τ,k))
    end
end
# needed for gravity_gradient
function az_squared_averages(ωe::Real, Id::Real, I::PrincipalInertias,
                              regime::Regime; N_τ::Int=512, σ::Integer=1)
    p      = az_amplitudes(ωe, Id, I, regime; σ = σ)   # once, not N_τ+1 times
    period = 2 * elliptic_K(p.k)                       # full period in τ
    dτ     = period / N_τ
    sum_   = SVector{3,Float64}(0.0, 0.0, 0.0)
    for j in 0:(N_τ-1)
        az   = _az_at((j + 0.5) * dτ, p.k, p.A, p.mode)   # midpoint rule
        sum_ = sum_ + az .^ 2
    end
    return sum_ / N_τ
end
