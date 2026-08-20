
# Earth gravitational parameter [m³/s²]; only used to form n_g = √(μ_e/R³).
const MU_EARTH = 3.986004418e14

#orbit mean motion n_g = √(μ_e/R³) for a circular orbit of radius R [m].
gg_mean_motion(R::Real) = sqrt(MU_EARTH / R^3)

# B&S Eq 6
function gg_torque(R̂_body, I::PrincipalInertias, n_g::Real)
    d = inertia_body(I)
    IR = SVector{3,Float64}(d[1]*R̂_body[1], d[2]*R̂_body[2], d[3]*R̂_body[3])
    return 3 * n_g^2 * cross(SVector{3,Float64}(R̂_body...), IR)
end

# Orbit Angles Eq 22-23
function orbit_angles(i::Real, Ω::Real; δ_e::Real = OBLIQUITY)
    δ  = acos(clamp(cos(δ_e)*cos(i) + sin(δ_e)*cos(Ω)*sin(i), -1.0, 1.0))
    a1 = cos(Ω)*cos(δ_e)*sin(i) - cos(i)*sin(δ_e)
    a2 = sin(Ω)*sin(i)
    return atan(a2, a1), δ
end

# Averaged inertia tensor (diagonal in H) Eqs. (29)–(31)

function averaged_inertia(ωe::Real, Id::Real, I::PrincipalInertias, regime::Regime;
                          N_τ::Int = 512)
    az = az_moments(Id, I, regime)
    Ii, Is, Il = I.Ii, I.Is, I.Il
    Īx = 0.5*(Ii*(1-az.a11) + Is*(1-az.a22) + Il*(1-az.a33))
    Īz = Ii*az.a11 + Is*az.a22 + Il*az.a33
    @assert isapprox(2Īx + Īz, Ii + Is + Il; rtol=1e-6) "avg-inertia trace identity failed"
    return Īx, Īx, Īz     # Ī_y = Ī_x
end

# G→H rotation (Eq. 19)

function hg_matrix(α::Real, β::Real, λ::Real, δ::Real)
    Mmid = @SMatrix [0.0 0.0 1.0; 0.0 -1.0 0.0; 1.0 0.0 0.0]
    return HO_mat(α, β) * Mmid * R3(-λ) * R1(-δ)
end

#averaged GG torque (Eqs. 32–34)
function averaged_gg_torque(α::Real, β::Real, λ::Real, δ::Real,
                            Īx::Real, Īz::Real, n_g::Real)
    b = hg_matrix(α, β, λ, δ)
    bx1, bx2 = b[1,1], b[1,2]
    by1, by2 = b[2,1], b[2,2]
    bz1, bz2 = b[3,1], b[3,2]
    f = 1.5 * n_g^2
    L̄x =  f * (Īz - Īx) * (by1*bz1 + by2*bz2)     # Ī_y = Ī_x
    L̄y = -f * (Īz - Īx) * (bx1*bz1 + bx2*bz2)
    L̄z = -f * (Īx - Īx) * (bx1*by1 + bx2*by2)     # ≡ 0
    return L̄x, L̄y, L̄z
end
