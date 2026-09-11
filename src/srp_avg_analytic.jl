# introducing the appendix B backend srp equations, this avoids stalling at the separatrix as K only appears in ratios bounding it

# elliptic functions averages CHECKED
function elliptic_function_averages(k::Real)
    m = k^2
    if m >= 1 - 1e-9
        # Separatrix limits: E(1) = 1, K → ∞.
        return (dn = 0.0, sn2 = 1.0, cn2 = 0.0, dn2 = 0.0, sn2dn = 0.0,
                cn2dn = 0.0, dn3 = 0.0, sn4 = 1.0, cn4 = 0.0, dn4 = 0.0,
                sn2cn2 = 0.0, sn2dn2 = 0.0, cn2dn2 = 0.0)
    elseif m < 1e-8
        # Uniform-spin limits (⟨sin²⟩ = 1/2, ⟨sin⁴⟩ = 3/8, ⟨sin²cos²⟩ = 1/8).
        return (dn = 1.0, sn2 = 0.5, cn2 = 0.5, dn2 = 1.0, sn2dn = 0.5,
                cn2dn = 0.5, dn3 = 1.0, sn4 = 0.375, cn4 = 0.375, dn4 = 1.0,
                sn2cn2 = 0.125, sn2dn2 = 0.5, cn2dn2 = 0.5)
    end
    Kv, Ev = K(m), E(m)            # Elliptic.jl parameter convention m = k²
    kp2 = 1 - m                    # k′²
    return (
        dn     = π / (2Kv),                                          # B3
        sn2    = (Kv - Ev) / (m * Kv),                               # B4
        cn2    = (Ev - kp2 * Kv) / (m * Kv),                         # B5
        dn2    = Ev / Kv,                                            # B6
        sn2dn  = π / (4Kv),                                          # B7
        cn2dn  = π / (4Kv),                                          # B8
        dn3    = (1 + kp2) * π / (4Kv),                              # B9
        sn4    = ((m + 2) * Kv - 2 * (m + 1) * Ev) / (3 * m^2 * Kv),         # B10
        cn4    = ((4m - 2) * Ev - kp2 * (3m - 2) * Kv) / (3 * m^2 * Kv),     # B11
        dn4    = (2 * (kp2 + 1) * Ev - kp2 * Kv) / (3Kv),            # B12
        sn2cn2 = ((1 + kp2) * Ev - 2 * kp2 * Kv) / (3 * m^2 * Kv),   # B13
        sn2dn2 = ((2m - 1) * Ev + kp2 * Kv) / (3 * m * Kv),          # B14
        cn2dn2 = ((1 + m) * Ev - kp2 * Kv) / (3 * m * Kv),           # B15
    )
end

# finding the az moments for both sam and lam to eventually find the torque

struct AzMoments
    regime::Symbol       # :LAM or :SAM (Separatrix uses the common k→1 limit)
    k::Float64           # elliptic modulus (Eq. A5 / A13)
    a2::Float64;    a3::Float64
    a11::Float64;   a22::Float64;   a33::Float64
    a112::Float64;  a113::Float64;  a223::Float64;  a233::Float64
    a222::Float64;  a333::Float64
    a1111::Float64; a2222::Float64; a3333::Float64
    a1122::Float64; a1133::Float64; a2233::Float64
end
#LAM
function az_moments(Id::Real, I::PrincipalInertias, regime::LAM; σ::Integer = 1)
    abs(σ) == 1 || throw(ArgumentError("branch σ must be +1 or -1"))
    k, _, _, B₁, B₂, B₃ = torquefree_params_LAM(1.0, Id, I)   # ωe = 1 ⇒ H = Id
    A1, A2, A3 = I.Ii * B₁ / Id, I.Is * B₂ / Id, σ * I.Il * B₃ / Id
    e = elliptic_function_averages(k)
    # LAM: a_z = (A₁ sn, A₂ cn, A₃ dn) — Eqs. A1 + A2 (σ = ±1 ⇒ LAM±)
    AzMoments(:LAM, k,
        0.0,              A3 * e.dn,
        A1^2 * e.sn2,     A2^2 * e.cn2,     A3^2 * e.dn2,
        0.0,              A1^2 * A3 * e.sn2dn,
        A2^2 * A3 * e.cn2dn,   0.0,
        0.0,              A3^3 * e.dn3,
        A1^4 * e.sn4,     A2^4 * e.cn4,     A3^4 * e.dn4,
        A1^2 * A2^2 * e.sn2cn2, A1^2 * A3^2 * e.sn2dn2, A2^2 * A3^2 * e.cn2dn2)
end
#SAM
function az_moments(Id::Real, I::PrincipalInertias, regime::SAM; σ::Integer = 1)
    abs(σ) == 1 || throw(ArgumentError("branch σ must be +1 or -1"))
    k, _, _, B₁, B₂, B₃ = torquefree_params_SAM(1.0, Id, I)
    A1, A2, A3 = I.Ii * B₁ / Id, σ * I.Is * B₂ / Id, I.Il * B₃ / Id
    e = elliptic_function_averages(k)
    # SAM: a_z = (A₁ sn, A₂ dn, A₃ cn) — Eqs. A1 + A11 (σ = ±1 ⇒ SAM±)
    AzMoments(:SAM, k,
        A2 * e.dn,        0.0,
        A1^2 * e.sn2,     A2^2 * e.dn2,     A3^2 * e.cn2,
        A1^2 * A2 * e.sn2dn,   0.0,
        0.0,              A2 * A3^2 * e.cn2dn,
        A2^3 * e.dn3,     0.0,
        A1^4 * e.sn4,     A2^4 * e.dn4,     A3^4 * e.cn4,
        A1^2 * A2^2 * e.sn2dn2, A1^2 * A3^2 * e.sn2cn2, A2^2 * A3^2 * e.cn2dn2)
end

az_moments(Id::Real, I::PrincipalInertias, ::Separatrix; σ::Integer = 1) =
    AzMoments(:SAM, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# Da BIGGEST ONE - all of the appendix B averaged products (Eqs. B16–B49).

_S_fn(f, n, az) = f[1]*n[1]*az.a11 + f[2]*n[2]*az.a22 + f[3]*n[3]*az.a33
_S_nn(n, az)    = n[1]^2*az.a11 + n[2]^2*az.a22 + n[3]^2*az.a33
function _Q_fn3(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    f1*n1^3*az.a1111 + f2*n2^3*az.a2222 + f3*n3^3*az.a3333 +
    3 * (f1*n1*(n2^2*az.a1122 + n3^2*az.a1133) +
         f2*n2*(n1^2*az.a1122 + n3^2*az.a2233) +
         f3*n3*(n1^2*az.a1133 + n2^2*az.a2233))
end

# <f_z> B16 (LAM) & B33 (SAM) Checked

_avg_fz(f, n, az::AzMoments) =
    az.regime === :LAM ? f[3] * az.a3 : f[2] * az.a2

# ⟨fx nx⟩ B17 = B34 (identical) Checked
function _avg_fxnx(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    (1/2)*(f3*n3 - f1*n1)*az.a11 + (1/2)*(f3*n3 - f2*n2)*az.a22 +
    (1/2)*(f1*n1 + f2*n2)
end

# ⟨fy nx⟩ B18 (LAM) / B35 (SAM) Checked
function _avg_fynx(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    az.regime === :LAM ? (1/2)*(f2*n1 - f1*n2)*az.a3 :
                         (1/2)*(f1*n3 - f3*n1)*az.a2
end

# ⟨fz nz⟩ — B19 = B36 (identical) checked
_avg_fznz(f, n, az::AzMoments) =
    f[1]*n[1]*az.a11 + f[2]*n[2]*az.a22 + f[3]*n[3]*az.a33

# ⟨fx nx nz⟩ — B20 (LAM) / B37 (SAM) checked
function _avg_fxnxnz(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    if az.regime === :LAM
        (1/2)*(f1*n1*n3 + f2*n2*n3)*az.a3 -
        (1/2)*(f3*n1^2 + 2*f1*n1*n3 - f3*n3^2)*az.a113 -
        (1/2)*(f3*n2^2 + 2*f2*n2*n3 - f3*n3^2)*az.a223
    else
        (1/2)*(f2*n3^2 - f2*n1^2 - 2*f1*n2*n1 + 2*f3*n2*n3)*az.a112 +
        (1/2)*(f2*n3^2 - f2*n2^2 + 2*f3*n2*n3)*az.a222 +
        (1/2)*(f2*n2^2 - f3*n2*n3 + f1*n1*n2 - f2*n3^2)*az.a2
    end
end

# ⟨fy nx nz⟩  B21 (LAM) / B38 (SAM) checked
function _avg_fynxnz(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    if az.regime === :LAM
        (1/2)*(f3*n1*n2 - f2*n1*n3)*az.a11 +
        (1/2)*(f1*n2*n3 - f3*n1*n2)*az.a22 +
        (1/2)*(f2*n1*n3 - f1*n2*n3)*az.a33
    else
        (1/2)*(f1*n2*n3 - 2*f2*n1*n3 + f3*n1*n2)*az.a11 +
        (1/2)*(2*f1*n2*n3 - f2*n1*n3 - f3*n1*n2)*az.a22 +
        (1/2)*(f2*n1*n3 - f1*n2*n3)
    end
end

# ⟨fz nx²⟩  B22 (LAM) / B39 (SAM) checked
function _avg_fznx2(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    if az.regime === :LAM
        (1/2)*(f3*n1^2 + f3*n2^2)*az.a3 -
        (1/2)*(f3*n1^2 + 2*f1*n1*n3 - f3*n3^2)*az.a113 -
        (1/2)*(f3*n2^2 + 2*f2*n2*n3 - f3*n3^2)*az.a223
    else
        (1/2)*(f2*n3^2 - f2*n1^2 - 2*f1*n2*n1 + 2*f3*n2*n3)*az.a112 +
        (1/2)*(f2*n3^2 - f2*n2^2 + 2*f3*n2*n3)*az.a222 +
        (1/2)*(f2*n1^2 + f2*n2^2 - 2*f3*n3*n2)*az.a2
    end
end

# ⟨fz nz²⟩ B23 (LAM) / B40 (SAM) checked
function _avg_fznz2(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    if az.regime === :LAM
        (f3*n1^2 + 2*f1*n1*n3)*az.a113 + (f3*n2^2 + 2*f2*n2*n3)*az.a223 +
        f3*n3^2*az.a333
    else
        (f2*n1^2 + 2*f1*n2*n1 - f2*n3^2 - 2*f3*n2*n3)*az.a112 +
        (f2*n2^2 - 2*f3*n2*n3 - f2*n3^2)*az.a222 +
        (f2*n3^2 + 2*f3*n2*n3)*az.a2
    end
end

# ⟨fx nx³⟩  B24 (LAM) / B41 (SAM).  - checked
function _avg_fxnx3(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    if az.regime === :LAM
        (3/8)*(3*f3*n1^2*n3 - 2*f1*n1^3 - 3*f2*n1^2*n2 - f1*n1*n2^2 + 3*f1*n1*n3^2
            + f3*n2^2*n3 + f2*n2*n3^2) * az.a11 +
        (3/8)*(f2*n1^2*n2 + f3*n1^2*n3 - f1*n1*n2^2 + f1*n1*n3^2 - 2*f2*n2^3
            + 3*f3*n2^2*n3 + 3*f2*n2*n3^2) * az.a22 +
        (3/8)*(f1*n1^3 - 3*f3*n1^2*n3 - 3*f1*n1*n3^2 + f3*n3^3) * az.a1111 +
        (3/8)*(f2*n2^3 - 3*f3*n2^2*n3 - 3*f2*n2*n3^2 + f3*n3^3) * az.a2222 +
        (3/8)*(3*f2*n1^2*n2 - 3*f3*n1^2*n3 + 3*f1*n1*n2^2 - 3*f1*n1*n3^2
            - 3*f3*n2^2*n3 - 3*f2*n2*n3^2 + 2*f3*n3^3) * az.a1122 +
        (3/8)*(f1*n1^3 + f2*n1^2*n2 + f1*n1*n2^2 + f2*n2^3)
    else
        (1/8)*(3*f1*n1^3 - 9*f3*n1^2*n3 - 9*f1*n1*n3^2 + 3*f3*n3^3) * az.a1111 +
        (3/8)*(3*f2*n2^3 - 3*f3*n2^2*n3 - 3*f2*n2*n3^2 + f3*n3^3) * az.a2222 +
        (9/8)*(9*f2*n1^2*n2 - f3*n1^2*n3 + f1*n1*n2^2 - 9*f1*n1*n3^2 - 9*f3*n2^2*n3
            - 9*f2*n2*n3^2 + (2/3)*f3*n3^3) * az.a1122 +
        (3/8)*(-2*f1*n1^3 - f2*n1^2*n2 + 3*f3*n1^2*n3 - f1*n1*n2^2 + 3*f1*n1*n3^2
            + f3*n2^2*n3 + f2*n2*n3^2) * az.a11 +
        (3/8)*(-f2*n1^2*n2 + f3*n1^2*n3 - f1*n1*n2^2 + f1*n1*n3^2 - 2*f2*n2^3
            + 3*f3*n2^2*n3 + 3*f2*n2*n3^2) * az.a22 +
        (3/8)*(3*f1*n1^3 + 3*f2*n1^2*n2 + 3*f1*n1*n2^2 + 3*f2*n2^3)
    end

end

# ⟨fx nx nz²⟩ B25 (LAM) / B42 (SAM). checked
function _avg_fxnxnz2(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    if az.regime === :LAM
        (1/2)*(f1*n1^3 - 2*f3*n1^2*n3 + f2*n2*n1^2 - 4*f1*n1*n3^2 + f3*n3^3
            - f2*n2*n3^2) * az.a11 +
        (1/2)*(f2*n2^3 - 2*f3*n2^2*n3 + f1*n1*n2^2 - 4*f2*n2*n3^2 + f3*n3^3
            - f1*n1*n3^2) * az.a22 +
        (1/2)*(-f1*n1^3 + 3*f3*n1^2*n3 + 3*f1*n1*n3^2 - f3*n3^3) * az.a1111 +
        (3/2)*(f2*n1^2*n2 + f3*n1^2*n3 - f1*n1*n2^2 + f1*n1*n3^2 + f3*n2^2*n3
            + f2*n2*n3^2 - (2/3)*f3*n3^3) * az.a1122 +
        (1/2)*(-f2*n2^3 + 3*f3*n2^2*n3 + 3*f2*n2*n3^2 - f3*n3^3) * az.a2222 +
        (1/2)*(f1*n1*n3^2 + f2*n2*n3^2)
    else
        (1/2)*(-f1*n1^3 + 3*f3*n1^2*n3 + 3*f1*n1*n3^2 - f3*n3^3) * az.a1111 +
        (1/2)*(-f2*n2^3 + 3*f3*n2^2*n3 + 3*f2*n2*n3^2 - f3*n3^3) * az.a2222 +
        (3/2)*(-f2*n1^2*n2 + f3*n1^2*n3 - f1*n1*n2^2 + f1*n1*n3^2 + f3*n2^2*n3
            + f2*n2*n3^2 - (2/3)*f3*n3^3) * az.a1122 +
        (1/2)*(f1*n1^3 - 2*f3*n1^2*n3 + f2*n2*n1^2 - 4*f1*n1*n3^2 + f3*n3^3
            - f2*n2*n3^2) * az.a11 +
        (1/2)*(f2*n2^3 - 2*f3*n2^2*n3 + f1*n1*n2^2 - 4*f2*n2*n3^2 + f3*n3^3
            - f1*n1*n3^2) * az.a22 +
        (1/2)*(f1*n1*n3^2 + f2*n2*n3^2)
    end
end

# ⟨fy nx³⟩  B26 (LAM) / B43 (SAM) checked
function _avg_fynx3(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    if az.regime === :LAM
        -(3/8)*(f2*n1^3 - f1*n2*n1^2 - 3*f2*n1*n3^2 + 2*f3*n2*n1*n3 +
                f1*n2*n3^2)*az.a113 +
         (3/8)*(f1*n2^3 - f2*n1*n2^2 - 3*f1*n2*n3^2 + 2*f3*n1*n2*n3 +
                f2*n1*n3^2)*az.a223 -
         (3/8)*(-f2*n1^3 + f1*n1^2*n2 - f2*n1*n2^2 + f1*n2^3)*az.a3
    else
        (3/8)*(f3*n1^3 - f1*n1^2*n3 - 2*f3*n1*n2^2 + 4*f2*n1*n2*n3 -
               f3*n1*n3^2 - 2*f1*n2^2*n3 + f1*n3^3)*az.a112 +
        (3/8)*(-3*f1*n2^2*n3 + f3*n1*n2^2 + 2*f2*n1*n2*n3 + f1*n3^3 -
               f3*n1*n3^2)*az.a222 +
        (3/8)*(-f3*n1^3 + f1*n3*n1^2 - f3*n1*n2^2 - 2*f2*n3*n1*n2 +
               3*f1*n3*n2^2)*az.a2
    end
end

# ⟨fy nx nz²⟩ B27 (LAM) / B44 (SAM) checked
function _avg_fynxnz2(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    if az.regime === :LAM
        (1/2)*(f2*n1^3 - f1*n2*n1^2 - 3*f2*n1*n3^2 + 2*f3*n2*n1*n3 +
               f1*n2*n3^2)*az.a113 -
        (1/2)*(f1*n2^3 - f2*n1*n2^2 - 3*f1*n2*n3^2 + 2*f3*n1*n2*n3 +
               f2*n1*n3^2)*az.a223 -
        (1/2)*(f1*n2 - f2*n1)*n3^2*az.a3
    else
        (1/2)*(-f3*n1^3 + f1*n1^2*n3 + 2*f3*n1*n2^2 - 4*f2*n1*n2*n3 +
               f3*n1*n3^2 + 2*f1*n2^2*n3 - f1*n3^3)*az.a112 +
        (1/2)*(3*f1*n2^2*n3 - f3*n1*n2^2 - 2*f2*n1*n2*n3 - f1*n3^3 +
               f3*n1*n3^2)*az.a222 +
        (1/2)*(-2*f1*n2^2*n3 + 2*f2*n1*n2*n3 + f1*n3^3 - f3*n1*n3^2)*az.a2
    end
end

# ⟨fz nx² nz⟩ B28 (LAM) / B45(SAM) (identical) checked
function _avg_fznx2nz(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    (1/2)*(f1*n1^3 - 4*f3*n1^2*n3 + f1*n1*n2^2 - 2*f1*n1*n3^2 -
        f3*n2^2*n3 + f3*n3^3)*az.a11 +
    (1/2)*(f2*n1^2*n2 - f3*n1^2*n3 + f2*n2^3 - 4*f3*n2^2*n3 -
        2*f2*n2*n3^2 + f3*n3^3)*az.a22 +
    (1/2)*(f3*n3*n1^2 + f3*n3*n2^2) +
    (1/2)*(-f1*n1^3 + 3*f3*n1^2*n3 + 3*f1*n1*n3^2 - f3*n3^3)*az.a1111 +
    (1/2)*(-f2*n2^3 + 3*f3*n2^2*n3 + 3*f2*n2*n3^2 - f3*n3^3)*az.a2222 +
    (3/2)*(-f2*n1^2*n2 + f3*n1^2*n3 - f1*n1*n2^2 + f1*n1*n3^2 +
        f3*n2^2*n3 + f2*n2*n3^2 - (2/3)*f3*n3^3)*az.a1122
end

# ⟨fz nz³⟩  B29 = B46 (identical content) checked
function _avg_fznz3(f, n, az::AzMoments)
    f1, f2, f3 = f; n1, n2, n3 = n
    (3*f3*n1^2*n3 + 3*f1*n1*n3^2 - 2*f3*n3^3)*az.a11 +
    (3*f3*n2^2*n3 + 3*f2*n2*n3^2 - 2*f3*n3^3)*az.a22 +
    (f1*n1^3 - 3*f3*n1^2*n3 - 3*f1*n1*n3^2 + f3*n3^3)*az.a1111 +
    (f2*n2^3 - 3*f3*n2^2*n3 - 3*f2*n2*n3^2 + f3*n3^3)*az.a2222 +
    3*(f2*n1^2*n2 - f3*n1^2*n3 + f1*n1*n2^2 - f1*n1*n3^2 -
       f3*n2^2*n3 - f2*n2*n3^2 + (2/3)*f3*n3^3)*az.a1122 +
    f3*n3^3
end

# ⟨g a_{zℓ} δ_ℓ⟩ B30–B32 (LAM) / B47–B49 (SAM), with δ = r × û in body
# components and g the Eq. 21 Fourier illumination.  Closed forms are in
# H-frame sun components ux = −sinβ, uz = cosβ. checked
function _avg_g_az_delta(ℓ::Integer, n, r, ux::Real, uz::Real, az::AzMoments)
    n1, n2, n3 = n
    r1, r2, r3 = r
    if az.regime === :LAM
        if ℓ == 1        # B30
            (2/(3π))*(6*n1*n3*r2*ux^2*uz - 4*n1*n3*r2*uz^3)*az.a1111 +
            (1/4)*(2*n1*r2*uz^2 - n1*r2*ux^2)*az.a113 +
            (2/(3π))*(6*n1*n2*r3*ux^2*uz - 4*n1*n3*r2*uz^3 -
                      4*n1*n2*r3*uz^3 + 6*n1*n3*r2*ux^2*uz)*az.a1122 +
            (4/(3π))*(2*n1*n3*r2*uz^3 - n1*n2*r3*ux^2*uz -
                      2*n1*n3*r2*ux^2*uz)*az.a11
        elseif ℓ == 2    # B31
            (2/(3π))*(4*n1*n2*r3*uz^3 + 4*n2*n3*r1*uz^3 -
                      6*n1*n2*r3*ux^2*uz - 6*n2*n3*r1*ux^2*uz)*az.a1122 +
            (2/(3π))*(4*n2*n3*r1*uz^3 - 6*n2*n3*r1*ux^2*uz)*az.a2222 +
            (1/4)*(n2*r1*ux^2 - 2*n2*r1*uz^2)*az.a223 +
            (4/(3π))*(n1*n2*r3*ux^2*uz - 2*n2*n3*r1*uz^3 +
                      2*n2*n3*r1*ux^2*uz)*az.a22
        else             # B32
            (2/(3π))*(6*n1*n3*r2*ux^2*uz - 4*n1*n3*r2*uz^3)*az.a1133 +
            (1/4)*(n1*r2*ux^2 - 2*n1*r2*uz^2)*az.a113 +
            (2/(3π))*(4*n2*n3*r1*uz^3 - 6*n2*n3*r1*ux^2*uz)*az.a2233 -
            (1/4)*(n2*r1*ux^2 - 2*n2*r1*uz^2)*az.a223 +
            (4/(3π))*(n2*n3*r1*ux^2*uz - n1*n3*r2*ux^2*uz)*az.a33 -
            (1/(4π))*(n1*r2*ux^2 - n2*r1*ux^2)*az.a3
        end
    else
        if ℓ == 1        # B47
            (2/(3π))*(6*n1*n3*r2*ux^2*uz - 4*n1*n3*r2*uz^3)*az.a1111 +
            (1/4)*(n1*r3*ux^2 - 2*n1*r3*uz^2)*az.a112 +
            (2/(3π))*(6*n1*n2*r3*ux^2*uz - 4*n1*n3*r2*uz^3 -
                      4*n1*n2*r3*uz^3 + 6*n1*n3*r2*ux^2*uz)*az.a1122 +
            (4/(3π))*(2*n1*n3*r2*uz^3 - n1*n2*r3*ux^2*uz -
                      2*n1*n3*r2*ux^2*uz)*az.a11
        elseif ℓ == 2    # B48
            (2/(3π))*(4*n1*n2*r3*uz^3 + 4*n2*n3*r1*uz^3 -
                      6*n1*n2*r3*ux^2*uz - 6*n2*n3*r1*ux^2*uz)*az.a1122 -
            (3/12)*(n1*r3*ux^2 + n3*r1*ux^2 - 2*n1*r3*uz^2 -
                    2*n3*r1*uz^2)*az.a112 +
            (2/(3π))*(4*n2*n3*r1*uz^3 - 6*n2*n3*r1*ux^2*uz)*az.a2222 -
            (1/4)*(n3*r1*ux^2 - 2*n3*r1*uz^2)*az.a222 +
            (4/(3π))*(n1*n2*r3*ux^2*uz - 2*n2*n3*r1*uz^3 +
                      2*n2*n3*r1*ux^2*uz)*az.a22 +
            (1/4)*(n1*r3*ux^2 - 2*n3*r1*uz^2)*az.a2
        else             # B49
            (2/(3π))*(6*n1*n3*r2*ux^2*uz - 4*n1*n3*r2*uz^3)*az.a1133 +
            (2/(3π))*(4*n2*n3*r1*uz^3 - 6*n2*n3*r1*ux^2*uz)*az.a2233 -
            (1/4)*(n3*r1*ux^2 - 2*n3*r1*uz^2)*az.a233 +
            (4/(3π))*(n2*n3*r1*ux^2*uz - n1*n3*r2*ux^2*uz)*az.a33
        end
    end
end

# Averaged torque assembly (B&S 2021 Eqs. 21–26) 
# bringing everything together

function averaged_srp_torques_analytic(shape::ShapeModel, β::Real, Id::Real,
                                       I::PrincipalInertias, regime::Regime;
                                       P_SRP::Real = P_SRP_1AU, σ::Integer = 1)
    az = az_moments(Id, I, regime; σ = σ)
    ux, uz = -sin(β), cos(β)

    Mx = My = Mz = 0.0
    azM = MVector{3,Float64}(0.0, 0.0, 0.0)

    for fc in shape.facets
        n = fc.normal
        r = fc.centroid
        d = cross(r, n)                                # d = r × n̂ (Eq. 22)
        cs = 2 * fc.ρ * fc.s                           # Eq. 22
        ca = 1 - fc.ρ * fc.s                           # Eq. 22
        cd = B_LAMBERT * (1 - fc.s * fc.ρ)             # Eq. 11 (= B(1−s)ρ + B(1−ρ))
        w  = P_SRP * fc.area

        c1 = ux * (cd/2 + cs/(3π))
        c2 = ux * uz * (cs + 8cd/(3π))
        c3 = (4/(3π)) * cs * ux^3
        c4 = (4/π) * cs * ux * uz^2

        # Eq. 23
        Mx -= w * ( c1*_avg_fxnx(d, n, az) + c2*_avg_fxnxnz(d, n, az) +
                    c3*_avg_fxnx3(d, n, az) + c4*_avg_fxnxnz2(d, n, az) +
                    (1/2)*ca*ux*uz*_avg_fynx(r, n, az) +
                    (8/(3π))*ca*ux*uz^2*_avg_fynxnz(r, n, az) )

        # Eq. 24
        My -= w * ( c1*_avg_fynx(d, n, az) + c2*_avg_fynxnz(d, n, az) +
                    c3*_avg_fynx3(d, n, az) + c4*_avg_fynxnz2(d, n, az) +
                    (1/(3π))*ca*ux*_avg_fz(r, n, az) -
                    (1/2)*ca*ux*uz*_avg_fxnx(r, n, az) +
                    (1/2)*ca*ux*uz*_avg_fznz(r, n, az) -
                    (8/(3π))*ca*ux*uz^2*_avg_fxnxnz(r, n, az) +
                    (4/(3π))*ca*ux^3*_avg_fznx2(r, n, az) +
                    (4/(3π))*ca*ux*uz^2*_avg_fznz2(r, n, az) )

        # Eq. 25
        Mz -= w * ( (1/(3π))*cd*_avg_fz(d, n, az) +
                    uz*(cd/2 + cs/(3π))*_avg_fznz(d, n, az) +
                    (cs/2 + 4cd/(3π)) *
                        (ux^2*_avg_fznx2(d, n, az) + uz^2*_avg_fznz2(d, n, az)) +
                    (4/π)*cs*ux^2*uz*_avg_fznx2nz(d, n, az) +
                    (4/(3π))*cs*uz^3*_avg_fznz3(d, n, az) -
                    (1/2)*ca*ux^2*_avg_fynx(r, n, az) -
                    (8/(3π))*ca*ux^2*uz*_avg_fynxnz(r, n, az) )

        # Eq. 26: mask d to component ℓ to retain the d_ℓ terms.
        for ℓ in 1:3
            dm = SVector{3,Float64}(ℓ == 1 ? d[1] : 0.0,
                                    ℓ == 2 ? d[2] : 0.0,
                                    ℓ == 3 ? d[3] : 0.0)
            azM[ℓ] -= w * ( (1/(3π))*cd*_avg_fz(dm, n, az) +
                            uz*(cd/2 + cs/(3π))*_avg_fznz(dm, n, az) +
                            (cs/2 + 4cd/(3π)) *
                                (ux^2*_avg_fznx2(dm, n, az) +
                                 uz^2*_avg_fznz2(dm, n, az)) +
                            (4/π)*cs*ux^2*uz*_avg_fznx2nz(dm, n, az) +
                            (4/(3π))*cs*uz^3*_avg_fznz3(dm, n, az) +
                            ca*_avg_g_az_delta(ℓ, n, r, ux, uz, az) )
        end
    end
    return AveragedTorques(SVector{3,Float64}(Mx, My, Mz),
                           SVector{3,Float64}(azM[1], azM[2], azM[3]))
end


function averaged_srp_torques_analytic(shape::ShapeModel, state::OsculatingState,
                                       I::PrincipalInertias; kwargs...)
    regime = classify_regime(state.Id, I)
    return averaged_srp_torques_analytic(shape, state.β, state.Id, I, regime;
                                         kwargs...)
end
