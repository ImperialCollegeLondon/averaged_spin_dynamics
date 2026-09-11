
# basic types - use parallel axis theorem from Schaub Junkins to determine principal inertias

abstract type MassPrimitive end

# S&J rectangular paralleliped
struct SolidBox <: MassPrimitive
    mass::Float64
    dims::SVector{3,Float64}
    centroid::SVector{3,Float64}
    R::SMatrix{3,3,Float64,9}
end

"""
About its own centroid, in local axes:

    I_xx = m b²/12,  I_yy = m a²/12,  I_zz = m(a² + b²)/12

"""
struct ThinPlate <: MassPrimitive
    mass::Float64
    dims::SVector{2,Float64}
    centroid::SVector{3,Float64}
    R::SMatrix{3,3,Float64,9}
end

"""
Solid circular cylinder of uniform density, symmetry axis along its OWN LOCAL
ẑ.  About its own centroid, in local axes:

    I_xx = I_yy = m(3r² + h²)/12,   I_zz = m r²/2

"""
struct SolidCylinder <: MassPrimitive
    mass::Float64
    dims::SVector{2,Float64}      # (radius, height)
    centroid::SVector{3,Float64}
    R::SMatrix{3,3,Float64,9}
end

struct PointMass <: MassPrimitive
    mass::Float64
    centroid::SVector{3,Float64}
    R::SMatrix{3,3,Float64,9}
end

const _EYE3 = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)

SolidBox(mass::Real, dims; centroid = (0.0, 0.0, 0.0), R = _EYE3) =
    SolidBox(Float64(mass), SVector{3,Float64}(dims), SVector{3,Float64}(centroid),
             SMatrix{3,3,Float64,9}(R))

ThinPlate(mass::Real, dims; centroid = (0.0, 0.0, 0.0), R = _EYE3) =
    ThinPlate(Float64(mass), SVector{2,Float64}(dims), SVector{3,Float64}(centroid),
              SMatrix{3,3,Float64,9}(R))

SolidCylinder(mass::Real, dims; centroid = (0.0, 0.0, 0.0), R = _EYE3) =
    SolidCylinder(Float64(mass), SVector{2,Float64}(dims), SVector{3,Float64}(centroid),
                  SMatrix{3,3,Float64,9}(R))

PointMass(mass::Real, centroid; R = _EYE3) =
    PointMass(Float64(mass), SVector{3,Float64}(centroid), SMatrix{3,3,Float64,9}(R))

mass(p::MassPrimitive)     = p.mass
centroid(p::MassPrimitive) = p.centroid

# Analytic tensors about centroid


function inertia_local(p::SolidBox)
    m = p.mass; a, b, c = p.dims
    return SMatrix{3,3,Float64,9}(m*(b^2 + c^2)/12, 0, 0,
                                  0, m*(a^2 + c^2)/12, 0,
                                  0, 0, m*(a^2 + b^2)/12)
end

function inertia_local(p::ThinPlate)
    m = p.mass; a, b = p.dims
    return SMatrix{3,3,Float64,9}(m*b^2/12, 0, 0,
                                  0, m*a^2/12, 0,
                                  0, 0, m*(a^2 + b^2)/12)
end

function inertia_local(p::SolidCylinder)
    m = p.mass; r, h = p.dims
    return SMatrix{3,3,Float64,9}(m*(3r^2 + h^2)/12, 0, 0,
                                  0, m*(3r^2 + h^2)/12, 0,
                                  0, 0, m*r^2/2)
end

inertia_local(::PointMass) = zero(SMatrix{3,3,Float64,9})


# Iᵦ = R' * I_local * R

inertia_primitive(p::MassPrimitive) = p.R' * inertia_local(p) * p.R

# Parallel-axis theorem

"""
    parallel_axis(I_c, m, d) → SMatrix{3,3}

Shift an inertia tensor `I_c`, taken about a body's centroid, to a reference
point offset by `d` FROM that centroid (i.e. `d = r_centroid − r_reference`),
both expressed in the same frame:

    I_O = I_c + m ( (d·d) 𝟙 − d dᵀ )

The dyadic `d dᵀ` term is what makes this the full tensor form.  It is
non-zero off-diagonal whenever `d` is not parallel to a coordinate axis, so an
asymmetric or boom-mounted component correctly generates products of inertia —
the on-axis scalar version `I + m d²` would silently drop them.
"""
function parallel_axis(I_c::AbstractMatrix, m::Real, d::AbstractVector)
    dv = SVector{3,Float64}(d)
    return SMatrix{3,3,Float64,9}(I_c) + m * (dot(dv, dv) * _EYE3 - dv * dv')
end

# Composite result

"""
    MassProperties

Everything `composite_inertia` computed, so intermediate stages stay
inspectable instead of being collapsed into three numbers.

  `mass`        total mass [kg]
  `com`         composite centre of mass, in the input body frame [m]
  `I_com`       full inertia tensor about `com`, in the input body frame [kg m²]
  `I_origin`    same tensor about the input body-frame ORIGIN [kg m²]
  `principal`   `PrincipalInertias(Il, Ii, Is)` — the long-axis mapping
  `axes`        `(b̂₁, b̂₂, b̂₃)` as body-frame column vectors, right-handed
  `R_bp`        DCM body → principal: `R_bp * I_com * R_bp' = diag(Ii, Is, Il)`

`R_bp`'s rows are b̂₁ᵀ, b̂₂ᵀ, b̂₃ᵀ, so the diagonal comes out in the project's
(b̂₁,b̂₂,b̂₃) ↔ (I_i,I_s,I_l) order, not in ascending-eigenvalue order.
"""
struct MassProperties
    mass::Float64
    com::SVector{3,Float64}
    I_com::SMatrix{3,3,Float64,9}
    I_origin::SMatrix{3,3,Float64,9}
    principal::PrincipalInertias
    axes::NTuple{3,SVector{3,Float64}}
    R_bp::SMatrix{3,3,Float64,9}
end

function Base.show(io::IO, mp::MassProperties)
    P = mp.principal
    print(io, "MassProperties(m = ", round(mp.mass, digits=2), " kg, ",
              "Il = ", round(P.Il, digits=1), ", ",
              "Ii = ", round(P.Ii, digits=1), ", ",
              "Is = ", round(P.Is, digits=1), " kg·m²)")
end

# Realisability 

# Checks if all moments are positive and teh inequality holds
function check_realisable(Il::Real, Ii::Real, Is::Real;
                          rtol::Real = 1e-10, on_violation::Symbol = :error)
    msgs = String[]
    for (nm, v) in (("Il", Il), ("Ii", Ii), ("Is", Is))
        v > 0 || push!(msgs, "$nm = $v is not positive")
    end
    scale = max(abs(Is), abs(Ii), abs(Il), eps())
    tol   = rtol * scale
    Is ≤ Ii + Il + tol || push!(msgs, "triangle inequality violated: Is = $Is > Ii + Il = $(Ii + Il)")
    Ii ≤ Is + Il + tol || push!(msgs, "triangle inequality violated: Ii = $Ii > Is + Il = $(Is + Il)")
    Il ≤ Is + Ii + tol || push!(msgs, "triangle inequality violated: Il = $Il > Is + Ii = $(Is + Ii)")

    isempty(msgs) && return true
    text = "unphysical inertia tensor: " * join(msgs, "; ")
    on_violation === :error  && throw(ArgumentError(text))
    on_violation === :warn   && @warn text
    return false
end

check_realisable(P::PrincipalInertias; kwargs...) =
    check_realisable(P.Il, P.Ii, P.Is; kwargs...)

# The composite calculation - building the complete tensor


function composite_inertia(primitives::AbstractVector{<:MassPrimitive};
                           about::Symbol = :com,
                           on_violation::Symbol = :error,
                           rtol::Real = 1e-10)
    isempty(primitives) && throw(ArgumentError("composite_inertia: no primitives given"))
    any(p -> p.mass < 0, primitives) &&
        throw(ArgumentError("composite_inertia: negative primitive mass"))

    M = sum(p.mass for p in primitives)
    M > 0 || throw(ArgumentError("composite_inertia: total mass is zero"))
    r_cm = sum(p.mass * p.centroid for p in primitives) / M

    ref = about === :com    ? r_cm :
          about === :origin ? zero(SVector{3,Float64}) :
          throw(ArgumentError("composite_inertia: `about` must be :com or :origin, got :$about"))

    # Shift every primitive to the SAME reference point before summing.
    I_ref = zero(SMatrix{3,3,Float64,9})
    for p in primitives
        I_ref += parallel_axis(inertia_primitive(p), p.mass, p.centroid - ref)
    end

    # Both reference points are cheap to carry, and having them side by side
    # makes an accidentally off-centre body frame obvious.  Going origin → com
    # SUBTRACTS the shift term (the theorem only adds when moving away from the
    # centroid), so it is written out rather than reusing `parallel_axis`.
    shift = M * (dot(r_cm, r_cm) * _EYE3 - r_cm * r_cm')
    I_com    = about === :com ? I_ref : I_ref - shift
    I_origin = about === :com ? I_ref + shift : I_ref

    I_sym = Symmetric(Array((I_ref + I_ref') / 2))
    F     = eigen(I_sym)                      # ascending eigenvalues
    λ     = F.values
    V     = F.vectors

    Il, Ii, Is = λ[1], λ[2], λ[3]
    b3 = SVector{3,Float64}(V[:, 1])          # minimum  → I_l
    b1 = SVector{3,Float64}(V[:, 2])          # intermediate → I_i
    b2 = SVector{3,Float64}(V[:, 3])          # maximum  → I_s

    # Eigenvector signs are arbitrary; pick the right-handed triad (b̂₁,b̂₂,b̂₃).
    dot(cross(b1, b2), b3) < 0 && (b3 = -b3)

    check_realisable(Il, Ii, Is; rtol = rtol, on_violation = on_violation)

    # Clamp the ordering against eigen-solver round-off before the
    # PrincipalInertias constructor (which demands Il ≤ Ii ≤ Is strictly).
    Ii = max(Ii, Il)
    Is = max(Is, Ii)
    P  = PrincipalInertias(Il, Ii, Is)

    R_bp = SMatrix{3,3,Float64,9}(vcat(b1', b2', b3'))
    return MassProperties(M, r_cm, I_com, I_origin, P, (b1, b2, b3), R_bp)
end

composite_inertia(primitives::MassPrimitive...; kwargs...) =
    composite_inertia(collect(MassPrimitive, primitives); kwargs...)
principal_inertias(primitives; kwargs...) = composite_inertia(primitives; kwargs...).principal
