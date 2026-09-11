

goes8_inertia_albuja() = PrincipalInertias(980.5133, 3440.9438, 3561.0894)

# CoM (Albuja x,y,z) and reconstruction anchors
const _GOES8_COM   = SVector{3,Float64}(1.15837, 0.1626, 0.0125)
const _GOES8_PANEL_CX = 4.7      # solar-panel centroid x (Albuja frame) [m]
const _GOES8_TAB_CX   = 7.8      # trim-tab centroid x [m]
const _GOES8_SAIL_CX  = -17.5    # solar-sail centroid x [m]

# Albuja (x,y,z) → body (b̂₁,b̂₂,b̂₃) with sail on +b̂₃ (matches B&S Fig. 3).
# Proper rotation (det +1).  Two choices satisfy the +b̂₃ + inertia assignment
# (they differ by 180° about b̂₃): (y,−z,−x) and (−y,z,−x).  We use (−y,z,−x): it
# reproduces the β-shape of B&S 2021 Fig. 8 M̄_y (the SAM/LAM branch).  The M̄_x,M̄_z
# come out negated vs B&S — a 2-fold H-direction (ẑ_H) convention; the Fig-8 overlay
# script maps to the B&S H-frame (−M̄ₓ,M̄_y,−M̄_z).  See docs/verification.md Stage-5.
_goes8_perm(v) = SVector{3,Float64}(-v[2], v[3], -v[1])

_unit(v) = (u = SVector{3,Float64}(v...); u / norm(u))
_mag_from_x(r̂, cx) = (cx - _GOES8_COM[1]) / r̂[1]      # |r| from centroid x-position

# Rotation about body b̂₃ by angle γ (active).  θ_sa about −b̂₃ ⇒ γ = −θ_sa.
function _rotz(γ)
    c, s = cos(γ), sin(γ)
    @SMatrix [c -s 0.0; s c 0.0; 0.0 0.0 1.0]
end

function goes8_shape_full(; θ_sa::Real = deg2rad(17.0), optical::Symbol = :bs,
                          sail::Symbol = :albuja, n_sail_sides::Int = 4)
    optical in (:bs, :albuja) || throw(ArgumentError("optical must be :bs or :albuja"))
    sail in (:albuja, :cylinder, :cone) ||
        throw(ArgumentError("sail must be :albuja, :cylinder or :cone"))
    (sail === :albuja || n_sail_sides >= 3) ||
        throw(ArgumentError("n_sail_sides must be ≥ 3"))
    fs = Facet[]

    # optical props (ρ, s) per component/face
    op = optical === :bs ?
        (panel_f=(0.27,1.0), panel_b=(0.07,0.0), tab_f=(0.83,1.0), tab_b=(0.07,0.0),
         bus=(0.60,1.0), sail_side=(0.66,1.0), sail_base=(0.83,1.0)) :
        (panel_f=(0.21,0.2), panel_b=(0.82,0.2), tab_f=(0.88,0.2), tab_b=(0.07,0.2),
         bus=(0.93,0.2), sail_side=(0.66,0.2), sail_base=(0.66,0.2))

    Rsa = _rotz(-θ_sa)   # array rotation about −b̂₃

    # Solar panel (front/back), Table 5; rotated by θ_sa about −b̂₃
    let n̂ = _unit((-0.004,-0.217,0.976)), r̂ = _unit((0.979,0.171,-0.119))
        r = _mag_from_x(r̂, _GOES8_PANEL_CX) * r̂
        n_b = Rsa * _goes8_perm(n̂); r_b = Rsa * _goes8_perm(r)
        push!(fs, Facet(4.81*2.68, n_b,  r_b, op.panel_f...))
        push!(fs, Facet(4.81*2.68, -n_b, r_b, op.panel_b...))
    end
    # Trim tab (front/back), Table 5 Albuja
    let n̂ = _unit((-0.004,-0.217,0.976)), r̂ = _unit((0.992,0.104,-0.075))
        r = _mag_from_x(r̂, _GOES8_TAB_CX) * r̂
        n_b = _goes8_perm(n̂); r_b = _goes8_perm(r)
        push!(fs, Facet(1.30*1.30, n_b,  r_b, op.tab_f...))
        push!(fs, Facet(1.30*1.30, -n_b, r_b, op.tab_b...))
    end
    # Bus: box at bus_c, axes/half-extents from Table 5 normals
    let bus_c = SVector(1.01, 1.14, -0.62),
        x̂ = _unit((0.999,-0.013,0.001)), B̂ = _unit((0.006,0.537,0.844)), Ĉ = _unit((-0.012,-0.844,0.537))
        for (n̂, he, A) in ((x̂,1.31,2.43*2.43), (-x̂,1.31,2.43*2.43),
                            (B̂,1.215,2.62*2.43), (-B̂,1.215,2.62*2.43),
                            (Ĉ,1.215,2.62*2.43), (-Ĉ,1.215,2.62*2.43))
            r = (bus_c + he*n̂) - _GOES8_COM
            push!(fs, Facet(A, _goes8_perm(n̂), _goes8_perm(r), op.bus...))
        end
    end
    # Solar sail: base disc + lateral surface
    let R = 1.635, h = 1.63,
        base_A = π*R^2, lat_A = π*R*sqrt(R^2 + h^2)
        if sail === :albuja
            # Transcribed Table-5 facets, verbatim (4 radial-normal side panels).
            side_A = lat_A / 4
            sail_t = (((-0.999,0.013,-0.001), (-0.997,0.064,-0.034), base_A, op.sail_base),
                      (( 0.006,0.537, 0.844), (-0.996,0.093, 0.002), side_A, op.sail_side),
                      (( 0.012,0.844,-0.537), (-0.992,0.107,-0.062), side_A, op.sail_side),
                      ((-0.006,-0.537,-0.844), (-0.996,0.044,-0.076), side_A, op.sail_side),
                      ((-0.012,-0.844,0.537), (-0.999,0.0294,-0.012), side_A, op.sail_side))
            for (n, r̂v, A, ρs) in sail_t
                n̂ = _unit(n); r̂ = _unit(r̂v)
                r = _mag_from_x(r̂, _GOES8_SAIL_CX) * r̂
                push!(fs, Facet(A, _goes8_perm(n̂), _goes8_perm(r), ρs...))
            end
        else
            # Idealised lateral surface with `n_sail_sides` area-preserving panels.
            n̂_base = _unit((-0.999, 0.013, -0.001))
            r̂_base = _unit((-0.997, 0.064, -0.034))
            r_base = _mag_from_x(r̂_base, _GOES8_SAIL_CX) * r̂_base     # base-disc centre
            push!(fs, Facet(base_A, _goes8_perm(n̂_base), _goes8_perm(r_base), op.sail_base...))

            â = n̂_base                       # base outward normal = apex→base axis
            û = _unit(cross(â, SVector{3,Float64}(0.0, 0.0, 1.0)))
            v̂ = cross(â, û)                  # (û, v̂, â) right-handed
            apex = r_base - h * â
            Δ = π / n_sail_sides
            for j in 0:(n_sail_sides - 1)
                φ = 2π * (j + 0.5) / n_sail_sides
                ê = cos(φ) * û + sin(φ) * v̂
                if sail === :cone
                    # true cone outward normal: (h ê − R â)/√(h²+R²)
                    n̂  = _unit(h * ê - R * â)
                    P1 = r_base + R * (cos(φ - Δ) * û + sin(φ - Δ) * v̂)
                    P2 = r_base + R * (cos(φ + Δ) * û + sin(φ + Δ) * v̂)
                    cen = (apex + P1 + P2) / 3          # flat-panel centroid
                else                                    # :cylinder
                    n̂  = ê
                    cen = r_base - (h/2) * â + R * ê
                end
                push!(fs, Facet(lat_A / n_sail_sides, _goes8_perm(n̂),
                                _goes8_perm(cen), op.sail_side...))
            end
        end
    end

    tag = sail === :albuja ? "Albuja-15facet" :
          "sail=$(sail)×$(n_sail_sides) ($(length(fs)) facets)"
    return ShapeModel(fs; name = "GOES-8 $tag (θ_sa=$(round(rad2deg(θ_sa),digits=1))°, $optical)")
end
