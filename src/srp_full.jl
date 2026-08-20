
# defined in B&S 2021 p.751 under EQ. 11
const P_SRP_1AU = 4.56e-6        # [N/m²] nominal SRP at 1 AU (B&S 2021 §II.D)
const B_LAMBERT = 2.0 / 3.0      # scattering coefficient for Lambertian reflection

# Eq 11 B&S 2021
function srp_facet_force(f::Facet, û; P_SRP::Real = P_SRP_1AU)
    u = SVector{3,Float64}(û[1], û[2], û[3])
    μ = dot(u, f.normal)                      # û·n̂ᵢ
    μ ≤ 0 && return SVector{3,Float64}(0.0, 0.0, 0.0)   # illumination function

    c_d = B_LAMBERT * (1.0 - f.s * f.ρ)       # diffuse + thermal re-emission
    # [ρs(2 n̂n̂ᵀ − U) + U]·û = (1 − ρs) û + 2ρs (û·n̂) n̂
    reflected = (1.0 - f.ρ * f.s) * u + (2.0 * f.ρ * f.s * μ + c_d) * f.normal
    return -P_SRP * f.area * μ * reflected
end
# eq 12 to give the total force 
function srp_force_torque(shape::ShapeModel, û; P_SRP::Real = P_SRP_1AU)
    F = SVector{3,Float64}(0.0, 0.0, 0.0)
    M = SVector{3,Float64}(0.0, 0.0, 0.0)
    for f in shape.facets
        fi = srp_facet_force(f, û; P_SRP = P_SRP)
        F += fi
        M += cross(f.centroid, fi)
    end
    return F, M
end
# return the body frame torque from srp
srp_torque(shape::ShapeModel, û; P_SRP::Real = P_SRP_1AU) =
    srp_force_torque(shape, û; P_SRP = P_SRP)[2]
