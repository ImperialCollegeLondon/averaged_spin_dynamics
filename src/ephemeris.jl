
const SECONDS_PER_YEAR = 365.25 * 86400.0   # [s]
const OBLIQUITY        = deg2rad(23.4393)    # ecliptic obliquity (unused if frame=ecliptic)

# sun direction

function sun_direction_ecliptic(t::Real; t0::Real = 0.0)
    λ = 2π * (t - t0) / SECONDS_PER_YEAR     # ecliptic longitude
    return SVector{3,Float64}(cos(λ), sin(λ), 0.0)
end

# quaternion

function quat_normalize(β)
    q = SVector{4,Float64}(β[1], β[2], β[3], β[4])
    return q / norm(q)
end

function quat_to_dcm(β)
    β0, β1, β2, β3 = β[1], β[2], β[3], β[4]
    return @SMatrix [
        β0^2+β1^2-β2^2-β3^2   2(β1*β2+β0*β3)        2(β1*β3-β0*β2);
        2(β1*β2-β0*β3)        β0^2-β1^2+β2^2-β3^2   2(β2*β3+β0*β1);
        2(β1*β3+β0*β2)        2(β2*β3-β0*β1)        β0^2-β1^2-β2^2+β3^2
    ]
end

# B&S 2021, Eq. 10
function quat_kinematics(β, ω)
    β0, β1, β2, β3 = β[1], β[2], β[3], β[4]
    ω1, ω2, ω3 = ω[1], ω[2], ω[3]
    return SVector{4,Float64}(
        0.5 * (-β1*ω1 - β2*ω2 - β3*ω3),
        0.5 * ( β0*ω1 - β3*ω2 + β2*ω3),
        0.5 * ( β3*ω1 + β0*ω2 - β1*ω3),
        0.5 * (-β2*ω1 + β1*ω2 + β0*ω3),
    )
end

function sun_direction_body(β, t::Real; t0::Real = 0.0)
    û_inertial = sun_direction_ecliptic(t; t0 = t0)
    return quat_to_dcm(β) * û_inertial
end
