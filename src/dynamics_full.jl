using DifferentialEquations

inertia_body(I::PrincipalInertias) = SVector{3,Float64}(I.Ii, I.Is, I.Il)

#equations 9-10 in B&S 2021
function euler_rates(β, ω, Idiag, M)
    I1, I2, I3 = Idiag
    ω1, ω2, ω3 = ω[1], ω[2], ω[3]
    dω = SVector{3,Float64}(
        (M[1] - (I3 - I2) * ω2 * ω3) / I1,
        (M[2] - (I1 - I3) * ω3 * ω1) / I2,
        (M[3] - (I2 - I1) * ω1 * ω2) / I3,
    )
    return quat_kinematics(β, ω), dω
end

function _full_rhs(u, p, t)
    Idiag, torque_fn = p
    β = SVector{4,Float64}(u[1], u[2], u[3], u[4])
    ω = SVector{3,Float64}(u[5], u[6], u[7])
    M = torque_fn(β, t)
    dβ, dω = euler_rates(β, ω, Idiag, M)
    return SVector{7,Float64}(dβ[1], dβ[2], dβ[3], dβ[4], dω[1], dω[2], dω[3])
end

# torque free model start
zero_torque(β, t) = SVector{3,Float64}(0.0, 0.0, 0.0)

#srp torque setup
srp_torque_fn(shape::ShapeModel; P_SRP::Real = P_SRP_1AU, t0::Real = 0.0) =
    (β, t) -> srp_torque(shape, sun_direction_body(β, t; t0 = t0); P_SRP = P_SRP)

# propagation of the dynamics
function propagate_full(I::PrincipalInertias, β0, ω0, tspan;
                        torque = zero_torque,
                        solver = Vern9(), reltol = 1e-12, abstol = 1e-12,
                        kwargs...)
    Idiag = inertia_body(I)
    β = quat_normalize(β0)
    u0 = SVector{7,Float64}(β[1], β[2], β[3], β[4], ω0[1], ω0[2], ω0[3])
    prob = ODEProblem(_full_rhs, u0, tspan, (Idiag, torque))
    return solve(prob, solver; reltol = reltol, abstol = abstol, kwargs...)
end
# rotational kinetic energy
function kinetic_energy_full(I::PrincipalInertias, ω)
    Idiag = inertia_body(I)
    return 0.5 * (Idiag[1]*ω[1]^2 + Idiag[2]*ω[2]^2 + Idiag[3]*ω[3]^2)
end
# angular momentum
function angular_momentum_full(I::PrincipalInertias, ω)
    Idiag = inertia_body(I)
    return sqrt((Idiag[1]*ω[1])^2 + (Idiag[2]*ω[2])^2 + (Idiag[3]*ω[3])^2)
end
