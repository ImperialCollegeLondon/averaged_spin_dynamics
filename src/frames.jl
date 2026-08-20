"""
defining the reference frames for the model as need to define the moment of inertias in the body frame to determine the AXIS MODE.
given values in the inertial frame we use a frame chain to reach the body frame, with rotations defined by B&S 2021. 
This chain is Inertial (N) -> Heliocentric (O) -> angular momentum (H) -> Body (B)
The convention for the inertial axes is Is (short axis, highest inertia) = b̂₂, Il (long axis, lowest inertia) = b̂₃, Ii (intermediate axis) = b̂₁, Is>Ii>Il
"""

# Elementary Rotations --------------------------------------------------------------------------------------------------
"""
we can define the rotations as a combination of the three elementary rotations about the x,y, and z axes.
We will start by defining these three.
"""
function R1(θ::Real)
    c, s = cos(θ), sin(θ)
    @SMatrix [ 1.0 0.0 0.0;
               0.0  c   s; 
               0.0 -s   c; ]
end
"rotation about the x axis in theta (radians)"

function R2(θ::Real)
    c, s = cos(θ), sin(θ)
    @SMatrix [  c  0.0 -s ;
               0.0 1.0 0.0; 
                s  0.0  c ; ]
end
"rotation about the y axis in theta (radians)"

function R3(θ::Real)
    c, s = cos(θ), sin(θ)
    @SMatrix [  c   s  0.0;
               -s   c  0.0; 
               0.0 0.0 1.0; ]
end
"rotation about the z axis in theta (radians)"

# Reference frame transformations --------------------------------------------------------------------------------------
"these will be taken as defined in II.A in B&S 2021"

"from O frame to H frame"
function HO_mat(α::Real, β::Real)
    R2(β)*R3(α)
end

"from H frame to B frame"
function BH_mat(ϕ::Real, θ::Real, ψ::Real)
    R3(ϕ)*R1(θ)*R3(ψ)
end

# Getting the direction of the angular momentum vector from the spherical coordinates ----------------------------------
function H_alpha_beta(α::Real, β::Real)
    sb, cb = sin(β), cos(β);
    sa, ca = sin(α), cos(α);
    SVector{3, Float64}(sb*ca, sb*sa, cb)
end

"the inverse i.e. getting alpha and beta from H"

function a_b_H(H::SVector{3, Float64})
    β = acos(clamp(H[3], -1.0, 1.0))
    α = atan(H[2], H[1])
    return α, β
end

# preventing singularity ---------------------------------------------------------------------------
"when H points along the Z axis beta will tend to zero resulting in a singularity so need a way of transforming between coord systems"

function to_cart(α::Real, β::Real)
    sb = sin(β)
    Hx = cos(α)*sb
    Hy = sin(α)*sb
    return Hx, Hy
end

function from_cart(Hx::Real, Hy::Real)
    β = asin(clamp(sqrt(Hx^2 + Hy^2), 0.0, 1.0))
    α = atan(Hy, Hx)
    return α, β
end

"to prevent any propagation issues/singularity by checking proximity to the singularity"
function sinβ_prev(β::Real; tol::Real=1e-6)
    abs(sin(β)) < tol
end


# Geocentric frame -------------------------------------------------------------------------------
"""necessary for the gravity gradient torque from B&S 2022 to be calculated 
getting the geocentric frame from the orbital elements"""
function G_from_orbel(Ω::Real, i::Real)
    R1(i)*R3(Ω)
end

"need to find H from geo frame in the heliocentric O frame"
function H_G_in_O(λ::Real,δ::Real)
    cd, sd = cos(δ), sin(δ)
    cl, sl = cos(λ), sin(λ)
    SVector{3, Float64}(cd*cl, cd*sl, sd)
end