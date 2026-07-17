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




