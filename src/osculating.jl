"""
The osculating elements describe the physical state using α, β, |H|, and I_d or the dynamic moment of inertia.
I_d is how we define whether it is in long axis mode (LAM) or short axis mode (SAM) and the following file will be using the established frames
to derive these quantities in the correct reference and with the correct equations.
"""

# Principal inertias -----------------------------------------------------------------
"""
The Principal inertias are the moment of inertias along the principal axes of the body
and are how we determine which mode the tumbling regime is in.
Il<Ii<Is
"""

struct PrincipalInertias
    Il::Float64
    Ii::Float64
    Is::Float64
    function PrincipalInertias(Il, Ii,Is)
        Il ≤ Ii ≤ Is || throw(ArgumentError("principal inertias not valid"))
        new(Float64(Il), Float64(Ii), Float64(Is))
    end
end

# Osculating state -------------------------------------------------------------------
"setup state elements"
struct OsculatingState
    α::Float64
    β::Float64
    H::Float64
    Id::Float64
end

"need to translate it into a form the differential solver understands and then extract it again"
to_svector(s::OsculatingState) = SVector{4,Float64}(s.α, s.β, s.H, s.Id)
from_svector(::Type{OsculatingState}, v) = OsculatingState(v[1], v[2], v[3], v[4])

# DA BIG ONE (regime classification) -------------------------------------------------

abstract type Regime end
struct LAM <: Regime end #Long axis mode, rotates about the long axis or b^3
struct SAM <: Regime end #short axis mode, rotates about the short axis or b^2
struct Separatrix <: Regime end 

function classify_regime(Id::Real, I::PrincipalInertias)
    if Id < I.Ii - eps(I.Ii)
        return LAM()
    elseif Id > I.Ii + eps(I.Ii)
        return SAM()
    else
        return Separatrix() #this may be where the issue with the Separatrix happens
    end
end

# Perturbation setup for the osculating values

omega_e(H::Real, Id::Real) = H / Id
angular_momentum(ωe::Real, Id::Real) = Id * ωe
kinetic_energy(H::Real, Id::Real) = H^2 / (2 * Id)
Id_from_energy_momentum(T::Real, H::Real) = H^2 / (2 * T)

Base.@kwdef struct PerturbationConfig
    srp::Bool             = true
    dissipation::Bool     = true
    gravity_gradient::Bool = true
    srp_backend::Symbol   = :numeric    # :numeric | :analytic | :fourier
    σ_branch::Int         = 1           # LAM±/SAM± spin branch
    resonant::Bool        = false       # ended up not using due to unsolved errors
    μ::Float64            = 1.0e-3      # get these from B&S
    J::Float64            = 1.0
    orbit_i::Float64      = 0.0
    orbit_Ω::Float64      = 0.0
    orbit_R::Float64      = 4.2575e7
end

