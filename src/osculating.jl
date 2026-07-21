"""
The osculating elements describe the physical state using α, β, |H|, and I_d or the dynamic moment of inertia.
I_d is how we define whether it is in long axis mode (LAM) or short axis mode (SAM) and the following file will be using the established frames
to derive these quantities in the correct reference and with the correct equations.
"""

# Principal inertias -----------------------------------------------------------------
"""
The Principal inertias are the moment of inertias along the principal axes of the body
and are how we determine which mode the tumbling regime is in.
Il<Ii<Is so let's set that up
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

