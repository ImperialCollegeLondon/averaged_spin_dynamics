
# flat plate that will be used to 'build' the satellite
struct Facet
    area::Float64
    normal::SVector{3,Float64}     # n̂ᵢ, body frame, unit
    centroid::SVector{3,Float64}   # rᵢ, CoM→centroid [m]
    ρ::Float64                     # total reflectivity
    s::Float64                     # specular fraction
    function Facet(area::Real, normal, centroid, ρ::Real, s::Real)
        n  = SVector{3,Float64}(normal[1], normal[2], normal[3])
        nn = norm(n)
        nn > 0          || throw(ArgumentError("facet normal must be non-zero"))
        area > 0        || throw(ArgumentError("facet area must be positive, got $area"))
        0 ≤ ρ ≤ 1       || throw(ArgumentError("ρ ∈ [0,1], got $ρ"))
        0 ≤ s ≤ 1       || throw(ArgumentError("s ∈ [0,1], got $s"))
        c = SVector{3,Float64}(centroid[1], centroid[2], centroid[3])
        new(Float64(area), n / nn, c, Float64(ρ), Float64(s))
    end
end

# collects the facets together into a 'shape model', i.e. satellite approximation
struct ShapeModel
    facets::Vector{Facet}
    name::String
end
ShapeModel(facets::Vector{Facet}; name::String="unnamed") = ShapeModel(facets, name)

n_facets(m::ShapeModel)   = length(m.facets)
total_area(m::ShapeModel) = sum(f.area for f in m.facets)

# all taken from B&S 2021 Table 1
const OPTICAL_PROPERTIES = Dict{Symbol,NamedTuple{(:ρ, :s),Tuple{Float64,Float64}}}(
    :bus               => (ρ = 0.60, s = 1.0),   # MLI
    :solar_array_front => (ρ = 0.27, s = 1.0),   # solar cell
    :solar_array_back  => (ρ = 0.07, s = 0.0),   # graphite
    :trim_tab_front    => (ρ = 0.83, s = 1.0),   # Al tape
    :trim_tab_back     => (ρ = 0.07, s = 0.0),   # graphite
    :solar_sail_side   => (ρ = 0.66, s = 1.0),   # Al Kapton
    :solar_sail_base   => (ρ = 0.83, s = 1.0),   # Al tape
)

optical(component::Symbol) = OPTICAL_PROPERTIES[component]

# prinicpal inertias for GOES 8 given in B&S 2021, diag(Il, Ii, Is)
goes8_inertia() = PrincipalInertias(980.5, 3432.1, 3570.0)
