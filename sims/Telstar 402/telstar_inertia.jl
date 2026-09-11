
using .master
using LinearAlgebra, Printf, StaticArrays


# ASSUMED INPUTS - Edit here


# [SOURCED, FROM THE SISTER SHIP]
              
const BUS_DIMS_M = (4.08, 2.22, 2.54)

const BUS_AXIS_ORDER = (1, 2, 3)      # permutation applied to BUS_DIMS_M

#  Masses 

const TOTAL_DRY_MASS_KG   = 1700.0

const BUS_MASS_FRACTION   = 0.90                         

# Solar arrays
# Array wing geometry is not published.  Modelled as two identical thin flat
# rectangular plates, one per wing, deployed along ±ŷ_s.
const N_PANELS            = 2
const PANEL_SPAN_M        = 7.0     # along the boom axis ŷ_s  
const PANEL_CHORD_M       = 2.4     # across the boom, along x̂_s 
# Gap between the bus face and the INBOARD edge of the panel (yoke / boom).
const BOOM_OFFSET_M       = 1.0     
const ARRAY_ANGLE_DEG     = 0.0


const ASSUMPTIONS = Tuple{String,String,String}[]   # (label, value, provenance)
note!(label, value, prov) = push!(ASSUMPTIONS, (label, value, prov))

# Build the primitive list in the spacecraft-designer frame


dx, dy, dz = BUS_DIMS_M[BUS_AXIS_ORDER[1]], BUS_DIMS_M[BUS_AXIS_ORDER[2]],
             BUS_DIMS_M[BUS_AXIS_ORDER[3]]

m_bus   = BUS_MASS_FRACTION * TOTAL_DRY_MASS_KG
m_panel = (1 - BUS_MASS_FRACTION) * TOTAL_DRY_MASS_KG / N_PANELS

# Panel centroid: bus half-width + boom + half the span, along ±ŷ_s.
y_panel = dy/2 + BOOM_OFFSET_M + PANEL_SPAN_M/2

# R maps body → panel-local components; rotating about ŷ_s tilts the panel
# plane while leaving the span along the boom.
R_panel = R2(deg2rad(ARRAY_ANGLE_DEG))

parts = MassPrimitive[
    SolidBox(m_bus, (dx, dy, dz)),
    (ThinPlate(m_panel, (PANEL_CHORD_M, PANEL_SPAN_M);
               centroid = (0.0, s * y_panel, 0.0), R = R_panel)
     for s in (+1.0, -1.0))...
]

mp = composite_inertia(parts)          # errors if the result is unphysical
P  = mp.principal

# Report
println("="^78)
println("Telstar 401 — APPROXIMATE principal inertias from a primitive assembly")
println("="^78)

@printf("\ntotal mass            %10.2f kg\n", mp.mass)
@printf("centre of mass (x,y,z) [%.4f, %.4f, %.4f] m  (designer frame)\n",
        mp.com[1], mp.com[2], mp.com[3])

println("\ninertia tensor about the centre of mass, designer frame [kg m²]:")
for i in 1:3
    @printf("   %14.2f %14.2f %14.2f\n", mp.I_com[i,1], mp.I_com[i,2], mp.I_com[i,3])
end
offdiag = maximum(abs.(mp.I_com - Diagonal(diag(mp.I_com))))
@printf("max |off-diagonal|    %10.4e kg m²  (%.2e of I_s)\n", offdiag, offdiag/P.Is)
println(offdiag / P.Is < 1e-12 ?
        "  → the assumed layout is symmetric, so the designer frame IS principal." :
        "  → products of inertia are non-zero; the eigen-solve below is doing real work.")

println("\nprincipal inertias, project long-axis convention [kg m²]:")
@printf("   I_l (min, b̂₃) %12.2f\n", P.Il)
@printf("   I_i (mid, b̂₁) %12.2f\n", P.Ii)
@printf("   I_s (max, b̂₂) %12.2f\n", P.Is)
@printf("   ratios:  I_i/I_s = %.4f   I_l/I_s = %.4f\n", P.Ii/P.Is, P.Il/P.Is)

println("\nprincipal axes in designer-frame components:")
for (nm, v) in zip(("b̂₁ (I_i)", "b̂₂ (I_s)", "b̂₃ (I_l)"), mp.axes)
    @printf("   %-9s [%+.6f, %+.6f, %+.6f]\n", nm, v[1], v[2], v[3])
end

println("\nrealizability:")
@printf("   all moments positive          %s\n", all(x -> x > 0, (P.Il, P.Ii, P.Is)) ? "yes" : "NO")
@printf("   I_s ≤ I_i + I_l               %s  (%.2f ≤ %.2f)\n",
        P.Is ≤ P.Ii + P.Il ? "yes" : "NO", P.Is, P.Ii + P.Il)
@printf("   trace check tr(I) = ΣI_k      %.6e vs %.6e\n",
        tr(mp.I_com), P.Il + P.Ii + P.Is)

# Context: how this body compares with the one satellite we DO have published
# inertias for.
G = goes8_inertia()
println("\nshape-class comparison with GOES-8 (published, B&S 2021 Table 1):")
@printf("   %-12s %10s %10s\n", "", "Telstar*", "GOES-8")
@printf("   %-12s %10.4f %10.4f\n", "I_i/I_s", P.Ii/P.Is, G.Ii/G.Is)
@printf("   %-12s %10.4f %10.4f\n", "I_l/I_s", P.Il/P.Is, G.Il/G.Is)
@printf("   %-12s %10.1f %10.1f\n", "I_s [kg m²]", P.Is, G.Is)
println("   (* approximate, built from the assumptions listed below)")

# Assumption ledger
note!("bus box dimensions", @sprintf("%.2f × %.2f × %.2f m", BUS_DIMS_M...),
      "TELSTAR 402 (sister ship) — assumed identical bus [FLAG-TELSTAR-BUS]")
note!("bus dimension → axis map", string(BUS_AXIS_ORDER) * " as (x̂_s, ŷ_s, ẑ_s)",
      "ASSUMED — source gives dimensions, not orientations [FLAG-TELSTAR-AXES]")
note!("total dry mass", @sprintf("%.1f kg", TOTAL_DRY_MASS_KG),
      "ASSUMED — no traceable source; all moments scale linearly [FLAG-TELSTAR-MASS]")
note!("bus mass fraction", @sprintf("%.2f (bus %.1f kg, each wing %.1f kg)",
                                    BUS_MASS_FRACTION, m_bus, m_panel),
      "ASSUMED — split not published [FLAG-TELSTAR-MASS]")
note!("panels", @sprintf("%d × thin plate, %.2f m span × %.2f m chord",
                         N_PANELS, PANEL_SPAN_M, PANEL_CHORD_M),
      "ASSUMED — array geometry not published [FLAG-TELSTAR-ARRAY]")
note!("boom offset", @sprintf("%.2f m bus face → inboard panel edge", BOOM_OFFSET_M),
      "ASSUMED [FLAG-TELSTAR-ARRAY]")
note!("array angle at EOL", @sprintf("%.1f° about the boom axis", ARRAY_ANGLE_DEG),
      "ASSUMED — GOES-8's θ_sa = 17° has no Telstar equivalent [FLAG-TELSTAR-ARRAY]")
note!("panel thickness", "zero (thin-plate limit)",
      "MODEL CHOICE — a real ~3 cm honeycomb panel adds <0.1% to I")
note!("everything else", "no tanks, antennas, booms, MLI or trim tabs modelled",
      "MODEL CHOICE — bus + 2 wings only")

println("\n" * "="^78)
println("ASSUMED VALUES USED IN THIS ESTIMATE — none of these is a measurement")
println("="^78)
for (k, v, p) in ASSUMPTIONS
    @printf("  %-26s %s\n", k, v)
    @printf("  %-26s   ↳ %s\n", "", p)
end
println("""
The only externally sourced number above is the bus box, and it belongs to
Telstar 402.  Treat the resulting tensor as a shape-class placeholder for
sensitivity work, not as Telstar 401's inertia.""")
