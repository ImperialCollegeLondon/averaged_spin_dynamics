
using .master
using LinearAlgebra, Printf, StaticArrays

const ECHO_INERTIA_SCRIPT = false
const INERTIA_SCRIPT = joinpath(@__DIR__, "telstar_inertia")

if !isdefined(Main, :BUS_DIMS_M)
    if ECHO_INERTIA_SCRIPT
        Base.include(Main, INERTIA_SCRIPT)
    else
        redirect_stdout(devnull) do
            Base.include(Main, INERTIA_SCRIPT)
        end
    end
end


const INERTIA_REF = Main.mp


# OPTICAL PROPERTIES
#   bus faces  → :bus               (MLI,        ρ=0.60, s=1.0)
#   panel sun  → :solar_array_front (Si cell,    ρ=0.27, s=1.0)
#   panel dark → :solar_array_back  (graphite,   ρ=0.07, s=0.0)
const OPTICAL_BUS         = :bus
const OPTICAL_PANEL_FRONT = :solar_array_front
const OPTICAL_PANEL_BACK  = :solar_array_back

const N_FACETS_EXPECTED = 6 + 2 * N_PANELS


function telstar401_primitives(array_angle_deg::Real = ARRAY_ANGLE_DEG)
    dx = BUS_DIMS_M[BUS_AXIS_ORDER[1]]
    dy = BUS_DIMS_M[BUS_AXIS_ORDER[2]]
    dz = BUS_DIMS_M[BUS_AXIS_ORDER[3]]

    m_bus   = BUS_MASS_FRACTION * TOTAL_DRY_MASS_KG
    m_panel = (1 - BUS_MASS_FRACTION) * TOTAL_DRY_MASS_KG / N_PANELS

    # Panel centroid: bus half-width + boom + half the span, along ±ŷ_s.
    y_panel = dy / 2 + BOOM_OFFSET_M + PANEL_SPAN_M / 2

    # R maps body → panel-local; rotating about ŷ_s tilts the panel plane while
    # leaving the span along the boom.  (Same call as the inertia script.)
    R_panel = R2(deg2rad(array_angle_deg))

    prims = MassPrimitive[
        SolidBox(m_bus, (dx, dy, dz)),
        (ThinPlate(m_panel, (PANEL_CHORD_M, PANEL_SPAN_M);
                   centroid = (0.0, s * y_panel, 0.0), R = R_panel)
         for s in (+1.0, -1.0))...
    ]

    return (prims = prims, dx = dx, dy = dy, dz = dz,
            m_bus = m_bus, m_panel = m_panel,
            y_panel = y_panel, R_panel = R_panel)
end

# Re-express a primitive in a rotated common frame: if v_new = R * v_old then a
# centroid maps as R*c, and a primitive whose own DCM took old→local now needs
# new→local, i.e. R_p * R'.  Used only by the frame check.
_reframe(p::SolidBox,  R) = SolidBox(p.mass, p.dims;  centroid = R * p.centroid, R = p.R * R')
_reframe(p::ThinPlate, R) = ThinPlate(p.mass, p.dims; centroid = R * p.centroid, R = p.R * R')
_reframe(p::PointMass, R) = PointMass(p.mass,          R * p.centroid;           R = p.R * R')

# Inertia accessor 

const _TELSTAR401_INERTIA = let
    P   = composite_inertia(telstar401_primitives().prims).principal
    ref = INERTIA_REF.principal
    (P.Il == ref.Il && P.Ii == ref.Ii && P.Is == ref.Is) || error("""
        telstar401_inertia(): rebuilt tensor differs from telstar401_inertia.jl's
        own composite_inertia result.  This function would be a second, divergent
        source of the same numbers — refusing to define it.
          rebuilt: (Il,Ii,Is) = $((P.Il, P.Ii, P.Is))
          INERTIA_REF:         $((ref.Il, ref.Ii, ref.Is))""")
    P
end


telstar401_inertia() = _TELSTAR401_INERTIA


function telstar401_shape(; array_angle_deg::Real = ARRAY_ANGLE_DEG,
                            frame::Symbol = :principal,
                            check::Bool = true,
                            warn_inert::Bool = true,
                            atol::Real = 1e-9)
    frame in (:principal, :designer) ||
        throw(ArgumentError("telstar401_shape: `frame` must be :principal or :designer, got :$frame"))

    G   = telstar401_primitives(array_angle_deg)
    mpd = composite_inertia(G.prims)          # designer-frame assembly

    # Guard against this file and the inertia script drifting apart: at the
    # default array angle they must describe the SAME body, bit for bit.
    if array_angle_deg == ARRAY_ANGLE_DEG
        ref = INERTIA_REF.principal
        (isapprox(mpd.principal.Il, ref.Il; rtol = 1e-12) &&
         isapprox(mpd.principal.Ii, ref.Ii; rtol = 1e-12) &&
         isapprox(mpd.principal.Is, ref.Is; rtol = 1e-12)) || error("""
            telstar401_shape: the primitive assembly rebuilt here does NOT match
            telstar401_inertia.jl's.  The two files have drifted apart.
              here: (Il,Ii,Is) = $((mpd.principal.Il, mpd.principal.Ii, mpd.principal.Is))
              there:(Il,Ii,Is) = $((ref.Il, ref.Ii, ref.Is))""")
    end

    R = frame === :principal ? mpd.R_bp : one(SMatrix{3,3,Float64,9})

    facets = Facet[]

    #  Bus: six faces of the box, half-extents from the FULL side lengths 
    hx, hy, hz = G.dx / 2, G.dy / 2, G.dz / 2
    ρb, sb = optical(OPTICAL_BUS)
    box_faces = (                                     # (normal, centroid, area)
        (SVector( 1.0, 0.0, 0.0), SVector( hx, 0.0, 0.0), G.dy * G.dz),
        (SVector(-1.0, 0.0, 0.0), SVector(-hx, 0.0, 0.0), G.dy * G.dz),
        (SVector( 0.0, 1.0, 0.0), SVector(0.0,  hy, 0.0), G.dx * G.dz),
        (SVector( 0.0,-1.0, 0.0), SVector(0.0, -hy, 0.0), G.dx * G.dz),
        (SVector( 0.0, 0.0, 1.0), SVector(0.0, 0.0,  hz), G.dx * G.dy),
        (SVector( 0.0, 0.0,-1.0), SVector(0.0, 0.0, -hz), G.dx * G.dy),
    )
    for (n, c, A) in box_faces
        push!(facets, Facet(A, R * n, R * c, ρb, sb))
    end

    # Solar arrays: front/back pair per wing, at the plate centroid
    # Zero thickness, so both faces share one centroid (as `goes8_shape` does).
    # The panel-local +ẑ is the sun side; its designer-frame components are
    # R_panel' * ẑ = (sinθ, 0, cosθ), i.e. ẑ_s at ARRAY_ANGLE_DEG = 0.
    A_panel  = PANEL_SPAN_M * PANEL_CHORD_M
    n_front  = G.R_panel' * SVector(0.0, 0.0, 1.0)
    ρf, sf = optical(OPTICAL_PANEL_FRONT)
    ρr, sr = optical(OPTICAL_PANEL_BACK)
    for sgn in (+1.0, -1.0)
        c = SVector(0.0, sgn * G.y_panel, 0.0)
        push!(facets, Facet(A_panel, R *  n_front, R * c, ρf, sf))
        push!(facets, Facet(A_panel, R * -n_front, R * c, ρr, sr))
    end

    length(facets) == N_FACETS_EXPECTED ||
        error("telstar401_shape: built $(length(facets)) facets, expected $N_FACETS_EXPECTED")

    if check
        _assert_frame_consistent(G, mpd, R, frame; atol = atol)
    end

    if warn_inert
        @warn """
        telstar401_shape: this geometry has IDENTICALLY ZERO net SRP torque \
        (~1e-21 N·m vs ~3e-4 N·m for goes8_shape) — the bus box self-cancels \
        and the two symmetric wings cancel each other.  It cannot drive a YORP \
        or spin-up study.  See [FLAG-TELSTAR-INERT] at the top of \
        scripts/validation/telstar401_shape.jl.""" maxlog = 1
    end

    nm = frame === :principal ? "Telstar 401 (box-wing approx)" :
                                "Telstar 401 (box-wing approx, DESIGNER FRAME — not principal)"
    return ShapeModel(facets; name = nm)
end


function _assert_frame_consistent(G, mpd, R, frame::Symbol; atol::Real = 1e-9)
    mpr = composite_inertia([_reframe(p, R) for p in G.prims])

    # Moments are frame-invariant — a cheap check that _reframe is not lying.
    scale = mpd.principal.Is
    for (nm, a, b) in (("Il", mpr.principal.Il, mpd.principal.Il),
                       ("Ii", mpr.principal.Ii, mpd.principal.Ii),
                       ("Is", mpr.principal.Is, mpd.principal.Is))
        abs(a - b) ≤ atol * scale ||
            error("telstar401_shape frame check: $nm changed under rotation " *
                  "($a vs $b) — the reframing is wrong, not the frame.")
    end

    if frame === :principal
        # The real assertion: in the facet frame the tensor must ALREADY be
        # principal, in the project's (b̂₁,b̂₂,b̂₃) → (I_i,I_s,I_l) order.
        dev = maximum(abs.(mpr.R_bp - one(SMatrix{3,3,Float64,9})))
        dev ≤ 1e-8 || error("""
            telstar401_shape: facet frame is NOT the principal frame of the
            paired inertia tensor (max |R_bp − 𝟙| = $dev).  Refusing to return a
            shape whose normals and inertia axes disagree.
            R_bp in the facet frame =
            $(mpr.R_bp)""")

        offd = maximum(abs.(mpr.I_com - Diagonal(diag(mpr.I_com))))
        offd ≤ 1e-8 * scale || error("telstar401_shape: residual products of " *
                                     "inertia in the facet frame ($offd kg·m²)")
    end
    return mpr
end

# Report + sanity checks (only when run as a script)

function facet_mass_centroid(shape::ShapeModel, G)
    n_box = 6
    A_box = sum(f.area for f in shape.facets[1:n_box])
    m = Float64[]
    for (k, f) in enumerate(shape.facets)
        push!(m, k ≤ n_box ? G.m_bus * f.area / A_box : G.m_panel / 2)
    end
    M = sum(m)
    c = sum(m[k] * shape.facets[k].centroid for k in eachindex(m)) / M
    return c, M
end


function srp_torque_survey(shape::ShapeModel; n::Int = 200)
    Tmax, Tsum, Fmax = 0.0, 0.0, 0.0
    ga = π * (3 - sqrt(5.0))                      # golden angle
    for k in 0:(n-1)
        z  = 1 - 2(k + 0.5) / n
        r  = sqrt(max(0.0, 1 - z^2))
        φ  = ga * k
        û  = SVector(r*cos(φ), r*sin(φ), z)
        F, T = srp_force_torque(shape, û)
        nT = norm(T); Tmax = max(Tmax, nT); Tsum += nT
        Fmax = max(Fmax, norm(F))
    end
    return (Tmax, Tsum / n, Fmax)
end

function main()
    shape = telstar401_shape(; warn_inert = false)   # reported in full below
    G     = telstar401_primitives()
    mpd   = composite_inertia(G.prims)
    P     = mpd.principal

    println("="^78)
    println("Telstar 401 — APPROXIMATE box-wing facet model")
    println("="^78)
    println("""
This shape is the SRP counterpart of scripts/validation/telstar401_inertia.jl
and shares every geometry and mass constant with it.  Both are placeholders.""")

    # Frame
    println("\n── FRAME ──────────────────────────────────────────────────────")
    println("R_bp (designer → project body), from composite_inertia:")
    for i in 1:3
        @printf("   %+8.5f %+8.5f %+8.5f\n", mpd.R_bp[i,1], mpd.R_bp[i,2], mpd.R_bp[i,3])
    end
    println("rows are b̂₁ᵀ, b̂₂ᵀ, b̂₃ᵀ in designer-frame components, i.e.")
    for (nm, v) in zip(("b̂₁ (I_i)", "b̂₂ (I_s)", "b̂₃ (I_l)"), mpd.axes)
        @printf("   %-9s = [%+.4f, %+.4f, %+.4f] in (x̂_s, ŷ_s, ẑ_s)\n", nm, v[1], v[2], v[3])
    end
    @printf("det(R_bp) = %+.12f   (must be +1: right-handed, no reflection)\n", det(mpd.R_bp))

    # The check that would otherwise be invisible: run it and say so out loud.
    mpr = _assert_frame_consistent(G, mpd, mpd.R_bp, :principal)
    dev = maximum(abs.(mpr.R_bp - one(SMatrix{3,3,Float64,9})))
    println("\nframe verification (approach (b), run in addition to (a)):")
    @printf("   re-eigendecomposed in the FACET frame: max |R_bp − 𝟙| = %.3e  → %s\n",
            dev, dev ≤ 1e-8 ? "PASS" : "FAIL")
    @printf("   principal moments unchanged by the rotation:            %s\n",
            (isapprox(mpr.principal.Il, P.Il; rtol=1e-12) &&
             isapprox(mpr.principal.Ii, P.Ii; rtol=1e-12) &&
             isapprox(mpr.principal.Is, P.Is; rtol=1e-12)) ? "PASS" : "FAIL")
    @printf("   matches telstar401_inertia.jl's own tensor:             %s\n",
            (P.Il == INERTIA_REF.principal.Il &&
             P.Ii == INERTIA_REF.principal.Ii &&
             P.Is == INERTIA_REF.principal.Is) ? "PASS" : "FAIL")

    # Facets
    println("\n── FACETS ─────────────────────────────────────────────────────")
    @printf("facet count  %d  (expected %d: 6 bus faces + 2 faces × %d wings)\n",
            n_facets(shape), N_FACETS_EXPECTED, N_PANELS)
    @printf("%-4s %-20s %8s   %-24s %-26s %6s %5s\n",
            "#", "component", "area m²", "n̂ (b̂₁,b̂₂,b̂₃)", "r (b̂₁,b̂₂,b̂₃) [m]", "ρ", "s")
    labels = vcat(fill("bus box face", 6),
                  reduce(vcat, [["array wing $(w) front", "array wing $(w) back"]
                                for w in 1:N_PANELS]))
    for (k, f) in enumerate(shape.facets)
        @printf("%-4d %-20s %8.3f   [%+.3f %+.3f %+.3f]     [%+7.3f %+7.3f %+7.3f]  %5.2f %5.2f\n",
                k, labels[k], f.area,
                f.normal[1], f.normal[2], f.normal[3],
                f.centroid[1], f.centroid[2], f.centroid[3], f.ρ, f.s)
    end

    # Areas 
    # `total_area` sums every facet
    A_box_closed = 2 * (G.dx*G.dy + G.dy*G.dz + G.dz*G.dx)
    A_panels_1s  = N_PANELS * PANEL_SPAN_M * PANEL_CHORD_M
    A_proj_mean  = A_box_closed / 4 + A_panels_1s / 2

    g      = goes8_shape()
    Gbox   = 2 * (2.0*2.2 + 2.2*3.4 + 3.4*2.0)     # goes8_shape bus, full extents
    Gproj  = Gbox / 4 + 12.0 / 2 + 1.5 / 2         # + one wing + trim tab

    println("\n── AREAS ──────────────────────────────────────────────────────")
    @printf("total_area (all facets, both plate sides)   %8.2f m²\n", total_area(shape))
    @printf("  bus, six faces                            %8.2f m²\n", A_box_closed)
    @printf("  arrays, %d wings × 2 sides                 %8.2f m²\n",
            N_PANELS, 2 * A_panels_1s)
    @printf("mean projected area (box/4 + plates/2)      %8.2f m²   ← the 'order 10–30 m²' figure\n",
            A_proj_mean)
    @printf("area-to-mass (projected / %.0f kg dry)      %8.5f m²/kg\n",
            TOTAL_DRY_MASS_KG, A_proj_mean / TOTAL_DRY_MASS_KG)
    println("\nsame quantities for goes8_shape(), for scale:")
    @printf("   total_area %6.2f m²   mean projected %6.2f m²   A/m %.5f m²/kg (972 kg)\n",
            total_area(g), Gproj, Gproj / 972.0)
    println("""
   → Telstar is the larger body (bigger bus, two wings vs one), and its
     projected area sits at the top of the 10–30 m² band while its
     area-to-mass ratio lands within a few % of GOES-8's.  A summed
     `total_area` of ~117 m² is NOT off by 10×; it is the same number counted
     the other way.""")

    # Mass-weighted facet centroid 
    c_f, M_f = facet_mass_centroid(shape, G)
    println("\n── MASS-WEIGHTED FACET CENTROID (frame check) ─────────────────")
    @printf("mass accounted for   %10.2f kg  (assembly total %.2f kg)\n", M_f, mpd.mass)
    @printf("facet centroid       [%+.3e, %+.3e, %+.3e] m  in (b̂₁,b̂₂,b̂₃)\n",
            c_f[1], c_f[2], c_f[3])
    @printf("‖centroid‖           %.3e m   vs body scale %.2f m  → %s\n",
            norm(c_f), G.y_panel + PANEL_SPAN_M/2,
            norm(c_f) < 1e-9 ? "at the origin, as required" : "OFF-CENTRE — investigate")
    @printf("composite CoM from the mass model   [%+.3e, %+.3e, %+.3e] m\n",
            mpd.com[1], mpd.com[2], mpd.com[3])
    println("""
   (Exactly zero is expected: the assumed layout is symmetric about the CoM in
    all three axes, so this confirms the rotation was applied to centroids as
    well as normals, rather than probing an asymmetry that isn't there.)""")

    # Inertia pairing
    println("\n── PAIRED INERTIA (from telstar401_inertia.jl) ────────────────")
    @printf("   I_l (min, b̂₃) %10.2f     I_i (mid, b̂₁) %10.2f     I_s (max, b̂₂) %10.2f kg m²\n",
            P.Il, P.Ii, P.Is)
    @printf("   I_i/I_s = %.4f   I_l/I_s = %.4f   (GOES-8: %.4f, %.4f)\n",
            P.Ii/P.Is, P.Il/P.Is,
            goes8_inertia().Ii/goes8_inertia().Is, goes8_inertia().Il/goes8_inertia().Is)
    println("   b̂₃ (minimum inertia) lies along the solar-array boom, as it must")
    println("   for a bus-plus-two-wings body — an independent sign that the")
    println("   eigen-solve and the facet layout agree about which axis is which.")

    # YORP asymmetry: the check that decides whether this shape is usable 
    println("\n── NET SRP TORQUE  [FLAG-TELSTAR-INERT] ───────────────────────")
    sub(idx) = ShapeModel(shape.facets[idx]; name = "subset")
    Tt, Tt_m, Ft = srp_torque_survey(shape)
    Tg, Tg_m, Fg = srp_torque_survey(g)
    @printf("%-28s %12s %12s %12s\n", "shape", "max|T| N m", "mean|T|", "max|F| N")
    @printf("%-28s %12.4e %12.4e %12.4e\n", "Telstar 401 (this file)", Tt, Tt_m, Ft)
    @printf("%-28s %12.4e %12.4e %12.4e\n", "goes8_shape() reference", Tg, Tg_m, Fg)
    @printf("ratio Telstar/GOES-8 (max|T|)   %.3e\n", Tt / Tg)

    println("\nwhere the torque lives (max|T| over the same 200 directions):")
    for (nm, idx) in (("bus box alone (facets 1-6)",  1:6),
                      ("both wings (facets 7-10)",    7:10),
                      ("one wing alone (facets 7-8)", 7:8))
        @printf("   %-30s %12.4e N m\n", nm, srp_torque_survey(sub(collect(idx)))[1])
    end

    # (dᵢ/2)·Aᵢ = V/2 on every face of a box — the exact reason the bus cancels.
    @printf("\nbus moment-arm × area product, all six faces (V/2 = %.6f m³):\n",
            G.dx*G.dy*G.dz/2)
    @printf("   ")
    for f in shape.facets[1:6]
        @printf("%.4f  ", norm(f.centroid) * f.area)
    end
    println("\n   → identical, so the lit bus faces cancel in pairs for any û.")

    at_t = averaged_srp_torques(shape, deg2rad(60), 2π/1200, 0.5*(P.Ii + P.Is), P,
                                SAM(); N_φ = 45, N_τ = 90)
    Ig2  = goes8_inertia()
    at_g = averaged_srp_torques(g, deg2rad(60), 2π/1200, 0.5*(Ig2.Ii + Ig2.Is), Ig2,
                                SAM(); N_φ = 45, N_τ = 90)
    @printf("\ntumbling-averaged |M_H|:  Telstar %.4e N m   GOES-8 %.4e N m\n",
            norm(at_t.M_H), norm(at_g.M_H))

    println("""
   *** THIS SHAPE IS DYNAMICALLY INERT.""")

    # Assumption ledger
    println("\n" * "="^78)
    println("ASSUMED VALUES USED IN THIS SHAPE — none of these is a measurement")
    println("="^78)
    ledger = [
        ("bus facet optics (ρ, s)",
         @sprintf("(%.2f, %.1f)  — MLI", optical(OPTICAL_BUS)...),
         "BORROWED FROM GOES-8, B&S 2021 Table 1 — same-era, same-technology-class placeholder [FLAG-TELSTAR-OPTICAL]"),
        ("panel front optics (ρ, s)",
         @sprintf("(%.2f, %.1f)  — Si solar cell", optical(OPTICAL_PANEL_FRONT)...),
         "BORROWED FROM GOES-8 [FLAG-TELSTAR-OPTICAL]"),
        ("panel back optics (ρ, s)",
         @sprintf("(%.2f, %.1f)  — graphite substrate", optical(OPTICAL_PANEL_BACK)...),
         "BORROWED FROM GOES-8 [FLAG-TELSTAR-OPTICAL]"),
        ("facet decomposition",
         @sprintf("%d facets: 6 bus faces + front/back × %d wings", N_FACETS_EXPECTED, N_PANELS),
         "MODEL CHOICE — box-wing, mirroring goes8_shape(); no antennas, thrusters, trim tabs or self-shadowing"),
        ("panel thickness",
         "zero (both faces share one centroid)",
         "MODEL CHOICE — matches the ThinPlate used for the inertia"),
        ("facet frame",
         "rotated by composite_inertia's R_bp into (b̂₁,b̂₂,b̂₃)",
         "DERIVED, not assumed — recomputed from the shared constants and verified above"),
        ("net SRP torque",
         "IDENTICALLY ZERO — no asymmetry in the assumed geometry",
         "CONSEQUENCE, not an assumption — the shape cannot drive a YORP study until an asymmetry is chosen and added to telstar401_inertia.jl [FLAG-TELSTAR-INERT]"),
    ]
    for (k, v, p) in ledger
        @printf("  %-26s %s\n", k, v)
        @printf("  %-26s   ↳ %s\n", "", p)
    end
    println("""
GEOMETRY AND MASS ASSUMPTIONS ARE NOT REPEATED HERE. 
""")

    return shape
end

if abspath(PROGRAM_FILE) == @__FILE__      # `@__FILE__` swallows the rest of a
    main()                                 # `&&` line, so this is written out.
end
