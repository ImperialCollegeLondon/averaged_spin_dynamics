
using .master
using LinearAlgebra, Printf, StaticArrays


# SHARED GEOMETRY — pulled from skynet_inertia

const ECHO_INERTIA_SCRIPT = false
const INERTIA_SCRIPT = joinpath(@__DIR__, "skynet_inertia")

if !isdefined(Main, :DRUM_RADIUS_M)
    if ECHO_INERTIA_SCRIPT
        Base.include(Main, INERTIA_SCRIPT)
    else
        redirect_stdout(devnull) do
            Base.include(Main, INERTIA_SCRIPT)
        end
    end
end

# `mp` is the MassProperties from THAT script's own composite_inertia call, for
# the ASYMMETRIC configuration — the reference the frame check is made against.
const INERTIA_REF = Main.mp

# FACETING

const N_DRUM_SIDE_FACETS = 24

# OPTICAL PROPERTIES
#   drum sides + caps → :bus            (MLI blanket,  ρ=0.60, s=1.0)
#   horn             → :trim_tab_front  (Al tape,      ρ=0.83, s=1.0)

const OPTICAL_DRUM = :bus
const OPTICAL_HORN = :trim_tab_front

const N_FACETS_EXPECTED = N_DRUM_SIDE_FACETS + 2 + 6   # sides + 2 caps + horn box

# Primitive assembly — rebuilt from the shared constants


function skynet1a_primitives()
    G = Main.skynet_primitives(:asymmetric)
    return (prims = G.prims,
            r_drum = Main.DRUM_RADIUS_M, h_drum = Main.DRUM_HEIGHT_M,
            horn_dims = Main.HORN_DIMS_M,
            horn_centroid = SVector{3,Float64}(Main.HORN_RADIAL_OFFSET_M, 0.0, G.z_top),
            m_drum = G.m_drum, m_horn = G.m_antenna)
end

# Re-express a primitive in a rotated common frame (same helper as telstar_shape).
_reframe(p::SolidBox,      R) = SolidBox(p.mass, p.dims;      centroid = R * p.centroid, R = p.R * R')
_reframe(p::ThinPlate,     R) = ThinPlate(p.mass, p.dims;     centroid = R * p.centroid, R = p.R * R')
_reframe(p::SolidCylinder, R) = SolidCylinder(p.mass, p.dims; centroid = R * p.centroid, R = p.R * R')
_reframe(p::PointMass,     R) = PointMass(p.mass,             R * p.centroid;            R = p.R * R')

# Inertia accessor 
const _SKYNET1A_INERTIA = let
    P   = composite_inertia(skynet1a_primitives().prims).principal
    ref = INERTIA_REF.principal
    (P.Il == ref.Il && P.Ii == ref.Ii && P.Is == ref.Is) || error("""
        skynet1a_inertia(): rebuilt tensor differs from skynet_inertia's own
        composite_inertia result.  This would be a second, divergent source of
        the same numbers — refusing to define it.
          rebuilt: (Il,Ii,Is) = $((P.Il, P.Ii, P.Is))
          INERTIA_REF:         $((ref.Il, ref.Ii, ref.Is))""")
    P
end



skynet1a_inertia() = _SKYNET1A_INERTIA


# The shape

function skynet1a_shape(; n_side::Int = N_DRUM_SIDE_FACETS,
                          frame::Symbol = :principal,
                          uniform_optics::Bool = false,
                          check::Bool = true,
                          atol::Real = 1e-9)
    frame in (:principal, :designer) ||
        throw(ArgumentError("skynet1a_shape: `frame` must be :principal or :designer, got :$frame"))
    n_side ≥ 3 || throw(ArgumentError("skynet1a_shape: n_side must be ≥ 3, got $n_side"))

    G   = skynet1a_primitives()
    mpd = composite_inertia(G.prims)

    # Guard against this file and the inertia script drifting apart.
    ref = INERTIA_REF.principal
    (isapprox(mpd.principal.Il, ref.Il; rtol = 1e-12) &&
     isapprox(mpd.principal.Ii, ref.Ii; rtol = 1e-12) &&
     isapprox(mpd.principal.Is, ref.Is; rtol = 1e-12)) || error("""
        skynet1a_shape: the primitive assembly rebuilt here does NOT match
        skynet_inertia's.  The two files have drifted apart.
          here: (Il,Ii,Is) = $((mpd.principal.Il, mpd.principal.Ii, mpd.principal.Is))
          there:(Il,Ii,Is) = $((ref.Il, ref.Ii, ref.Is))""")

    R   = frame === :principal ? mpd.R_bp : one(SMatrix{3,3,Float64,9})
    com = mpd.com                                   # designer frame

    # Every facet centroid goes through this: shift to the CoM, THEN rotate.
    place(c) = R * (SVector{3,Float64}(c) - com)

    r, h = G.r_drum, G.h_drum
    facets = Facet[]

    ρd, sd = optical(OPTICAL_DRUM)
    ρh, sh = uniform_optics ? (ρd, sd) : optical(OPTICAL_HORN)

    #  Drum sides: regular n-gon prism, area-matched to the true cylinder 
    A_side = 2π * r * h / n_side
    for j in 0:(n_side - 1)
        θ = 2π * (j + 0.5) / n_side          # sector midpoint
        n̂ = SVector(cos(θ), sin(θ), 0.0)
        c  = SVector(r * cos(θ), r * sin(θ), 0.0)
        push!(facets, Facet(A_side, R * n̂, place(c), ρd, sd))
    end

    # Drum end caps
    A_cap = π * r^2
    push!(facets, Facet(A_cap, R * SVector(0.0, 0.0,  1.0), place((0.0, 0.0,  h/2)), ρd, sd))
    push!(facets, Facet(A_cap, R * SVector(0.0, 0.0, -1.0), place((0.0, 0.0, -h/2)), ρd, sd))

    # Horn: six box faces
    hd = G.horn_dims; hc = G.horn_centroid
    hx, hy, hz = hd[1]/2, hd[2]/2, hd[3]/2
    horn_faces = (
        (SVector( 1.0, 0.0, 0.0), SVector( hx, 0.0, 0.0), hd[2]*hd[3]),
        (SVector(-1.0, 0.0, 0.0), SVector(-hx, 0.0, 0.0), hd[2]*hd[3]),
        (SVector( 0.0, 1.0, 0.0), SVector(0.0,  hy, 0.0), hd[1]*hd[3]),
        (SVector( 0.0,-1.0, 0.0), SVector(0.0, -hy, 0.0), hd[1]*hd[3]),
        (SVector( 0.0, 0.0, 1.0), SVector(0.0, 0.0,  hz), hd[1]*hd[2]),
        (SVector( 0.0, 0.0,-1.0), SVector(0.0, 0.0, -hz), hd[1]*hd[2]),
    )
    for (n̂, off, A) in horn_faces
        push!(facets, Facet(A, R * n̂, place(hc + off), ρh, sh))
    end

    expected = n_side + 2 + 6
    length(facets) == expected ||
        error("skynet1a_shape: built $(length(facets)) facets, expected $expected")

    check && _assert_frame_consistent(G, mpd, R, frame; atol = atol)

    nm = frame === :principal ? "Skynet 1A (drum + off-axis horn)" :
                                "Skynet 1A (drum + off-axis horn, DESIGNER FRAME — not principal)"
    return ShapeModel(facets; name = nm)
end

"""
    _assert_frame_consistent(G, mpd, R, frame; atol)

Re-expresses the primitive assembly in the frame the facets were just written
in, re-runs `composite_inertia`, and demands the principal frame come out as the
identity
"""
function _assert_frame_consistent(G, mpd, R, frame::Symbol; atol::Real = 1e-9)
    mpr = composite_inertia([_reframe(p, R) for p in G.prims])

    scale = mpd.principal.Is
    for (nm, a, b) in (("Il", mpr.principal.Il, mpd.principal.Il),
                       ("Ii", mpr.principal.Ii, mpd.principal.Ii),
                       ("Is", mpr.principal.Is, mpd.principal.Is))
        abs(a - b) ≤ atol * scale ||
            error("skynet1a_shape frame check: $nm changed under rotation " *
                  "($a vs $b) — the reframing is wrong, not the frame.")
    end

    if frame === :principal
        dev = maximum(abs.(mpr.R_bp - one(SMatrix{3,3,Float64,9})))
        dev ≤ 1e-8 || error("""
            skynet1a_shape: facet frame is NOT the principal frame of the paired
            inertia tensor (max |R_bp − 𝟙| = $dev).  Refusing to return a shape
            whose normals and inertia axes disagree.
            R_bp in the facet frame =
            $(mpr.R_bp)""")

        offd = maximum(abs.(mpr.I_com - Diagonal(diag(mpr.I_com))))
        offd ≤ 1e-8 * scale || error("skynet1a_shape: residual products of " *
                                     "inertia in the facet frame ($offd kg·m²)")
    end
    return mpr
end

# Measurements + report


"""
    facet_mass_centroid(shape, G; n_side) → (centroid, total_mass)

Mass-weighted centroid of the FACETS, using the same mass split as
`skynet_inertia`: the drum's mass spread over its sides and caps in proportion
to area, the horn's over its six faces likewise.
"""
function facet_mass_centroid(shape::ShapeModel, G; n_side::Int = N_DRUM_SIDE_FACETS)
    n_drum = n_side + 2
    A_drum = sum(f.area for f in shape.facets[1:n_drum])
    A_horn = sum(f.area for f in shape.facets[(n_drum+1):end])
    m = Float64[]
    for (k, f) in enumerate(shape.facets)
        push!(m, k ≤ n_drum ? G.m_drum * f.area / A_drum : G.m_horn * f.area / A_horn)
    end
    M = sum(m)
    c = sum(m[k] * shape.facets[k].centroid for k in eachindex(m)) / M
    return c, M
end

"""
    srp_torque_survey(shape; n = 200) → (max|T|, mean|T|, max|F|)

Peak and mean instantaneous SRP torque magnitude over `n` sun directions spread
quasi-uniformly on the sphere (Fibonacci spiral — deterministic, not
pole-biased), plus the peak force for scale.  Same measurement telstar_shape
uses, so the three bodies are directly comparable.
"""
function srp_torque_survey(shape::ShapeModel; n::Int = 200)
    Tmax, Tsum, Fmax = 0.0, 0.0, 0.0
    ga = π * (3 - sqrt(5.0))
    for k in 0:(n-1)
        z = 1 - 2(k + 0.5) / n
        rr = sqrt(max(0.0, 1 - z^2))
        φ = ga * k
        û = SVector(rr*cos(φ), rr*sin(φ), z)
        F, T = srp_force_torque(shape, û)
        nT = norm(T); Tmax = max(Tmax, nT); Tsum += nT
        Fmax = max(Fmax, norm(F))
    end
    return (Tmax, Tsum / n, Fmax)
end

"""
    averaged_torque_survey(shape, I; nβ, n_band) → NamedTuple

The measurement that actually decides whether this body can drive a spin-rate
study, as opposed to merely having a non-zero instantaneous torque.
"""
function averaged_torque_survey(shape::ShapeModel, I::PrincipalInertias;
                                nβ::Int = 13, n_band::Int = 9, σ::Integer = 1)
    lam(f) = I.Il/I.Is + f * (I.Ii - I.Il) / I.Is
    sam(f) = I.Ii/I.Is + f * (I.Is - I.Ii) / I.Is
    rs = [lam.(range(0.05, 0.95, length = n_band)) ; sam.(range(0.05, 0.95, length = n_band))]
    βs = range(0.0, π, length = nβ)

    mz_max, mz_sum, mt_max, mt_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    for β in βs, rr in rs
        Id = rr * I.Is
        st = OsculatingState(0.0, β, 1.0, Id)      # H irrelevant: M̄ is H-independent
        T  = averaged_srp_torques_analytic(shape, st, I; σ = σ)
        mz = abs(T.M_H[3]) / I.Is
        mt = hypot(T.M_H[1], T.M_H[2]) / I.Is
        mz_max = max(mz_max, mz); mz_sum += mz
        mt_max = max(mt_max, mt); mt_sum += mt
        n += 1
    end
    return (mz_max = mz_max, mz_mean = mz_sum/n,
            mt_max = mt_max, mt_mean = mt_sum/n, n = n)
end

function main()
    G     = skynet1a_primitives()
    mpd   = composite_inertia(G.prims)
    shape = skynet1a_shape()
    I     = skynet1a_inertia()

    println("="^78)
    println("Skynet 1A — faceted SRP shape model (asymmetric / off-axis-horn variant)")
    println("="^78)
    @printf("\n%s\n", shape.name)
    @printf("facets                %4d   (%d drum sides + 2 caps + 6 horn faces)\n",
            n_facets(shape), N_DRUM_SIDE_FACETS)
    @printf("total facet area      %10.4f m²\n", total_area(shape))
    @printf("  true cylinder area  %10.4f m²  (2πrh + 2πr² = lateral + caps)\n",
            2π*G.r_drum*G.h_drum + 2π*G.r_drum^2)
    @printf("  horn box area       %10.4f m²\n",
            2*(G.horn_dims[1]*G.horn_dims[2] + G.horn_dims[1]*G.horn_dims[3] +
               G.horn_dims[2]*G.horn_dims[3]))

    #  Frame 
    println("\n" * "─"^78)
    println("FRAME CONSISTENCY")
    println("─"^78)
    println("R_bp (designer → project body), from composite_inertia:")
    for i in 1:3
        @printf("   %+9.6f %+9.6f %+9.6f\n", mpd.R_bp[i,1], mpd.R_bp[i,2], mpd.R_bp[i,3])
    end
    @printf("det(R_bp) = %+.12f   (must be +1: right-handed, no reflection)\n", det(mpd.R_bp))
    @printf("composite CoM (designer frame) = [%+.6f, %+.6f, %+.6f] m\n",
            mpd.com[1], mpd.com[2], mpd.com[3])
    @printf("  |CoM| = %.6f m — NOT zero, so the CoM shift in `place()` is load-bearing\n",
            norm(mpd.com))

    mpr = _assert_frame_consistent(G, mpd, mpd.R_bp, :principal)
    dev = maximum(abs.(mpr.R_bp - one(SMatrix{3,3,Float64,9})))
    @printf("re-eigendecomposed in the FACET frame: max |R_bp − 𝟙| = %.3e  → %s\n",
            dev, dev ≤ 1e-8 ? "PASS, facets and tensor share axes" : "FAIL")

    c, M = facet_mass_centroid(shape, G)
    @printf("facet mass-weighted centroid = [%+.3e, %+.3e, %+.3e] m  (|c| = %.3e)\n",
            c[1], c[2], c[3], norm(c))
    @printf("   → %s (exact answer is 0; a dropped CoM shift would give %.4f m)\n",
            norm(c) < 1e-12 ? "PASS" : "CHECK", norm(mpd.com))
    @printf("facet mass total %.4f kg vs primitive total %.4f kg\n", M, mpd.mass)

    #Triaxiality restated 
    println("\n" * "─"^78)
    println("PAIRED INERTIA (asymmetric variant)")
    println("─"^78)
    t = Main.triaxiality(I)
    @printf("   I_l = %.4f   I_i = %.4f   I_s = %.4f  kg m²\n", I.Il, I.Ii, I.Is)
    @printf("   (I_i−I_l)/I_s = %.4e   (I_s−I_i)/I_s = %.4e   tol = %.0e\n",
            t.d_li, t.d_is, Main.TRIAX_TOL)
    @printf("   → %s\n", t.degenerate ? "DEGENERATE" : "genuinely triaxial, safe to classify")

    # Faceting convergence 
    println("\n" * "─"^78)
    println("FACETING CONVERGENCE  [FLAG-SKYNET-FACETING]")
    println("─"^78)
    @printf("   %-6s %14s %14s %10s\n", "n_side", "max|T| [N m]", "mean|T| [N m]", "Δ vs 96")
    ref96 = srp_torque_survey(skynet1a_shape(; n_side = 96))[1]
    for n in (12, 24, 48, 96)
        Tm, Tmean, _ = srp_torque_survey(skynet1a_shape(; n_side = n))
        @printf("   %-6d %14.6e %14.6e %9.2f%%\n", n, Tm, Tmean, 100*(Tm-ref96)/ref96)
    end

    # The torque measurement
    println("\n" * "="^78)
    println("NET SRP TORQUE — MEASURED, NOT ASSUMED")
    println("="^78)
    println("""
    Triaxiality alone does NOT guarantee a net torque: Telstar's assumed body is
    fully triaxial and has IDENTICALLY ZERO net SRP torque, because a box's six
    faces all carry the same arm×area = V/2 and cancel in pairs.  So the torque
    is measured directly here rather than inferred from the inertia tensor.""")

    g = goes8_shape_full(; θ_sa = deg2rad(17), optical = :bs)
    Ts, Ts_m, Fs = srp_torque_survey(shape)
    Tg, Tg_m, Fg = srp_torque_survey(g)
    shape_u = skynet1a_shape(; uniform_optics = true)
    Tu, Tu_m, _ = srp_torque_survey(shape_u)

    println("\nINSTANTANEOUS torque over 200 sun directions:")
    @printf("   %-34s %14s %14s %14s\n", "", "max|T| [N m]", "mean|T| [N m]", "max|F| [N]")
    @printf("   %-34s %14.6e %14.6e %14.6e\n", "Skynet 1A (drum + off-axis horn)", Ts, Ts_m, Fs)
    @printf("   %-34s %14.6e %14.6e %14s\n", "  same, UNIFORM optics", Tu, Tu_m, "—")
    @printf("   %-34s %14.6e %14.6e %14.6e\n", "GOES-8 (26-facet, θ_sa=17°)", Tg, Tg_m, Fg)
    @printf("\n   Skynet/GOES-8 max|T| ratio = %.4f\n", Ts/Tg)
    @printf("   NON-ZERO? %s   (Telstar's is ~1e-21 N m, i.e. round-off)\n",
            Ts > 1e-12 ? "YES — this body is not inert" : "NO — INERT, unusable for YORP")
    @printf("   geometric vs optical: uniform-optics torque is %.1f%% of the full value,\n",
            100*Tu/Ts)
    println("     so the net torque survives removing the invented optical contrast —")
    println("     it comes from the horn's OFFSET, not from its assumed reflectivity.")

    # Where it comes from
    sub(idx) = ShapeModel(shape.facets[idx]; name = "sub")
    n_drum = N_DRUM_SIDE_FACETS + 2
    println("\n   per-component instantaneous max|T| (about the COMPOSITE CoM):")
    @printf("     %-28s %14.6e N m\n", "drum only (sides + caps)",
            srp_torque_survey(sub(1:n_drum))[1])
    @printf("     %-28s %14.6e N m\n", "horn only (6 faces)",
            srp_torque_survey(sub((n_drum+1):n_facets(shape)))[1])
    println("""
     Both are non-zero because both sit OFF the composite CoM: each component's
     own self-torque cancels, but its net force acts through a centroid offset
     from the CoM, giving c × F.  That is the whole mechanism here.""")

    # The averaged torque — the one that matters for RQ1
    println("\n" * "="^78)
    println("AVERAGED TORQUE M̄ — THE COMPONENT THAT ACTUALLY DRIVES SPIN CHANGE")
    println("="^78)
    println("""
    A non-zero instantaneous torque is necessary but NOT sufficient.  The
    averaged model propagates M̄ in the H frame, and only M̄_z (along Ĥ) changes
    |H| and therefore the spin rate; M̄_x, M̄_y only precess Ĥ.  Both are swept
    below over β ∈ [0,π] and I_d across the LAM and SAM bands, normalised by I_s
    so bodies of different size compare directly.""")

    A = averaged_torque_survey(shape, I)
    Ig = goes8_inertia()
    Ag = averaged_torque_survey(g, Ig)

    @printf("\n   %-22s %14s %14s %14s %14s\n", "",
            "max|M̄_z|/I_s", "mean|M̄_z|/I_s", "max|M̄_⊥|/I_s", "mean|M̄_⊥|/I_s")
    @printf("   %-22s %14.4e %14.4e %14.4e %14.4e\n", "Skynet 1A",
            A.mz_max, A.mz_mean, A.mt_max, A.mt_mean)
    @printf("   %-22s %14.4e %14.4e %14.4e %14.4e\n", "GOES-8",
            Ag.mz_max, Ag.mz_mean, Ag.mt_max, Ag.mt_mean)
    @printf("\n   ratio Skynet/GOES-8:   M̄_z %.3e      M̄_⊥ %.3e\n",
            A.mz_max/Ag.mz_max, A.mt_max/Ag.mt_max)
    @printf("   Skynet's own anisotropy: max|M̄_z| / max|M̄_⊥| = %.3e\n",
            A.mz_max/A.mt_max)
    println("""
    READ THIS INTO THE FIGURE 10 CAPTION.  If M̄_z/I_s is orders of magnitude
    below GOES-8's while M̄_⊥ is comparable, the torque is essentially PURELY
    TRANSVERSE: it precesses Ĥ but does not spin the body up or down, so ω_e
    evolves under dissipation alone and the ω_e panel will look like Telstar's
    relaxation rather than GOES-8's spin-up.  That is a real physical result for
    a spinner, not a modelling failure — but it means Skynet tests the RQ1
    forgetting question in α and β, NOT in ω_e.""")

    println("\n" * "="^78)
    println("ASSUMPTIONS ADDED BY THIS FILE (on top of skynet_inertia's ledger)")
    println("="^78)
    for (k, v, p) in [
        ("drum faceting", @sprintf("%d-gon prism, area-matched, + 2 caps", N_DRUM_SIDE_FACETS),
         "MODEL CHOICE — a cylinder has no facets; convergence checked above [FLAG-SKYNET-FACETING]"),
        ("drum optics", string(OPTICAL_DRUM) * @sprintf(" (ρ=%.2f, s=%.1f)", optical(OPTICAL_DRUM)...),
         "BORROWED from GOES-8 Table 1 — material class only; GOES-8 is 1990s, Skynet is 1969, so the era argument does NOT hold [FLAG-SKYNET-OPTICAL]"),
        ("horn optics", string(OPTICAL_HORN) * @sprintf(" (ρ=%.2f, s=%.1f)", optical(OPTICAL_HORN)...),
         "BORROWED + ASSUMED metallic; the drum/horn contrast is invented, and its effect is measured above [FLAG-SKYNET-OPTICAL]"),
        ("centroid reference", "composite CoM, not designer origin",
         "REQUIRED — this body's CoM is off-origin, unlike Telstar's; verified by facet_mass_centroid above"),
        ("symmetric config", "no shape built for it at all",
         "DELIBERATE — degenerate symmetric top, cannot be classified [FLAG-SKYNET-AXISYMMETRIC]"),
    ]
        @printf("  %-22s %s\n", k, v)
        @printf("  %-22s   ↳ %s\n", "", p)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__      # `@__FILE__` swallows the rest of a
    main()                                 # `&&` line, so this is written out
end                                        # (same idiom as telstar_shape).
