
using .master
using LinearAlgebra, Printf, StaticArrays


# SOURCED INPUTS


const TOTAL_MASS_KG = 422.0
const ASTRONAUTIX_ALTERNATIVE_KG = 243.0   # recorded, not used

const DRUM_HEIGHT_M   = 0.810
const DRUM_DIAMETER_M = 1.370
const DRUM_RADIUS_M   = DRUM_DIAMETER_M / 2

# INVENTED INPUTS - Edit here


# Mass split

const DRUM_MASS_FRACTION    = 0.85
const ANTENNA_MASS_FRACTION = 1.0 - DRUM_MASS_FRACTION

# Symmetric baseline platform — configuration (1)

const PLATFORM_DISC_RADIUS_M    = 0.550
const PLATFORM_DISC_THICKNESS_M = 0.100

# Off-axis horn — configuration (2) 

const HORN_DIMS_M          = (0.450, 0.450, 0.600)  # (along x̂_s, ŷ_s, ẑ_s)
const HORN_RADIAL_OFFSET_M = 0.350                  # from the spin axis, along +x̂_s

# [MODEL CHOICE] Solid vs shell
# RIGID-BODY ASSUMPTION


const ASSUMPTIONS = Tuple{String,String,String}[]   # (label, value, provenance)
note!(label, value, prov) = push!(ASSUMPTIONS, (label, value, prov))

const m_drum    = DRUM_MASS_FRACTION    * TOTAL_MASS_KG
const m_antenna = ANTENNA_MASS_FRACTION * TOTAL_MASS_KG

"""
    skynet_primitives(config) → (prims = …, …)

Build the primitive assembly in the designer frame.

`config = :symmetric`  drum + coaxial platform disc — configuration (1).
`config = :asymmetric` drum + off-axis horn box    — configuration (2).

Returned as a NamedTuple carrying the derived dimensions too, so the shape
builder never re-derives them independently.
"""
function skynet_primitives(config::Symbol)
    config in (:symmetric, :asymmetric) ||
        throw(ArgumentError("skynet_primitives: config must be :symmetric or :asymmetric, got :$config"))

    drum = SolidCylinder(m_drum, (DRUM_RADIUS_M, DRUM_HEIGHT_M))   # centred at origin

    if config === :symmetric
        # Platform disc sits ON TOP of the drum, coaxial: its centroid is one
        # drum half-height plus one disc half-thickness up the spin axis.
        z = DRUM_HEIGHT_M/2 + PLATFORM_DISC_THICKNESS_M/2
        plat = SolidCylinder(m_antenna, (PLATFORM_DISC_RADIUS_M, PLATFORM_DISC_THICKNESS_M);
                             centroid = (0.0, 0.0, z))
        return (prims = MassPrimitive[drum, plat], config = config,
                z_top = z, x_off = 0.0,
                m_drum = m_drum, m_antenna = m_antenna)
    else
        # Horn sits on top AND off to one side.  Both offsets matter: the axial
        # one moves the composite CoM up, the radial one is what destroys the
        # axisymmetry.
        z = DRUM_HEIGHT_M/2 + HORN_DIMS_M[3]/2
        horn = SolidBox(m_antenna, HORN_DIMS_M;
                        centroid = (HORN_RADIAL_OFFSET_M, 0.0, z))
        return (prims = MassPrimitive[drum, horn], config = config,
                z_top = z, x_off = HORN_RADIAL_OFFSET_M,
                m_drum = m_drum, m_antenna = m_antenna)
    end
end

#  Both configurations, computed 
const MP_SYM = composite_inertia(skynet_primitives(:symmetric).prims)
const MP_ASY = composite_inertia(skynet_primitives(:asymmetric).prims)

# Triaxiality test 

const TRIAX_TOL = 1e-6
function triaxiality(P::PrincipalInertias)
    d_li = (P.Ii - P.Il) / P.Is
    d_is = (P.Is - P.Ii) / P.Is
    return (d_li = d_li, d_is = d_is,
            degenerate = (d_li < TRIAX_TOL) || (d_is < TRIAX_TOL))
end

# The exported accessor 
const _SKYNET1A_INERTIA = let
    P = MP_ASY.principal
    t = triaxiality(P)
    t.degenerate && error("""
        skynet1a_inertia(): the ASYMMETRIC configuration came out degenerate
        (d_li = $(t.d_li), d_is = $(t.d_is), tol = $TRIAX_TOL).  The invented
        horn offset is not actually breaking the axisymmetry — refusing to
        define an accessor that would hand a symmetric top to the LAM/SAM
        machinery.  Increase HORN_RADIAL_OFFSET_M or HORN_DIMS_M asymmetry.""")
    P
end
skynet1a_inertia() = _SKYNET1A_INERTIA


# Report

function _report_config(name, mp, G)
    P = mp.principal
    t = triaxiality(P)
    println("\n" * "─"^78)
    println(name)
    println("─"^78)
    @printf("total mass            %10.2f kg   (drum %.1f + antenna %.1f)\n",
            mp.mass, G.m_drum, G.m_antenna)
    @printf("centre of mass (x,y,z) [%+.5f, %+.5f, %+.5f] m  (designer frame)\n",
            mp.com[1], mp.com[2], mp.com[3])

    println("\ninertia tensor about the centre of mass, designer frame [kg m²]:")
    for i in 1:3
        @printf("   %12.4f %12.4f %12.4f\n", mp.I_com[i,1], mp.I_com[i,2], mp.I_com[i,3])
    end
    offdiag = maximum(abs.(mp.I_com - Diagonal(diag(mp.I_com))))
    @printf("max |off-diagonal|    %10.4e kg m²  (%.2e of I_s)\n", offdiag, offdiag/P.Is)

    println("\nprincipal inertias, project long-axis convention [kg m²]:")
    @printf("   I_l (min, b̂₃) %12.4f\n", P.Il)
    @printf("   I_i (mid, b̂₁) %12.4f\n", P.Ii)
    @printf("   I_s (max, b̂₂) %12.4f\n", P.Is)
    @printf("   ratios:  I_i/I_s = %.6f   I_l/I_s = %.6f\n", P.Ii/P.Is, P.Il/P.Is)

    println("\nTRIAXIALITY TEST  (tol = $(TRIAX_TOL), normalised by I_s):")
    @printf("   (I_i − I_l)/I_s = %.6e   %s\n", t.d_li,
            t.d_li < TRIAX_TOL ? "◄── DEGENERATE" : "ok")
    @printf("   (I_s − I_i)/I_s = %.6e   %s\n", t.d_is,
            t.d_is < TRIAX_TOL ? "◄── DEGENERATE" : "ok")
    @printf("   verdict: %s\n", t.degenerate ? "SYMMETRIC TOP — UNUSABLE" : "genuinely triaxial")

    println("\nrealizability:")
    @printf("   all moments positive   %s\n", all(x -> x > 0, (P.Il, P.Ii, P.Is)) ? "yes" : "NO")
    @printf("   I_s ≤ I_i + I_l        %s  (%.4f ≤ %.4f)\n",
            P.Is ≤ P.Ii + P.Il ? "yes" : "NO", P.Is, P.Ii + P.Il)
    @printf("   trace tr(I) = ΣI_k     %.8e vs %.8e\n",
            tr(mp.I_com), P.Il + P.Ii + P.Is)
    return t
end

function main()
    println("="^78)
    println("Skynet 1A — APPROXIMATE principal inertias from a primitive assembly")
    println("="^78)
    println("""
    Mass 422 kg and drum 810 mm × 1370 mm ⌀ are SOURCED.  Everything about the
    despun antenna platform is INVENTED.  Two configurations follow.""")

    t_sym = _report_config("CONFIGURATION (1) — SYMMETRIC BASELINE  [drum + coaxial disc]",
                           MP_SYM, skynet_primitives(:symmetric))
    println("""
    [FLAG-SKYNET-AXISYMMETRIC]  This configuration is UNUSABLE with this
    framework.  A drum plus a coaxial disc is exactly axisymmetric about the
    spin axis, so two principal moments are equal to round-off and the body is a
    degenerate symmetric top.  The LAM/SAM/separatrix classification this
    project is built on presumes three DISTINCT moments: the elliptic modulus
    and the tumbling period P_ψ both involve differences of the moments, and a
    symmetric top has no separatrix to be either side of.  Nothing downstream
    may propagate it.

    NOTE THE FRAMEWORK DOES NOT STOP YOU.  `PrincipalInertias`'s constructor
    tests `Il ≤ Ii ≤ Is`, NOT strict inequality (src/osculating.jl:19), so this
    tensor CONSTRUCTS WITHOUT COMPLAINT and would fail later, deeper, and less
    legibly.  That is why the explicit `triaxiality` test above exists and why
    `skynet1a_inertia()` refuses to return this configuration.""")

    t_asy = _report_config("CONFIGURATION (2) — ASYMMETRIC VARIANT  [drum + off-axis horn]",
                           MP_ASY, skynet_primitives(:asymmetric))
    println("""
    [FLAG-SKYNET-INVENTED-HORN]  Triaxiality here is MANUFACTURED by an invented
    horn offset ($(HORN_RADIAL_OFFSET_M) m radially).  It is physically motivated —
    a despun platform exists to point a directional antenna, and a directional
    antenna is an off-axis structure — but the numbers are chosen, not sourced.
    This is the configuration every downstream script uses.""")

    # ── Which degeneracy actually occurred ───────────────────────────────────
    println("\n" * "="^78)
    println("WHICH PAIR OF MOMENTS IS DEGENERATE — a result, not an assumption")
    println("="^78)
    Ps = MP_SYM.principal
    println("""
    The symmetric baseline is OBLATE, not prolate: the drum is wider than it is
    tall (1.370 m ⌀ vs 0.810 m), so the spin-axis moment is the LARGEST and the
    two equal transverse moments are the two SMALLEST.  The degeneracy is
    therefore

        I_l = I_i  <  I_s        (measured: I_l = $(round(Ps.Il, digits=4)), I_i = $(round(Ps.Ii, digits=4)), I_s = $(round(Ps.Is, digits=4)))

    NOT I_i = I_s.  Both are "a symmetric top" and both are equally unusable,
    but they are different degeneracies: I_i = I_s is the PROLATE case, which a
    tall narrow drum would have given.  Which one you get is set by the aspect
    ratio, and for Skynet's dimensions it is the oblate one.""")

    # ── Comparison with the project's other bodies ───────────────────────────
    G = goes8_inertia()
    Pa = MP_ASY.principal
    println("\nshape-class comparison (asymmetric variant vs the project's other bodies):")
    @printf("   %-12s %12s %12s\n", "", "Skynet 1A*", "GOES-8")
    @printf("   %-12s %12.4f %12.4f\n", "I_i/I_s", Pa.Ii/Pa.Is, G.Ii/G.Is)
    @printf("   %-12s %12.4f %12.4f\n", "I_l/I_s", Pa.Il/Pa.Is, G.Il/G.Is)
    @printf("   %-12s %12.1f %12.1f\n", "I_s [kg m²]", Pa.Is, G.Is)
    println("""
   (* approximate, built from the assumptions below)
   Skynet is a much rounder body than GOES-8 (I_l/I_s = $(round(Pa.Il/Pa.Is,digits=3)) vs $(round(G.Il/G.Is,digits=3)))
   and ~40× lighter in I_s — it is a genuinely different point in inertia-ratio
   space, which is the reason for including it.""")

    # ── Assumption ledger ────────────────────────────────────────────────────
    note!("total mass", @sprintf("%.1f kg", TOTAL_MASS_KG),
          "SOURCED — two independent sources agree; Astronautix gives $(ASTRONAUTIX_ALTERNATIVE_KG) kg instead (42% lower), unresolved [FLAG-SKYNET-MASS-SRC]")
    note!("drum dimensions", @sprintf("%.3f m high × %.3f m ⌀ (r = %.4f m)",
                                      DRUM_HEIGHT_M, DRUM_DIAMETER_M, DRUM_RADIUS_M),
          "SOURCED — both agreeing sources")
    note!("drum mass model", "solid cylinder, uniform density",
          "MODEL CHOICE — a thin shell is equally defensible but needs an unsourced wall thickness; solid is the conservative choice for triaxiality [FLAG-SKYNET-SOLIDNESS]")
    note!("drum / antenna mass split", @sprintf("%.2f / %.2f (%.1f kg / %.1f kg)",
                                                DRUM_MASS_FRACTION, ANTENNA_MASS_FRACTION,
                                                m_drum, m_antenna),
          "INVENTED — no source gives the platform's mass or fraction [FLAG-SKYNET-MASS-SPLIT]")
    note!("platform disc (config 1)", @sprintf("r = %.3f m, t = %.3f m, coaxial",
                                               PLATFORM_DISC_RADIUS_M, PLATFORM_DISC_THICKNESS_M),
          "INVENTED — and axisymmetric by construction, hence unusable [FLAG-SKYNET-AXISYMMETRIC]")
    note!("horn box (config 2)", @sprintf("%.3f × %.3f × %.3f m", HORN_DIMS_M...),
          "INVENTED GEOMETRY — no source gives the platform's shape or size [FLAG-SKYNET-INVENTED-HORN]")
    note!("horn radial offset", @sprintf("%.3f m from the spin axis", HORN_RADIAL_OFFSET_M),
          "INVENTED GEOMETRY — this single number is what makes the body triaxial; results are most sensitive to it [FLAG-SKYNET-INVENTED-HORN]")
    note!("rigid-body treatment", "drum + platform as ONE rigid body",
          "ASSUMPTION defended ONLY for the current defunct state: no power, despin motor dead, bearing assumed SEIZED (a free bearing is the unmodelled other limit) [FLAG-SKYNET-TWOBODY]")
    note!("everything else", "no tanks, thrusters, solar cells, MLI or booms modelled",
          "MODEL CHOICE — drum + one platform primitive only")

    println("\n" * "="^78)
    println("ASSUMED VALUES USED IN THIS ESTIMATE")
    println("="^78)
    for (k, v, p) in ASSUMPTIONS
        @printf("  %-26s %s\n", k, v)
        @printf("  %-26s   ↳ %s\n", "", p)
    end
    println("""
The only externally sourced numbers above are the total mass and the drum's two
dimensions.  The entire antenna platform — the part that makes this body
triaxial and therefore the part every dynamical result depends on — is
invented.  Treat the resulting tensor as a shape-class placeholder for a
spin-stabilised drum, not as Skynet 1A's inertia.""")

    return (sym = t_sym, asy = t_asy)
end

# `mp` is the name the paired shape script reads back, matching the Telstar
# pattern.  It is the ASYMMETRIC configuration — the usable one.
const mp = MP_ASY

# Run the report at include time, matching telstar_inertia's idiom: this file is
# a REPORT SCRIPT, and the paired shape script suppresses the output with
# `redirect_stdout` when it only wants the constants and `mp`.
const REPORT = main()
