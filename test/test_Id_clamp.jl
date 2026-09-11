#=
test_Id_clamp.jl — regression test for the I_d → I_s sqrt DomainError in
`torquefree_params_SAM` (src/torque_free.jl).

THE BUG.  The function computed
    B₁ = ωe * sqrt(Id * (Is − Id) / (Ii * (Is − Ii)))
    B₃ = ωe * sqrt(Id * (Is − Id) / (Il * (Is − Il)))
with no guard on (Is − Id).  I_d = H²/2T is bounded above by I_s, and I_d = I_s
is an exact fixed point of the averaged flow, so I_d > I_s is finite-step
overshoot of the dissipative attractor with no physical content — but it threw
a DomainError from inside an RK stage and killed the solve outright.

THE FIX.  `Id = min(Id, Is)` at entry, so B₁, B₃, τ_rate, k² and n_param all
describe the same I_d.  (The k² line has always clamped its own radicand; the
amplitudes simply never got the same treatment.)

Run:  julia --project=".." test/test_Id_clamp.jl      (from the repo root)

Three tests, in increasing order of how much they exercise:

  1. UNIT — the clamp is what does the work.  Calls the function with
     I_d = I_s(1 + 1e-12) and, alongside it, evaluates the ORIGINAL unguarded
     expression on the same numbers to confirm that expression really does
     throw.  Without that second half the test would pass just as well against
     code that never had the bug.
  2. LIMIT — at I_d = I_s the returned parameters are the physical uniform-
     rotation limit: B₁ = B₃ = 0, B₂ = ωe, so ω = (0, ωe, 0) about b̂₂.  This is
     what makes the clamp a limiting value rather than a fudge.
  3. END-TO-END — replays Skynet 1A at J = 10, documented in
     sims/Skynet 1A/traj_skynet1a.jl [FLAG-SKYNET-IDIS] as dying at
     t = 0.0153 yr with 1 − I_d/I_s = 1.1e−7.  Same body, same IC, same
     backend.  Asserts it now reaches the horizon AND lands on the closed-form
     destination ω_e0·(I_d0/I_s), which holds for this body because M̄_z ≈ 0.
=#
# torquefree_params_SAM is internal to the module (not in master.jl's export list),
# so it is reached qualified as master.torquefree_params_SAM below.
include(joinpath(@__DIR__, "..", "src", "master.jl"))
using .master
using Test, Printf

include(joinpath(@__DIR__, "..", "sims", "Skynet 1A", "skynet_shape"))

Isk = skynet1a_inertia()
shk = skynet1a_shape()

@testset "I_d → I_s clamp in torquefree_params_SAM" begin

    Il, Ii, Is = Isk.Il, Isk.Ii, Isk.Is
    ωe = 2π / (20.0 * 60.0)

    @testset "1. overshoot no longer throws, and the raw expression still would" begin
        # 1.1e-7 relative is the overshoot measured for Skynet J = 10; 1e-12 is
        # the far more common roundoff-scale case (the ensemble CSVs record
        # radicands of order -3e-15 on an I_s of ~2e3).
        for rel in (1e-12, 1.1e-7, 1e-4)
            Id = Is * (1 + rel)

            # The ORIGINAL line, verbatim, on the same inputs.  If this stops
            # throwing, the test below has become vacuous and this @test fails
            # loudly rather than passing silently.
            @test_throws DomainError sqrt(Id * (Is - Id) / (Ii * (Is - Ii)))

            k, n, τ_rate, B₁, B₂, B₃ = master.torquefree_params_SAM(ωe, Id, Isk)
            @test all(isfinite, (k, n, τ_rate, B₁, B₂, B₃))
            @test 0.0 ≤ k ≤ 1.0
        end
    end

    @testset "2. clamped value is the uniform-rotation limit" begin
        k, n, τ_rate, B₁, B₂, B₃ = master.torquefree_params_SAM(ωe, Is, Isk)
        # ω = B₁ sn, B₂ dn, B₃ cn.  At I_d = I_s: pure spin about b̂₂ at ωe.
        @test B₁ ≈ 0.0 atol = 1e-12 * ωe
        @test B₃ ≈ 0.0 atol = 1e-12 * ωe
        @test B₂ ≈ ωe   rtol = 1e-12
        @test k  ≈ 0.0  atol = 1e-12
        @test n  ≈ 0.0  atol = 1e-12
        # τ_rate is NOT zero here and is not supposed to be — its radicand
        # (Eq. A12) carries no (I_s − I_d) factor, so τ keeps advancing.  What
        # makes the limit uniform rotation is k → 0, at which sn → sin, cn → cos,
        # dn → 1, so ω = (B₁ sin τ, B₂, B₃ cos τ) = (0, ωe, 0) for EVERY τ.
        # Assert that, which is the physical statement, rather than τ_rate = 0,
        # which is not true.
        for τ in (0.0, 0.3, 1.7, 12.0)
            ω = torque_free_body_rates(τ * τ_rate, ωe, Is, Isk, SAM(); σ = -1)
            @test ω[1] ≈ 0.0 atol = 1e-12 * ωe
            @test abs(ω[2]) ≈ ωe rtol = 1e-12
            @test ω[3] ≈ 0.0 atol = 1e-12 * ωe
        end

        # And the overshooting call must agree with it to roundoff, i.e. the
        # clamp returns the attractor rather than some nearby wrong branch.
        k2, n2, τ2, C₁, C₂, C₃ = master.torquefree_params_SAM(ωe, Is * (1 + 1e-9), Isk)
        @test (C₁, C₂, C₃) == (B₁, B₂, B₃)
    end

    @testset "3. Skynet 1A J = 10 reaches the horizon (was: died at 0.0153 yr)" begin
        # IC copied from sims/Skynet 1A/traj_skynet1a.jl: LAM_F = 0.1, β₀ = 75°,
        # P_e = 20 min, σ = -1, μ/J = 1e-3.
        lam_frac(f) = Il/Is + f * (Ii - Il) / Is
        Id0_rel = lam_frac(0.1)
        Id0     = Id0_rel * Is
        ωe0     = 2π / (20.0 * 60.0)
        st0     = OsculatingState(0.0, deg2rad(75.0), Id0 * ωe0, Id0)

        YEARS = 20.0
        tf    = YEARS * SECONDS_PER_YEAR
        J     = 10.0
        cfg   = PerturbationConfig(srp = true, dissipation = true,
                                   gravity_gradient = true,
                                   μ = 1.0e-3 * J, J = J, orbit_R = 4.2575e7,
                                   srp_backend = :analytic, σ_branch = -1,
                                   resonant = false)

        sol = propagate_averaged(Isk, st0, (0.0, tf); shape = shk, cfg = cfg,
                  reltol = 1e-8, abstol = 1e-10, maxiters = Int(1e7),
                  callback = spin_termination_callbacks(ωe_floor = 1e-8,
                                                        ωe_ceiling = 1.0))

        te_yr = sol.t[end] / SECONDS_PER_YEAR
        @printf("    ran %.4f / %.0f yr (pre-fix: 0.0153 yr, DomainError)\n", te_yr, YEARS)
        @test te_yr > 0.0153 * 10          # nowhere near the old failure point
        @test te_yr ≈ YEARS rtol = 1e-6    # full horizon

        uend = sol(sol.t[end])
        ωe_final, Id_final = uend[3] / uend[4], uend[4]
        ωe_pred = ωe0 * Id0_rel            # M̄_z ≈ 0 ⇒ |H| conserved
        @printf("    ω_e %.6e → %.6e  (closed form %.6e, %+.2e rel);  I_d/I_s = %.12f\n",
                ωe0, ωe_final, ωe_pred, (ωe_final - ωe_pred)/ωe_pred, Id_final/Is)
        @test ωe_final ≈ ωe_pred rtol = 1e-6
        @test Id_final ≤ Is                # the clamp's whole point
        @test Id_final / Is ≈ 1.0 rtol = 1e-9
    end
end
