
module master

using StaticArrays
using LinearAlgebra

# M1
include("frames.jl")
include("osculating.jl")
include("torque_free.jl")
include("mass_properties.jl")
# M2 
include("shape.jl")
include("goes_shape_full.jl")
include("ephemeris.jl")
include("srp_full.jl")
include("dynamics_full.jl")

# M3 
include("srp_avg_numeric")
include("averaged_dynamics.jl")

# M4 (B&S 2022: internal energy dissipation + gravity gradient) 
include("dissipation.jl")
include("gravgradient.jl")

# M5
include("equilibria.jl")
include("fate.jl")

# M6
include("srp_avg_analytic")

# M8 (B&S 2021 Resonance)
include("resonance.jl")

#  Exports 
export PrincipalInertias, OsculatingState, LAM, SAM, Separatrix
# M1: composite rigid-body inertia (mass_properties.jl).  The file was included
# but nothing in it was exported, so every caller outside the module hit
# `UndefVarError: SolidBox not defined`.
export MassPrimitive, SolidBox, ThinPlate, PointMass, SolidCylinder, MassProperties
export inertia_local, inertia_primitive, parallel_axis
export composite_inertia, principal_inertias, check_realizable
export PerturbationConfig
export classify_regime
export R1, R2, R3
export HO_matrix, BH_matrix
export alpha_beta_from_Hhat, Hhat_from_alpha_beta
export sinβ_guard, to_vw, from_vw
export omega_e, angular_momentum, kinetic_energy, Id_from_energy_momentum
export to_svector, from_svector
export elliptic_K, elliptic_Pi_branchtracked
export torque_free_rates, torque_free_body_rates
export direction_cosines_az, az_squared_averages, az_amplitudes
export tumbling_periods

# M2: full Euler model
export Facet, ShapeModel, n_facets, total_area
export OPTICAL_PROPERTIES, optical
export goes8_shape, goes8_inertia
export goes8_shape_full, goes8_inertia_albuja
export SECONDS_PER_YEAR, sun_direction_ecliptic, sun_direction_body
export quat_normalize, quat_to_dcm, quat_kinematics
export P_SRP_1AU, B_LAMBERT, srp_facet_force, srp_force_torque, srp_torque
export inertia_body, euler_rates, propagate_full
export zero_torque, srp_torque_fn
export kinetic_energy_full, angular_momentum_full

# M3: tumbling-averaged model
export AveragedTorques, averaged_srp_torques
export MEAN_MOTION, averaged_eom, propagate_averaged
export MAX_MU_OVER_J, epsilon_beta
export averaging_validity_callback, separatrix_crossing_callback

# M5: averaged equilibria + linear stability (B&S 2022 §V.B)
export AveragedEquilibrium, find_equilibrium, equilibrium_family
export equilibrium_jacobian, equilibrium_eigenvalues, is_stable
export SpinOutcome, spin_termination_callbacks, classify_spin_outcome

# M6: analytic (closed-form) SRP averaging backend (B&S 2021 App. B)
export elliptic_function_averages, AzMoments, az_moments
export averaged_srp_torques_analytic

# M4: internal energy dissipation (B&S 2022 §II.C, §IV.A, Appendix A)
export slug_rates, propagate_full_slug, OmegaAverages, omega_averages, t_coefficients, solve_A
export averaged_dissipation_rate, h_d, residual_D, dbar2

# M4: gravity gradient (B&S 2022 §II.D, §IV.B)
export MU_EARTH, gg_mean_motion, gg_torque, orbit_angles
export averaged_inertia, hg_matrix, averaged_gg_torque

# M8: resonance-averaged dynamics (B&S Dec-2021)
export period_ratio, resonance_Id
export phibar_dot, phibar_ddot, taur_dot, taur_ddot, gamma_ddot
export resonance_averaged_torques, ResonanceTable, resonance_table, interp_table
export resonance_eom, propagate_resonance, resonance_selfcheck

end # module
