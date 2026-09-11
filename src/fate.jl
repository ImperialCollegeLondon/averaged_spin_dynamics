

struct SpinOutcome
    fate::Symbol
    ωe_final::Float64
    ωe_peak::Float64
    ωe_min::Float64
    ωe_mean_tail::Float64
    ripple::Float64
    t_end::Float64
    reached_horizon::Bool
    β_final::Float64
    Id_final::Float64
    α_final::Float64
end
# Seeing if any despun

function spin_termination_callbacks(; ωe_floor::Real = 1e-8,
                                     ωe_ceiling::Real = 1.0)
    ωe(u) = u[3] / u[4]
    lower = ContinuousCallback((u, t, integ) -> ωe(u) - ωe_floor,
                               integ -> terminate!(integ))
    upper = ContinuousCallback((u, t, integ) -> ωe(u) - ωe_ceiling,
                               integ -> terminate!(integ))
    return CallbackSet(lower, upper)
end
# Seeing what the ultimate spin state is, spin up, despin, or cycling
function classify_spin_outcome(sol, tf::Real;
                               ωe_floor::Real = 1e-8, ωe_ceiling::Real = 1.0,
                               nsample::Int = 2000, ripple_tol::Real = 1e-3)
    t_end = sol.t[end]
    reached = t_end >= tf - 1.0

    ts  = range(sol.t[1], t_end; length = nsample)
    ωes = [(u = sol(t); u[3] / u[4]) for t in ts]
    uf  = sol(t_end)
    ωe_f = uf[3] / uf[4]

    tail   = ωes[max(1, end - nsample ÷ 10 + 1):end]
    m_tail = sum(tail) / length(tail)
    ripple = (maximum(tail) - minimum(tail)) / max(abs(m_tail), eps())

    fate = if !reached
        if ωe_f <= 2 * ωe_floor
            :despin
        elseif ωe_f >= 0.5 * ωe_ceiling
            :spinup
        else
            :incomplete
        end
    else
        ripple < ripple_tol ? :equilibrium : :cycling
    end

    return SpinOutcome(fate, ωe_f, maximum(ωes), minimum(ωes), m_tail, ripple,
                       t_end, reached, uf[2], uf[4], uf[1])
end
