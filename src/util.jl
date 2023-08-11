function sample_patenthz(x0, hz, c; β=.95, ν=2)
    obsolence = QuasiMonteCarlo.sample(N,T-1,QuasiMonteCarlo.HaltonSample())'
    ishock = QuasiMonteCarlo.sample(N,1,QuasiMonteCarlo.HaltonSample())'
    return patenthz(
        x0,
        hz,
        ishock,
        obsolence,
        c,
        β=β,
        ν=ν
    )
end

function create_simul_hz(par)
    simulation_shocks = QuasiMonteCarlo.sample(N,T,QuasiMonteCarlo.HaltonSample())'
    obsolence = QuasiMonteCarlo.sample(N,T-1,QuasiMonteCarlo.HaltonSample())'
    ishock = QuasiMonteCarlo.sample(N,1,QuasiMonteCarlo.HaltonSample())'
    return simulate_patenthz(
        par,
        simulation_shocks,
        obsolence,
        c,
        ishock
    )
end