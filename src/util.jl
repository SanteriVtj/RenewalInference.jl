function sample_patenthz(x0, hz, c; β=.95, ν=2, N=200, T=17, alg=QuasiMonteCarlo.HaltonSample())
    obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = QuasiMonteCarlo.sample(N,1,alg)'
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

function create_simul_hz(par, c; N=200, T=17, alg=QuasiMonteCarlo.HaltonSample())
    simulation_shocks = QuasiMonteCarlo.sample(N,T,alg)'
    obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = QuasiMonteCarlo.sample(N,1,alg)'
    return simulate_patenthz(
        par,
        simulation_shocks,
        obsolence,
        c,
        ishock
    )
end