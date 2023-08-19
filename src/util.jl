function _patenthz(x0, hz, c, inventor_data; β=.95, ν=2, N=200, T=17, alg=QuasiMonteCarlo.HaltonSample())
    obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = quantile.(Normal(), QuasiMonteCarlo.sample(N÷2,1,alg)')
    ishock = [ishock ; ishock]
    return patenthz(
        x0,
        hz,
        ishock,
        obsolence,
        c,
        inventor_data,
        β=β,
        ν=ν
    )
end

function _simulate_patenthz(par, c, inventor_data; N=200, T=17, alg=QuasiMonteCarlo.HaltonSample())
    simulation_shocks = QuasiMonteCarlo.sample(N,T,alg)'
    obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = quantile.(Normal(), QuasiMonteCarlo.sample(N,1,alg)')
    return simulate_patenthz(
        par,
        simulation_shocks,
        obsolence,
        c,
        ishock,
        inventor_data
    )
end

function _thresholds(par,c; β=.95, N=200, T=17, alg=QuasiMonteCarlo.HaltonSample())
    obsolence = QuasiMonteCarlo.sample(N,T-1,alg)'
    ishock = quantile.(Normal(), QuasiMonteCarlo.sample(N,1,alg)')
    return thresholds(par, c, ishock, obsolence, β)
end