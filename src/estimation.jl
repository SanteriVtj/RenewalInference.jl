function patentSMM(f, N, init_distribution)
    opt_patent = OptimizationFunction(
        f,
        Optimization.AutoForwardDiff()
    )

    res = Dict()
    @time for i in 1:N
        @show i
        x0 = collect(Iterators.flatten(rand.(init_distribution, 1)))
        optp = OptimizationProblem(
            opt_patent,
            x0,
            [0],
            lb = [0.,0,0,0,0],
            ub = [1.,100_000,1,1,1]
        )

        res[i] = solve(
            optp,
            LBFGS(linesearch=LineSearches.BackTracking()),
            maxtime = 60
        )
    end
    return res
end