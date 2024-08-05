function computehz(expirations)
    min = 1
    max = maximum(expirations)

    cm = countmap(expirations)
    exp_count = [get(cm, i, 0) for i=min:max]
    surv_count = sum(exp_count).-cumsum(exp_count)
    surv_count .= [sum(exp_count);surv_count[1:end-1]]
    hz = exp_count ./ surv_count
    [0; hz[1:end-1]]
end

function modelhz(x, N)
    cum_lapse = N .- x
    lapse = cum_lapse .- [zero(eltype(x)); cum_lapse[1:end-1]]
    hz = lapse ./ [N; x[1:end-1]]
    hz[isnan.(hz)] .= 0
    return hz
end