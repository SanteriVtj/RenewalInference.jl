function computehz(expirations)
    min = minimum(expirations)
    max = maximum(expirations)

    cm = countmap(expirations)
    exp_count = [get(cm, i, 0) for i=min:max]
    surv_count = sum(exp_count).-cumsum(exp_count)
    surv_count .= [sum(exp_count);surv_count[1:end-1]]
    hz = exp_count ./ surv_count
    [0; hz[1:end-1]]
end