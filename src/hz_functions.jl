function modelhz(x, N)
    cum_lapse = N .- x
    lapse = cum_lapse .- [zero(eltype(x)); cum_lapse[1:end-1]]
    hz = lapse ./ [N; x[1:end-1]]
    hz[isnan.(hz)] .= 0
    return hz
end