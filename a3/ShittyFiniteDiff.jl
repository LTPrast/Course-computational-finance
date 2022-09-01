central_difference(f, x, h) = (f(x + h) - f(x - h))/2h

function cd_relative_errors(lo, hi, count)
    hs = 10 .^ LinRange(lo, hi, count)
    cds = central_difference.(exp, 1.0, hs)
    relerrs = abs.(cds .- ℯ) ./ ℯ
end
