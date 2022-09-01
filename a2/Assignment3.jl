using Random, Distributions
using LaTeXStrings
StandardNormal = Normal(0.0, 1.0)

Φ(x) = cdf(StandardNormal, x)

struct SimulationParameters{F<:Real, I<:Integer}
    S₀::F
    T::F
    M::I
    K::F
    r::F
    σ::F
end

function calc_d₁(s::SimulationParameters)
    (log(s.S₀/s.K) + (s.r + 0.5s.σ^2)s.T)/(s.σ√(s.T))
end

function calc_d₂(s::SimulationParameters)
    calc_d₁(s) - s.σ*√(s.T)
end

function calc_d₂(s::SimulationParameters, d₁::F) where {F<:Real}
    d₁ - s.σ*√(s.T)
end

function euro_call_option_value(s::SimulationParameters)
    d₁ = calc_d₁(s)
    d₂ = calc_d₂(s, d₁)
    s.S₀*Φ(d₁) - exp(-s.r * s.T) * s.K * Φ(d₂)
end

function agm_call_option_value(s::SimulationParameters)
    σₐ = s.σ√((2s.M + 1)/(6(s.M + 1)))
    rₐ = 0.5 * ((s.r - 0.5s.σ^2) + σₐ^2)
    sₐ = SimulationParameters(s.S₀, s.T, s.M, s.K, rₐ, σₐ)
    exp((rₐ - s.r)s.T) * euro_call_option_value(sₐ)
end

function euler_GBM(s::SimulationParameters)
    standard_normal = Normal(zero(typeof(s.S₀)), one(typeof(s.S₀)))
    δt = s.T / s.M
    Sₜs = Vector{typeof(s.S₀)}(undef, s.M+1)
    Sₜs[1] = s.S₀
    for n in 1:s.M
        zₙ = rand(standard_normal)
        Sₜs[n+1] = Sₜs[n] * (1 + s.r*δt + √(δt) * s.σ * zₙ)
    end
    Sₜs
end

function euler_GBM!(s::SimulationParameters, Sₜs)
    standard_normal = Normal(zero(typeof(s.S₀)), one(typeof(s.S₀)))
    if length(Sₜs) < s.M+1
        resize!(Sₜs, s.M+1)
    end
    δt = s.T / s.M
    Sₜs[1] = s.S₀
    for n in 1:s.M
        zₙ = rand(standard_normal)
        Sₜs[n+1] = Sₜs[n] * (1 + s.r*δt + √(δt) * s.σ * zₙ)
    end;
end

function vanilla_call_payoff(S, K)
    max(S-K, 0)
end

function gmean(xs)
    exp(mean(log.(xs)))
end

function aam_call_option_value_e(Sₜs, s::SimulationParameters)
    Aa = mean(Sₜs)
    vanilla_call_payoff(Aa, s.K) * exp(-s.r * s.T)
end

function agm_call_option_value_e(Sₜs, s::SimulationParameters)
    Ag = gmean(Sₜs)
    vanilla_call_payoff(Ag, s.K) * exp(-s.r * s.T)
end

function mc_asian_gm(s::SimulationParameters, nsim)
    gm_calls = Vector{typeof(s.S₀)}(undef, nsim)
    Sₜs = Vector{typeof(s.S₀)}(undef, s.M+1)
    for i in 1:nsim
        exact_GBM!(s, Sₜs)
        @inbounds gm_calls[i] = agm_call_option_value_e(Sₜs, s)
    end
    gm_calls
end

function mc_asian_am(s::SimulationParameters, nsim)
    gm_calls = Vector{typeof(s.S₀)}(undef, nsim)
    Sₜs = Vector{typeof(s.S₀)}(undef, s.M+1)
    for i in 1:nsim
        exact_GBM!(s, Sₜs)
        @inbounds gm_calls[i] = aam_call_option_value_e(Sₜs, s)
    end
    gm_calls
end

function exact_GBM!(s::SimulationParameters, Sₜs)
    standard_normal = Normal(zero(typeof(s.S₀)), one(typeof(s.S₀)))
    if length(Sₜs) < s.M+1
        resize!(Sₜs, s.M+1)
    end
    δt = s.T / s.M
    Sₜs[1] = s.S₀
    for n in 1:s.M
        zₙ = rand(standard_normal)
        Sₜs[n+1] = Sₜs[n] * exp((s.r - 0.5 * s.σ^2)*δt + s.σ*√(δt)*zₙ)
    end;
end

function standard_error(xs)
    std(xs)/√length(xs)
end

function asian_gm_convergence(s::SimulationParameters, nsim_values)
    gm_means_stderrs = Array{typeof(s.S₀)}(undef, length(nsim_values), 2)
    for (i,nsim) in enumerate(nsim_values)
        gm_calls = mc_asian_gm(s, nsim)
        @inbounds gm_means_stderrs[i,1] = mean(gm_calls)
        @inbounds gm_means_stderrs[i,2] = standard_error(gm_calls)
    end
    gm_means_stderrs
end

function asian_am_convergence(s::SimulationParameters, nsim_values)
    am_means_stderrs = Array{typeof(s.S₀)}(undef, length(nsim_values), 2)
    for (i,nsim) in enumerate(nsim_values)
        am_calls = mc_asian_am(s, nsim)
        @inbounds am_means_stderrs[i,1] = mean(am_calls)
        @inbounds am_means_stderrs[i,2] = standard_error(am_calls)
    end
    am_means_stderrs
end

function mc_asian_am_cv(s::SimulationParameters, nsim)
    am_calls = Vector{typeof(s.S₀)}(undef, nsim)
    gm_calls = Vector{typeof(s.S₀)}(undef, nsim)
    Sₜs = Vector{typeof(s.S₀)}(undef, s.M+1)
    for i in 1:nsim
        exact_GBM!(s, Sₜs)
        @inbounds am_calls[i] = aam_call_option_value_e(Sₜs, s)
        @inbounds gm_calls[i] = agm_call_option_value_e(Sₜs, s)
    end
    gm_var = var(gm_calls)
    am_gm_cov = cov(am_calls, gm_calls)
    β = am_gm_cov / gm_var
    gm_analytical = agm_call_option_value(s)
    am_calls .-= β * (gm_calls .- gm_analytical)
    am_calls
end

function asian_am_convergence_cv(s::SimulationParameters, nsim_values)
    amcv_means_stderrs = Array{typeof(s.S₀)}(undef, length(nsim_values), 2)
    for (i,nsim) in enumerate(nsim_values)
        amcv_calls = mc_asian_am_cv(s, nsim)
        amcv_means_stderrs[i, 1] = mean(amcv_calls)
        amcv_means_stderrs[i, 2] = standard_error(amcv_calls)
    end
    amcv_means_stderrs
end

function sample_variance(xs)
    x̄ = mean(xs)
    sum((xs .- x̄).^2) / (length(xs) - 1)
end

function estimate_β(s::SimulationParameters, nsim)
    gm_calls = Vector{typeof(s.S₀)}(undef, nsim)
    am_calls = Vector{typeof(s.S₀)}(undef, nsim)
    Sₜs = Vector{typeof(s.S₀)}(undef, s.M+1)
    for i in 1:nsim
        exact_GBM!(s, Sₜs)
        @inbounds gm_calls[i] = agm_call_option_value_e(Sₜs, s)
        @inbounds am_calls[i] = aam_call_option_value_e(Sₜs, s)
    end
    gm_var = var(gm_calls)
    am_gm_covar = cov(am_calls, gm_calls)
    am_gm_covar/gm_var
end

function mc_asian_am_cv(s::SimulationParameters, nsim, β)
    amcv_calls = Vector{typeof(s.S₀)}(undef, nsim)
    gm_analytical = agm_call_option_value(s)
    Sₜs = Vector{typeof(s.S₀)}(undef, s.M+1)
    for i in 1:nsim
        exact_GBM!(s, Sₜs)
        am_call = aam_call_option_value_e(Sₜs, s)
        gm_call = agm_call_option_value_e(Sₜs, s)
        @inbounds amcv_calls[i] = am_call - β * (gm_call - gm_analytical)
    end
    amcv_calls
end

function asian_am_convergence_cv(s::SimulationParameters, nsim_values, β)
    amcv_means_stderrs = Array{typeof(s.S₀)}(undef, length(nsim_values), 2)
    for (i,nsim) in enumerate(nsim_values)
        amcv_calls = mc_asian_am_cv(s, nsim, β)
        amcv_means_stderrs[i, 1] = mean(amcv_calls)
        amcv_means_stderrs[i, 2] = standard_error(amcv_calls)
    end
    amcv_means_stderrs
end

#= Generate some plots =#
# Default simulation simulation parameters
S₀ = 100.0
K = 99.0
r = 0.06
σ = 0.2
T = 1.0
M = 350
s = SimulationParameters(S₀, T, M, K, r, σ)
# Convergence of Asian GM option Monte Carlo
Random.seed!(Int64(0xdeadbeef))
nsim_values = 10 .^(2:7)  # 100, 1000, ... 10⁷
gm_calls_stderrs = asian_gm_convergence(s, nsim_values)
gm_analytical = agm_call_option_value(s)

# Convergence of Asian AM option without control variates
Random.seed!(Int64(0xb00b5))
am_calls_stderrs = asian_am_convergence(s, nsim_values)

# Estimate β parameter for method of control variates
Random.seed!(Int64(0xc0ffee))
β = estimate_β(s, 10_000_000)

# Convergence of Asian AM option with control variates
Random.seed!(Int64(0xb1ac8c0ffee))
amcv_calls_stderrs = asian_am_convergence_cv(s, nsim_values, β)

#= PLOTS =#
gm_analytical_repeater(x) = agm_call_option_value(s)

using Plots
gr()
gm_convergence_plot = plot(nsim_values, gm_calls_stderrs[:,1]; seriestype=:scatter, xscale=:log10, yerror=gm_calls_stderrs[:,2], title="Asian GM Option", label = "MC estimate", legend=true)
plot!(gm_convergence_plot, gm_analytical_repeater, label="analytical")
xlabel!(gm_convergence_plot, "Number of paths")
ylabel!(gm_convergence_plot, "Option Price")
xticks!(gm_convergence_plot, nsim_values)
savefig(gm_convergence_plot, "gm_convergence.pdf")


begin
am_convergence_plot = plot(nsim_values, am_calls_stderrs[:,1]; seriestype=:scatter, xscale=:log10, yerror=am_calls_stderrs[:,2], title="Asian AM Option", legend=false)
xlabel!(am_convergence_plot, "Number of paths")
ylabel!(am_convergence_plot, "Option Price")
xticks!(am_convergence_plot, nsim_values)
savefig(am_convergence_plot, "am_convergence.pdf")
end

begin
amcv_convergence_plot = plot(nsim_values, am_calls_stderrs[:,1]; seriestype=:scatter, xscale=:log10, yerror=am_calls_stderrs[:,2], title="Asian AM Option", label = "without control variate", legend=true)
plot!(nsim_values, amcv_calls_stderrs[:,1]; yerror=amcv_calls_stderrs[:,2], seriestype=:scatter, label="with control variate")
xlabel!(amcv_convergence_plot, "Number of paths")
ylabel!(amcv_convergence_plot, "Option Price")
xticks!(amcv_convergence_plot, nsim_values)
savefig(amcv_convergence_plot, "amcv_convergence.pdf")
end

#= Last part of 3.3 
Julia's metaprogramming would be very useful here, but unfortunately I do not
have time to learn it at the moment topkek =#

#= Vary number of points in average, M =#
default_s = SimulationParameters(100.0, 1.0, 100, 99.0, 0.06, 0.2)
M_values = 50:50:300

function estimate_βs_M(s::SimulationParameters, M_values, nsim)
    n = length(M_values)
    βs = Vector{typeof(s.S₀)}(undef, n)
    for i in 1:n
        @inbounds sₙ = SimulationParameters(s.S₀, s.T, M_values[i], s.K, s.r, s.σ)
        @inbounds βs[i] = estimate_β(sₙ, nsim)
    end
    βs
end

nsim = 1_000_000

βs = estimate_βs_M(default_s, M_values, nsim)

function aam_M_convergence(s::SimulationParameters, M_values, nsim, βs)
    n = min(length(M_values), length(βs))
    amcv_calls_stderrs = Array{typeof(s.S₀)}(undef, n, 2)
    am_calls_stderrs = Array{typeof(s.S₀)}(undef, n, 2)
    for i in 1:n
        sₙ = SimulationParameters(s.S₀, s.T, M_values[i], s.K, s.r, s.σ)
        am_calls = mc_asian_am(sₙ, nsim)
        @inbounds am_calls_stderrs[i,1] = mean(am_calls)
        @inbounds am_calls_stderrs[i,2] = standard_error(am_calls)
        amcv_calls = mc_asian_am_cv(sₙ, nsim, β)
        @inbounds amcv_calls_stderrs[i,1] = mean(amcv_calls)
        @inbounds amcv_calls_stderrs[i,2] = standard_error(amcv_calls)
    end
    (am_calls_stderrs, amcv_calls_stderrs)
end

nsim = 1_000_000
(am_calls_stderrs, amcv_calls_stderrs) = aam_M_convergence(default_s, M_values, nsim, βs)

amcv_M_plot = plot(M_values, am_calls_stderrs[:,1];  yerror=am_calls_stderrs[:,2], label="without control variate", title=L"Varying $K$")
plot!(amcv_M_plot, M_values, amcv_calls_stderrs[:,1]; yerror=amcv_calls_stderrs[:,2], label="with control variate")
xlabel!(amcv_M_plot, L"$K$")
ylabel!(amcv_M_plot, "Option Value")
savefig(amcv_M_plot, "amcv_M_plot.pdf")

#= Vary strike price =#
function estimate_βs_K(s::SimulationParameters, K_values, nsim)
    n = length(K_values)
    βs = Vector{typeof(s.S₀)}(undef, n)
    for i in 1:n
        @inbounds sₙ = SimulationParameters(s.S₀, s.T, s.M, K_values[i], s.r, s.σ)
        @inbounds βs[i] = estimate_β(sₙ, nsim)
    end
    βs
end

nsim = 1_000_000
K_values = 50.0:10.0:150.0
βs = estimate_βs_K(default_s, K_values, nsim)

function aam_K_convergence(s::SimulationParameters, K_values, nsim, βs)
    n = min(length(K_values), length(βs))
    amcv_calls_stderrs = Array{typeof(s.S₀)}(undef, n, 2)
    am_calls_stderrs = Array{typeof(s.S₀)}(undef, n, 2)
    for i in 1:n
        sₙ = SimulationParameters(s.S₀, s.T, s.M, s.K, s.r, s.σ)
        am_calls = mc_asian_am(sₙ, nsim)
        @inbounds am_calls_stderrs[i,1] = mean(am_calls)
        @inbounds am_calls_stderrs[i,2] = standard_error(am_calls)
        amcv_calls = mc_asian_am_cv(sₙ, nsim, β)
        @inbounds amcv_calls_stderrs[i,1] = mean(amcv_calls)
        @inbounds amcv_calls_stderrs[i,2] = standard_error(amcv_calls)
    end
    (am_calls_stderrs, amcv_calls_stderrs)
end

nsim = 1_000_000
(am_calls_stderrs, amcv_calls_stderrs) = aam_K_convergence(default_s, K_values, nsim, βs)

amcv_K_plot = plot(K_values, am_calls_stderrs[:,1];  yerror=am_calls_stderrs[:,2], label="without control variate", title="Varying Points in Average")
plot!(amcv_K_plot, K_values, amcv_calls_stderrs[:,1]; yerror=amcv_calls_stderrs[:,2], label="with control variate")
xlabel!(amcv_K_plot, "Number of points in average")
ylabel!(amcv_K_plot, "Option Value")
savefig(amcv_K_plot, "amcv_K_plot.pdf")


 #= Varying σ =#
 function estimate_βs_σ(s::SimulationParameters, σ_values, nsim)
    n = length(σ_values)
    βs = Vector{typeof(s.S₀)}(undef, n)
    for i in 1:n
        @inbounds sₙ = SimulationParameters(s.S₀, s.T, s.M, s.K, s.r, σ_values[i])
        @inbounds βs[i] = estimate_β(sₙ, nsim)
    end
    βs
end

nsim = 1_000_000
σ_values = 0.1:0.1:1.0
βs = estimate_βs_σ(default_s, σ_values, nsim)

function aam_σ_convergence(s::SimulationParameters, σ_values, nsim, βs)
    n = min(length(σ_values), length(βs))
    amcv_calls_stderrs = Array{typeof(s.S₀)}(undef, n, 2)
    am_calls_stderrs = Array{typeof(s.S₀)}(undef, n, 2)
    for i in 1:n
        sₙ = SimulationParameters(s.S₀, s.T, s.M, s.K, s.r, σ_values[i])
        am_calls = mc_asian_am(sₙ, nsim)
        @inbounds am_calls_stderrs[i,1] = mean(am_calls)
        @inbounds am_calls_stderrs[i,2] = standard_error(am_calls)
        amcv_calls = mc_asian_am_cv(sₙ, nsim, β)
        @inbounds amcv_calls_stderrs[i,1] = mean(amcv_calls)
        @inbounds amcv_calls_stderrs[i,2] = standard_error(amcv_calls)
    end
    (am_calls_stderrs, amcv_calls_stderrs)
end

nsim = 1_000_000
(am_calls_stderrs, amcv_calls_stderrs) = aam_σ_convergence(default_s, σ_values, nsim, βs)

amcv_σ_plot = plot(σ_values, am_calls_stderrs[:,1];  yerror=am_calls_stderrs[:,2], label="without control variate", title="Varying Points in Average")
plot!(amcv_σ_plot, σ_values, amcv_calls_stderrs[:,1]; yerror=amcv_calls_stderrs[:,2], label="with control variate")
xlabel!(amcv_σ_plot, L"\sigma")
ylabel!(amcv_σ_plot, "Option Value")
savefig(amcv_σ_plot, "amcv_σ_plot.pdf") 