using LinearAlgebra
using Plots
using Distributions
using LaTeXStrings

struct OptionParameters{F <: Real}
    S₀::F  # initial asset price
    r::F  # risk-free interest rate
    σ::F  # volatility
    K::F  # strike 
    T::F  # maturity
end

struct FDMParameters{F <: Real, I <: Integer}
    S_min::F
    S_max::F
    Nₓ::I
    Nₜ::I
end

StandardNormal = Normal(0, 1)
Φ(x) = cdf(StandardNormal, x)

function bs_option_analytical(opt::OptionParameters)
    d₁ = (log(opt.S₀/opt.K) + (opt.r + opt.σ^2/2))/(opt.σ * √(opt.T))
    d₂ = d₁ - opt.σ * √(opt.T)
    opt.S₀*Φ(d₁) - exp(-opt.r * opt.T) * opt.K * Φ(d₂)
end

function calc_derived_fdm_params(opt::OptionParameters, fdm::FDMParameters)
    Δτ = opt.T / fdm.Nₜ
    X_max = log(fdm.S_max)
    X_min = log(fdm.S_min)
    Δx = (X_max - X_min)fdm.Nₓ
    (X_min, X_max, Δx, Δτ)
end

euro_call_payoff(S, K) = max(S - K, 0)

function explicit_FTCS_el(opt::OptionParameters, fdm::FDMParameters)
    σ = opt.σ
    r = opt.r
    (_, _, Δx, Δτ) = calc_derived_fdm_params(opt, fdm)
    kₘ =  1 - r * Δτ  - σ^2*Δτ/(Δx^2)
    kₗ = Δτ/(2*Δx) * (σ^2/Δx - (r - σ^2/2))
    kᵤ = Δτ/(2*Δx) * ((r - σ^2/2) + σ^2/Δx)
    (kₗ, kₘ, kᵤ)
end

function matrix_FTCS_explicit(opt::OptionParameters, fdm::FDMParameters)
    X_min = log(fdm.S_min)
    X_max = log(fdm.S_max)
    Δx = (X_max - X_min) / fdm.Nₓ
    Δτ = opt.T/fdm.Nₜ
    σ = opt.σ
    r = opt.r
    (kₗ, kₘ, kᵤ) = explicit_FTCS_el(opt, fdm)
    main_diagonal = fill(kₘ, fdm.Nₓ)
    lower_subdiag = fill(kₗ, fdm.Nₓ-1)
    upper_subdiag = fill(kᵤ, fdm.Nₓ-1)
    main_diagonal[1] = 0
    main_diagonal[end] = 0
    upper_subdiag[1] = 0
    lower_subdiag[end] = 0
    Tridiagonal(lower_subdiag, main_diagonal, upper_subdiag)
end

function matrix_FTCS_implicit(opt::OptionParameters, fdm::FDMParameters)
    X_min = log(fdm.S_min)
    X_max = log(fdm.S_max)
    Δx = (X_max - X_min) / fdm.Nₓ
    σ = opt.σ
    r = opt.r
    Δx = (X_max - X_min)/fdm.Nₓ
    Δτ = opt.T/fdm.Nₜ
    r_minus_σ²_over_2 = r - σ^2/2
    σ²_over_Δx² = σ^2/(Δx^2)
    Δτ_over_2Δx = Δτ / (2Δx)
    kᵤ = -Δτ_over_2Δx * (r_minus_σ²_over_2 + σ^2/Δx)
    kₗ = -Δτ_over_2Δx * (σ^2/Δx - r_minus_σ²_over_2)
    kₘ = 1 + Δτ*(σ²_over_Δx² + r)
    main_diagonal = fill(kₘ, fdm.Nₓ - 2)
    lower_subdiag = fill(kₗ, fdm.Nₓ - 3)
    upper_subdiag = fill(kᵤ, fdm.Nₓ - 3)
    Tridiagonal(lower_subdiag, main_diagonal, upper_subdiag)
end

function FTCS_implicit(opt::OptionParameters, fdm::FDMParameters)
    A = matrix_FTCS_implicit(opt, fdm)
    X_min = log(fdm.S_min)
    X_max = log(fdm.S_max)
    Δx = (X_max - X_min) / fdm.Nₓ
    Δτ = opt.T/fdm.Nₜ
    Xs = LinRange(X_min, X_max, fdm.Nₓ)
    V₀ = euro_call_payoff.(exp.(Xs), opt.K)  # τ = 0
    V₀[1] = 0  # should be satisfied already, but just in case
    V_values = similar(V₀, fdm.Nₓ, fdm.Nₜ)   # store all V values
    V_values[:,1] = V₀  # first column boundary condition
    # TODO: set the bottom row here using V_Xmax = exp(X_max) - opt.K * exp(-opt.r * τ)
    τ = Δτ
    for i in 2:length(eachcol(V_values))
        V_Xmax = exp(X_max) - opt.K * exp(-opt.r * τ)  # boundary condition
        # V₀[2:end-1] .= V_values[2:end-1,i-1]
        # V₀[end] -= V_Xmax
        V_values[2:end-1,i] = A\(V_values[2:end-1,i-1])
        V_values[1,i] = 0
        V_values[end,i] = V_Xmax
        τ += Δτ
    end
    V_values
end

function FTCS_explicit(opt::OptionParameters, fdm::FDMParameters)
    A = matrix_FTCS_explicit(opt, fdm)
    X_min = log(fdm.S_min)
    X_max = log(fdm.S_max)
    Δx = (X_max - X_min) / fdm.Nₓ
    Δτ = opt.T/fdm.Nₜ
    Xs = LinRange(X_min, X_max, fdm.Nₓ)
    V₀ = euro_call_payoff.(exp.(Xs), opt.K)
    V₀[1] = 0  # should be satisfied already, but just in case
    V_values = similar(V₀, fdm.Nₓ, fdm.Nₜ+1)   # store all V values
    V_values[:,1] = V₀
    τ = Δτ
    for i in 2:length(eachcol(V_values))
        V_values[:,i] = A*V_values[:,i-1]
        V_values[end,i] = fdm.S_max - opt.K * exp(-r*(τ))  # TODO: is this correct?
        τ += Δτ
    end 
    V_values
end

S₀ = 100.0
r = 0.04
σ = 0.3
T = 1.0
K = 110.0

opt_default = OptionParameters(S₀, r, σ, K, T)
fdm_default = FDMParameters(1.0, 500.0, 2000, 1000)

explicit_FTCS_result = FTCS_explicit(opt_default, fdm_default)

implicit_FTCS_result = FTCS_implicit(opt_default, fdm_default)

function plot_final_results(V_values, opt::OptionParameters, fdm::FDMParameters; method_label="<method-name>")
    X_min = log(fdm.S_min)
    X_max = log(fdm.S_max)
    Xs = LinRange(X_min, X_max, fdm.Nₓ)
    Ss = exp.(Xs)
    opts = OptionParameters.(Ss, opt.r, opt.σ, opt.K, opt.T)
    t0_plot = plot(Ss, V_values[:,end]; label=method_label * L", $t=T$", lw=2)
    xlabel!(t0_plot, L"Asset Price, $S$")
    ylabel!(t0_plot, L"Option Value, $V$")
    plot!(t0_plot, Ss, V_values[:,1]; label=method_label * L", $t=0$", lw=2)
    plot!(t0_plot, Ss, bs_option_analytical.(opts); lw=2, label="Analytical")
    t0_plot
end

explicit_2d_plot = plot_final_results(explicit_FTCS_result, opt_default, fdm_default; method_label="Explicit FTCS")