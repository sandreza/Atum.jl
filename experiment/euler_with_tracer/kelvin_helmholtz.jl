using Atum
using Revise
using Atum.EulerWithTracer
using Bennu: fieldarray
using BenchmarkTools
using CUDA
using StaticArrays: SVector
using WriteVTK
using LinearAlgebra
import Atum: boundarystate
# using GLMakie

# include("../kernels.jl")
# include("../new_dgsem.jl")
# include("../shallow_water.jl")
#=
function Atum.twopointflux(::Atum.EntropyConservativeFlux, law::EulerWithTracerLaw, q₁, aux₁, q₂, aux₂)

    γ = constants(law).γ
    ρ₁, ρu⃗₁, ρe₁, ρc₁ = EulerWithTracer.unpackstate(law, q₁)
    ρ₂, ρu⃗₂, ρe₂, ρc₂ = EulerWithTracer.unpackstate(law, q₂)

    Φ₁ = EulerWithTracer.geopotential(law, aux₁)
    u⃗₁ = ρu⃗₁ / ρ₁
    c₁ = ρc₁ / ρ₁
    p₁ = EulerWithTracer.pressure(law, ρ₁, ρu⃗₁, ρe₁, Φ₁)
    b₁ = ρ₁ / 2p₁

    Φ₂ = EulerWithTracer.geopotential(law, aux₂)
    u⃗₂ = ρu⃗₂ / ρ₂
    c₂ = ρc₂ / ρ₂
    p₂ = EulerWithTracer.pressure(law, ρ₂, ρu⃗₂, ρe₂, Φ₂)
    b₂ = ρ₂ / 2p₂

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    c_avg = Atum.avg(c₁, c₂)
    b_avg = Atum.avg(b₁, b₂)
    Φ_avg = Atum.avg(Φ₁, Φ₂)

    u²_avg = Atum.avg(u⃗₁' * u⃗₁, u⃗₂' * u⃗₂)
    ρ_log = Atum.logavg(ρ₁, ρ₂)
    b_log = Atum.logavg(b₁, b₂)

    fρ = u⃗_avg * ρ_log
    fρu⃗ = u⃗_avg * fρ' + ρ_avg / 2b_avg * I
    fρe = (1 / (2 * (γ - 1) * b_log) - u²_avg / 2 + Φ_avg) * fρ + fρu⃗ * u⃗_avg
    # fρc = c_avg * fρ' # check this 
    fρc = u⃗_avg * (ρ_avg * c_avg) # use kennedy gruber flux instead

    hcat(fρ, fρu⃗, fρe, fρc)
end

function Atum.surfaceflux(::Atum.MatrixFlux, law::EulerWithTracerLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
    FT = eltype(law)
    γ = constants(law).γ
    ecflux = Atum.surfaceflux(Atum.EntropyConservativeFlux(), law, n⃗, q⁻, aux⁻, q⁺, aux⁺) # defined in surface fluxes automagically

    ρ⁻, ρu⃗⁻, ρe⁻, ρθ⁻ = EulerWithTracer.unpackstate(law, q⁻)
    Φ⁻ = EulerWithTracer.geopotential(law, aux⁻)
    u⃗⁻ = ρu⃗⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    p⁻ = EulerWithTracer.pressure(law, ρ⁻, ρu⃗⁻, ρe⁻, Φ⁻)
    b⁻ = ρ⁻ / 2p⁻

    ρ⁺, ρu⃗⁺, ρe⁺, ρθ⁺ = EulerWithTracer.unpackstate(law, q⁺)
    Φ⁺ = EulerWithTracer.geopotential(law, aux⁺)
    u⃗⁺ = ρu⃗⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    p⁺ = EulerWithTracer.pressure(law, ρ⁺, ρu⃗⁺, ρe⁺, Φ⁺)
    b⁺ = ρ⁺ / 2p⁺

    Φ_avg = Atum.avg(Φ⁻, Φ⁺)
    ρ_log = Atum.logavg(ρ⁻, ρ⁺)
    b_log = Atum.logavg(b⁻, b⁺)
    u⃗_avg = Atum.avg(u⃗⁻, u⃗⁺)
    θ_avg = Atum.avg(θ⁻, θ⁺)
    p_avg = Atum.avg(ρ⁻, ρ⁺) / 2Atum.avg(b⁻, b⁺)
    u²_bar = 2 * norm(u⃗_avg) - Atum.avg(norm(u⃗⁻), norm(u⃗⁺))
    h_bar = γ / (2 * b_log * (γ - 1)) + u²_bar / 2 + Φ_avg
    c_bar = sqrt(γ * p_avg / ρ_log)

    u⃗mc = u⃗_avg - c_bar * n⃗
    u⃗pc = u⃗_avg + c_bar * n⃗
    u_avgᵀn = u⃗_avg' * n⃗

    v⁻ = Atum.entropyvariables(law, q⁻, aux⁻)
    v⁺ = Atum.entropyvariables(law, q⁺, aux⁺)
    Δv = v⁺ - v⁻

    λ1 = abs(u_avgᵀn - c_bar) * ρ_log / 2γ
    λ2 = abs(u_avgᵀn) * ρ_log * (γ - 1) / γ
    λ3 = abs(u_avgᵀn + c_bar) * ρ_log / 2γ
    λ4 = abs(u_avgᵀn) * p_avg

    Δv_ρ, Δv_ρu⃗, Δv_ρe, Δv_ρc = EulerWithTracer.unpackstate(law, Δv)
    u⃗ₜ = u⃗_avg - u_avgᵀn * n⃗

    w1 = λ1 * (Δv_ρ + u⃗mc' * Δv_ρu⃗ + (h_bar - c_bar * u_avgᵀn) * Δv_ρe)
    w2 = λ2 * (Δv_ρ + u⃗_avg' * Δv_ρu⃗ + (u²_bar / 2 + Φ_avg) * Δv_ρe)
    w3 = λ3 * (Δv_ρ + u⃗pc' * Δv_ρu⃗ + (h_bar + c_bar * u_avgᵀn) * Δv_ρe)

    Dρ = w1 + w2 + w3

    Dρu⃗ = (w1 * u⃗mc +
            w2 * u⃗_avg +
            w3 * u⃗pc +
            λ4 * (Δv_ρu⃗ - n⃗' * (Δv_ρu⃗) * n⃗ + Δv_ρe * u⃗ₜ))

    Dρe = (w1 * (h_bar - c_bar * u_avgᵀn) +
           w2 * (u²_bar / 2 + Φ_avg) +
           w3 * (h_bar + c_bar * u_avgᵀn) +
           λ4 * (u⃗ₜ' * Δv_ρu⃗ + Δv_ρe * (u⃗_avg' * u⃗_avg - u_avgᵀn^2)))

    c = c_bar
    ρ = ρ_log
    c⁻² = 1 / c^2
    uₙ = u_avgᵀn

    Δp = p⁺ - p⁻
    Δu⃗ = u⃗⁺ - u⃗⁻
    Δuₙ = Δu⃗' * n⃗
    Δρθ = ρθ⁺ - ρθ⁻

    ww1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² * 0.5
    ww3 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² * 0.5
    ww5 = abs(uₙ) * (Δρθ - θ_avg  * Δp * c⁻²)
    Dρc = (ww1 + ww3) * θ_avg + ww5


    ecflux - SVector(Dρ, Dρu⃗..., Dρe, Dρc) / 2
end

=#


function boundarystate(law::EulerWithTracerLaw, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρe⁻, ρc⁻ = EulerWithTracer.unpackstate(law, q⁻)
    ρ⁺, ρe⁺, ρc⁺ = ρ⁻, ρe⁻, ρc⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρe⁺, ρc⁺), aux⁻
end

function kelvin_helmholtz(law, x⃗, parameters)
    FT = eltype(law)
    x, z = x⃗

    Δρ = FT(parameters.Δρ)
    ρ₀ = FT(parameters.ρ₀)
    z₁ = FT(parameters.z₁)
    z₂ = FT(parameters.z₂)
    a = FT(parameters.a)
    A = FT(parameters.A)
    σ² = FT(parameters.σ²)
    P₀ = FT(parameters.P₀)
    uᶠ = FT(parameters.uᶠ)
    γ = FT(parameters.γ)

    ρ = 1 + (Δρ / ρ₀) * 0.5 * (tanh((z - z₁) / a) - tanh((z - z₂) / a))
    ρu = ρ * (uᶠ * (tanh((z - z₁) / a) - tanh((z - z₂) / a) - 1))
    ρv = ρ * (A * sin(2π * x) * (exp(-(z - z₁)^2 / σ²) + exp(-(z - z₂)^2 / σ²)))
    u⃗ = SVector(ρu / ρ, ρv / ρ)
    ρe = P₀ / (γ - 1) + 0.5 * ρ * u⃗' * u⃗
    ρc = ρ * (1 + 0.5 * (tanh((z - z₂) / a) - tanh((z - z₁) / a)))

    SVector(ρ, ρu, ρv, ρe, ρc)
end

parameters = (
    Δρ=1.0,
    ρ₀=1.0,
    z₁=0.5,
    z₂=1.5,
    a=0.05,
    A=0.01,
    σ²=0.04,
    P₀=10.0,
    uᶠ=1.0,
    γ=5 / 3,
)

## Specify the Grid 
A = CuArray
FT = Float64
N = 4
K = 800

Nq = N + 1

law = EulerWithTracerLaw{FT,2}()

cell = LobattoCell{FT,A}(Nq, Nq)
vx = range(FT(0), stop=FT(1), length=K + 1)
vy = range(FT(0), stop=FT(2), length=K + 1)
grid = brickgrid(cell, (vx, vy); periodic=(true, true))

cpucell = LobattoCell{FT,Array}(Nq, Nq)
cpugrid = brickgrid(cpucell, (vx, vy); periodic=(true, true))

# volume_form = FluxDifferencingForm(KennedyGruberFlux())
# surface_numericalflux = RoeFlux()
volume_form = FluxDifferencingForm(EntropyConservativeFlux())
surface_numericalflux = MatrixFlux()

dg = DGSEM(; law, grid, volume_form, surface_numericalflux)
# volume_form = EntropyConservativeFlux()
# nbl = NewEulerWithTracerLaw{FT,2}()
# new_dg = ModifiedFluxDifferencing(; law=nbl, grid, volume_form, surface_numericalflux=RoeFlux())
##
cfl = FT(14 // 8) # for lsrk14, roughly a cfl of 0.125 per stage

c = sqrt(parameters.P₀ / parameters.ρ₀)
dt = cfl * min_node_distance(grid) / c
# timeend = @isdefined(_testing) ? 10dt : FT(200)
timeend = 3.2
println("dt is ", dt)
q = fieldarray(undef, law, grid)
qc = fieldarray(undef, law, cpugrid)
qc .= kelvin_helmholtz.(Ref(law), points(cpugrid), Ref(parameters))
gpu_components = components(q)
cpu_components = components(qc)
for i in eachindex(gpu_components)
    gpu_components[i] .= A(cpu_components[i])
end

do_output = function (step, time, q)
    if step % ceil(Int, timeend / 100 / dt) == 0
        println(" currently on time ", time)
        ρ, ρu, ρv, ρe, ρc = components(q)
        println("extrema ", extrema(ρu))
    end
end

odesolver = LSRK144(dg, q, dt)
println("outputing now")
tic = Base.time()
solve!(q, timeend, odesolver) # after_step=do_output)
toc = Base.time()
ρ, ρu, ρv, ρe, ρc = components(q)
println("maximum density ", maximum(ρ))
println("minimum density ", minimum(ρ))
println("new kernel: The time for the simulation is ", toc - tic)
