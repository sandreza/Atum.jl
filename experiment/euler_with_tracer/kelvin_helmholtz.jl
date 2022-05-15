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
    Δρ=0.0,
    ρ₀=1.0,
    z₁=0.5,
    z₂=1.5,
    a=0.05,
    A=0.01,
    σ²=0.04,
    P₀=10.0,
    uᶠ=1.0,
    γ= 5/3,
)

## Specify the Grid 
A = CuArray
FT = Float64
N = 7
K = 128

Nq = N + 1

law = EulerWithTracerLaw{FT,2}()

cell = LobattoCell{FT,A}(Nq, Nq)
vx = range(FT(0), stop=FT(1), length=K + 1)
vy = range(FT(0), stop=FT(2), length=K + 1)
grid = brickgrid(cell, (vx, vy); periodic=(true, true))

cpucell = LobattoCell{FT,Array}(Nq, Nq)
cpugrid = brickgrid(cpucell, (vx, vy); periodic=(true, true))

volume_form = FluxDifferencingForm(KennedyGruberFlux())
surface_numericalflux = RoeFlux()
# volume_form = FluxDifferencingForm(EntropyConservativeFlux())
# surface_numericalflux = MatrixFlux()

dg = DGSEM(; law, grid, volume_form, surface_numericalflux);
##
cfl = FT(18 // 8) # for lsrk14, roughly a cfl of 0.125 per stage

c = sqrt(parameters.P₀ / parameters.ρ₀ * 2.0)
dt = cfl * min_node_distance(grid) / c
# timeend = @isdefined(_testing) ? 10dt : FT(200)
timeend = 2
println("dt is ", dt)
q = fieldarray(undef, law, grid)
qc = fieldarray(undef, law, cpugrid)
q2 = fieldarray(undef, law, cpugrid)
q4 = fieldarray(undef, law, cpugrid)
q6 = fieldarray(undef, law, cpugrid)

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


qsaves = (q2, q4, q6)
for (i, qsave) in enumerate(qsaves)
    println("for ", i)
    odesolver = LSRK144(dg, q, dt)
    println("outputing now")
    tic = Base.time()
    solve!(q, timeend, odesolver)
    toc = Base.time()
    ρ, ρu, ρv, ρe, ρc = components(q)
    println("maximum density ", maximum(ρ))
    println("minimum density ", minimum(ρ))
    println("new kernel: The time for the simulation is ", toc - tic)

    gpu_components = components(q)
    cpu_components = components(qsave)
    for i in eachindex(gpu_components)
        cpu_components[i] .= Array(gpu_components[i])
    end

end

##