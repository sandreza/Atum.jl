using Atum
using Atum.EulerTotalEnergy
using Atum.EulerGravity
using Random
using StaticArrays
using StaticArrays: SVector, MVector
using WriteVTK
using Statistics
using BenchmarkTools
using Revise
using CUDA

include("../kernels.jl")
include("../new_dgsem.jl")
include("../euler.jl")

import Atum: boundarystate, source!
Random.seed!(12345)
# for lazyness 
const parameters = (
    R=287.0,
    pₒ=1e5, # get_planet_parameter(:MSLP),
    g=9.81,
    cp=1e3,
    γ=1.4,
    cv=1e3 / 1.4,
    T_0=0.0,
    xmax=3e3,
    ymax=3e3,
    zmax=3e3,
    Tₛ=300.0, # 300.0,
    ρₛ=1.27,
    scale_height=8e3,
    Δθ=10.0,
    Q₀=100.0, # W/m²
    r_ℓ=100.0, # radiation length scale
    s_ℓ=100.0, # sponge exponential decay length scale
    λ=1 / 10.0, # sponge relaxation timescale
)

@inline function modified_boundarystate(law::DryTotalEnergyLaw, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρe⁻ = unpackstate(law, q⁻)
    ρ⁺, ρe⁺ = ρ⁻, ρe⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

@inline function modified_source!(law::DryTotalEnergyLaw, source, state, aux)
    # Extract the state
    ρ, ρu⃗, ρe = unpackstate(law, state)

    z = aux[3]

    Q₀ = 100.0   # convective_forcing.Q₀
    r_ℓ = 100.0  # convective_forcing.r_ℓ
    s_ℓ = 100.0  # convective_forcing.s_ℓ
    λ = 1 / 10.0 # convective_forcing.λ
    L = 3e3      # convective_forcing.L

    radiation_profile = Q₀ / r_ℓ * exp(-z / r_ℓ)

    damping_profile = -exp(-(L - z) / s_ℓ)

    # Apply convective forcing and sponge layer damping
    source[2] = λ * damping_profile * ρu⃗[1]
    source[3] = λ * damping_profile * ρu⃗[2]
    source[4] = λ * damping_profile * ρu⃗[3]
    source[5] = ρ * radiation_profile

    return nothing
end

θ₀(z) = parameters.Tₛ + parameters.Δθ / parameters.zmax * z
p₀(z) = parameters.pₒ * (parameters.g / (-parameters.Δθ / parameters.zmax * parameters.cp) * log(θ₀(z) / parameters.Tₛ) + 1)^(parameters.cp / parameters.R)
T₀(z) = (p₀(z) / parameters.pₒ)^(parameters.R / parameters.cp) * θ₀(z)
ρ₀(z) = p₀(z) / (parameters.R * T₀(z))

ρu₀(x, y, z) = 0.01 * @SVector [randn(), randn(), randn()]

e_pot(z) = parameters.g * z
e_int(z) = parameters.cv * (T₀(z) - parameters.T_0)
e_kin(x, y, z) = 0.5 * (ρu₀(x, y, z)' * ρu₀(x, y, z)) / ρ₀(z)^2

ρe₀(x, y, z) = ρ₀(z) * (e_kin(x, y, z) + e_int(z) + e_pot(z))

function initial_condition(law, x⃗)
    FT = eltype(law)
    x, y, z = x⃗

    ρ = ρ₀(z)
    ρu, ρv, ρw = ρu₀(x, y, z)
    ρe = ρe₀(x, y, z)

    SVector(ρ, ρu, ρv, ρw, ρe)
end

A = CuArray
FT = Float64
N = 3

K = 16
println("DOFs per direction = ", (N + 1) * K, " with VF ", vf)

volume_form = vf

outputvtk = false
Nq = N + 1

law = EulerGravityLaw{FT,3}()
bl = law
# pp = 2
cell = LobattoCell{FT,A}(Nq, Nq, Nq)
v1d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v2d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v3d = range(FT(0), stop=FT(3.0e3), length=K + 1)
grid = brickgrid(cell, (v1d, v2d, v3d); periodic=(true, true, false))
x⃗ = points(grid)

dg = ModifiedFluxDifferencing(; law=DryTotalEnergyLaw{FT,3}(), grid, volume_form=KennedyGruberFlux(), surface_numericalflux=RoeFlux(), auxstate=dg.auxstate)

cfl = 1.8 # FT(15 // 8) # for lsrk14, roughly a cfl of 0.125 per stage

c = 330.0 # [m/s]
dt = cfl * min_node_distance(grid) / c * 1.0
println(" the dt is ", dt)
timeend = 60 * 60 * 4  #

q = fieldarray(undef, law, grid)
q0 = fieldarray(undef, law, grid)
q .= initial_condition.(Ref(law), points(grid))
q0 .= initial_condition.(Ref(law), points(grid))
qq = initial_condition.(Ref(law), points(grid))
dqq = initial_condition.(Ref(law), points(grid))

do_output = function (step, time, q)
    if step % ceil(Int, timeend / 100 / dt) == 0
        println("simulation is ", time / timeend * 100, " percent complete")
    end
end

odesolver_dg = LSRK144(dg, q, dt)

tic = Base.time()
solve!(q, timeend, odesolver_dg, after_step=do_output)
toc = Base.time()
println("The time for the original simulation is ", toc - tic)
println(components(q)[1][1])
