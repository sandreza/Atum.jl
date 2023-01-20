using Atum
using Atum.ShallowWater
using Bennu: fieldarray
using BenchmarkTools
using CUDA
using StaticArrays: SVector
using WriteVTK
using LinearAlgebra
import Atum: boundarystate

include("../kernels.jl")
include("../new_dgsem.jl")
include("../lorenz.jl")

function ellipse(law::LorenzAttractor{S,T,V,W}, x⃗) where {S,T,V,W}
    x, y, z = x⃗
    ρ, σ, β = W
    bounding_set = (x / ρ)^2 + (y / ρ)^2 + (z / ρ - 1 - σ / ρ)^2
    # highly unlikely to be found bounding_set ≥ 1
    ρ₀ = exp(-5 * bounding_set)
    return SVector(ρ₀)
end

function lorenz_speed(law::LorenzAttractor{S,T,V,W}, x⃗) where {S,T,V,W}
    x₁, y₁, z₁ = x⃗
    ρ, σ, β = W
    u⃗₁ = SVector(-σ * (x₁ - y₁), -y₁ - x₁ * z₁ + ρ * x₁, -β * z₁ + x₁ * y₁)
    return sqrt(u⃗₁' * u⃗₁)
end

## Specify the Grid 
A = Array
FT = Float64
N = 1
K = 4

Nq = N + 1

law = LorenzAttractor{FT,3}(ρ=1.0, σ=1.0, β=1.0)
getparams(law::LorenzAttractor{S,T,V,W}) where {S,T,V,W} = W
parameters = getparams(law)

cell = LobattoCell{FT,A}(Nq, Nq, Nq)
vx = range(FT(-2 * parameters.ρ), stop=FT(2 * parameters.ρ), length=K + 1)
vy = range(FT(-2 * parameters.ρ), stop=FT(2 * parameters.ρ), length=K + 1)
vz = range(FT(-2), stop=FT(2 * parameters.ρ), length=K + 1)
grid = brickgrid(cell, (vx, vy, vz); periodic=(false, false, false))

volume_form = KennedyGruberFlux()
surface_numericalflux = RoeFlux()
x⃗ = points(grid)
dg = ModifiedFluxDifferencing(; law, grid, volume_form, surface_numericalflux, auxstate = x⃗)

##
cfl = FT(15 // 8) # for lsrk14, roughly a cfl of 0.125 per stage
c⃗ = lorenz_speed.(Ref(law), x⃗)
c = maximum(c⃗)
dt = 1.0 # cfl * min_node_distance(grid) / c 
# timeend = @isdefined(_testing) ? 10dt : FT(200)
timeend = 1 * dt
println("dt is ", dt)
q = fieldarray(undef, law, grid)
q .= ellipse.(Ref(law), x⃗)

normalization = sum(q .* dg.MJ) 

# now it can be thought of as a probability density
q .*= 1 / normalization


do_output = function (step, time, q)
    if step % ceil(Int, timeend / 100 / dt) == 0
        println(" currently on time ", time)
        println("extrema is ", extrema(components(q)[1]))
    end
end

odesolver = LSRK144(dg, q, dt)

tic = Base.time()
solve!(q, timeend, odesolver, after_step=do_output)
toc = Base.time()
println("The time for the original simulation is ", toc - tic)

sum(q .* dg.MJ) 