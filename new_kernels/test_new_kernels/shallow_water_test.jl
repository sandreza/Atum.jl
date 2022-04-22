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
include("../shallow_water.jl")

function boundarystate(law::ShallowWaterLaw, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρθ⁻ = ShallowWater.unpackstate(law, q⁻)
    ρ⁺, ρθ⁺ = ρ⁻, ρθ⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρθ⁺), aux⁻
end

function bickleyjet(law, x⃗)
    FT = eltype(law)
    x, y = x⃗

    ϵ = FT(1 / 10)
    l = FT(1 / 2)
    k = FT(1 / 2)

    U = cosh(y)^(-2)

    Ψ = exp(-(y + l / 10)^2 / (2 * (l^2))) * cos(k * x) * cos(k * y)

    u = Ψ * (k * tan(k * y) + y / (l^2))
    v = -Ψ * k * tan(k * x)

    ρ = FT(1)
    ρu = ρ * (U + ϵ * u)
    ρv = ρ * (ϵ * v)
    ρθ = ρ * sin(k * y)

    SVector(ρ, ρu, ρv, ρθ)
end

## Specify the Grid 
A = Array
FT = Float64
N = 3
K = 16

Nq = N + 1

law = ShallowWaterLaw{FT,2}()

cell = LobattoCell{FT,A}(Nq, Nq)
v1d = range(FT(-2π), stop=FT(2π), length=K + 1)
grid = brickgrid(cell, (v1d, v1d); periodic=(true, true))

volume_form = FluxDifferencingForm(EntropyConservativeFlux())
dg = DGSEM(; law, grid, volume_form, surface_numericalflux=RoeFlux())
volume_form = EntropyConservativeFlux()
nbl = NewShallowWaterLaw{FT,2}()
new_dg = ModifiedFluxDifferencing(; law=nbl, grid, volume_form, surface_numericalflux=RoeFlux())
##
cfl = FT(15 // 8) # for lsrk14, roughly a cfl of 0.125 per stage

c = sqrt(constants(law).grav)
dt = cfl * min_node_distance(grid) / c
# timeend = @isdefined(_testing) ? 10dt : FT(200)
timeend = 200.0
println("dt is ", dt)
q = fieldarray(undef, law, grid)
q .= bickleyjet.(Ref(law), points(grid))

do_output = function (step, time, q)
    if step % ceil(Int, timeend / 100 / dt) == 0
        println(" currently on time ", time)
        ρ, ρu, ρv = components(q)
        println("extrema ", extrema(ρu))
    end
end

odesolver = LSRK144(new_dg, q, dt)
println("outputing now")
# solve!(q, timeend, odesolver; after_step=do_output)
tic = Base.time()
solve!(q, timeend, odesolver) #  after_step=do_output)
toc = Base.time()
ρ, ρu, ρv = components(q)
println("extrema ", extrema(ρu))
println("new kernel: The time for the simulation is ", toc - tic)

odesolver = LSRK144(dg, q, dt)
println("outputing now")
# solve!(q, timeend, odesolver; after_step=do_output)
tic = Base.time()
solve!(q, timeend, odesolver) #  after_step=do_output)
toc = Base.time()
ρ, ρu, ρv = components(q)
println("extrema ", extrema(ρu))
println("old kernel: The time for the simulation is ", toc - tic)

println("----------")
println("running again")

odesolver = LSRK144(new_dg, q, dt)
println("outputing now")
# solve!(q, timeend, odesolver; after_step=do_output)
tic = Base.time()
solve!(q, timeend, odesolver) #  after_step=do_output)
toc = Base.time()
ρ, ρu, ρv = components(q)
println("extrema ", extrema(ρu))
println("new kernel: The time for the simulation is ", toc - tic)


odesolver = LSRK144(dg, q, dt)
println("outputing now")
# solve!(q, timeend, odesolver; after_step=do_output)
tic = Base.time()
solve!(q, timeend, odesolver) #  after_step=do_output)
toc = Base.time()
ρ, ρu, ρv = components(q)
println("extrema ", extrema(ρu))
println("old kernel: The time for the simulation is ", toc - tic)


##
n⃗ = SVector(0.0, 0.0)
flux = MArray{Tuple{4},FT}(undef)
fill!(flux, -zero(FT))
q⁺ = q[1]
q⁻ = q[2]
nbl = NewShallowWaterLaw{FT,2}()
tims_mod = @benchmark modified_surfaceflux!(Atum.RoeFlux(), nbl, flux, n⃗, q⁻, 0.0, q⁺, 0.0)
bl = ShallowWaterLaw{FT,2}()
tims_orig = @benchmark rflux = Atum.surfaceflux(Atum.RoeFlux(), bl, n⃗, q⁻, 0.0, q⁺, 0.0)

modified_surfaceflux!(Atum.RoeFlux(), nbl, flux, n⃗, q⁻, 0.0, q⁺, 0.0)
rflux = Atum.surfaceflux(Atum.RoeFlux(), bl, n⃗, q⁻, 0.0, q⁺, 0.0)
rflux - flux

##
f = zeros(2, 4)
q1 = randn(4)
q2 = randn(4)

modified_volumeflux!(Atum.EntropyConservativeFlux(), nbl, f, q⁻, 0.0, q⁺, 0.0)
f2 = Atum.twopointflux(Atum.EntropyConservativeFlux(), bl, q⁻, 0.0, q⁺, 0.0)
f - f2

numflux = Atum.EntropyConservativeFlux()
tims_mod = @benchmark modified_volumeflux!(numflux, nbl, f, q⁻, 0.0, q⁺, 0.0)
tims_original = @benchmark f2 = Atum.twopointflux(numflux, bl, q⁻, 0.0, q⁺, 0.0)

#modified_volumeflux!(volume_numericalflux, law, flux[:, :], q1[:], aux1[:], q2, aux2)
##
q1 = fieldarray(undef, law, grid)
q1 .= bickleyjet.(Ref(law), points(grid))
q2 = fieldarray(undef, law, grid)
q2 .= bickleyjet.(Ref(law), points(grid))
q3 = fieldarray(undef, law, grid)
q3 .= bickleyjet.(Ref(law), points(grid))
q4 = fieldarray(undef, law, grid)
q4 .= bickleyjet.(Ref(law), points(grid))

function boundarystate(law::NewShallowWaterLaw, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρθ⁻ = q⁻[1], SVector(q⁻[2], q⁻[3]), q⁻[4] # unpackstate(law, q⁻)
    ρ⁺, ρθ⁺ = ρ⁻, ρθ⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρθ⁺), aux⁻
end

volume_form = FluxDifferencingForm(EntropyConservativeFlux())
dg = DGSEM(; law, grid, volume_form, surface_numericalflux=RoeFlux())
volume_form = EntropyConservativeFlux()
new_dg = ModifiedFluxDifferencing(; law=nbl, grid, volume_form, surface_numericalflux=RoeFlux())

dg(q1, q2, 0.0, increment=false)
new_dg(q3, q2, 0.0, increment=false)

norm(parent(components(q1 - q3)[1]))

oldtime = @benchmark CUDA.@sync begin
    dg(q1, q2, 0.0, increment=false)
end

newtime = @benchmark CUDA.@sync begin
    new_dg(q3, q2, 0.0, increment=false)
end

##
using UnPack
odesolver = LSRK144(new_dg, q, dt)
old_odesolver = LSRK144(dg, q, dt)
after_stage(args...) = nothing
dosteptime = @benchmark Atum.dostep!(q, odesolver, after_stage)
olddosteptime = @benchmark Atum.dostep!(q, old_odesolver, after_stage)

@unpack rhs!, dq, rka, rkb, rkc, dt, time = odesolver

caq = parent(components(q)[1])
cadq = parent(components(dq)[1])
modifieddostep = @benchmark begin
    caq = parent(components(q)[1])
    cadq = parent(components(dq)[1])
    for stage = 1:14
        cadq .*= rka[stage]
        rhs!(dq, q, 0.0)
        @. caq += rkb[stage] * dt * cadq
    end
end

unmodifieddostep = @benchmark begin
    for stage = 1:14
        dq .*= rka[stage]
        rhs!(dq, q, 0.0)
        @. q += rkb[stage] * dt * dq
    end
end
