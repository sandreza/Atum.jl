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

function boundarystate(law::EulerGravityLaw, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
    ρ⁺, ρe⁺ = ρ⁻, ρe⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

function source!(law::EulerGravityLaw, source, state, aux, dim, directions)
    # Extract the state
    ρ, ρu⃗, ρe = EulerGravity.unpackstate(law, state)

    z = aux[3]

    Q₀ = 100.0   # convective_forcing.Q₀
    r_ℓ = 100.0  # convective_forcing.r_ℓ
    s_ℓ = 100.0  # convective_forcing.s_ℓ
    λ = 1 / 10.0 # convective_forcing.λ
    L = 3e3      # convective_forcing.L

    radiation_profile = Q₀ / r_ℓ * exp(-z / r_ℓ)

    damping_profile = -exp(-(L - z) / s_ℓ)

    # Apply convective forcing
    source[2] += λ * damping_profile * ρu⃗[1]
    source[3] += λ * damping_profile * ρu⃗[2]
    source[4] += λ * damping_profile * ρu⃗[3]
    source[5] += ρ * radiation_profile

    return nothing
end

function boundarystate(law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρe⁻ = EulerTotalEnergy.unpackstate(law, q⁻)
    ρ⁺, ρe⁺ = ρ⁻, ρe⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

function source!(law::EulerTotalEnergyLaw, source, state, aux, dim, directions)
    # Extract the state
    ρ, ρu⃗, ρe = EulerTotalEnergy.unpackstate(law, state)

    z = aux[3]

    Q₀ = 100.0   # convective_forcing.Q₀
    r_ℓ = 100.0  # convective_forcing.r_ℓ
    s_ℓ = 100.0  # convective_forcing.s_ℓ
    λ = 1 / 10.0 # convective_forcing.λ
    L = 3e3      # convective_forcing.L

    radiation_profile = Q₀ / r_ℓ * exp(-z / r_ℓ)

    damping_profile = -exp(-(L - z) / s_ℓ)

    # Apply convective forcing
    source[2] += λ * damping_profile * ρu⃗[1]
    source[3] += λ * damping_profile * ρu⃗[2]
    source[4] += λ * damping_profile * ρu⃗[3]
    source[5] += ρ * radiation_profile

    return nothing
end

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

K = 4 * 4 * 2
vf = FluxDifferencingForm(KennedyGruberFlux())
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
fsdg = FluxSource(; law, grid, volume_form=KennedyGruberFlux(), surface_numericalflux=RoeFlux())
dg = DGSEM(; law, grid, volume_form=FluxDifferencingForm(KennedyGruberFlux()), surface_numericalflux=RoeFlux())
new_dg = ModifiedFluxDifferencing(; law=DryTotalEnergyLaw{FT,3}(), grid, volume_form=KennedyGruberFlux(), surface_numericalflux=RoeFlux(), auxstate=dg.auxstate)

cfl = 1.8 # FT(15 // 8) # for lsrk14, roughly a cfl of 0.125 per stage

c = 330.0 # [m/s]
dt = cfl * min_node_distance(grid) / c * 1.0
println(" the dt is ", dt)
timeend = 30 #

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
odesolver_fsdg = LSRK144(fsdg, q, dt)
odesolver_new = LSRK144(new_dg, q, dt)

tic = Base.time()
solve!(q, timeend, odesolver_dg) # ; after_step=do_output)
toc = Base.time()
println("The time for the original simulation is ", toc - tic)

tic = Base.time()
solve!(q, timeend, odesolver_fsdg) # ; after_step=do_output)
toc = Base.time()
println("The time for the modified simulation is ", toc - tic)

tic = Base.time()
solve!(q, timeend, odesolver_new) # ; after_step=do_output)
toc = Base.time()
println("The time for the new simulation is ", toc - tic)

println("running again ")
odesolver_dg = LSRK144(dg, q, dt)
odesolver_fsdg = LSRK144(fsdg, q, dt)
odesolver_new = LSRK144(new_dg, q, dt)

tic = Base.time()
solve!(q, timeend, odesolver_dg) # ; after_step=do_output)
toc = Base.time()
println("The time for the original simulation is ", toc - tic)

tic = Base.time()
solve!(q, timeend, odesolver_fsdg) # ; after_step=do_output)
toc = Base.time()
println("The time for the modified simulation is ", toc - tic)

tic = Base.time()
solve!(q, timeend, odesolver_new) # ; after_step=do_output)
toc = Base.time()
println("The time for the new simulation is ", toc - tic)


println(q[1])


##
# Test Volume Flux 
println("Testing the volume flux")
bl = EulerTotalEnergyLaw{FT,3}()
nbl = DryTotalEnergyLaw{FT,3}()

f = zeros(3, 5)
g = copy(f)
q1 = randn(5)
q2 = randn(5)
aux1 = randn(5)
aux2 = randn(5)

q⁺ = q[1]
q⁻ = q[2]

modified_volumeflux!(Atum.KennedyGruberFlux(), nbl, f, q⁻, aux1, q⁺, aux2)
f2 = Atum.twopointflux(Atum.KennedyGruberFlux(), bl, q⁻, aux1, q⁺, aux2)
f - f2

pressure(nbl, q1[1], SVector(q1[2:4]...), q1[end], aux1[end])
Atum.EulerTotalEnergy.pressure(bl, q1[1], SVector(q1[2:4]...), q1[end], aux1[end])
Atum.EulerTotalEnergy.geopotential(bl, aux1)
geopotential(nbl, aux1)

numflux = Atum.KennedyGruberFlux()
volume_newtime = @benchmark begin
    modified_volumeflux!(numflux, nbl, f, q⁻, aux1, q⁺, aux2)
end

volume_oldtime = @benchmark begin
    f2 = Atum.twopointflux(numflux, bl, q⁻, aux1, q⁺, aux2)
end

##
# Test Surface Flux 
println("testing the surface flux")
f = zeros(5)
g = copy(f)
q1 = randn(5)
q2 = randn(5)
aux1 = randn(5)
aux2 = randn(5)
n⃗ = SVector(0.0, 0.0, 0.0)
q⁺ = q[1]
q⁻ = q[2]
aux⁺ = randn(5)
aux⁻ = randn(5)

numerical_surfaceflux = Atum.RoeFlux()

modified_surfaceflux!(numerical_surfaceflux, nbl, f, n⃗, q⁻, aux⁻, q⁺, aux⁺)
f2 = Atum.surfaceflux(numerical_surfaceflux, bl, n⃗, q⁻, aux⁻, q⁺, aux⁺)
f - f2

surface_oldtime = @benchmark begin
    f2 = Atum.surfaceflux(numerical_surfaceflux, bl, n⃗, q⁻, aux⁻, q⁺, aux⁺)
end

surface_newtime = @benchmark begin
    modified_surfaceflux!(numerical_surfaceflux, nbl, f, n⃗, q⁻, aux⁻, q⁺, aux⁺)
end

## 
println("Testing the rhs operator")
# Test Operator 
q1 = fieldarray(undef, law, grid);
q1 .= initial_condition.(Ref(law), points(grid));

q2 = fieldarray(undef, law, grid);
q2 .= initial_condition.(Ref(law), points(grid));

q3 = fieldarray(undef, law, grid);
q3 .= initial_condition.(Ref(law), points(grid));

q4 = fieldarray(undef, law, grid);
q4 .= initial_condition.(Ref(law), points(grid));



volume_form = FluxDifferencingForm(EntropyConservativeFlux())
dg = DGSEM(; law, grid, volume_form=FluxDifferencingForm(KennedyGruberFlux()), surface_numericalflux=RoeFlux())
fsdg = FluxSource(; law=EulerTotalEnergyLaw{FT,3}(), grid, volume_form=KennedyGruberFlux(), surface_numericalflux=RoeFlux(), auxstate=dg.auxstate)
new_dg = ModifiedFluxDifferencing(; law=DryTotalEnergyLaw{FT,3}(), grid, volume_form=KennedyGruberFlux(), surface_numericalflux=RoeFlux(), auxstate=dg.auxstate)

dg(q2, q1, 0.0, increment=false)
fsdg(q3, q1, 0.0, increment=false)
new_dg(q4, q1, 0.0, increment=false)

for i in 1:5
    ndif = norm(parent(components(q3 - q4)[i]), Inf)
    println("The error is ", ndif)
end

oldtime = @benchmark CUDA.@sync begin
    dg(q1, q2, 0.0, increment=false)

end

flux_sourcetime = @benchmark CUDA.@sync begin
    fsdg(q1, q2, 0.0, increment=false)
end

newtime = @benchmark CUDA.@sync begin
    new_dg(q1, q2, 0.0, increment=false)
end

##
# Test Timestep
println("Testing the timestepping")
dq = fieldarray(undef, law, grid)
dq .= initial_condition.(Ref(law), points(grid))
using UnPack
odesolver = LSRK144(new_dg, q, dt)
fs_odesolver = LSRK144(fsdg, q, dt)
old_odesolver = LSRK144(dg, q, dt)
after_stage(args...) = nothing
dosteptime = @benchmark CUDA.@sync Atum.dostep!(q, odesolver, after_stage)
fsdosteptime = @benchmark CUDA.@sync Atum.dostep!(q, fs_odesolver, after_stage)
olddosteptime = @benchmark CUDA.@sync Atum.dostep!(q, old_odesolver, after_stage)

rhs! = old_odesolver.rhs!
dq = old_odesolver.dq
rka = old_odesolver.rka
rkb = old_odesolver.rkb
rkc = old_odesolver.rkc
dt = old_odesolver.dt

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
