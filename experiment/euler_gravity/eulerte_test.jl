using Atum
using Atum.EulerTotalEnergy
using Random
using StaticArrays
using StaticArrays: SVector, MVector
using WriteVTK
using Statistics
using BenchmarkTools
using Revise
using CUDA
using LinearAlgebra

import Atum: boundarystate, source!
Random.seed!(12345)
# for lazyness 
const parameters = (
    R=287,
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

K = 8*2
vf = (KennedyGruberFlux(), KennedyGruberFlux(), KennedyGruberFlux())
surface_numericalflux = (RoeFlux(), RoeFlux(), RoeFlux())
l_vf = Atum.LinearizedKennedyGruberFlux()
println("DOFs = ", (N + 1) * K, " with VF ", vf)

volume_form = vf

outputvtk = false
Nq = N + 1

law = EulerTotalEnergyLaw{FT,3}()
# pp = 2
cell = LobattoCell{FT,A}(Nq, Nq, Nq)
v1d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v2d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v3d = range(FT(0), stop=FT(3.0e3), length=K + 1)
grid = brickgrid(cell, (v1d, v2d, v3d); periodic=(true, true, false),
                   ordering = StackedOrdering{CartesianOrdering}())
x⃗ = points(grid)
dg = DGSEM(; law, grid, volume_form, surface_numericalflux=RoeFlux())
fsdg = FluxSource(; law, grid, volume_form, surface_numericalflux=RoeFlux())
dg_r = DGSEM(; law, grid, volume_form, surface_numericalflux=RusanovFlux())
dg_sd = SingleDirection(; law, grid, volume_form = l_vf, surface_numericalflux = Atum.LinearizedRefanovFlux())
fsdg2 = FluxSource(; law, grid, volume_form, surface_numericalflux= surface_numericalflux)
cfl = FT(15 // 8) # for lsrk14, roughly a cfl of 0.125 per stage
# cfl = 1.75
c = 330.0 # [m/s]
dt = cfl * min_node_distance(grid) / c * 1.0
println(" the dt is ", dt)
timeend = 60 * 6

Random.seed!(1234)
q = fieldarray(undef, law, grid)
q0 = fieldarray(undef, law, grid)
q .= initial_condition.(Ref(law), points(grid))
q0 .= initial_condition.(Ref(law), points(grid))
qq = initial_condition.(Ref(law), points(grid))
dqq = initial_condition.(Ref(law), points(grid))

aux = Atum.auxiliary.(Ref(law), x⃗, q)
fsdg.auxstate .= aux
fsdg2.auxstate .= aux
dg_sd.auxstate .= aux

fsdg(q0, q, 0.0)

odesolver = LSRK144(fsdg, q, dt)

do_output = function (step, time, q)
    if step % ceil(Int, timeend / 100 / dt) == 0
        println("simulation is ", time / timeend * 100, " percent complete")
    end
end

tic = time()
solve!(q, timeend, odesolver; after_step=do_output)
toc = time()
println("The time for the simulation is ", toc - tic)
println(q[1])
fsdg(q0,q, 0.0)

#=
for α ∈ [-8.0, 0.0, 2.0, 3.5, π, 100.0]
    local αq = q .* α
    local Lαq = q .* α
    local v = q .* 1.0
    local Lv = q .* 1.0
    local αqpv =  αq  .+ v
    local Lαqpv =  αq  .+ v

    dg_sd(Lαq, αq , 0.0, increment = false)
    dg_sd(Lv, v , 0.0, increment = false)
    dg_sd(Lαqpv, αqpv , 0.0, increment = false)

    println(" The error is ", norm(Lαqpv .- (Lαq .+ Lv), Inf))
end
=#


#=
aux = Atum.auxiliary.(Ref(law), x⃗, q)
fsdg.auxstate .= aux
dg_sd.auxstate .= aux

function tnomatvec!(dq, q, event)
    wait(event)
    dg_sd(dq, q, 0.0; increment=false) # make sure using same function here as later
    return Atum.KernelAbstractions.Event(Atum.getdevice(dg))
end

device = Atum.getdevice(dg)
comp_stream = Atum.KernelAbstractions.Event(device)

tnomatvec!(q0, q, comp_stream)

mat = Bennu.batchedbandedmatrix(tnomatvec!, dg.grid, q0, q, 1, 256) # default 1024 too much

α = 2.0
αq = q .* α
Lαq = q .* α
v = q .* 1.0
Lv = q .* 1.0
αqpv =  αq  .+ v
Lαqpv =  αq  .+ v

q .= initial_condition.(Ref(law), points(grid))
arrayq = Bennu.get_batched_array(q, Nq^2, K^2)
q0 = fieldarray(undef, law, grid)
q0 .= 0 * q0
arrayq0 = Bennu.get_batched_array(q0, Nq^2, K^2)
mul!(arrayq0, mat, arrayq)
q1 = 0.0 * q
dg_sd(q1, q, 0.0; increment=false)
norm(q1 - q0, Inf)

mid = cld(size(mat.data, 2), 2)
imrka = 0.5
mat.data .*= -dt * imrka 
mat.data[:, mid, :, :] .+= 1

testq = fieldarray(undef, law, grid)
dg_sd(q1, q, 0.0; increment=false)
mul!(arrayq0, mat, arrayq)
testq .= q - dt * imrka * q1
println("the error in applying the operator ", norm(testq - q0, Inf))

# The linear operator is  q0 = q - dt * imrka * dg(q0, q, 0.0)
lumat = batchedbandedlu!(mat.data) # does indeed mutate the matrix

ldiv!(q0, lumat, testq)


println("The absolute error in solving the linear system ", norm(q - q0, Inf))
=#