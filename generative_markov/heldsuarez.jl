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
using JLD2
using BenchmarkTools
using ProgressBars
using HDF5

import Atum: boundarystate, source!

spinup = false
grab_states = false
compute_embedding = true

include(pwd() * "/experiment/euler_gravity/sphere_utils.jl")
include(pwd() * "/experiment/euler_gravity/interpolate.jl")
include(pwd() * "/experiment/euler_gravity/sphere_statistics_functions.jl")

# CURRENTLY DAMPING EVERYTHING
const X = 80.0 # 20.0; # small planet parmaeter # X = 40 is interesting, X = 80 is stil good
FT = Float64

hs_p = (
    a=6378e3 / X,
    Ω=2π / 86400 * X,
    g=9.81,
    R_d=287.0,
    γ=1.4,
    pₛ=1e5,
    cp_d=287 / (1 - 1 / 1.4),
    cv_d=287 / (1 - 1 / 1.4) - 287.0,
    H=30e3,
    Tₛ=285,
    ρₛ=1e5 / (287 * 285),
    gravc=6.67408e-11,
    mearth=5.9722e24,
)
# hs_p = Tuple([FT(hs) for hs in hs_p])
# We take our initial condition to be an isothermal atmosphere at rest 
# this allows for a gentle and stable start to the simulation where we can 
# test things like hydrostatic balance

# intial condition for velocity
uˡᵒⁿ(hs_p, λ, ϕ, r) = 0.0
uˡᵃᵗ(hs_p, λ, ϕ, r) = 0.0
uʳᵃᵈ(hs_p, λ, ϕ, r) = 0.0

# the reference value for the geopotential doesn't matter, but is here so that 
# it can relate to the shallow geopoential g * r
# we want the linearization to still be 9.81 * r
#  (r - a + a)⁻¹ = 1/a - (r-a)/a²
# 2/a - (r - a + a)⁻¹ = r / a^2
geo(hs_p, r) = (2 * hs_p.gravc * hs_p.mearth / hs_p.a - hs_p.gravc * hs_p.mearth / r) / X^2
# rhoish(hs_p, λ, ϕ, r) = hs_p.ρₛ * exp(-(10 * r - 10 * hs_p.a)) / (hs_p.R_d * hs_p.Tₛ))
ρ₀(hs_p, λ, ϕ, r) = hs_p.ρₛ * exp(-(geo(hs_p, r) - geo(hs_p, hs_p.a)) / (hs_p.R_d * hs_p.Tₛ))
ρuˡᵒⁿ(hs_p, λ, ϕ, r) = ρ₀(hs_p, λ, ϕ, r) * uˡᵒⁿ(hs_p, λ, ϕ, r)
ρuˡᵃᵗ(hs_p, λ, ϕ, r) = ρ₀(hs_p, λ, ϕ, r) * uˡᵃᵗ(hs_p, λ, ϕ, r)
ρuʳᵃᵈ(hs_p, λ, ϕ, r) = ρ₀(hs_p, λ, ϕ, r) * uʳᵃᵈ(hs_p, λ, ϕ, r)

pressure_init(hs_p, λ, ϕ, r) = ρ₀(hs_p, λ, ϕ, r) * hs_p.R_d * hs_p.Tₛ
e_int(hs_p, λ, ϕ, r) = hs_p.cv_d * pressure_init(hs_p, λ, ϕ, r) / (ρ₀(hs_p, λ, ϕ, r) * hs_p.R_d)
e_kin(hs_p, λ, ϕ, r) = 0.5 * (uˡᵒⁿ(hs_p, λ, ϕ, r)^2 + uˡᵃᵗ(hs_p, λ, ϕ, r)^2 + uʳᵃᵈ(hs_p, λ, ϕ, r)^2)
e_pot(hs_p, λ, ϕ, r) = geo(hs_p, r)

ρe(hs_p, λ, ϕ, r) = ρ₀(hs_p, λ, ϕ, r) * (e_int(hs_p, λ, ϕ, r) + e_kin(hs_p, λ, ϕ, r) + e_pot(hs_p, λ, ϕ, r))

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(hs_p, x...) = ρ₀(hs_p, lon(x...), lat(x...), rad(x...))
ρu₀ᶜᵃʳᵗ(hs_p, x...) = (ρuʳᵃᵈ(hs_p, lon(x...), lat(x...), rad(x...)) * r̂(x...)
                       + ρuˡᵃᵗ(hs_p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                       + ρuˡᵒⁿ(hs_p, lon(x...), lat(x...), rad(x...)) * λ̂(x...))
ρe₀ᶜᵃʳᵗ(hs_p, x...) = ρe(hs_p, lon(x...), lat(x...), rad(x...))

# correction auxstate for sphere
function sphere_auxiliary(law::EulerTotalEnergyLaw, hs_p, x⃗, state)
    ix_ρ, ix_ρu⃗, ix_ρe = Atum.EulerTotalEnergy.varsindices(law)
    ix_x, ix_y, ix_z, _, _, _, _ = Atum.EulerTotalEnergy.auxindices(law)
    r = sqrt(x⃗' * x⃗)
    r̂ = x⃗ ./ r
    tmp = (I - r̂ * (r̂')) * state[ix_ρu⃗] # get rid of rapidly fluctuating vertical component
    ϕ = geo(hs_p, r)
    @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], state[ix_ρ], tmp..., state[ix_ρe], ϕ)
end

N = 5
Nq = N + 1
Nq⃗ = (Nq, Nq, Nq)
dim = 3

FT = Float64
A = CuArray

Nq⃗ = (7, 7, 7)
Kv = 4 # 10     # 4
Kh = 6 # 12 * 2 # 18 * 2

law = EulerTotalEnergyLaw{FT,dim}()
cell = LobattoCell{FT,A}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
cpu_cell = LobattoCell{FT,Array}(Nq⃗[1], Nq⃗[2], Nq⃗[3])

vert_coord = range(FT(hs_p.a), stop=FT(hs_p.a + hs_p.H), length=Kv + 1)

grid = cubedspheregrid(cell, vert_coord, Kh)
x⃗ = points(grid)

cpu_grid = cubedspheregrid(cpu_cell, vert_coord, Kh)
cpu_x⃗ = points(cpu_grid)

function held_suarez_init(x⃗, param)
    x, y, z = x⃗
    hs_ρ = ρ₀(param, lon(x⃗...), lat(x⃗...), rad(x⃗...))
    hs_ρuᵣ = ρuʳᵃᵈ(param, lon(x⃗...), lat(x⃗...), rad(x⃗...)) * r̂ⁿᵒʳᵐ(x, y, z) * @SVector([x, y, z])
    hs_ρuₗ = ϕ̂ⁿᵒʳᵐ(x, y, z) * @SVector([x * z, y * z, -(x^2 + y^2)]) * ρuˡᵃᵗ(param, lon(x⃗...), lat(x⃗...), rad(x⃗...))
    hs_ρuₗₗ = ρuˡᵒⁿ(param, lon(x⃗...), lat(x⃗...), rad(x⃗...)) * λ̂(x⃗...)
    hs_ρu⃗ = hs_ρuᵣ + hs_ρuₗ + hs_ρuₗₗ
    hs_ρe = ρe(param, lon(x⃗...), lat(x⃗...), rad(x⃗...))
    SVector(hs_ρ, hs_ρu⃗..., hs_ρe)
end

state = fieldarray(undef, law, grid)
test_state = fieldarray(undef, law, grid)
stable_state = fieldarray(undef, law, grid)
old_state = fieldarray(undef, law, grid)
cpu_state = fieldarray(undef, law, cpu_grid)
cpu_state .= held_suarez_init.(cpu_x⃗, Ref(hs_p))
gpu_components = components(state)
cpu_components = components(cpu_state)
for i in eachindex(gpu_components)
    gpu_components[i] .= A(cpu_components[i])
end

# state .= held_suarez_init.(x⃗, Ref(hs_p)) # this line gives the error
test_state .= state
old_state .= state
aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
hs_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), state, aux)
hs_density = components(state)[1]
hs_soundspeed = Atum.EulerTotalEnergy.soundspeed.(Ref(law), hs_density, hs_pressure)
c_max = maximum(hs_soundspeed)

function boundarystate(law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρe⁻ = EulerTotalEnergy.unpackstate(law, q⁻)
    ρ⁺, ρe⁺ = ρ⁻, ρe⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗ # free-slip
    # ρu⃗⁺ = - ρu⃗⁻ # no-slip
    SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

function source!(law::EulerTotalEnergyLaw, source, state, aux, dim, directions)
    # Extract the state
    ρ, ρu, ρe = EulerTotalEnergy.unpackstate(law, state)
    Φ = EulerTotalEnergy.geopotential(law, aux)
    FT = Float64

    # First Coriolis 
    Ω = @SVector [-0, -0, 2π / 86400 * X]
    coriolis = -2Ω × ρu

    # Then Held-Suarez Forcing 
    day = 86400
    k_a = FT(1 / (40 * day))
    k_f = FT(1 / day)
    k_s = FT(1 / (4 * day))
    ΔT_y = FT(60)
    Δθ_z = FT(10)
    T_equator = FT(315)
    T_min = FT(200)
    σ_b = FT(7 / 10)

    pₛ = FT(1e5)
    R_d = FT(287)
    cp_d = FT(287 / (1 - 1 / 1.4))
    cv_d = FT(287 / (1 - 1 / 1.4) - 287.0)

    x = aux[1]
    y = aux[2]
    z = aux[3]
    coord = @SVector [x, y, z]

    # Held-Suarez forcing
    p = EulerTotalEnergy.pressure(law, ρ, ρu, ρe, Φ)
    T = p / (R_d * ρ)

    φ = @inbounds asin(coord[3] / norm(coord, 2))

    σ = p / pₛ
    exner_p = σ^(R_d / cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ) * sin(φ) - Δθ_z * log(σ) * cos(φ) * cos(φ)) * exner_p
    T_equil = max(T_min, T_equil)

    k_T = k_a + (k_s - k_a) * height_factor * cos(φ) * cos(φ) * cos(φ) * cos(φ)
    k_v = k_f * height_factor

    # horizontal projection option
    k = coord / norm(coord)
    P = I - k * k' # technically should project out pressure normal
    source_ρu = -X * k_v * P * ρu # - top_sponge * ρu

    source_ρe = -X * k_T * ρ * cv_d * (T - T_equil)

    Ω = @SVector [-0, -0, FT(2π / 86400 * X)]
    tmp = k' * Ω
    Ω = tmp * k
    coriolis = -2Ω × ρu

    source[2] = coriolis[1] + source_ρu[1]
    source[3] = coriolis[2] + source_ρu[2]
    source[4] = coriolis[3] + source_ρu[3]
    source[5] = source_ρe

    return nothing
end


vf = (KennedyGruberFlux(), KennedyGruberFlux(), KennedyGruberFlux())
sf_explicit = (RoeFlux(), RoeFlux(), RoeFlux())

dg_explicit = FluxSource(; law, grid, volume_form=vf, surface_numericalflux=sf_explicit)

vcfl = 2 * 1.8 # 0.25
hcfl = 1.8 # 0.25 # hcfl = 0.25 for a long run

Δx = min_node_distance(grid, dims=1)
Δy = min_node_distance(grid, dims=2)
Δz = min_node_distance(grid, dims=3)
vdt = vcfl * Δz / c_max
hdt = hcfl * Δx / c_max
ΔΩ = minimum([Δx, Δy, Δz])
dt = min(vdt, hdt)
println(" the dt is ", dt)
println(" the vertical cfl is ", dt * c_max / Δz)
println(" the horizontal cfl is ", dt * c_max / Δx)
println(" the minimum grid spacing in the horizontal is ", Δx)
println(" the minimum grid spacing in the vertical is ", Δz)

test_state .= old_state
stable_state .= old_state

aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, test_state)
xp = components(aux)[1]
yp = components(aux)[2]
zp = components(aux)[3]

# test_state .= state
endday = 30.0 * 40 / X
tmp_ρ = components(test_state)[1]
ρ̅_start = sum(tmp_ρ .* dg_explicit.MJ) / sum(dg_explicit.MJ)

fmvar = mean_variables.(Ref(law), state, aux)
fmvar .*= 0.0
smvar = second_moment_variables.(fmvar)

##
# interpolate fields 
println("precomputing interpolation data")
scale = 1
θlength = 32 
ϕlength = 32
rlist = range(vert_coord[1] + 000, vert_coord[end], length= 3 * scale)
θlist = range(-π, π, length = θlength)
ϕlist = range(0,  π, length = ϕlength)

xlist = [sphericaltocartesian(θ, ϕ, r) for θ in θlist, ϕ in ϕlist, r in rlist]

elist = zeros(Int, length(xlist))
ξlist = [SVector(0.0, 0.0, 0.0) for i in eachindex(xlist)]
for kk in eachindex(xlist)
    x = xlist[kk]
    (x̂, cell_coords) = Bennu.cubedspherereference(x, vert_coord, Kh)
    elist[kk] = cube_single_element_index(cell_coords, Kv, Kh)
    ξlist[kk] = SVector(x̂)
end
println("done precomputing interpolation data")
d_elist = A(elist)
d_ξlist = A(ξlist)
r = Tuple([cell.points_1d[i][:] for i in eachindex(cell.points_1d)])
ω = Tuple([A(baryweights(cell.points_1d[i][:])) for i in eachindex(cell.points_1d)])
newf = A(zeros(size(xlist)))

oldlist = components(test_state)

meanoldlist = components(fmvar)
secondoldlist = components(smvar)
meanlist = [A(zeros(size(xlist))) for i in 1:length(meanoldlist)]
secondlist = [A(zeros(size(xlist))) for i in 1:length(secondoldlist)]
gathermeanlist = copy(meanlist)
gathersecondlist = copy(secondlist)

##
tic = Base.time()
state .= test_state

if spinup == true
    partitions = 1:10
    for i in ProgressBar(partitions)
        aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
        dg_explicit.auxstate .= aux
        odesolver = LSRK144(dg_explicit, test_state, dt)
        end_time = 86400 * 365 / X / 100
        state .= test_state
        solve!(test_state, end_time, odesolver, adjust_final=false) 
    end
    hfile = h5open("starting_state.hdf5", "w")
    starting_state = Array{Float32}(components(test_state)[1].parent)
    hfile["starting_state"] = starting_state
    close(hfile)
else 
    hfile = h5open("starting_state.hdf5", "r")
    starting_state = read(hfile["starting_state"])
    close(hfile)
    for (i, component) in enumerate(components(test_state))
        component .= CuArray{Float64}(starting_state[:,i, :])
    end
end

#  grab markov states as one week decorrelated states
number_of_markov_states = 10;
gl_n, e_n = size(components(test_state)[1]);
markov_states = zeros(Float32, size(xlist)..., length(components(mean_variables.(Ref(law), test_state, aux))), number_of_markov_states);
if grab_states == true
    for i in ProgressBar(1:number_of_markov_states)
        aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
        dg_explicit.auxstate .= aux
        odesolver = LSRK144(dg_explicit, test_state, dt)
        end_time = 86400 * 7 / X
        state .= test_state
        solve!(test_state, end_time, odesolver, adjust_final=false) 

        fmvar .= mean_variables.(Ref(law), test_state, aux)
        for (j, component) in enumerate(components(fmvar))
            oldf = component
            interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
            markov_states[:, :, :, j, i] .= Array{Float32}(newf)
        end
    end
    normalization = [maximum(abs.(markov_states[:, :, :, i, :])) for i in 1:size(markov_states)[4]]
    rlist_v = Float32.(rlist)
    θlist_v = Float32.(θlist)
    ϕlist_v = Float32.(ϕlist)
    hfile = h5open("markov_state.hdf5", "w")
    hfile["markov_states"] = markov_states
    hfile["normalization"] = normalization
    hfile["r"] = rlist_v
    hfile["theta"] = θlist_v
    hfile["phi"] = ϕlist_v
    close(hfile)
end

hfile = h5open("markov_state.hdf5", "r")
temperature_states = read(hfile["markov_states"])[:, :, 1, end, :]
close(hfile)

steps = 60000
markov_indices = zeros(Int, steps)
labeled_states = zeros(eltype(temperature_states), size(temperature_states)[1:end-1]..., steps)
if compute_embedding == true
    for i in ProgressBar(1:steps)
        aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
        dg_explicit.auxstate .= aux
        odesolver = LSRK144(dg_explicit, test_state, dt)
        end_time = 86400 / X / 6
        state .= test_state
        solve!(test_state, end_time, odesolver, adjust_final=false) 

        fmvar .= mean_variables.(Ref(law), test_state, aux)
        oldf = components(fmvar)[end]
        interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
        newf_A = Array{Float32}(newf)[:,:,1]
        markov_index = argmin([norm(newf_A .- temperature_states[:, :, i]) for i in 1:size(temperature_states)[3]])
        markov_indices[i] = markov_index
        labeled_states[:, :, i] .= newf_A
    end
    hfile = h5open("embedding.hdf5", "w")
    hfile["states"] = labeled_states
    hfile["labels"] = markov_indices
    close(hfile)
end