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

include(pwd() * "/experiment/euler_gravity/sphere_utils.jl")
include(pwd() * "/experiment/euler_gravity/interpolate.jl")
include(pwd() * "/experiment/euler_gravity/sphere_statistics_functions.jl")

const X = 80.0 # 20.0; # small planet parameter # X = 40 is interesting

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



FT = Float64
A = Array
dim = 3

Nq⃗ = (7, 7, 7)
Kv = 4 # 10     # 4
Kh = 6 # 12 * 2 # 18 * 2

law = EulerTotalEnergyLaw{FT,dim}()
cell = LobattoCell{FT,A}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
vert_coord = range(FT(hs_p.a), stop=FT(hs_p.a + hs_p.H), length=Kv + 1)
grid = cubedspheregrid(cell, vert_coord, Kh)
x⃗ = points(grid)
function geopotential(hs_p, x⃗)
    r = sqrt(x⃗' * x⃗)
    r̂ = x⃗ ./ r
    Φ = (2 * hs_p.gravc * hs_p.mearth / hs_p.a - hs_p.gravc * hs_p.mearth / r) / X^2
    return Φ
end
Φ = geopotential.(Ref(hs_p), x⃗)
##
# interpolate fields 
println("precomputing interpolation data")
scale = 1
rlist = range(vert_coord[1] + 000, vert_coord[end], length=3 * scale)
θlist = range(-π, π, length=90 * scale)
ϕlist = range(0, π, length=45 * scale)

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

##
# load in files 
hfile = h5open("markov_model_even_time_nstate_100.h5")
markov_states = typeof(read(hfile["markov state 1"]))[]
for i in 1:100
    push!(markov_states, read(hfile["markov state $i"]))
end
close(hfile)

##
function mean_variables(state, x⃗, Φ)
    ρ = state[1]
    ρu⃗ = SVector(state[2], state[3], state[4])
    ρe = state[5]
    x = x⃗[1]
    y = x⃗[2]
    z = x⃗[3]
    # spherical vectors
    r⃗ = SVector(x, y, z)
    ϕ⃗ = SVector(x * z, y * z, -(x^2 + y^2))
    λ⃗ = SVector(-y, x, 0)
    # normalize (using nested functions gives error)
    r⃗_norm = sqrt(r⃗' * r⃗)
    r⃗_norm = r⃗_norm ≈ 0.0 ? 1.0 : r⃗_norm
    ϕ⃗_norm = sqrt(ϕ⃗' * ϕ⃗)
    ϕ⃗_norm = ϕ⃗_norm ≈ 0.0 ? 1.0 : ϕ⃗_norm
    λ⃗_norm = sqrt(λ⃗' * λ⃗)
    λ⃗_norm = λ⃗_norm ≈ 0.0 ? 1.0 : λ⃗_norm
    u⃗ = ρu⃗ / ρ
    u = (λ⃗' * u⃗) / λ⃗_norm
    v = (ϕ⃗' * u⃗) / ϕ⃗_norm
    w = (r⃗' * u⃗) / r⃗_norm
    γ = 1.4
    p = (γ - 1) * (ρe - 0.5 * ρ * u⃗' * u⃗ - ρ * Φ)
    T = p / (ρ * 287)

    return [u, v, w, p, T]
end
##
state = markov_states[1]
u, v, w, p, T = mean_variables(state[1, 1, :], x⃗[1, 1], Φ[1, 1])
##
sphere_state = zeros(size(state))
for i in 1:size(state)[1], j in 1:size(state)[2]
    sphere_state[i, j, :] .= mean_variables(state[i, j, :], x⃗[i, j], Φ[i, j])
end
interpolated_markov_state = zeros(size(xlist)..., 5)
##
hfile2 = h5open("viz_fields_100.h5", "w")
for (i, state) in ProgressBar(enumerate(markov_states))
    for i in 1:size(state)[1], j in 1:size(state)[2]
        sphere_state[i, j, :] .= mean_variables(state[i, j, :], x⃗[i, j], Φ[i, j])
    end
    for s in 1:5
        oldf = view(sphere_state, :, :, s)
        newf = view(interpolated_markov_state, :, :, :, s)
        interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CPU())
    end
    hfile2["zonal mean zonal wind $i"] = mean(interpolated_markov_state[:, :, :, 1], dims=1)[:, :, :]
    hfile2["pressure $i"] = mean(interpolated_markov_state[:, :, :, 4], dims=1)[:, :, :]
    hfile2["surface field $i"] = interpolated_markov_state[:, :, 1, :]
end
hfile2["rlist"] = collect(rlist)
hfile2["thetalist"] = collect(θlist)
hfile2["philist"] = collect(ϕlist)
close(hfile2)
