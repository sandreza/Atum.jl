using Atum, Atum.EulerGravity
using Random, StaticArrays
using StaticArrays: SVector, MVector
using Statistics, Revise, CUDA, ProgressBars
using GLMakie

import Atum: boundarystate, source!
Random.seed!(12345)
# for lazyness 
const parameters = (
    R=287,
    pₒ=1e5, # get_planet_parameter(:MSLP),
    g=9.81,
    cp=287 / (1 - 1 / 1.4),
    γ=1.4,
    cv=287 / (1 - 1 / 1.4) - 287.0,
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

    Q₀ = 100.0 /4  # convective_forcing.Q₀
    r_ℓ = 100.0  # convective_forcing.r_ℓ
    s_ℓ = 100.0  # convective_forcing.s_ℓ
    λ = 1 / 10.0 # convective_forcing.λ
    L = 3e3      # convective_forcing.L

    radiation_profile = Q₀ / r_ℓ * exp(-z / r_ℓ)

    damping_profile = -exp(-(L - z) / s_ℓ)

    # Apply convective forcing
    # source[2] += λ * damping_profile * ρu⃗[1]
    # source[3] += λ * damping_profile * ρu⃗[2]
    # source[4] += λ * damping_profile * ρu⃗[3]
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
# A = Array
FT = Float64 # N = 3, K = 50 looks nice
N = 3

K = 8
vf = FluxDifferencingForm(KennedyGruberFlux())
println("DOFs = ", (N + 1) * K, " with VF ", vf)

volume_form = vf

outputvtk = false
Nq = N + 1

println("constructing grid")
law = EulerGravityLaw{FT,3}()
# pp = 2
cell = LobattoCell{FT,A}(Nq, Nq, Nq)
cpu_cell = LobattoCell{FT,Array}(Nq, Nq, Nq)
v1d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v2d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v3d = range(FT(0), stop=FT(3.0e3), length=K + 1)
grid = brickgrid(cell, (v1d, v2d, v3d); periodic=(true, true, false))
cpu_grid = brickgrid(cpu_cell, (v1d, v2d, v3d); periodic=(true, true, false))
x⃗ = points(grid)
println("constructing rhs")
dg = DGSEM(; law, grid, volume_form, surface_numericalflux=RoeFlux())

cfl = FT(14 // 8) # for lsrk14, roughly a cfl of 0.125 per stage

c = 330.0 # [m/s]
dt = cfl * min_node_distance(grid) / c * 1.0
println(" the dt is ", dt)
timeend = 1.5 * 60 * 60

q = fieldarray(undef, law, grid)
qc = fieldarray(undef, law, cpu_grid)
q0 = fieldarray(undef, law, grid)
q .= initial_condition.(Ref(law), points(grid))
q0 .= initial_condition.(Ref(law), points(grid))
qq = initial_condition.(Ref(law), points(grid))
dqq = initial_condition.(Ref(law), points(grid))
println("initial conditions")

# needs to be constructed after the grid is constructed
include("convection_interpolate_function.jl")



odesolver = LSRK144(dg, q, dt)

timeend = 90 * 60
timeend = 1 * 60

thavg_timeseries = Vector{Float64}[]
avgwth_timeseries = Vector{Float64}[]
thavg, wavg, ththavg, avgwth = compute_field_averages(newgrid, ξlist, elist, r, ω, q, grid)
push!(thavg_timeseries, thavg)
push!(avgwth_timeseries, avgwth)

for i in ProgressBar(1:4 * 90)
    solve!(q, i * timeend, odesolver)
    thavg, wavg, ththavg, avgwth = compute_field_averages(newgrid, ξlist, elist, r, ω, q, grid)
    push!(thavg_timeseries, thavg)
    push!(avgwth_timeseries, avgwth)
end

begin
    fig = Figure()
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])
    time_slider = Slider(fig[2, 1:2], range=2:4*90, startvalue=2, horizontal=true)
    time_index = time_slider.value
    field = @lift(thavg_timeseries[$time_index])
    field2 = @lift($field[2:end] - $field[1:end-1])
    # field2 = @lift(avgwth_timeseries[$time_index])
    scatter!(ax1, field, zlist)
    scatter!(ax2, field2, zlist[2:end])
    xlims!(ax2, (-0.2, 0.2))
    display(fig)
end

##
maxind_t = [argmax(thavg_timeseries[2:end] - thavg_timeseries[1:end-1]) for thavg_timeseries in thavg_timeseries] # maximum slope for entrainment depth
heat_in = (mean(thavg_timeseries[end]) - mean(thavg_timeseries[1]) ) * parameters.zmax / (4 * 90 * 60) # not 25 / 717.5
tlist = collect(range(0, 4 * 90 * 60, length=4 * 90 + 1))
entrainment_layer_depth = sqrt.(3 * heat_in / parameters.Δθ * parameters.zmax * tlist )

begin
mldepth_fig = Figure()
ax1 = Axis(mldepth_fig[1, 1]; xlabel = "time in minutes", ylabel = "entrainment depth in m")
ln = lines!(ax1, tlist  ./ 60, entrainment_layer_depth, color = :red, label = "Theoretical")
sc = scatter!(ax1, tlist ./ 60, zlist[maxind_t], color = :blue, label = "Numerical")
axislegend(ax1, position  = :rt)
display(mldepth_fig)
end
##