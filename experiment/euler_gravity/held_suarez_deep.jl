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

import Atum: boundarystate, source!

statistic_save = true

include("sphere_utils.jl")
include("sphere_statistics_functions.jl")

hs_p = (
    a=6378e3,
    Ω=2π / 86400,
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
# We take our initial condition to be an isothermal atmosphere at rest 
# this allows for a gentle and stable start to the simulation where we can 
# test things like hydrostatic balance

# intial condition for velocity
uˡᵒⁿ(hs_p, λ, ϕ, r) = 0.0
uˡᵃᵗ(hs_p, λ, ϕ, r) = 0.0
uʳᵃᵈ(hs_p, λ, ϕ, r) = 0.0

# the reference value for the geopotential doesn't matter, but is here so that 
# it can relate to the shallow geopoential g * r
geo(hs_p, r) = 2 * hs_p.gravc * hs_p.mearth / hs_p.a - hs_p.gravc * hs_p.mearth / r

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

N = 3
Nq = N + 1
Nq⃗ = (Nq, Nq, Nq)
dim = 3

FT = Float64
A = CuArray

Kv = 7 # 12 
Kh = 15 # 12 
# N = 4, Kv = 12, Kh = 12 for paper resolution

law = EulerTotalEnergyLaw{FT,dim}()
cell = LobattoCell{FT,A}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
cpu_cell = LobattoCell{FT,Array}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
vert_coord = range(FT(hs_p.a), stop=FT(hs_p.a + hs_p.H), length=Kv + 1)
# vert_coord = FT(hs_p.a) .+ [0, 2e3, 5e3, 1e4, 1.8e4, 3e4]
vert_coord = FT(hs_p.a) .+ [0, 1.75e3, 4.5e3, 7.25e3, 1.1e4, 1.5e4, 2e4, 3e4]
# vert_coord = FT(hs_p.a) .+ [0, 2e3, 5e3, 1e4, 1.5e4, 2.25e4, 3e4]

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
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

function source!(law::EulerTotalEnergyLaw, source, state, aux, dim, directions)
    # Extract the state
    ρ, ρu, ρe = EulerTotalEnergy.unpackstate(law, state)
    Φ = EulerTotalEnergy.geopotential(law, aux)
    # _, source_ρu⃗, _ = EulerTotalEnergy.unpackstate(law, source)
    FT = Float64

    # First Coriolis 
    Ω = @SVector [-0, -0, 2π / 86400]
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

    pₛ = 1e5
    R_d = 287
    cp_d = 287 / (1 - 1 / 1.4)
    cv_d = 287 / (1 - 1 / 1.4) - 287.0

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
    source_ρu = -k_v * P * ρu

    # source_ρu = -k_v * ρu # damping everything is consistent with hydrostatic balance
    source_ρe = -k_T * ρ * cv_d * (T - T_equil)
    source_ρe += (ρu' * source_ρu) / ρ

    source[2] = coriolis[1] + source_ρu[1]
    source[3] = coriolis[2] + source_ρu[2]
    source[4] = coriolis[3] + source_ρu[3]
    source[5] = source_ρe

    return nothing
end


vf = (KennedyGruberFlux(), KennedyGruberFlux(), KennedyGruberFlux())
sf = (RoeFlux(), RoeFlux(), Atum.RefanovFlux(1.0))

linearized_vf = Atum.LinearizedKennedyGruberFlux()
linearized_sf = Atum.LinearizedRefanovFlux(1.0)

dg_sd = SingleDirection(; law, grid, volume_form=linearized_vf, surface_numericalflux=linearized_sf)
dg_fs = FluxSource(; law, grid, volume_form=vf, surface_numericalflux=sf)

vcfl = Inf
hcfl = 0.35 # for polynomial order 4 hcfl = 0.15, 3 hcfl = 0.25
Δx = min_node_distance(grid, dims=1)
Δy = min_node_distance(grid, dims=2)
Δz = min_node_distance(grid, dims=3)
vdt = vcfl * Δz / c_max
hdt = hcfl * Δx / c_max
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
endday = 30.0 * 40
tmp_ρ = components(test_state)[1]
ρ̅_start = sum(tmp_ρ .* dg_fs.MJ) / sum(dg_fs.MJ)


fmvar = mean_variables.(Ref(law), state, aux)
smvar = second_moment_variables.(fmvar)
fmvartmp = mean_variables.(Ref(law), state, aux)

fmvar .*= 0.0
smvar .*= 0.0

##
display_skip = 50
tic = time()
partitions = 1:endday*36*2 # 24*3  # 1:24*endday*3 for updating every 20 minutes

current_time = 0.0 # 
save_partition = 1
save_time = 0.0
averaging_counter = 0.0
statistic_counter = 0.0

state .= test_state 

for i in partitions
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_fs.auxstate .= aux
    dg_sd.auxstate .= aux
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
    end_time = 60 * 60 * 24 * endday / partitions[end]
    state .= test_state
    solve!(test_state, end_time, odesolver, adjust_final=false) # otherwise last step is wrong since linear solver isn't updated
    # midpoint type extrapolation
    α = 1.5
    state .= α * (test_state) + (1 - α) * state

    timeend = odesolver.time
    global current_time += timeend

    if statistic_save & (current_time / 86400 > 200) & (i % 15 == 0)
        println("gathering statistics at time ", current_time / 86400)
        global statistic_counter += 1.0
        global fmvartmp .= mean_variables.(Ref(law), test_state, aux)
        global smvar .+= second_moment_variables.(fmvartmp)
        global fmvar .+= mean_variables.(Ref(law), test_state, aux)
    end

    if i % display_skip == 0
        println("--------")
        println("done with ", display_skip * timeend / 60, " minutes")
        println("partition ", i, " out of ", partitions[end])
        local ρ, ρu, ρv, ρw, ρet = components(test_state)
        u = ρu ./ ρ
        v = ρv ./ ρ
        w = ρw ./ ρ
        println("maximum x-velocity ", maximum(u))
        println("maximum y-velocity ", maximum(v))
        println("maximum z-velocity ", maximum(w))
        uʳ = @. (xp * u + yp * v + zp * w) / sqrt(xp^2 + yp^2 + zp^2)
        minuʳ = minimum(uʳ)
        maxuʳ = maximum(uʳ)
        println("extrema vertical velocity ", (minuʳ, maxuʳ))
        hs_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), test_state, aux)
        hs_density = components(test_state)[1]
        hs_soundspeed = Atum.EulerTotalEnergy.soundspeed.(Ref(law), hs_density, hs_pressure)
        speed = @. sqrt(u^2 + v^2 + w^2)
        c_max = maximum(hs_soundspeed)
        mach_number = maximum(speed ./ hs_soundspeed)
        println("The maximum soundspeed is ", c_max)
        println("The largest mach number is ", mach_number)
        println(" the vertical cfl is ", dt * c_max / Δz)
        println(" the horizontal cfl is ", dt * c_max / Δx)
        println("The dt is now ", dt)
        println("The current day is ", current_time / 86400)
        ρ̅ = sum(ρ .* dg_fs.MJ) / sum(dg_fs.MJ)
        println("The average density of the system is ", ρ̅)
        toc = time()
        println("The runtime for the simulation is ", (toc - tic) / 60, " minutes")

        if isnan(ρ[1]) | isnan(ρu[1]) | isnan(ρv[1]) | isnan(ρw[1]) | isnan(ρet[1]) | isnan(ρ̅)
            println("The simulation NaNed, decreasing timestep and using stable state")
            local i = save_partition
            global current_time = save_time
            test_state .= stable_state
            state .= stable_state
            global dt *= 0.9
        else
            if (abs(minuʳ) + abs(maxuʳ)) < 2.0
                println("creating backup state")
                stable_state .= test_state
                global save_partition = i
                global save_time = current_time
            end
        end
        println("-----")
    end
end
##
toc = time()
println("The time for the simulation is ", toc - tic)
tmp_ρ = components(test_state)[1]
ρ̅_end = sum(tmp_ρ .* dg_fs.MJ) / sum(dg_fs.MJ)

# normalize statistics
fmvar .*= 1 / statistic_counter
smvar .*= 1 / statistic_counter

state .*= 1 / averaging_counter

println("The conservation of mass error is ", (ρ̅_start - ρ̅_end) / ρ̅_end)

gpu_components = components(test_state)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end
statenames = ("ρ", "ρu", "ρv", "ρw", "ρe")

filepath = "HeldSuarezDeepStretched_" * "Nev" * string(Kv) * "_Neh" * string(Kh) * "_Nq" * string(Nq⃗[1]) * ".jld2"
file = jldopen(filepath, "a+")
JLD2.Group(file, "state")
JLD2.Group(file, "averagedstate")
JLD2.Group(file, "grid")
for (i, statename) in enumerate(statenames)
    file["state"][statename] = cpu_components[i]
end

gpu_components = components(state)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end

for (i, statename) in enumerate(statenames)
    file["averagedstate"][statename] = cpu_components[i]
end

file["grid"]["vertical_coordinate"] = vert_coord
file["grid"]["gauss_lobatto_points"] = Nq⃗
file["grid"]["vertical_element_number"] = Kv
file["grid"]["horizontal_element_number"] = Kh

close(file)