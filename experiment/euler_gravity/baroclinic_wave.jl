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

include("sphere_utils.jl")

bw_p = (
    a=6378e3,
    Ω=2π / 86400,
    g=9.81,
    R_d=287,
    γ=1.4,
    pₛ=1e5,
    cv_d=1e3 / 1.4,
    cp_d=1e3,
    κ=287 / (1e3),
    T_0=0.0,
    H=30e3,
    k=3.0,
    Γ=0.005,
    T_E=310.0,
    T_P=240.0,
    b=2.0,
    z_t=15e3,
    λ_c=π / 9,
    ϕ_c=2 * π / 9,
    V_p=1.0,
)

# Initial Conditions for Baroclinic Wave 
T₀(bw_p) = 0.5 * (bw_p.T_E + bw_p.T_P)
bw_A(bw_p) = 1.0 / bw_p.Γ
bw_B(bw_p) = (T₀(bw_p) - bw_p.T_P) / T₀(bw_p) / bw_p.T_P
bw_C(bw_p) = 0.5 * (bw_p.k + 2) * (bw_p.T_E - bw_p.T_P) / bw_p.T_E / bw_p.T_P
bw_H(bw_p) = bw_p.R_d * T₀(bw_p) / bw_p.g
d_0(bw_p) = bw_p.a / 6

# convenience functions that only depend on height
τ_z_1(bw_p, r) = exp(bw_p.Γ * (r - bw_p.a) / T₀(bw_p))
τ_z_2(bw_p, r) = 1 - 2 * ((r - bw_p.a) / bw_p.b / bw_H(bw_p))^2
τ_z_3(bw_p, r) = exp(-((r - bw_p.a) / bw_p.b / bw_H(bw_p))^2)
τ_1(bw_p, r) = 1 / T₀(bw_p) * τ_z_1(bw_p, r) + bw_B(bw_p) * τ_z_2(bw_p, r) * τ_z_3(bw_p, r)
τ_2(bw_p, r) = bw_C(bw_p) * τ_z_2(bw_p, r) * τ_z_3(bw_p, r)
τ_int_1(bw_p, r) = bw_A(bw_p) * (τ_z_1(bw_p, r) - 1) + bw_B(bw_p) * (r - bw_p.a) * τ_z_3(bw_p, r)
τ_int_2(bw_p, r) = bw_C(bw_p) * (r - bw_p.a) * τ_z_3(bw_p, r)
F_z(bw_p, r) = (1 - 3 * ((r - bw_p.a) / bw_p.z_t)^2 + 2 * ((r - bw_p.a) / bw_p.z_t)^3) * ((r - bw_p.a) ≤ bw_p.z_t)

# convenience functions that only depend on longitude and latitude
d(bw_p, λ, ϕ) = bw_p.a * acos(sin(ϕ) * sin(bw_p.ϕ_c) + cos(ϕ) * cos(bw_p.ϕ_c) * cos(λ - bw_p.λ_c))
c3(bw_p, λ, ϕ) = cos(π * d(bw_p, λ, ϕ) / 2 / d_0(bw_p))^3
s1(bw_p, λ, ϕ) = sin(π * d(bw_p, λ, ϕ) / 2 / d_0(bw_p))
cond(bw_p, λ, ϕ) = (0 < d(bw_p, λ, ϕ) < d_0(bw_p)) * (d(bw_p, λ, ϕ) != bw_p.a * π)

# base-state thermodynamic variables
I_T(bw_p, ϕ, r) = (cos(ϕ) * r / bw_p.a)^bw_p.k - bw_p.k / (bw_p.k + 2) * (cos(ϕ) * r / bw_p.a)^(bw_p.k + 2)
T(bw_p, ϕ, r) = (τ_1(bw_p, r) - τ_2(bw_p, r) * I_T(bw_p, ϕ, r))^(-1) * (bw_p.a / r)^2
pressure_init(bw_p, ϕ, r) = bw_p.pₛ * exp(-bw_p.g / bw_p.R_d * (τ_int_1(bw_p, r) - τ_int_2(bw_p, r) * I_T(bw_p, ϕ, r)))
θ(bw_p, ϕ, r) = T(bw_p, ϕ, r) * (bw_p.pₛ / p(bw_p, ϕ, r))^bw_p.κ

# base-state velocity variables
U(bw_p, ϕ, r) = bw_p.g * bw_p.k / bw_p.a * τ_int_2(bw_p, r) * T(bw_p, ϕ, r) * ((cos(ϕ) * r / bw_p.a)^(bw_p.k - 1) - (cos(ϕ) * r / bw_p.a)^(bw_p.k + 1))
u(bw_p, ϕ, r) = -bw_p.Ω * r * cos(ϕ) + sqrt((bw_p.Ω * r * cos(ϕ))^2 + r * cos(ϕ) * U(bw_p, ϕ, r))
v(bw_p, ϕ, r) = 0.0
w(bw_p, ϕ, r) = 0.0

# velocity perturbations
δu(bw_p, λ, ϕ, r) = -16 * bw_p.V_p / 3 / sqrt(3) * F_z(bw_p, r) * c3(bw_p, λ, ϕ) * s1(bw_p, λ, ϕ) * (-sin(bw_p.ϕ_c) * cos(ϕ) + cos(bw_p.ϕ_c) * sin(ϕ) * cos(λ - bw_p.λ_c)) / sin(d(bw_p, λ, ϕ) / bw_p.a) * cond(bw_p, λ, ϕ)
δv(bw_p, λ, ϕ, r) = 16 * bw_p.V_p / 3 / sqrt(3) * F_z(bw_p, r) * c3(bw_p, λ, ϕ) * s1(bw_p, λ, ϕ) * cos(bw_p.ϕ_c) * sin(λ - bw_p.λ_c) / sin(d(bw_p, λ, ϕ) / bw_p.a) * cond(bw_p, λ, ϕ)
δw(bw_p, λ, ϕ, r) = 0.0

# compute the total energy
uˡᵒⁿ(bw_p, λ, ϕ, r) = u(bw_p, ϕ, r) + δu(bw_p, λ, ϕ, r)
uˡᵃᵗ(bw_p, λ, ϕ, r) = v(bw_p, ϕ, r) + δv(bw_p, λ, ϕ, r)
uʳᵃᵈ(bw_p, λ, ϕ, r) = w(bw_p, ϕ, r) + δw(bw_p, λ, ϕ, r)

e_int(bw_p, λ, ϕ, r) = (bw_p.R_d / bw_p.κ - bw_p.R_d) * (T(bw_p, ϕ, r) - bw_p.T_0)
e_kin(bw_p, λ, ϕ, r) = 0.5 * (uˡᵒⁿ(bw_p, λ, ϕ, r)^2 + uˡᵃᵗ(bw_p, λ, ϕ, r)^2 + uʳᵃᵈ(bw_p, λ, ϕ, r)^2)
e_pot(bw_p, λ, ϕ, r) = bw_p.g * r

ρ₀(bw_p, λ, ϕ, r) = pressure_init(bw_p, ϕ, r) / bw_p.R_d / T(bw_p, ϕ, r)
ρuˡᵒⁿ(bw_p, λ, ϕ, r) = ρ₀(bw_p, λ, ϕ, r) * uˡᵒⁿ(bw_p, λ, ϕ, r)
ρuˡᵃᵗ(bw_p, λ, ϕ, r) = ρ₀(bw_p, λ, ϕ, r) * uˡᵃᵗ(bw_p, λ, ϕ, r)
ρuʳᵃᵈ(bw_p, λ, ϕ, r) = ρ₀(bw_p, λ, ϕ, r) * uʳᵃᵈ(bw_p, λ, ϕ, r)

ρe(bw_p, λ, ϕ, r) = ρ₀(bw_p, λ, ϕ, r) * (e_int(bw_p, λ, ϕ, r) + e_kin(bw_p, λ, ϕ, r) + e_pot(bw_p, λ, ϕ, r))

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(bw_p, x...) = ρ₀(bw_p, lon(x...), lat(x...), rad(x...))
ρu₀ᶜᵃʳᵗ(bw_p, x...) = (ρuʳᵃᵈ(bw_p, lon(x...), lat(x...), rad(x...)) * r̂(x...)
                       + ρuˡᵃᵗ(bw_p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                       + ρuˡᵒⁿ(bw_p, lon(x...), lat(x...), rad(x...)) * λ̂(x...))
ρe₀ᶜᵃʳᵗ(bw_p, x...) = ρe(bw_p, lon(x...), lat(x...), rad(x...))



# correction auxstate for sphere
function sphere_auxiliary(law::EulerTotalEnergyLaw, x⃗, state)
    ix_ρ, ix_ρu⃗, ix_ρe = Atum.EulerTotalEnergy.varsindices(law)
    ix_x, ix_y, ix_z, _, _, _, _ = Atum.EulerTotalEnergy.auxindices(law)
    r = sqrt(x⃗' * x⃗)
    @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], state[ix_ρ], state[ix_ρu⃗]..., state[ix_ρe], 9.81 * r)
end

N = 3
Nq = N + 1
Nq⃗ = (Nq, Nq, Nq)
dim = 3

FT = Float64
A = CuArray

Kv = 8
Kh = 12

law = EulerTotalEnergyLaw{FT,dim}()
cell = LobattoCell{FT,A}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
cpu_cell = LobattoCell{FT,Array}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
vert_coord = range(FT(bw_p.a), stop=FT(bw_p.a + bw_p.H), length=Kv + 1)
grid = cubedspheregrid(cell, vert_coord, Kh)
x⃗ = points(grid)

cpu_grid = cubedspheregrid(cpu_cell, vert_coord, Kh)
cpu_x⃗ = points(cpu_grid)

function baroclinic_wave(x⃗, param)
    x, y, z = x⃗
    bw_ρ = ρ₀(param, lon(x⃗...), lat(x⃗...), rad(x⃗...))
    bw_ρuᵣ = ρuʳᵃᵈ(param, lon(x⃗...), lat(x⃗...), rad(x⃗...)) * r̂ⁿᵒʳᵐ(x, y, z) * @SVector([x, y, z])
    bw_ρuₗ = ϕ̂ⁿᵒʳᵐ(x, y, z) * @SVector([x * z, y * z, -(x^2 + y^2)]) * ρuˡᵃᵗ(param, lon(x⃗...), lat(x⃗...), rad(x⃗...))
    bw_ρuₗₗ = ρuˡᵒⁿ(param, lon(x⃗...), lat(x⃗...), rad(x⃗...)) * λ̂(x⃗...)
    bw_ρu⃗ = bw_ρuᵣ + bw_ρuₗ + bw_ρuₗₗ
    bw_ρe = ρe(param, lon(x⃗...), lat(x⃗...), rad(x⃗...))
    SVector(bw_ρ, bw_ρu⃗..., bw_ρe)
end

state = fieldarray(undef, law, grid)
test_state = fieldarray(undef, law, grid)
stable_state = fieldarray(undef, law, grid)
old_state = fieldarray(undef, law, grid)
cpu_state = fieldarray(undef, law, cpu_grid)
cpu_state .= baroclinic_wave.(cpu_x⃗, Ref(bw_p))
gpu_components = components(state)
cpu_components = components(cpu_state)
for i in eachindex(gpu_components)
    gpu_components[i] .= A(cpu_components[i])
end
# state .= baroclinic_wave.(x⃗, Ref(bw_p)) # this line gives the error
test_state .= state
old_state .= state
aux = sphere_auxiliary.(Ref(law), x⃗, state)
old_aux = sphere_auxiliary.(Ref(law), x⃗, state)
bw_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), state, aux)
bw_density = components(state)[1]
bw_soundspeed = Atum.EulerTotalEnergy.soundspeed.(Ref(law), bw_density, bw_pressure)
c_max = maximum(bw_soundspeed)

function boundarystate(law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, _)
    ρ⁻, ρu⃗⁻, ρe⁻ = EulerTotalEnergy.unpackstate(law, q⁻)
    ρ⁺, ρe⁺ = ρ⁻, ρe⁻
    ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
    SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

function source!(law::EulerTotalEnergyLaw, source, state, aux, dim, directions)
    # Extract the state
    _, ρu⃗, _ = EulerTotalEnergy.unpackstate(law, state)
    # _, source_ρu⃗, _ = EulerTotalEnergy.unpackstate(law, source)

    Ω = @SVector [-0, -0, 2π / 86400]

    coriolis = -2Ω × ρu⃗
    # source[2:4] += -2Ω × ρu⃗
    source[2] = coriolis[1]
    source[3] = coriolis[2]
    source[4] = coriolis[3]

    return nothing
end

vf = (KennedyGruberFlux(), KennedyGruberFlux(), KennedyGruberFlux())
sf = (RoeFlux(), RoeFlux(), Atum.RefanovFlux(0.1))  # KennedyGruberFlux()) 

linearized_vf = Atum.LinearizedKennedyGruberFlux()
linearized_sf = Atum.LinearizedRefanovFlux(0.1)# Atum.LinearizedKennedyGruberFlux() 

dg_sd = SingleDirection(; law, grid, volume_form=linearized_vf, surface_numericalflux=linearized_sf)
dg_fs = FluxSource(; law, grid, volume_form=vf, surface_numericalflux=sf)

dg_sd.auxstate .= aux
dg_fs.auxstate .= aux

vcfl = 120.0
hcfl = 0.25
Δx = min_node_distance(grid, dims=1)
Δy = min_node_distance(grid, dims=2)
Δz = min_node_distance(grid, dims=3)
vdt = vcfl * Δz / c_max
hdt = hcfl * Δx / c_max
dt = min(vdt, hdt)
println(" the dt is ", dt)
println(" the vertical cfl is ", dt * c_max / Δz)
println(" the horizontal cfl is ", dt * c_max / Δx)
timeend = 60 * 60 * 24 * 10 # 24 * 15 # seconds

odesolver = ARK23(dg_fs, dg_sd, fieldarray(state), dt; split_rhs=false, paperversion=false)

# odesolver = LSRK144(dg_fs, state, dt)


do_output = function (step, time, q)
    if step % ceil(Int, timeend / 100 / dt) == 0
        println("simulation is ", time / timeend * 100, " percent complete")
        ρ, ρu, ρv, ρw, _ = components(q)
        println("maximum x-velocity ", maximum(ρu ./ ρ))
        println("maximum y-velocity ", maximum(ρv ./ ρ))
        println("maximum z-velocity ", maximum(ρw ./ ρ))
        println("the time is ", time)
    end
end


tic = time()
solve!(state, timeend, odesolver; after_step=do_output)
toc = time()
println("The time for the simulation is ", toc - tic)

ρ, ρu, ρv, ρw, _ = components(state)
println("maximum x-velocity ", maximum(ρu ./ ρ))
println("maximum y-velocity ", maximum(ρv ./ ρ))
println("maximum z-velocity ", maximum(ρw ./ ρ))

##
# 30 minute updates and hcfl of 0.25 is stable
vcfl = 120.0
hcfl = 0.5
Δx = min_node_distance(grid, dims=1)
Δy = min_node_distance(grid, dims=2)
Δz = min_node_distance(grid, dims=3)
vdt = vcfl * Δz / 350
hdt = hcfl * Δx / 350
dt = min(vdt, hdt)

test_state .= old_state
# test_state .= state
endday = 10
##
tic = time()
partitions = 1:24*endday*2
for i in partitions
    aux = sphere_auxiliary.(Ref(law), x⃗, test_state)
    dg_fs.auxstate .= aux
    dg_sd.auxstate .= aux
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
    timeend = 60 * 60 * 24 * endday / partitions[end]
    # solve!(test_state, timeend, odesolver; after_step=do_output)
    solve!(test_state, timeend, odesolver)
    if i % 4 == 0
        println("--------")
        println("done with ", timeend)
        println("partition ", i, " out of ", partitions[end])
        ρ, ρu, ρv, ρw, ρet = components(test_state)
        println("maximum x-velocity ", maximum(ρu ./ ρ))
        println("maximum y-velocity ", maximum(ρv ./ ρ))
        println("maximum z-velocity ", maximum(ρw ./ ρ))
        bw_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), test_state, aux)
        bw_density = components(test_state)[1]
        bw_soundspeed = Atum.EulerTotalEnergy.soundspeed.(Ref(law), bw_density, bw_pressure)
        c_max = maximum(bw_soundspeed)
        println("The maximum soundspeed is ", c_max)
        println("The dt is now ", dt)
        if isnan(ρ[1]) | isnan(ρu[1]) | isnan(ρv[1]) | isnan(ρw[1]) | isnan(ρet[1])
            nothing
        else
            # stable_state .= test_state
        end
        println("-----")
    end
end
toc = time()
println("The time for the simulation is ", toc - tic)

