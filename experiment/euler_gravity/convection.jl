using Atum
using Atum.EulerGravity
using Random
using StaticArrays
using StaticArrays: SVector, MVector
using WriteVTK

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

# To do: put coordinate in aux⁻

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
    source[2:4] += λ * damping_profile * ρu
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

# using CUDA
A = Array
# A = Array
FT = Float64
N = 3

K = 4

vf = FluxDifferencingForm(KennedyGruberFlux())
println("DOFs = ", (3 + 1) * K, " with VF ", vf)

volume_form = vf

outputvtk = false
Nq = N + 1

law = EulerGravityLaw{FT,3}()
# pp = 2
cell = LobattoCell{FT,A}(Nq, Nq, Nq)
v1d = range(FT(-1.5e3), stop=FT(1.5e3), length=K+1)
v2d = range(FT(-1.5e3), stop=FT(1.5e3), length=K+1)
v3d = range(FT(0), stop=FT(3.0e3), length=K + 1)
grid = brickgrid(cell, (v1d, v2d, v3d); periodic=(true, true, false))
x⃗ = points(grid)
dg = DGSEM(; law, grid, volume_form, surface_numericalflux=RoeFlux())

cfl = FT(15 // 8) # for lsrk14, roughly a cfl of 0.125 per stage

c = 330.0 # [m/s]
dt = cfl * min_node_distance(grid) / c
println(" the dt is ", dt)
timeend = 2 * 60 * 60

q = fieldarray(undef, law, grid)
q .= initial_condition.(Ref(law), points(grid))

if outputvtk
    vtkdir = joinpath("output", "gravity_euler", "convection")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
end

do_output = function (step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
        filename = "step$(lpad(step, 6, '0'))"
        vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
        P = Bennu.toequallyspaced(cell)
        ρθ = last(components(q))
        vtkfile["ρθ"] = vec(Array(P * ρθ))
        vtk_save(vtkfile)
        pvd[time] = vtkfile
    elseif step % ceil(Int, timeend / 100 / dt) == 0
        println("simulation is ", time / timeend * 100, " percent complete")
    end
end

odesolver = LSRK144(dg, q, dt)

outputvtk && do_output(0, FT(0), q)
tic = time()
solve!(q, timeend, odesolver; after_step=do_output)
outputvtk && vtk_save(pvd)
toc = time()
println("The time for the simulation is ", toc - tic)
println(q[1])

##

x, y, z = components(x⃗)
xC = mean(x, dims=1)[:]
yC = mean(y, dims=1)[:]
zC = mean(z, dims=1)[:]
