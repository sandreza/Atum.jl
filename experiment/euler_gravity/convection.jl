using Atum
using Atum.EulerGravity

using StaticArrays: SVector, MVector
using WriteVTK

import Atum: boundarystate, source

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

    z = aux.z

    Q₀ = 100.0   # convective_forcing.Q₀
    r_ℓ = 100.0  # convective_forcing.r_ℓ
    s_ℓ = 100.0  # convective_forcing.s_ℓ
    λ = 1 / 10.0 # convective_forcing.λ
    L = 3e3      # convective_forcing.L

    radiation_profile = Q₀ / r_ℓ * exp(-z / r_ℓ)

    damping_profile = -exp(-(L - z) / s_ℓ)

    # Apply convective forcing
    source.ρu += λ * damping_profile * ρu
    source.ρe += ρ * radiation_profile

    return nothing
end

θ₀(p, x, y, z) = p.Tₛ + p.Δθ / p.zmax * z
p₀(p, x, y, z) = p.pₒ * (p.g / (-p.Δθ / p.zmax * p.cp_d) * log(θ₀(p, x, y, z) / p.Tₛ) + 1)^(p.cp_d / p.R_d)
T₀(p, x, y, z) = (p₀(p, x, y, z) / p.pₒ)^(p.R_d / p.cp_d) * θ₀(p, x, y, z)
ρ₀(p, x, y, z) = p₀(p, x, y, z) / (p.R_d * T₀(p, x, y, z))

ρu₀(p, x, y, z) = 0.01 * @SVector [randn(), randn(), randn()]

e_pot(p, x, y, z) = p.g * z
e_int(p, x, y, z) = p.cv_d * (T₀(p, x, y, z) - p.T_0)
e_kin(p, x, y, z) = 0.5 * (ρu₀(p, x, y, z)' * ρu₀(p, x, y, z)) / ρ₀(p, x, y, z)^2

ρe₀(p, x, y, z) = ρ₀(p, x, y, z) * (e_kin(p, x, y, z) + e_int(p, x, y, z) + e_pot(p, x, y, z))

function initial_condition(law, x⃗)
    FT = eltype(law)
    x, y, z = x⃗

    ρ = ρ₀(p, x, y, z)
    ρu, ρv, ρw = ρu₀(p, x, y, z)
    ρe = ρe₀(p, x, y, z)

    SVector(ρ, ρu, ρv, ρw, ρe)
end



# using CUDA
A = Array
# A = Array
FT = Float64
N = 3

K = 4

vf = FluxDifferencingForm(EntropyConservativeFlux())
println("DOFs = ", (3 + 1) * K, " with VF ", vf)

volume_form = vf

outputvtk = false
Nq = N + 1

law = EulerGravityLaw{FT,3}()
# pp = 2
cell = LobattoCell{FT,A}(Nq, Nq, Nq)
v1d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v2d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v3d = range(FT(0), stop=FT(3.0e3), length=K + 1)
grid = brickgrid(cell, (v1d, v2d, v3d); periodic=(true, true, false))

dg = DGSEM(; law, grid, volume_form, surface_numericalflux=RoeFlux())

cfl = FT(15 // 8) # for lsrk14, roughly a cfl of 0.125 per stage

c = 330.0 # [m/s]
dt = cfl * min_node_distance(grid) / c
println(" the dt is ", dt)
timeend = @isdefined(_testing) ? 10dt : FT(200)

q = initial_condition.(Ref(law), points(grid))

if outputvtk
    vtkdir = joinpath("output", "shallow_water", "bickleyjet")
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

# odesolver = LSRK54(dg, q, dt)
odesolver = LSRK144(dg, q, dt)

outputvtk && do_output(0, FT(0), q)
tic = time()
solve!(q, timeend, odesolver; after_step=do_output)
outputvtk && vtk_save(pvd)
toc = time()
println("The time for the simulation is ", toc - tic)
println(q[1])
