using Atum
using Atum.ShallowWater
using Bennu: fieldarray
using CUDA

using StaticArrays: SVector
using WriteVTK

import Atum: boundarystate
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

function run(A, FT, N, K; volume_form=WeakForm(), outputvtk=true)
  Nq = N + 1

  law = ShallowWaterLaw{FT,2}()

  cell = LobattoCell{FT,A}(Nq, Nq)
  v1d = range(FT(-2π), stop=FT(2π), length=K + 1)
  grid = brickgrid(cell, (v1d, v1d); periodic=(true, true))

  dg = DGSEM(; law, grid, volume_form,
    surface_numericalflux=RoeFlux())

  cfl = FT(15 // 8) # for lsrk14, roughly a cfl of 0.125 per stage

  c = sqrt(constants(law).grav)
  dt = FT(cfl * min_node_distance(grid) / c)
  timeend = @isdefined(_testing) ? 10dt : FT(1)
  # timeend = 200.0
  println("dt is ", dt)
  q = fieldarray(undef, law, grid)
  q .= bickleyjet.(Ref(law), points(grid))

  if outputvtk
    vtkdir = joinpath("output", "shallow_water", "bickleyjet")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end

  do_output = function (step, time, q)
    if step % ceil(Int, timeend / 100 / dt) == 0
      println(" currently on time ", time)
      ρ, ρu, ρv = components(q)
      println("extrema ", extrema(ρu))
    end
  end
  # if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
  # println(" currently on time ", time)
  # filename = "step$(lpad(step, 6, '0'))"
  # vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
  # P = Bennu.toequallyspaced(cell)
  # ρθ = last(components(q))
  # vtkfile["ρθ"] = vec(Array(P * ρθ))
  # vtk_save(vtkfile)
  # pvd[time] = vtkfile
  # end
  # end

  odesolver = LSRK144(dg, q, dt)

  outputvtk && do_output(0, FT(0), q)
  println("outputing now")
  # solve!(q, timeend, odesolver; after_step=do_output)
  solve!(q, timeend, odesolver)
  ρ, ρu, ρv = components(q)
  println("extrema ", extrema(Array(ρu)))
  outputvtk && vtk_save(pvd)
end

let
  A = CuArray
  FT = Float32
  N = 3

  K = 16*16
  tic = Base.time()
  run(A, FT, N, K, volume_form=FluxDifferencingForm(EntropyConservativeFlux()), outputvtk = false)
  toc = Base.time()
  println("time for the simulation ", toc - tic)
end
