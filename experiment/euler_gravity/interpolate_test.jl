using Atum, LinearAlgebra
Nq = 4
cell = LobattoCell{FT,A}(Nq, Nq, Nq)
v1d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v2d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v3d = range(FT(0), stop=FT(3.0e3), length=K + 1)
grid = brickgrid(cell, (v1d, v2d, v3d); periodic=(true, true, false))
x⃗ = points(grid)
x, y, z = components(x⃗)

r = Tuple([cell.points_1d[i][:] for i in eachindex(cell.points_1d)])
ω = Tuple([CuArray(baryweights(cell.points_1d[i][:])) for i in eachindex(cell.points_1d)])

elist = CuArray(1:64)
ξlist = CuArray([@SVector[-1.0, 0.0, 0.0] for i in 1:64])
oldf = x
newf = CuArray(collect(1:64) .* 1.0)
Nq⃗ = (Nq, Nq, Nq)
interpolate_field!(newf, oldf, elist, ξlist, r, ω, Nq⃗, arch=CUDADevice())
norm(newf[:] - x[1, :])

##
nx = ny = nz = 8
newx = range(-1.5e3, 1.5e3, length=nx)
newy = range(-1.5e3, 1.5e3, length=ny)
newz = range(0, 3e3, length=nz)
newgrid = CuArray([@SVector[newx[i], newy[j], newz[k]] for i in 1:nx, j in 1:ny, k in 1:nz][:])

oldgrid = grid
ξlist, elist = cube_interpolate(newgrid, oldgrid)

newf = CuArray(zeros(nx,ny,nz))
oldf = z
interpolate_field!(newf, oldf, elist, ξlist, r, ω, Nq⃗, arch=CUDADevice())

