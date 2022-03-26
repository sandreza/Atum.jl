using Atum
Nq = 4
cell = LobattoCell{FT,A}(Nq, Nq, Nq)
v1d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v2d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v3d = range(FT(0), stop=FT(3.0e3), length=K + 1)
grid = brickgrid(cell, (v1d, v2d, v3d); periodic=(true, true, false))
x⃗ = points(grid)
x, y, z = components(x⃗)
xC = mean(x, dims=1)[:]
yC = mean(y, dims=1)[:]
zC = mean(z, dims=1)[:]

r = [cell.points_1d[i][:] for i in eachindex(cell.points_1d)]
ω = [baryweights(cell.points_1d[i][:]) for i in eachindex(cell.points_1d)]
# ω = [cell.weights_1d[i][:] for i in eachindex(cell.weights_1d)]
bw[1]
ξ = [0.0, 0.0, 0.0]
ξ = [-1.0, -1.0, -1.0]
icheck = checkgl(0.0, r[1])
newf = lagrange_eval(reshape(x[:, 1], Nq, Nq, Nq), ξ..., r..., ω...)

##
function get_element(xnew, ynew, znew, xinfo, yinfo, zinfo)
    xmin, xmax, nex = xinfo
    ymin, ymax, ney = yinfo
    zmin, zmax, nez = zinfo
    ex = ceil(Int, (xnew - xmin) / (xmax - xmin) * nex)
    ex = min(max(ex, 1), nex)
    ey = ceil(Int, (ynew - ymin) / (ymax - ymin) * ney)
    ey = min(max(ey, 1), ney)
    ez = ceil(Int, (znew - zmin) / (zmax - zmin) * nez)
    ez = min(max(ez, 1), nez)
    e = ex + nex * (ey - 1 + ney * (ez - 1))
    return e
end

rescale(x, xmin, xmax) = 2 * (x - xmin) / (xmax - xmin) - 1

function get_reference(x, y, z, oldx, oldy, oldz)
    xmin, xmax = extrema(oldx)
    ymin, ymax = extrema(oldy)
    zmin, zmax = extrema(oldz)

    ξ1 = rescale(x, xmin, xmax)
    ξ2 = rescale(y, ymin, ymax)
    ξ3 = rescale(z, zmin, zmax)

    return (ξ1, ξ2, ξ3)
end

function cube_interpolate(newgrid, oldgrid)
    ξlist = [[0.0, 0.0, 0.0] for i in eachindex(newgrid)]
    elist = zeros(Int, length(newgrid))
    x, y, z = components(grid.points)
    nex, ney, nez = (size(oldgrid.vertices) .- 1)
    xinfo = (extrema(x)..., nex)
    yinfo = (extrema(y)..., ney)
    zinfo = (extrema(z)..., nez)
    for I in eachindex(newgrid)
        xnew, ynew, znew = newgrid[I]
        e = get_element(xnew, ynew, znew, xinfo, yinfo, zinfo)
        oldx = view(x, :, e)
        oldy = view(y, :, e)
        oldz = view(z, :, e)
        ξ = get_reference(xnew, ynew, znew, oldx, oldy, oldz)
        ξlist[I] .= ξ
        elist[I] = e
    end
    return ξlist, elist
end

##
newgrid = [[x[1, e], y[1, e], z[1, e]] for e in 1:64]
newgrid = [[x[I], y[I], z[I]] for I in eachindex(x)]
nx = ny = nz = 8
newgrid = [[0.0, 0.0, 0.0] for I in 1:nx*ny*nz]
newx = range(-1.5e3, 1.5e3, nx)
newy = range(-1.5e3, 1.5e3, ny)
newz = range(0, 3e3, nz)
for i in 1:nx, j in 1:ny, k in 1:nz
    ijk = i + nx * (j - 1 + ny * (k - 1))
    newgrid[ijk] .= [newx[i], newy[j], newz[k]]
end

oldgrid = grid
ξlist, elist = cube_interpolate(newgrid, grid)

##
xnew, ynew, znew = (1000.0, 1000.0, 1000.0)
xmin, xmax = extrema(x)
ymin, ymax = extrema(y)
zmin, zmax = extrema(z)
nex, ney, nez = size(grid.vertices) .- 1
ex = ceil(Int, (xnew - xmin) / (xmax - xmin) * nex) # 4 [0, 1], [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
ey = ceil(Int, (ynew - ymin) / (ymax - ymin) * ney)
ez = ceil(Int, (znew - zmin) / (zmax - zmin) * nez)
e = ex + nex * (ey - 1 + ney * (ez - 1))
x[:, e]
y[:, e]
z[:, e]

##
@benchmark lagrange_eval(reshape(x[:, 1], Nq, Nq, Nq), ξ..., r..., ω...)
@benchmark lagrange_eval_3(x[:, 1], ξ, r, ω, (Nq, Nq, Nq))

##
#
e_num = 4
newgrid = zeros(e_num, e_num, e_num) # number of elements in grid
newf = 0 * newgrid
ξlist = [(-1.0, -1.0, -1.0) for i in eachindex(newgrid)] # same point in each element 
elist = collect(1:(e_num^3))
Nq = (4, 4, 4)
oldf = copy(y)
interpolate_field!(newf, oldf, elist, ξlist, r, ω, Nq)

for I in eachindex(newgrid)
    oldfijk = zeros(Nq)
    ξ = ξlist[I]
    e = elist[I]
    for II in eachindex(oldfijk)
        oldfijk[II] = oldf[II, e]
    end
    newf[I] = lagrange_eval_2(oldfijk, ξ, r, ω)
end

norm(newf[:] - oldf[:])
