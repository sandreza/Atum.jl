using Atum, LinearAlgebra
include("interpolate.jl")

# First check the cube
Nq = 4
FT = Float64
A = Array
cell = LobattoCell{FT,A}(Nq, Nq, Nq)
K = 2
v1d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v2d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v3d = range(FT(0), stop=FT(3.0e3), length=K + 1)
grid = brickgrid(cell, (v1d, v2d, v3d); periodic=(true, true, false))
x⃗ = points(grid)
x, y, z = components(x⃗)

r = Tuple([cell.points_1d[i][:] for i in eachindex(cell.points_1d)])
ω = Tuple([A(baryweights(cell.points_1d[i][:])) for i in eachindex(cell.points_1d)])

elist = A(1:K^3)
ξlist = A([@SVector[-1.0, 0.0, 0.0] for i in 1:K^3])
oldf = x
newf = A(collect(1:K^3) .* 1.0)
Nq⃗ = (Nq, Nq, Nq)
interpolate_field!(newf, oldf, elist, ξlist, r, ω, Nq⃗, arch=CPU())
norm(newf[:] - x[1, :])
##
Kv = 3
Kh = 2
Nq = 3
Nq⃗ = (Nq, Nq, Nq)
cell = LobattoCell{FT,A}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
cpu_cell = LobattoCell{FT,Array}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
vert_coord = range(0.5, 1, length=Kv + 1)
grid = cubedspheregrid(cell, vert_coord, Kh)
x⃗ = points(grid)
cpu_grid = cubedspheregrid(cpu_cell, vert_coord, Kh)

r = Tuple([cell.points_1d[i][:] for i in eachindex(cell.points_1d)])
ω = Tuple([A(baryweights(cell.points_1d[i][:])) for i in eachindex(cell.points_1d)])

function cube_single_element_index(cell_coords, Kv, Kh)
    ev  = cell_coords[1] # vertical element
    ehi = cell_coords[2] # local face i index
    ehj = cell_coords[3] # local face j index
    ehf = cell_coords[4] # local face index
    return ev + Kv * (ehi - 1 + Kh * (ehj - 1 + Kh * (ehf - 1)))
end

n_ijk, n_e = size(x⃗)
inside_cell_index = argmin(norm.(cell.points)) # if on boundaries it is ambiguous
for e in 1:n_e
    xcheck = x⃗[inside_cell_index, e]
    (x̂, cell_coords) = Bennu.cubedspherereference(xcheck, vert_coord, Kh)
    println("--------")
    println("The element is ", e)
    println(SVector(x̂) - cell.points[inside_cell_index])
    println(cell_coords) # last index is face, first index is vertical level, the next two is the i,j face index, the last one is face
    computed_e = cube_single_element_index(cell_coords, Kv, Kh)
    println("computed element ", computed_e)
    println("--------")
end

rlist = range(0.5, 1, length=Kv + 2)
θlist = range(-π, π, length=4)
ϕlist = range(0, π, length=4)

tocartesian(θ, ϕ, r) = SVector(r * cos(θ) * sin(ϕ), r * sin(θ) * sin(ϕ), r * cos(ϕ))
xlist = [tocartesian(θ, ϕ, r) for θ in θlist, ϕ in ϕlist, r in rlist]

elist = zeros(Int, length(xlist))
ξlist = [[0.0, 0.0, 0.0] for i in eachindex(xlist)]
for kk in eachindex(xlist)
    x = xlist[kk]
    (x̂, cell_coords) = Bennu.cubedspherereference(x, vert_coord, Kh)
    elist[kk] = cube_single_element_index(cell_coords, Kv, Kh)
    ξlist[kk] .= x̂
end

r = Tuple([cell.points_1d[i][:] for i in eachindex(cell.points_1d)])
ω = Tuple([A(baryweights(cell.points_1d[i][:])) for i in eachindex(cell.points_1d)])
for i in 1:3
    oldf = components(x⃗)[i]
    newf = zeros(size(xlist))
    interpolate_field!(newf, oldf, elist, ξlist, r, ω, Nq⃗, arch=CPU())
    exactf = [xs[i] for xs in xlist]
    println("The error is ", norm(exactf - newf))
end

