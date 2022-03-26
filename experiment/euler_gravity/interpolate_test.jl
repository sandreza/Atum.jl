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
@benchmark lagrange_eval(reshape(x[:, 1], Nq, Nq, Nq), ξ..., r..., ω...)
@benchmark lagrange_eval_3(x[:, 1], ξ, r, ω, (Nq, Nq, Nq))