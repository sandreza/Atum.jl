using JLD2, Atum
include("interpolate.jl")
## 
# grab file 
# jlfile = jldopen("HeldSuarezStatistics_Nev10_Neh15_Nq4.jld2")
jlfile = jldopen("HeldSuarezStatistics_Nev5_Neh10_Nq4.jld2")
ρ = jlfile["firstmoment"]["ρ"]
temperature = jlfile["firstmoment"]["T"]
u = jlfile["firstmoment"]["u"]
v = jlfile["firstmoment"]["v"]
w = jlfile["firstmoment"]["w"]
p = jlfile["firstmoment"]["p"]

FT = Float64
A = Array
vert_coord = jlfile["grid"]["vertical_coordinate"]
Kv = jlfile["grid"]["vertical_element_number"]
Kh = jlfile["grid"]["horizontal_element_number"]
Nq⃗ = jlfile["grid"]["gauss_lobatto_points"]

cell = LobattoCell{FT,A}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
cpu_cell = LobattoCell{FT,Array}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
cpu_grid = cubedspheregrid(cpu_cell, vert_coord, Kh)

rlist = range(vert_coord[1] + 000, vert_coord[end], length=3)
θlist = range(-π, π, length=180)
ϕlist = range(0, π, length=90)

xlist = [sphericaltocartesian(θ, ϕ, r) for θ in θlist, ϕ in ϕlist, r in rlist]

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


newT = zeros(size(xlist))
newP = similar(newT)
newU = similar(newT)
newV = similar(newT)
for (newf, oldf) in zip([newU, newV, newP, newT], [u, v, p, temperature])
    interpolate_field!(newf, oldf, elist, ξlist, r, ω, Nq⃗, arch=CPU())
end

# if GLMakie is loeaded one can run the following lines
#=
fig = Figure()
ax11 = Axis(fig[1, 1])
ax12 = Axis(fig[1, 2])
ax21 = Axis(fig[2, 1])
ax22 = Axis(fig[2, 2])
heatmap!(ax11, newT[:, :, 1], colormap=:afmhot, interpolate=true, colorrange=(265, 310))
heatmap!(ax21, newP[:, :, 1], colormap=:bone_1, interpolate=true)
heatmap!(ax12, newU[:, :, 1], colormap=:balance, interpolate=true, colorrange=(-30,30))
heatmap!(ax22, newV[:, :, 1], colormap=:balance, interpolate=true, colorrange=(-30, 30))
=#