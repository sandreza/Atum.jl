using JLD2, Atum, Random
include("interpolate.jl")
## 
# grab file 
# filename = "HeldSuarezStatistics_Nev10_Neh15_Nq4.jld2"
# filename = "HeldSuarezStatistics_Nev2_Neh4_Nq5.jld2"
# filename = "HeldSuarezStatistics_Nev5_Neh10_Nq4.jld2"
# filename = "HeldSuarezStatisticsNew_Nev2_Neh4_Nq5.jld2"
# filename = "HeldSuarezStatisticsEvenNewer_Nev2_Neh4_Nq5.jld2"
# filename = "HeldSuarezStatistics_Nev7_Neh10_Nq4.jld2"
# filename = "stable_check.jld2"
# filename = "HeldSuarezStatistics_Nev7_Neh15_Nq4.jld2"
filename = "HeldSuarezStatisticsMinimal_Nev5_Neh12_Nq4.jld2"
# filename = "HeldSuarezStatisticsPaper_Nev5_Neh12_Nq5.jld2" # garbage
# filename = "HeldSuarezStatisticsQuadtratic_Nev8_Neh20_Nq4.jld2"
# filename = "HeldSuarezStatisticsQuadtratic_Nev11_Neh13_Nq4.jld2"
# filename = "HeldSuarezStatisticsQuadtratic_Nev14_Neh7_Nq4.jld2"
# filename = "HeldSuarezStatisticsQuadtratic_Nev14_Neh8_Nq4.jld2"
filename = "HeldSuarezStatisticsQuadtratic_Nev20_Neh8_Nq4.jld2"
filename = "HeldSuarezStatistics_Nev8_Neh6_Nq5.jld2"
filename = "HeldSuarezStatistics_Nev6_Neh12_Nq5.jld2"
filename = "HeldSuarezStatistics_Nev10_Neh15_Nq4.jld2"
# filename = "HeldSuarezStatistics_Nev8_Neh9_Nq5.jld2"
filename = "HeldSuarezStatistics_Nev20_Neh20_Nq3.jld2"
# filename = "HeldSuarezStatistics_Nev10_Neh10_Nq3.jld2"
# filename = "HeldSuarezStatistics_Nev15_Neh15_Nq3.jld2"
# filename = "HeldSuarezStatistics_Nev20_Neh10_Nq6.jld2"
look_at_instaneous = false
println("looking at ", filename)
jlfile = jldopen(filename)
ρ = jlfile["firstmoment"]["ρ"]
T = jlfile["firstmoment"]["T"]
u = jlfile["firstmoment"]["u"]
v = jlfile["firstmoment"]["v"]
w = jlfile["firstmoment"]["w"]
p = jlfile["firstmoment"]["p"]

ρ_i = jlfile["instantaneous"]["ρ"]
u_i = jlfile["instantaneous"]["u"]
v_i = jlfile["instantaneous"]["v"]
w_i = jlfile["instantaneous"]["w"]
p_i = jlfile["instantaneous"]["p"]
T_i = jlfile["instantaneous"]["T"]

uu = jlfile["secondmoment"]["uu"]
uv = jlfile["secondmoment"]["uv"]
vv = jlfile["secondmoment"]["vv"]
vT = jlfile["secondmoment"]["vT"]
TT = jlfile["secondmoment"]["TT"]


FT = Float64
A = Array
vert_coord = jlfile["grid"]["vertical_coordinate"]
Kv = jlfile["grid"]["vertical_element_number"]
Kh = jlfile["grid"]["horizontal_element_number"]
Nq⃗ = jlfile["grid"]["gauss_lobatto_points"]

cell = LobattoCell{FT,A}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
cpu_cell = LobattoCell{FT,Array}(Nq⃗[1], Nq⃗[2], Nq⃗[3])
cpu_grid = cubedspheregrid(cpu_cell, vert_coord, Kh)

scale = 6
rlist = range(vert_coord[1] + 000, vert_coord[end], length=19)
rlist1 = range(vert_coord[1], vert_coord[1] + 2e3, length = 13)
rlist2 = range(rlist1[end] + 1e3, vert_coord[end], length=11)
rlist = [rlist1... , rlist2...]
θlist = range(-π, π, length=90 * scale)
ϕlist = range(0, π, length=45 * scale)

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
newW = similar(newT)
newRho = similar(newT)
# average quantities
avgU = similar(newT)
avgV = similar(newT)
avgT = similar(newT)
avgP = similar(newT)
avgUU = similar(newT)
avgUV = similar(newT)
avgVV = similar(newT)
avgVT = similar(newT)
avgTT = similar(newT)

if look_at_instaneous
    newlist = [newU, newV, newP, newT, newW, newRho]
    oldlist = [u_i, v_i, p_i, T_i, w_i, ρ_i]
else
    newlist = [avgU, avgV, avgT, avgP, avgUU, avgUV, avgVV, avgVT, avgTT]
    oldlist = [u, v, T, p, uu, uv, vv, vT, TT]
end
for (newf, oldf) in zip(newlist, oldlist)
    interpolate_field!(newf, oldf, elist, ξlist, r, ω, Nq⃗, arch=CPU())
end

# if GLMakie is loeaded one can run the following lines
#=
pcoord = sum(avgP, dims=(1, 2))[1, 1, :] / (length(ϕlist) * length(θlist))
fig = Figure()
ax11 = Axis(fig[1, 1])
ax12 = Axis(fig[1, 2])
ax13 = Axis(fig[1, 3])
ax21 = Axis(fig[2, 1])
ax22 = Axis(fig[2, 2])
ax23 = Axis(fig[2, 3])

U̅ = sum(avgU, dims=1)[1, :, :] / length(θlist)
T̅ = sum(avgT, dims=1)[1, :, :] / length(θlist)
V̅ = sum(avgV, dims=1)[1, :, :] / length(θlist)
UpUp = sum(avgUU, dims=1)[1, :, :] / length(θlist) .- U̅ .* U̅
VpVp = sum(avgVV, dims=1)[1, :, :] / length(θlist) .- V̅ .* V̅
UpVp = sum(avgUV, dims=1)[1, :, :] / length(θlist) .- U̅ .* V̅
TpTp = sum(avgTT, dims=1)[1, :, :] / length(θlist) .- T̅ .* T̅
HTKE = 0.5 .* (UpUp + VpVp)
VpTp = sum(avgVT, dims=1)[1, :, :] / length(θlist) .- V̅ .* T̅


heatmap!(ax11, ϕlist, -pcoord, U̅, colormap=:balance, colorrange=(-30, 30), interpolate=true)
heatmap!(ax12, ϕlist, -pcoord, T̅, colormap=:thermometer, colorrange=(190, 310), interpolate=true)
heatmap!(ax13, ϕlist, -pcoord, TpTp, colormap=:thermometer, colorrange=(0, 40), interpolate=true)
heatmap!(ax23, ϕlist, -pcoord, HTKE, colormap=:thermometer, colorrange=(0, 320), interpolate=true)
heatmap!(ax21, ϕlist, -pcoord, UpVp, colormap=:balance, colorrange=(-30, 30), interpolate=true)
heatmap!(ax22, ϕlist, -pcoord, VpTp, colormap=:balance, colorrange=(-20, 20), interpolate=true)
=#


# avgT = sum(newT, dims =1)[1,:,:] / length(θlist)
# pcoord = sum(newP, dims=(1,2))[1,1,:] / (length(rlist) * length(θlist))
# heatmap(ϕlist, -pcoord, avgT, colormap = :thermometer, colorrange = (190,310), interpolate = true)

if look_at_instaneous
    fig = Figure()
    ax11 = Axis(fig[1, 1])
    ax12 = Axis(fig[1, 2])
    ax21 = Axis(fig[2, 1])
    ax22 = Axis(fig[2, 2])
    height_index = Observable(1)
    # heatmap!(ax11, @lift(newT[:, :, $height_index]), colormap=:afmhot, interpolate=true, colorrange=(265, 310))
    # heatmap!(ax21, @lift(newP[:, :, $height_index]), colormap=:bone_1, interpolate=true)
    # heatmap!(ax12, @lift(newU[:, :, $height_index]), colormap=:balance, interpolate=true, colorrange=(-30, 30))
    # heatmap!(ax22, @lift(newV[:, :, $height_index]), colormap=:balance, interpolate=true, colorrange=(-30, 30))

    heatmap!(ax11, @lift(newT[:, :, $height_index]), colormap=:afmhot, interpolate=true)
    heatmap!(ax21, @lift(newP[:, :, $height_index]), colormap=:bone_1, interpolate=true)
    heatmap!(ax12, @lift(newU[:, :, $height_index]), colormap=:balance, interpolate=true)
    heatmap!(ax22, @lift(newV[:, :, $height_index]), colormap=:balance, interpolate=true)
end
