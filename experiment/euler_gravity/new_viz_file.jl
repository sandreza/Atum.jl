using JLD2, GLMakie, Random
include("oldhelpers.jl")

main_path = "/home/sandre/Repositories/Atum.jl/"
filename = "HeldSuarezStatistics_Nev6_Neh12_Nq1_5_Nq2_5_Nq3_5.jld2"
filename = "SmallHeldSuarezStatistics_Nev6_Neh12_Nq1_5_Nq2_5_Nq3_5_X_20.0.jld2"
filename = "SmallHeldSuarezStatistics_Nev6_Neh12_Nq1_5_Nq2_5_Nq3_5_X_100.0.jld2"
filepath = main_path * filename
jlfile = jldopen(filepath)

avgρ = jlfile["firstmoment"]["p"]
avgU = jlfile["firstmoment"]["u"]
avgV = jlfile["firstmoment"]["v"]
avgW = jlfile["firstmoment"]["w"]
avgP = jlfile["firstmoment"]["p"]
avgT = jlfile["firstmoment"]["T"]

avgUU = jlfile["secondmoment"]["uu"]
avgVV = jlfile["secondmoment"]["vv"]
avgWW = jlfile["secondmoment"]["ww"]
avgUV = jlfile["secondmoment"]["uv"]
avgVW = jlfile["secondmoment"]["vw"]
avgVT = jlfile["secondmoment"]["vT"]
avgWT = jlfile["secondmoment"]["wT"]
avgTT = jlfile["secondmoment"]["TT"]

close(jlfile)

# fig = Figure(resolution=(1700 + 0 * 600, 1000 + 0 * 400))
fig = Figure()
add_label = true

state_names = []
ϕ = ϕlist
pcoord = sum(avgP, dims=(1, 2))[1, 1, :] / (length(ϕlist) * length(θlist))
p_coord = pcoord

U̅ = sum(avgU, dims=1)[1, :, :] / length(θlist)
T̅ = sum(avgT, dims=1)[1, :, :] / length(θlist)
V̅ = sum(avgV, dims=1)[1, :, :] / length(θlist)
UpUp = sum(avgUU, dims=1)[1, :, :] / length(θlist) .- U̅ .* U̅
VpVp = sum(avgVV, dims=1)[1, :, :] / length(θlist) .- V̅ .* V̅
UpVp = sum(avgUV, dims=1)[1, :, :] / length(θlist) .- U̅ .* V̅
HTKE = 0.5 .* (UpUp + VpVp)
VpTp = sum(avgVT, dims=1)[1, :, :] / length(θlist) .- V̅ .* T̅
TpTp = sum(avgTT, dims=1)[1, :, :] / length(θlist) .- T̅ .* T̅

##
fig_inst = Figure()
ax11_inst = Axis(fig_inst[1, 1])
ax12_inst = Axis(fig_inst[1, 2])
ax21_inst = Axis(fig_inst[2, 1])
ax22_inst = Axis(fig_inst[2, 2])

sl_y = Slider(fig_inst[3, 1:2], range=eachindex(rlist), horizontal=true, startvalue=1)
height_index = sl_y.value
# heatmap!(ax11, @lift(newT[:, :, $height_index]), colormap=:afmhot, interpolate=true, colorrange=(265, 310))
# heatmap!(ax21, @lift(newP[:, :, $height_index]), colormap=:bone_1, interpolate=true)
# heatmap!(ax12, @lift(newU[:, :, $height_index]), colormap=:balance, interpolate=true, colorrange=(-30, 30))
# heatmap!(ax22, @lift(newV[:, :, $height_index]), colormap=:balance, interpolate=true, colorrange=(-30, 30))
# ρ, u, v, w, p, T
newT = Array(meanlist[end])
newP = Array(meanlist[end-1])
newU = Array(meanlist[2])
newV = Array(meanlist[3])
newW = Array(meanlist[4])

heatmap!(ax11_inst, θlist, ϕlist, @lift(newT[:, :, $height_index]), colormap=:afmhot, interpolate=true)
heatmap!(ax21_inst, θlist, ϕlist, @lift(newP[:, :, $height_index]), colormap=:bone_1, interpolate=true)
heatmap!(ax12_inst, θlist, ϕlist, @lift(newU[:, :, $height_index]), colormap=:balance, interpolate=true, colorrange=(-10, 10))
heatmap!(ax22_inst, θlist, ϕlist, @lift(newV[:, :, $height_index]), colormap=:balance, interpolate=true, colorrange=(-10, 10))

display(fig_inst)

##


i = 1
ii = (i - 1) % 3 + 1 # +1 on why 1 based indexing is wrong
jj = (i - 1) ÷ 3 + 1 # +1 on why 1 based indexing is wrong
s_string = "u"
slice_zonal = U̅
colorrange, contour_levels, s_string = plot_helper(s_string, slice_zonal)
colorrange = (-40, 40) # override

push!(state_names, s_string)
ax1 = fig[jj, ii] = Axis(fig, title=state_names[i], titlesize=40)
contour_heatmap!(ax1, ϕ, p_coord, slice_zonal,
    contour_levels, colorrange,
    add_labels=add_label, random_seed=1)

i = 2
ii = (i - 1) % 3 + 1 # +1 on why 1 based indexing is wrong
jj = (i - 1) ÷ 3 + 1 # +1 on why 1 based indexing is wrong
s_string = "T"
slice_zonal = T̅
colorrange, contour_levels, s_string = plot_helper(s_string, slice_zonal)
colorrange = (180, 310)

push!(state_names, s_string)
ax2 = fig[jj, ii] = Axis(fig, title=state_names[i], titlesize=40)
contour_heatmap!(ax2, ϕ, p_coord, slice_zonal, contour_levels, colorrange, add_labels=add_label, colormap=:thermometer)
hideydecorations!(ax2, grid=false)

i = 3
ii = (i - 1) % 3 + 1 # +1 on why 1 based indexing is wrong
jj = (i - 1) ÷ 3 + 1 # +1 on why 1 based indexing is wrong

s_string = "T'T'"
slice_zonal = TpTp
colorrange, contour_levels, s_string = plot_helper(s_string, slice_zonal)
colorrange = (-8, 48) # override

push!(state_names, s_string)
ax3 = fig[jj, ii] = Axis(fig, title=state_names[i], titlesize=40)
contour_heatmap!(ax3, ϕ, p_coord, slice_zonal, contour_levels,
    colorrange, add_labels=add_label,
    colormap=:thermometer, random_seed=12)
hideydecorations!(ax3, grid=false)


i = 4
ii = (i - 1) % 3 + 1 # +1 on why 1 based indexing is wrong
jj = (i - 1) ÷ 3 + 1 # +1 on why 1 based indexing is wrong
s_string = "u'v'"
slice_zonal = UpVp
colorrange, contour_levels, s_string = plot_helper(s_string, slice_zonal)
colorrange = (-60, 60)

push!(state_names, s_string)
ax4 = fig[jj, ii] = Axis(fig, title=state_names[i], titlesize=40)
contour_heatmap!(ax4, ϕ, p_coord, slice_zonal, contour_levels, colorrange, add_labels=add_label)

i = 5
ii = (i - 1) % 3 + 1 # +1 on why 1 based indexing is wrong
jj = (i - 1) ÷ 3 + 1 # +1 on why 1 based indexing is wrong
s_string = "v'T'"
slice_zonal = VpTp
colorrange, contour_levels, s_string = plot_helper(s_string, slice_zonal)
colorrange = (-24, 24) # override

push!(state_names, s_string)
ax5 = fig[jj, ii] = Axis(fig, title=state_names[i], titlesize=40)
contour_heatmap!(ax5, ϕ, p_coord, slice_zonal, contour_levels, colorrange, add_labels=add_label)
hideydecorations!(ax5, grid=false)

i = 6
ii = (i - 1) % 3 + 1 # +1 on why 1 based indexing is wrong
jj = (i - 1) ÷ 3 + 1 # +1 on why 1 based indexing is wrong

colorrange, contour_levels, s_string = plot_helper("u'u'", HTKE)
slice_zonal = HTKE
colorrange = (0, 360) # modify to extrema 

s_string = "⟨(u' u' + v' v')/2⟩"
push!(state_names, s_string)

ax6 = fig[jj, ii] = Axis(fig, title=state_names[i], titlesize=40)
contour_heatmap!(ax6, ϕ, p_coord, slice_zonal,
    contour_levels, colorrange, add_labels=add_label,
    colormap=:thermometer, random_seed=10)
hideydecorations!(ax6, grid=false)

display(fig)

##
filename = "HeldSuarezStatistics_Nev6_Neh12_Nq1_5_Nq2_5_Nq3_5.jld2"
filepath = main_path * filename
jlfile1 = jldopen(filepath)

filename = "SmallHeldSuarezStatistics_Nev6_Neh12_Nq1_5_Nq2_5_Nq3_5_X_20.0.jld2"
filename = "SmallHeldSuarezStatistics_Nev6_Neh12_Nq1_5_Nq2_5_Nq3_5_X_100.0.jld2"
filepath = main_path * filename
jlfile2 = jldopen(filepath)


avgW1 = jlfile1["firstmoment"]["w"]
avgW2 = jlfile2["firstmoment"]["w"]
avgT1 = jlfile1["firstmoment"]["T"]
avgT2 = jlfile2["firstmoment"]["T"]

avgWW1 = jlfile1["secondmoment"]["ww"]
avgWW2 = jlfile2["secondmoment"]["ww"]
avgWT1 = jlfile1["secondmoment"]["wT"]
avgWT2 = jlfile2["secondmoment"]["wT"]


W1 = sum(avgW1, dims=1)[1, :, :] / length(θlist)
W2 = sum(avgW2, dims=1)[1, :, :] / length(θlist)
T1 = sum(avgT1, dims=1)[1, :, :] / length(θlist)
T2 = sum(avgT2, dims=1)[1, :, :] / length(θlist)
WW1 = sum(avgWW1, dims=1)[1, :, :] / length(θlist)
WW2 = sum(avgWW2, dims=1)[1, :, :] / length(θlist)
WT1 = sum(avgWT1, dims=1)[1, :, :] / length(θlist)
WT2 = sum(avgWT2, dims=1)[1, :, :] / length(θlist)
wpwp1 = WW1 - W1 .* W1
wpwp2 = WW2 - W2 .* W2
wpTp1 = WT1 - W1 .* T1
wpTp2 = WT2 - W2 .* T2

figW = Figure()
ax1 = Axis(figW[1, 1], xlabel="Latitude", ylabel="Pressure", title="avgW1")
ax2 = Axis(figW[1, 2], xlabel="Latitude", ylabel="Pressure", title="avgW2")
ax3 = Axis(figW[2, 1], xlabel="Latitude", ylabel="Pressure", title="avgWW1")
ax4 = Axis(figW[2, 2], xlabel="Latitude", ylabel="Pressure", title="avgWW2")
heatmap!(ax1, ϕ, p_coord, W1, colorrange=(-0.2 / 20, 0.2 / 20), colormap=:balance, interpolate=true)
heatmap!(ax2, ϕ, p_coord, W2, colorrange=(-0.2, 0.2), colormap=:balance, interpolate=true)
heatmap!(ax3, ϕ, p_coord, wpwp1, colorrange=(0, 10 / 20^2), colormap=:balance, interpolate=true)
heatmap!(ax4, ϕ, p_coord, wpwp2, colorrange=(0, 10), colormap=:balance, interpolate=true)
display(figW)
close(jlfile1)
close(jlfile2)