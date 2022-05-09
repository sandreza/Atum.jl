using Random
include("oldhelpers.jl")
fig = Figure(resolution=(1700 + 600, 1000 + 400))
add_label = true
println("looking at ", filename)
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