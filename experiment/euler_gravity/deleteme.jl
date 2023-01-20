newU = Array(meanlist[2])
newV = Array(meanlist[3])
newW = Array(meanlist[4])
state_val = newU
title_string = "Vertical Velocity"
height_index_start = 1
height_index_end = 60

# clims = quantile(state_val[:, :, height_index_start:height_index_end][:], 0.99)
# clims = (-clims, clims)
climsU = quantile(state_val[:, :, height_index_start:height_index_end][:], 0.99)
climsL = quantile(state_val[:, :, height_index_start:height_index_end][:], 0.01)
clims = (climsL, climsU)

fig_volume = Figure(resolution=(1520, 980))
ax = LScene(fig_volume, scenekw=(camera=cam3d!, show_axis=true))
ax_text = Label(fig_volume, title_string,
    textsize=30, color=(:black, 0.85))

cmap = :balance # :Blues_9
cmapa = RGBAf.(to_colormap(cmap), 1);
cmap = vcat(cmapa[1:15], fill(RGBAf(0, 0, 0, 0), 10), cmapa[25:end])

v1 = volume!(ax, 0 .. 20, 0 .. 10, 0 .. 5, state_val[:, 20:180-20, height_index_start:height_index_end],
    colorrange=clims, algorithm=:absorption, absorption=10.0f0,
    colormap=cmap)
axis = ax.scene[OldAxis]
axis[:names, :axisnames] = ("longitude [ᵒ]", "latitude [ᵒ]", "height [km]")
tstyle = axis[:names] #  get the nested attributes and work directly with them

tstyle[:textsize] = 15
tstyle[:textcolor] = (:black, :black, :black)
tstyle[:font] = "helvetica"
tstyle[:gap] = 10
axis[:ticks][:textcolor] = :black
axis[:ticks][:textsize] = 10
cbar1 = Colorbar(fig_volume, v1, label=L" $w$ [m/s]", width=25, ticklabelsize=30,
    labelsize=30, ticksize=25, tickalign=1, height=Relative(3 / 4)
)

axis[:ticks][:ranges] = ([0.0, 5.0, 10.0, 15.0, 20.0], [0.0, 2.5, 5.0, 7.5, 10.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
axis[:ticks][:labels] = (["180W", "90W", "0", "90E", "180E"], ["60S", "30S", "0", "30N", "60N"], ["0", "6", "12", "18", "24", "30"])

fig_volume[2:10, 1:10] = ax
fig_volume[3:8, 11] = cbar1
fig_volume[1, 5:6] = ax_text

zoom!(ax.scene, 0.8)
display(fig_volume)