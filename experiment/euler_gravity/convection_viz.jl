using LinearAlgebra, GLMakie
include("interpolate.jl")


##
# First check the cube
x⃗ = points(grid)
x, y, z = components(x⃗)
##
gpu_components = components(q)
cpu_components = components(qc)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end

##
r = Tuple([cpu_cell.points_1d[i][:] for i in eachindex(cpu_cell.points_1d)])
ω = Tuple([baryweights(cpu_cell.points_1d[i][:]) for i in eachindex(cpu_cell.points_1d)])

M = 200
xlist = range(minimum(x), maximum(x), length=M)
ylist = range(minimum(y), maximum(y), length=M)
zlist = range(minimum(z), maximum(z), length=M)

newgrid = [SVector(x, y, z) for x in xlist, y in ylist, z in zlist]

ξlist, elist = cube_interpolate(newgrid, cpu_grid, arch=CPU())

new_θ = zeros(size(newgrid))
new_wθ = zeros(size(newgrid))
new_θθ = zeros(size(newgrid))
new_w = zeros(size(newgrid))

x, y, z = components(grid.points)
ρ, ρu, ρv, ρw, ρe = components(q)
ϕ = 9.81 * z
p = @. (0.4) * (ρe - (ρu^2 + ρv^2 + ρw^2) / (2ρ) - ρ * ϕ)
# p  = ρ R T
T = @. p / (ρ * parameters.R)
θ = @. (parameters.pₒ / p)^(parameters.R / parameters.cp) * T
w = @. ρw ./ ρ
wθ = @. w .* θ
θθ = @. θ^2

interpolate_field!(new_θ, Array(θ), elist, ξlist, r, ω, (Nq, Nq, Nq); arch=CPU(), blocksize=16)
interpolate_field!(new_w, Array(w), elist, ξlist, r, ω, (Nq, Nq, Nq); arch=CPU(), blocksize=16)
interpolate_field!(new_wθ, Array(wθ), elist, ξlist, r, ω, (Nq, Nq, Nq); arch=CPU(), blocksize=16)
interpolate_field!(new_θθ, Array(θθ), elist, ξlist, r, ω, (Nq, Nq, Nq); arch=CPU(), blocksize=16)


fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = Axis(fig[1, 3])
θ̅ = mean(new_θ, dims=(1, 2))[1, 1, :]
w̅ = mean(new_w, dims=(1, 2))[1, 1, :]
avg_θθ = mean(new_θθ, dims=(1, 2))[1, 1, :]
avg_wθ = mean(new_wθ, dims=(1, 2))[1, 1, :]
lines!(ax1, θ̅, zlist, color=:blue)
lines!(ax1, θ₀.(zlist), zlist, color=:red)
scatter!(ax2, avg_wθ - θ̅ .* w̅, zlist)
scatter!(ax3, avg_θθ - θ̅ .* θ̅, zlist)
display(fig)

##
begin
    options = (; titlesize=30, ylabelsize=32,
        xlabelsize=32, xgridstyle=:dash, ygridstyle=:dash, xtickalign=1,
        xticksize=10, ytickalign=1, yticksize=10,
        xticklabelsize=10, yticklabelsize=10, ylabel="z [km]")

    fig_V = Figure(resolution=(1300, 800))
    volume_width = 3
    height_index = floor(Int, M / 10 * 4) # 120 for M = 300
    updraftquantile = 0.83 # 0.83

    ax = LScene(fig_V[1:8, 2:volume_width+1], scenekw=(camera=cam3d!, show_axis=true))
    stat_ax1 = Axis(fig_V[2:3, volume_width+2]; title="⟨θ⟩ [K]", options...)
    stat_ax2 = Axis(fig_V[4:5, volume_width+2]; title="⟨θ'w'⟩ [K m s⁻¹]", options...)
    stat_ax3 = Axis(fig_V[6:7, volume_width+2]; title="⟨θ'θ'⟩ [K²]", options...)

    cmap = :linear_kryw_0_100_c71_n256

    cmapa = reverse(RGBAf.(to_colormap(cmap), 1))
    cmap = vcat(fill(RGBAf(0, 0, 0, 0), floor(Int, 40 * (1 / (1 - updraftquantile) - 1))), cmapa[1:35])

    plot_state = new_θ[:, :, 1:height_index]

    clims = (quantile(plot_state[:], 0.00), mean(plot_state[:, :, 1]))

    v1 = volume!(ax, 0 .. 1, 0 .. 1, 0 .. 1, plot_state,
        colorrange=clims, algorithm=:absorption, absorption=10.0f0,
        colormap=cmap)

    axis = ax.scene[OldAxis]
    axis[:names, :axisnames] = ("x [km]", "y [km]", "z [km]")
    tstyle = axis[:names] #  get the nested attributes and work directly with them

    tstyle[:textsize] = 05
    tstyle[:textcolor] = (:black, :black, :black)
    tstyle[:font] = "helvetica"
    tstyle[:gap] = 10
    axis[:ticks][:textcolor] = :black
    axis[:ticks][:textsize] = 05
    cbar1 = Colorbar(fig_V[2:7, 1], v1, label="θ [K]", width=25, ticklabelsize=20,
        labelsize=20, ticksize=25, tickalign=1, height=Relative(3 / 4)
    )

    axis[:ticks][:ranges] = ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    axis[:ticks][:labels] = (["-1.5", "0", "1.5"], ["-1.5", "0", "1.5"], ["0", "0.7", "1.4"])

    line_options = (; linewidth=3, color=cmapa[end-4])
    lines!(stat_ax1, θ̅, zlist ./ 1e3; line_options...)
    xlims!(stat_ax1, (303, 306))
    ylims!(stat_ax1, (0.0, 2.0))
    lines!(stat_ax2, avg_wθ - θ̅ .* w̅, zlist ./ 1e3; line_options...)
    ylims!(stat_ax2, (0.0, 2.0))
    xlims!(stat_ax2, (-0.03, 0.10))
    lines!(stat_ax3, avg_θθ - θ̅ .* θ̅, zlist ./ 1e3; line_options...)
    ylims!(stat_ax3, (0.0, 2.0))
    rotate_cam!(fig_V.scene.children[1], (π / 16, 0, 0))
    display(fig_V)
end

wpthp = avg_wθ - θ̅ .* w̅;
nothing
##
#=
begin
filename = "convection_visualization_more_rez.h5"
fid = h5open(filename, "w")
fid["wpthp"] = avg_wθ - θ̅ .* w̅
fid["thpthp"] = avg_θθ - θ̅ .* θ̅
fid["thm"] = θ̅
fid["th3d"] = new_θ
fid["z"] = collect(zlist)
fid["x"] = collect(xlist)
fid["y"] = collect(ylist)
fid["dt"] = dt
fid["minnode_dist"] = min_node_distance(grid)
fid["cfl"] = cfl
close(fid)
end
=#


