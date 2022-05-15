##
using LinearAlgebra, GLMakie
include("interpolate_2D.jl")

# First check the cube
x⃗ = points(grid)
x, z = components(x⃗)
##
gpu_components = components(q)
cpu_components = components(qc)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end

##
r = Tuple([cpucell.points_1d[i][:] for i in eachindex(cpucell.points_1d)])
ω = Tuple([baryweights(cpucell.points_1d[i][:]) for i in eachindex(cpucell.points_1d)])

xlist = range(0, 1, length=K * (Nq + 1))
ylist = range(0, 2, length=K * (Nq + 1))
newgrid = [SVector(x, y) for x in xlist, y in ylist]

ξlist, elist = cube_interpolate_2D(newgrid, cpugrid, arch=CPU())

new2 = zeros(size(newgrid))
old2 = components(q2)[end]
interpolate_field_2D!(new2, old2, elist, ξlist, r, ω, (Nq, Nq); arch=CPU(), blocksize=16)

new4 = zeros(size(newgrid))
old4 = components(q4)[end]
interpolate_field_2D!(new4, old4, elist, ξlist, r, ω, (Nq, Nq); arch=CPU(), blocksize=16)

new6 = zeros(size(newgrid))
old6 = components(q6)[end]
interpolate_field_2D!(new6, old6, elist, ξlist, r, ω, (Nq, Nq); arch=CPU(), blocksize=16)

## 
fig = Figure(resolution = (1500, 500))
ax = Axis(fig[1, 1]; title="tracer t = 2")
ax2 = Axis(fig[1, 2]; title="tracer t = 4")
ax3 = Axis(fig[1, 3]; title="tracer t = 6")

colorrange = (0, 1)
heatmap!(ax, xlist, ylist, new2, colormap=:balance, colorrange=colorrange, interpolate=true)
ylims!(ax, (0, 1))

colorrange2 = (0, 1)
heatmap!(ax2, xlist, ylist, new4, colormap=:balance, colorrange=colorrange2, interpolate=true)
ylims!(ax2, (0, 1))

colorrange3 = (0, 1)
heatmap!(ax3, xlist, ylist, new6, colormap=:balance, colorrange=colorrange2, interpolate=true)
ylims!(ax3, (0, 1))

display(fig)

