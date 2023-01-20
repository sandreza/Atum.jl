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
old2 = components(qc)[end]
interpolate_field_2D!(new2, old2, elist, ξlist, r, ω, (Nq, Nq); arch=CPU(), blocksize=16)


## 
fig = Figure(resolution = (1000, 1000))
ax = Axis(fig[1, 1]; title="tracer t = 3.2")

colorrange = (0, 1)
heatmap!(ax, xlist, ylist, new2, colormap=:balance, colorrange=colorrange, interpolate=true)
ylims!(ax, (0, 1))

save("instability_low_rez.png", fig)