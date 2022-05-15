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

newc = zeros(size(newgrid))
oldc = cpu_components[end]
interpolate_field_2D!(newc, oldc, elist, ξlist, r, ω, (Nq, Nq); arch=CPU(), blocksize=16)

newρ = zeros(size(newgrid))
oldρ = cpu_components[1]
interpolate_field_2D!(newρ, oldρ, elist, ξlist, r, ω, (Nq, Nq); arch=CPU(), blocksize=16)

## 
fig = Figure()
ax = Axis(fig[1, 1]; title = "tracer")
ax2 = Axis(fig[1, 2]; title = "density")
colorrange = (0, 1)
heatmap!(ax, xlist, ylist, newc, colormap=:balance, colorrange=colorrange)
ylims!(ax, (0, 1))

colorrange2 = (1, 2)
heatmap!(ax2, xlist, ylist, newρ, colormap= Reverse(:balance), colorrange=colorrange2)
ylims!(ax2, (0, 1))

display(fig)

