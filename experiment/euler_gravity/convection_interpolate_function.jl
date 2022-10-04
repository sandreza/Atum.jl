println("computing interpolate function")
include("interpolate.jl")
r = Tuple([cpu_cell.points_1d[i][:] for i in eachindex(cpu_cell.points_1d)])
ω = Tuple([baryweights(cpu_cell.points_1d[i][:]) for i in eachindex(cpu_cell.points_1d)])

M = 200
x, y, z = components(grid.points)
xlist = range(minimum(x), maximum(x), length=M)
ylist = range(minimum(y), maximum(y), length=M)
zlist = range(minimum(z), maximum(z), length=M)

newgrid = [SVector(x, y, z) for x in xlist, y in ylist, z in zlist]

ξlist, elist = cube_interpolate(newgrid, cpu_grid, arch=CPU())

function compute_field_averages(newgrid, ξlist, elist, r, ω, q, grid)
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

    θ̅ = mean(new_θ, dims=(1, 2))[:]
    w̅ = mean(new_w, dims=(1, 2))[:]
    avg_θθ = mean(new_θθ, dims=(1, 2))[:]
    avg_wθ = mean(new_wθ, dims=(1, 2))[:]
    return θ̅, w̅, avg_θθ, avg_wθ
end

