using GLMakie
println("precomputing interpolation data")
#=
scale = 4
rlist = range(vert_coord[1] + 000, vert_coord[end], length=15 * scale)
θlist = range(-π, π, length=90 * scale)
ϕlist = range(0, π, length=45 * scale)

xlist = [sphericaltocartesian(θ, ϕ, r) for θ in θlist, ϕ in ϕlist, r in rlist]

elist = zeros(Int, length(xlist))
ξlist = [SVector(0.0, 0.0, 0.0) for i in eachindex(xlist)]
for kk in eachindex(xlist)
    x = xlist[kk]
    (x̂, cell_coords) = Bennu.cubedspherereference(x, vert_coord, Kh)
    elist[kk] = cube_single_element_index(cell_coords, Kv, Kh)
    ξlist[kk] = SVector(x̂)
end
println("done precomputing interpolation data")
d_elist = A(elist)
d_ξlist = A(ξlist)
r = Tuple([cell.points_1d[i][:] for i in eachindex(cell.points_1d)])
ω = Tuple([A(baryweights(cell.points_1d[i][:])) for i in eachindex(cell.points_1d)])

oldlist = components(test_state)

meanoldlist = components(fmvar)
secondoldlist = components(smvar)
meanlist = [A(zeros(size(xlist))) for i in 1:length(meanoldlist)]
secondlist = [A(zeros(size(xlist))) for i in 1:length(secondoldlist)]
=#

@show "done computing interpolation data"

readfile = false 
if readfile 
    @show "reading file"
    ctest_state = components(test_state)
    tmp = h5open("restart_SmallHeldSuarezStatistics_Nev6_Neh12_Nq1_5_Nq2_5_Nq3_5_X_20.0.h5", "r")
    for i in eachindex(ctest_state)
        ctest_state[i] .= A(read(tmp["state_$i"]))
    end
    close(tmp)
end

totes_sim = 50
u_timeseries = []
T_timeseries = []
v_timeseries = []
rho_timeseries = []
w_timeseries = []
p_timeseries = []
θ_timeseries = []
u = Array(meanlist[2][:, :, 2])
height_index = 3
for i in ProgressBar(1:totes_sim)
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = 100 * dt
    solve!(test_state, end_time, odesolver, adjust_final=false)

    # ρ, u, v, w, p, T
    fmvar .= mean_variables.(Ref(law), test_state, aux)
    # save u
    oldf = components(fmvar)[2]
    newf = meanlist[2]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    u .= Array(newf[:, :, height_index])
    push!(u_timeseries, copy(u))
    # save T 
    oldf = components(fmvar)[end]
    newf = meanlist[end]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    T = Array(newf[:, :, height_index])
    push!(T_timeseries, T)
    # save v
    oldf = components(fmvar)[3]
    newf = meanlist[3]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    v = Array(newf[:, :, height_index])
    push!(v_timeseries, v)
    # save ρ 
    oldf = components(fmvar)[1]
    newf = meanlist[1]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    ρ = Array(newf[:, :, height_index])
    push!(rho_timeseries, ρ)
    # save p 
    oldf = components(fmvar)[5]
    newf = meanlist[5]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    p = Array(newf[:, :, height_index])
    push!(p_timeseries, p)
    # save θ
    push!(θ_timeseries, T .*( (p ./ 1e5) .^(-hs_p.R_d/hs_p.cp_d) ) )
    # save w
    oldf = components(fmvar)[4]
    newf = meanlist[4]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    w = Array(newf[:, :, height_index])
    push!(w_timeseries, w)
end

fig = Figure()
ax = Axis(fig[1, 1]; title="Zonal Velocity")
ax_T = Axis(fig[1, 2]; title="Temperature")
time_slider = Slider(fig[2, 1:2], range=1:totes_sim, startvalue=1, horizontal=true)
time_index = time_slider.value
u_plot = @lift(u_timeseries[$time_index])
T_plot = @lift(T_timeseries[$time_index])
heatmap!(ax, θlist, ϕlist, u_plot, colormap=:balance, colorrange=(-30, 30), interpolate=true)
heatmap!(ax_T, θlist, ϕlist, T_plot, colormap=:afmhot, colorrange=(270, 304), interpolate=true)



x = [sin(ϕ) * cos(θ) for θ in θlist, ϕ in ϕlist]
y = [sin(ϕ) * sin(θ) for θ in θlist, ϕ in ϕlist]
z = [cos(ϕ) for θ in θlist, ϕ in ϕlist]

fig = Figure()
ax_T = LScene(fig[1, 1], show_axis=false)
ax_u = LScene(fig[1, 2], show_axis=false)
ax_v = LScene(fig[2, 1], show_axis=false)
ax_rho = LScene(fig[2, 2], show_axis=false)
time_slider = Slider(fig[3, 1:2], range=1:totes_sim, startvalue=1, horizontal=true)
time_index = time_slider.value
surface!(ax_T, x, y, z, color=@lift(T_timeseries[$time_index]), colormap=:afmhot, colorrange=(270, 304), shading=false)
surface!(ax_u, x, y, z, color=@lift(u_timeseries[$time_index]), colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_v, x, y, z, color=@lift(w_timeseries[$time_index]), colormap=:balance, colorrange=(-2, 2), shading=false)
surface!(ax_rho, x, y, z, color=@lift(rho_timeseries[$time_index]), colormap=:bone_1, colorrange=(1.06, 1.2), shading=false)

# rotation = (0 * π / 5, π / 6, 0)
# rotate_cam!(ax.scene, rotation)

wzonalbands = zeros(360, 180, totes_sim)
θzonalbands = zeros(360, 180, totes_sim)
for i in 1:totes_sim
    wzonalbands[:, :, i] = w_timeseries[i][:, :, :]
    θzonalbands[:, :, i] = θ_timeseries[i][:, :, :]
end
histfig = Figure()
ax1 = Axis(histfig[1, 1])
hist!(ax1, wzonalbands[:, 90, :][:], bins=100, color=:blue, alpha=0.5, label="90")
hist!(ax1, wzonalbands[:, 45, :][:], bins=100, color=:red, alpha=0.5, label="76")
hist!(ax1, wzonalbands[:, 60, :][:], bins=100, color=:green, alpha=0.5, label="60")
# xlims!(ax1, (-1, 1))

for i in 1:5:91
    μ = mean(wzonalbands[:, i, :][:])
    σ = std(wzonalbands[:, i, :][:])
    median_μ = median(wzonalbands[:, i, :][:])
    skewness1 = mean((wzonalbands[:, i, :][:] .- μ) .^ 3) / σ^3
    skewness2 = 3 * (μ - median_μ) / σ
    println("-----")
    println("for i = $i")
    println("mean: $μ, std: $σ, median: $median_μ, skewness1: $skewness1, skewness2: $skewness2")
    println("------")
end

θ̄ = mean(θzonalbands[:, 90, :][:])
w̄ = mean(wzonalbands[:, 90, :][:])
wθ = [Tuple([wzonalbands[j, 90, i] - w̄, θzonalbands[j, 90, i] - θ̄]) for j in 1:360, i in 1:(totes_sim)]

wzonalbands[j, 90, i] - w̄, θzonalbands[j, 90, i] - θ̄]
##
#=
using HDF5
filename = "zonalbandw_small_planet.h5"
fid = h5open(filename, "w")
fid["wzonalband"] = wzonalbands
close(fid)
=#
#=
filepath = "restart_" * "SmallHeldSuarezStatistics_" * "Nev" * string(Kv) * "_Neh" * string(Kh)
filepath = filepath * "_Nq1_" * string(Nq⃗[1]) * "_Nq2_" * string(Nq⃗[2])
filepath = filepath * "_Nq3_" * string(Nq⃗[3]) * "_X_" * string(X) * ".h5"

fid = h5open(filepath, "w")
dustates = components(test_state)
for i in eachindex(dustates)
    fid["state_$i"] = Array(dustates[i])
end
close(fid)
=#

#=
create_group(fid, "T")
create_group(fid, "rho")
create_group(fid, "u")
create_group(fid, "v")
create_group(fid, "grid")
fid["grid"]["θlist"] = collect(θlist)
fid["grid"]["ϕlist"] = collect(ϕlist)
fid["grid"]["rlist"] = collect(rlist)
tic = time()
for i in eachindex(T_timeseries)
    fid["T"][string(i)] = T_timeseries[i]
    fid["rho"][string(i)] = rho_timeseries[i]
    fid["u"][string(i)] = u_timeseries[i]
    fid["v"][string(i)] = v_timeseries[i]
    toc = time()
    if toc - tic > 1
        println("currently at timestep ", i, " out of ", length(T_timeseries))
        tic = toc
    end
end
close(fid)
=#

##
#=
using HDF5
filename = "even_high_rez_hs_small.h5"
fid = h5open(filename, "w")
create_group(fid, "T")
create_group(fid, "rho")
create_group(fid, "u")
create_group(fid, "v")
create_group(fid, "grid")
fid["grid"]["θlist"] = collect(θlist)
fid["grid"]["ϕlist"] = collect(ϕlist)
fid["grid"]["rlist"] = collect(rlist)
tic = time()
for i in eachindex(T_timeseries)
    fid["T"][string(i)] = T_timeseries[i]
    fid["rho"][string(i)] = rho_timeseries[i]
    fid["u"][string(i)] = u_timeseries[i]
    fid["v"][string(i)] = v_timeseries[i]
    toc = time()
    if toc - tic > 1
        println("currently at timestep ", i, " out of ", length(T_timeseries))
        tic = toc
    end
end
close(fid)
=#