totes_sim = 4 * 5 * 400
u_timeseries = []
T_timeseries = []
v_timeseries = []
rho_timeseries = []
w_timeseries = []
u = Array(meanlist[2][:, :, 2])
for i in 1:totes_sim
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = 5 * dt
    solve!(test_state, end_time, odesolver, adjust_final=false)
    # ρ, u, v, w, p, T
    fmvar .= mean_variables.(Ref(law), test_state, aux)
    # save u
    oldf = components(fmvar)[2]
    newf = meanlist[2]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    u .= Array(newf[:, :, 2])
    push!(u_timeseries, copy(u))
    # save T 
    oldf = components(fmvar)[end]
    newf = meanlist[end]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    T = Array(newf[:, :, 2])
    push!(T_timeseries, T)
    # save v
    oldf = components(fmvar)[3]
    newf = meanlist[3]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    v = Array(newf[:, :, 2])
    push!(v_timeseries, v)
    # save ρ 
    oldf = components(fmvar)[1]
    newf = meanlist[1]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    ρ = Array(newf[:, :, 2])
    push!(rho_timeseries, ρ)
    # save w
    oldf = components(fmvar)[4]
    newf = meanlist[4]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    w = Array(newf[:, :, 2])
    push!(w_timeseries, w)

    if i % 10 == 0
        println("currently at timestep ", i, " out of ", totes_sim)
    end
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
surface!(ax_v, x, y, z, color=@lift(v_timeseries[$time_index]), colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_rho, x, y, z, color=@lift(rho_timeseries[$time_index]), colormap=:bone_1, colorrange=(1.06, 1.2), shading=false)

rotation = (0 * π / 5, π / 6, 0)
rotate_cam!(ax.scene, rotation)

##
using HDF5
filename = "even_high_rez_hs.h5"
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

#=
rlist = range(vert_coord[1] + 000, vert_coord[end], length=15 * scale)
θlist = range(-π, π, length=90 * scale)
ϕlist = range(0, π, length=45 * scale)
=#