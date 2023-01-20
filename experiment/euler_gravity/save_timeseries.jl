using HDF5, ProgressBars

println("precomputing interpolation data")
scale = 8
rlist = [vert_coord[1]]# range(vert_coord[1] + 000, vert_coord[end], length=15 * scale)
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

meanlist = [A(zeros(size(xlist))) for i in 1:length(components(fmvar))]
##
totes_sim = 3000

##
u_timeseries = zeros(length(θlist), length(ϕlist), totes_sim)
T_timeseries = copy(u_timeseries)
v_timeseries = copy(u_timeseries)
rho_timeseries = copy(u_timeseries)


for i in ProgressBar(1:totes_sim)
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = 86400 / 40 / 4 # basically every 6 hours
    solve!(test_state, end_time, odesolver, adjust_final=false)
    # ρ, u, v, w, p, T
    fmvar .= mean_variables.(Ref(law), test_state, aux)
    # save u
    oldf = components(fmvar)[2]
    newf = meanlist[2]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    u_timeseries[:, :, i] .= Array(newf[:, :, 1])

    # save T 
    oldf = components(fmvar)[end]
    newf = meanlist[end]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    T_timeseries[:, :, i] .= Array(newf[:, :, 1])

    # save v
    oldf = components(fmvar)[3]
    newf = meanlist[3]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    v_timeseries[:, :, i] .= Array(newf[:, :, 1])

    # save ρ 
    oldf = components(fmvar)[1]
    newf = meanlist[1]
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
    rho_timeseries[:, :, i] .= Array(newf[:, :, 1])

end


##
filename = "small_planet_hs_high_rez.h5"
fid = h5open(filename, "w")
fid["u"] = u_timeseries
fid["v"] = v_timeseries
fid["rho"] = rho_timeseries
fid["T"] = T_timeseries
close(fid)
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

x = [sin(ϕ) * cos(θ) for θ in θlist, ϕ in ϕlist]
y = [sin(ϕ) * sin(θ) for θ in θlist, ϕ in ϕlist]
z = [cos(ϕ) for θ in θlist, ϕ in ϕlist]

fig = Figure(resolution=(2000, 1000))
ax_T = LScene(fig[1, 1], show_axis=false)
ax_u = LScene(fig[1, 2], show_axis=false)
ax_v = LScene(fig[2, 1], show_axis=false)
ax_rho = LScene(fig[2, 2], show_axis=false)
time_slider = Slider(fig[3, 1:2], range=1:totes_sim, startvalue=1, horizontal=true)
time_index = time_slider.value
surface!(ax_T, x, y, z, color=@lift(T_timeseries[:, :, $time_index]), colormap=:thermometer, colorrange=(270, 304), shading=false)
surface!(ax_u, x, y, z, color=@lift(u_timeseries[:, :, $time_index]), colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_v, x, y, z, color=@lift(v_timeseries[:, :, $time_index]), colormap=:balance, colorrange=(-30, 30), shading=false)
surface!(ax_rho, x, y, z, color=@lift(rho_timeseries[:, :, $time_index]), colormap=:bone_1, colorrange=(1.1, 1.27), shading=false)

rotation = (π / 5, π / 6, 0)
rotation = (π / 16, π / 6, 0)
for ax in [ax_T, ax_u, ax_v, ax_rho]
    rotate_cam!(ax.scene, rotation)
end

##
framerate = 3 * 30
timestamps = 1:totes_sim

function record_the_scene()
    GLMakie.record(fig, "time_animation_hs_hgih_rez.mp4", timestamps;
        framerate=framerate) do t
        time_index[] = t
        nothing
    end;
    return nothing
end

##
function change_function(time)
    time_index[] = time
end

GLMakie.record(change_function, fig, "test_animation.mp4", timestamps; framerate=framerate)
