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
gathermeanlist = copy(meanlist)
gathersecondlist = copy(secondlist)

end_time = 60 * 60 * 24 * endday / partitions[end]

listostuff = []
tlist = []
global current_time = 0.0
endindex = 20000 * 3 * 3

println("starting the simulation")
state .= test_state
for i in 1:endindex
    #=
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_fs.auxstate .= aux
    dg_sd.auxstate .= aux
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
        solve!(test_state, end_time, odesolver, adjust_final=false) # otherwise last step is wrong since linear solver isn't updated
    =#
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = 60 * 60 * 24 * endday / partitions[end]
    state .= test_state
    solve!(test_state, end_time, odesolver, adjust_final=false)


    # current reference state α = 1.0
    # midpoint type extrapolation: α = 1.5
    # backward euler type extrapolation: α = 2.0
    α = 1.5 # 1.5
    state .= α * (test_state) + (1 - α) * state

    timeend = odesolver.time
    global current_time += timeend

    if i % 400 == 0
        println("saving at ", current_time)
        println("this is iteration ", i)
        println("the simulation is ", i / endindex * 100.0, " percent complete")
        global fmvar .= mean_variables.(Ref(law), test_state, aux)

        for (newf, oldf) in zip(meanlist, meanoldlist)
            interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
        end
        local u = meanlist[2]
        push!(listostuff, (sum(Array(u), dims=1)[1, :, :]./length(θlist))[floor(Int, length(ϕlist) / 2), :])
        push!(tlist, current_time)
    end
end

ntimes = length(listostuff)
qbo = zeros(ntimes, length(rlist))
for i in 1:ntimes
    qbo[i, :] = listostuff[i]
end

using GLMakie

fig = Figure()
ax = Axis(fig[1,1]; xlabel = "time", ylabel = "height")
heatmap!(ax, [tlist...] ./ (60 * 60 * 20), (collect(rlist) .- rlist[1]) ./ 1e3,  qbo, colormap=:balance, colorrange=(-10, 10), interpolate = true)


