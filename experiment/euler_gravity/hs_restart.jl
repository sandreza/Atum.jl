
vf = (KennedyGruberFlux(), KennedyGruberFlux(), KennedyGruberFlux())
sf = (RoeFlux(), RoeFlux(),  Atum.RefanovFlux(1.0))

linearized_vf = Atum.LinearizedKennedyGruberFlux()
linearized_sf = Atum.LinearizedRefanovFlux(1.0)


# vf = (KennedyGruberFlux(), KennedyGruberFlux(), CentralFlux())
# sf = (RoeFlux(), RoeFlux(),  Atum.CentralRefanovFlux(1.0))

# linearized_vf = Atum.LinearizedCentralFlux()
# linearized_sf = Atum.LinearizedCentralRefanovFlux(1.0)


dg_sd = SingleDirection(; law, grid, volume_form=linearized_vf, surface_numericalflux=linearized_sf)
dg_fs = FluxSource(; law, grid, volume_form=vf, surface_numericalflux=sf)
dt = 150.0
test_state .= stable_state
save_stable_state = true
partitions = 1:48*endday  # 1:24*endday*3
# current_time = 200 * 86400
# save_time = 200 * 86400

tic = time()
for i in partitions
    aux = sphere_auxiliary_3.(Ref(law), x⃗, test_state)
    dg_fs.auxstate .= aux
    dg_sd.auxstate .= aux
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
    end_time = 60 * 60 * 24 * endday / partitions[end] 
    solve!(test_state, end_time, odesolver, adjust_final=false) # otherwise last step is wrong since linear solver isn't updated
    timeend = odesolver.time
    if i % display_skip == 0
        println("--------")
        println("done with ", display_skip * timeend / 60, " minutes")
        println("partition ", i, " out of ", partitions[end])
        local ρ, ρu, ρv, ρw, ρet = components(test_state)
        u = ρu ./ ρ
        v = ρv ./ ρ
        w = ρw ./ ρ
        println("maximum x-velocity ", maximum(u))
        println("maximum y-velocity ", maximum(v))
        println("maximum z-velocity ", maximum(w))
        uʳ = @. (xp * u + yp * v + zp * w) / sqrt(xp^2 + yp^2 + zp^2)
        minuʳ = minimum(uʳ)
        maxuʳ = maximum(uʳ)
        println("extrema vertical velocity ", (minuʳ, maxuʳ))

        bw_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), test_state, aux)
        bw_density = components(test_state)[1]
        bw_soundspeed = Atum.EulerTotalEnergy.soundspeed.(Ref(law), bw_density, bw_pressure)
        speed = @. sqrt(u^2 + v^2 + w^2)
        c_max = maximum(bw_soundspeed)
        mach_number = maximum(speed ./ bw_soundspeed)
        println("The maximum soundspeed is ", c_max)
        println("The largest mach number is ", mach_number)
        println("The dt is now ", dt)
        global current_time += display_skip * timeend
        println("The current day is ", current_time / 86400)
        ρ̅ = sum(ρ .* dg_fs.MJ) / sum(dg_fs.MJ)
        println("The average density of the system is ", ρ̅)
        toc = time()
        println("The runtime for the simulation is ", (toc - tic) / 60, " minutes")

        if isnan(ρ[1]) | isnan(ρu[1]) | isnan(ρv[1]) | isnan(ρw[1]) | isnan(ρet[1])
            println("The simulation NaNed, decreasing timestep and using stable state")
            local i = save_partition
            global current_time = save_time
            test_state .= stable_state
            global dt *= 0.9
        else
            if save_stable_state & ((abs(minuʳ) + abs(maxuʳ)) < 2.0)
                println("creating backup state")
                stable_state .= test_state
                global save_partition = i
                global save_time = current_time
            end

            if current_time / 86400 >= 200
                println("averaging as well ")
                state .+= stable_state
            end
        end
        println("-----")
    end
end
toc = time()
println("The time for the simulation is ", toc - tic)
tmp_ρ = components(test_state)[1]
ρ̅_end = sum(tmp_ρ .* dg_fs.MJ) / sum(dg_fs.MJ)

println("The conservation of mass error is ", (ρ̅_start - ρ̅_end) / ρ̅_end)

#=
averaging_scale = round(Int,state[1][1] / stable_state[1][1])
state ./= averaging_scale

gpu_components = components(test_state)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end
statenames = ("ρ", "ρu", "ρv", "ρw", "ρe")

filepath = "HeldSuarez_" * "Nev" * string(Kv) * "_Neh" * string(Kv) * "_Nq" * string(Nq⃗[1]) * ".jld2"
file = jldopen(filepath, "a+")
JLD2.Group(file, "state")
JLD2.Group(file, "averagedstate")
for (i, statename) in enumerate(statenames)
    file["state"][statename] = cpu_components[i]
end

gpu_components = components(state)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end

for (i, statename) in enumerate(statenames)
    file["averagedstate"][statename] = cpu_components[i]
end
close(file)
=#