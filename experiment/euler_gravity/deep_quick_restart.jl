
statistic_counter = 1
state .= stable_state
test_state .= stable_state

aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, test_state)
fmvar .= mean_variables.(Ref(law), test_state, aux)
smvar .= second_moment_variables.(fmvar)

partitions = 1.0:1.0:172800.0
for i in partitions
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_fs.auxstate .= aux
    dg_sd.auxstate .= aux
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
    end_time = 60 * 60 * 24 * endday / partitions[end] 
    state .= test_state
    solve!(test_state, end_time, odesolver, adjust_final=false) # otherwise last step is wrong since linear solver isn't updated
    # current reference state α = 1.0
    # midpoint type extrapolation: α = 1.5
    # backward euler type extrapolation: α = 2.0
    α = 1.5
    state .= α * (test_state) + (1 - α) * state

    timeend = odesolver.time
    global current_time += timeend

    if statistic_save & (current_time / 86400 > 200) & (i % 15 == 0)
        println("gathering statistics at time ", current_time / 86400)
        global statistic_counter += 1.0
        global fmvartmp .= mean_variables.(Ref(law), test_state, aux)
        global smvar .+= second_moment_variables.(fmvartmp)
        global fmvar .+= mean_variables.(Ref(law), test_state, aux)
        println("statistic counter ", statistic_counter)
    end

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
        hs_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), test_state, aux)
        hs_density = components(test_state)[1]
        hs_soundspeed = Atum.EulerTotalEnergy.soundspeed.(Ref(law), hs_density, hs_pressure)
        speed = @. sqrt(u^2 + v^2 + w^2)
        c_max = maximum(hs_soundspeed)
        mach_number = maximum(speed ./ hs_soundspeed)
        println("The maximum soundspeed is ", c_max)
        println("The largest mach number is ", mach_number)
        println(" the vertical cfl is ", dt * c_max / Δz)
        println(" the horizontal cfl is ", dt * c_max / Δx)
        println("The dt is now ", dt)
        println("The current day is ", current_time / 86400)
        ρ̅ = sum(ρ .* dg_fs.MJ) / sum(dg_fs.MJ)
        println("The average density of the system is ", ρ̅)
        toc = time()
        println("The runtime for the simulation is ", (toc - tic) / 60, " minutes")

    if isnan(ρ[1]) | isnan(ρu[1]) | isnan(ρv[1]) | isnan(ρw[1]) | isnan(ρet[1]) | isnan(ρ̅)
        println("The simulation NaNed, decreasing timestep and using stable state")
        local i = save_partition
        global current_time = save_time
        test_state .= stable_state
        state .= stable_state
        global dt *= 0.9

        global statistic_counter = 1
        aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, test_state)
        fmvar .= mean_variables.(Ref(law), test_state, aux)
        smvar .= second_moment_variables.(fmvar)
    else
        if (abs(minuʳ) + abs(maxuʳ)) < 20.0
            println("creating backup state")
            stable_state .= test_state
            global save_partition = i
            global save_time = current_time
        end
    end
        println("-----")
    end
end
##
toc = time()
println("The time for the simulation is ", toc - tic)
tmp_ρ = components(test_state)[1]
ρ̅_end = sum(tmp_ρ .* dg_fs.MJ) / sum(dg_fs.MJ)

# normalize statistics
fmvar .*= 1 / statistic_counter
smvar .*= 1 / statistic_counter

state .*= 1 / averaging_counter

println("The conservation of mass error is ", (ρ̅_start - ρ̅_end) / ρ̅_end)

gpu_components = components(test_state)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end
statenames = ("ρ", "ρu", "ρv", "ρw", "ρe")

filepath = "HeldSuarezDeepStretched_" * "Nev" * string(Kv) * "_Neh" * string(Kh) * "_Nq" * string(Nq⃗[1]) * ".jld2"
file = jldopen(filepath, "a+")
JLD2.Group(file, "state")
JLD2.Group(file, "averagedstate")
JLD2.Group(file, "grid")
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

file["grid"]["vertical_coordinate"] = vert_coord
file["grid"]["gauss_lobatto_points"] = Nq⃗
file["grid"]["vertical_element_number"] = Kv
file["grid"]["horizontal_element_number"] = Kh

close(file)