state = fieldarray(undef, law, grid)
test_state = fieldarray(undef, law, grid)
stable_state = fieldarray(undef, law, grid)
old_state = fieldarray(undef, law, grid)
cpu_state = fieldarray(undef, law, cpu_grid)
cpu_state .= held_suarez_init.(cpu_x⃗, Ref(hs_p))
gpu_components = components(state)
cpu_components = components(cpu_state)
for i in eachindex(gpu_components)
    gpu_components[i] .= A(cpu_components[i])
end

test_state .= state
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
α = 1.5 # 1.5
state .= α * (test_state) + (1 - α) * state

timeend = odesolver.time
global current_time += timeend

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
    toc = Base.time()
    println("The runtime for the simulation is ", (toc - tic) / 60, " minutes")