

test_state .= stable_state
dt = 0.1 * Δx / 353

function sphere_auxiliary_2(law::EulerTotalEnergyLaw, hs_p, x⃗, state)
    ix_ρ, ix_ρu⃗, ix_ρe = Atum.EulerTotalEnergy.varsindices(law)
    ix_x, ix_y, ix_z, _, _, _, _ = Atum.EulerTotalEnergy.auxindices(law)
    r = sqrt(x⃗' * x⃗)
    r̂ = x⃗ ./ r
    tmp = state[ix_ρu⃗] # get rid of rapidly fluctuating vertical component
    ϕ = geo(hs_p, r)
    @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], state[ix_ρ], tmp..., state[ix_ρe], ϕ)
end

function sphere_auxiliary_3(law::EulerTotalEnergyLaw, hs_p, x⃗, state)
    ix_ρ, ix_ρu⃗, ix_ρe = Atum.EulerTotalEnergy.varsindices(law)
    ix_x, ix_y, ix_z, _, _, _, _ = Atum.EulerTotalEnergy.auxindices(law)
    r = sqrt(x⃗' * x⃗)
    r̂ = x⃗ ./ r
    tmp = 0 * state[ix_ρu⃗] # get rid of rapidly fluctuating vertical component
    ϕ = geo(hs_p, r)
    @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], state[ix_ρ], tmp..., state[ix_ρe], ϕ)
end

function sphere_auxiliary_4(law::EulerTotalEnergyLaw, hs_p, x⃗, state)
    ix_ρ, ix_ρu⃗, ix_ρe = Atum.EulerTotalEnergy.varsindices(law)
    ix_x, ix_y, ix_z, _, _, _, _ = Atum.EulerTotalEnergy.auxindices(law)
    r = sqrt(x⃗' * x⃗)
    r̂ = x⃗ ./ r
    tmp = (I - 2 * r̂ * (r̂')) * state[ix_ρu⃗] # get rid of rapidly fluctuating vertical component
    ϕ = geo(hs_p, r)
    @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], state[ix_ρ], tmp..., state[ix_ρe], ϕ)
end

function remove_vert(law::EulerTotalEnergyLaw, hs_p, x⃗, state)
    ix_ρ, ix_ρu⃗, ix_ρe = Atum.EulerTotalEnergy.varsindices(law)
    ix_x, ix_y, ix_z, _, _, _, _ = Atum.EulerTotalEnergy.auxindices(law)
    r = sqrt(x⃗' * x⃗)
    r̂ = x⃗ ./ r
    tmp = (I -  1 * r̂ * (r̂')) * state[ix_ρu⃗] # get rid of rapidly fluctuating vertical component
    @inbounds SVector(state[ix_ρ], tmp..., state[ix_ρe])
end

dg_sd = SingleDirection(; law, grid, volume_form=linearized_vf, surface_numericalflux=linearized_sf)
dg_fs = FluxSource(; law, grid, volume_form=vf, surface_numericalflux=sf)
aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, test_state)
dg_fs.auxstate .= aux
dg_sd.auxstate .= aux

state .= test_state
skippy = 8
totes_its = 100

tic = time()
for i in 1:totes_its
    println("step ", i)

    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_fs.auxstate .= aux
    dg_sd.auxstate .= aux
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=true)


    dothisthing = (x...) -> nothing
    if i == 1
        state .= test_state
        Atum.dostep!(test_state, odesolver, dothisthing)
    else
        for j in 1:skippy
            if j == div(skippy,2)
                state .= test_state
            end
            Atum.dostep!(test_state, odesolver, dothisthing)
        end
    end
    α = 1.5
    state .= α * (test_state) + (1-α) * state

    # state .+= test_state
    
    begin
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
        ρ̅ = sum(ρ .* dg_fs.MJ) / sum(dg_fs.MJ)
        println("The average density of the system is ", ρ̅)
        println("The maximum density is ", maximum(ρ))
    end
    
end
toc = time()
println("The time for the simulation is ", toc - tic, " seconds")
println("The ratio is ", dt * totes_its * skippy / (toc - tic)  )