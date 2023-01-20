tscale = 9
endscale = 8 # 8 is 1 day
timeend = 60 * 60 * 3 / tscale
dt = 400.0
# dt = 200 and timeend = 60 * 60 and i = 1:3 produces the bad state,

function sphere_auxiliary_3(law::EulerTotalEnergyLaw, x⃗, state)
    ix_ρ, ix_ρu⃗, ix_ρe = Atum.EulerTotalEnergy.varsindices(law)
    ix_x, ix_y, ix_z, _, _, _, _ = Atum.EulerTotalEnergy.auxindices(law)
    r = sqrt(x⃗' * x⃗)
    r̂ = x⃗ ./ r
    tmp = (I - r̂ * (r̂')) * state[ix_ρu⃗]
    @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], state[ix_ρ], tmp..., state[ix_ρe], 9.81 * r)
end

function sphere_auxiliary_4(law::EulerTotalEnergyLaw, x⃗, state)
    ix_ρ, ix_ρu⃗, ix_ρe = Atum.EulerTotalEnergy.varsindices(law)
    ix_x, ix_y, ix_z, _, _, _, _ = Atum.EulerTotalEnergy.auxindices(law)
    r = sqrt(x⃗' * x⃗)
    r̂ = x⃗ ./ r
    tmp = r̂ * (r̂' * state[ix_ρu⃗])
    @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], state[ix_ρ], tmp..., state[ix_ρe], 9.81 * r)
end

do_output = function (step, time, q)
    if step % ceil(Int, timeend / 50 / dt) == 0
        println("simulation is ", time / timeend * 100, " percent complete")
        ρ, ρu, ρv, ρw, _ = components(q)
        println("maximum x-velocity ", maximum(ρu ./ ρ))
        println("maximum y-velocity ", maximum(ρv ./ ρ))
        println("maximum z-velocity ", maximum(ρw ./ ρ))
        println("the time is ", time)
    end
end

vf = (KennedyGruberFlux(), KennedyGruberFlux(), KennedyGruberFlux())
# sf = (RoeFlux(), RoeFlux(), Atum.RefanovFlux(1.0))
sf = (Atum.RefanovFlux(0.1), Atum.RefanovFlux(0.1), Atum.RefanovFlux(1.0))
linearized_vf = Atum.LinearizedKennedyGruberFlux()
linearized_sf = Atum.LinearizedRefanovFlux(1.0)

dg_sd = SingleDirection(; law, grid, volume_form=linearized_vf, surface_numericalflux=linearized_sf)
dg_fs = FluxSource(; law, grid, volume_form=vf, surface_numericalflux=sf)


ρ, ρu, ρv, ρw, ρet = components(test_state)
xp = components(aux)[1]
yp = components(aux)[2]
zp = components(aux)[3]

test_state .= stable_state
tic_global = time()
for i in 1:endscale*tscale
    aux = sphere_auxiliary_3.(Ref(law), x⃗, test_state)
    dg_fs.auxstate .= aux
    dg_sd.auxstate .= aux
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
    println("starting simulation ", i)
    tic = time()
    solve!(test_state, timeend, odesolver, adjust_final=false)
    toc = time()
    println("The time to simulate ", timeend / 60, " minutes is ", toc - tic, " seconds")
    u = ρu ./ ρ
    v = ρv ./ ρ
    w = ρw ./ ρ

    minu = minimum(u)
    maxu = maximum(u)
    minv = minimum(v)
    maxv = maximum(v)
    minw = minimum(w)
    maxw = maximum(w)
    uʳ = @. (xp * u + yp * v + zp * w) / sqrt(xp^2 + yp^2 + zp^2)
    minuʳ = minimum(uʳ)
    maxuʳ = maximum(uʳ)
    println("extrema x-velocity ", (minu, maxu))
    println("extrema y-velocity ", (minv, maxv))
    println("extrema z-velocity ", (minw, maxw))
    println("extrema vertical velocity ", (minuʳ, maxuʳ))

    speed = @. sqrt(u^2 + v^2 + w^2)
    println("max speed is ", maximum(speed))
    ρ̅ = sum(ρ .* dg_fs.MJ) / sum(dg_fs.MJ)
    println("The average density of the system is ", ρ̅)
end

ρ, ρu, ρv, ρw, ρet = components(test_state)
s_ρ, s_ρu, s_ρv, s_ρw, s_ρet = components(stable_state)
println(sum(isnan.(ρ)))

#=
vf = (KennedyGruberFlux(), KennedyGruberFlux(), KennedyGruberFlux())
sf = (RoeFlux(), RoeFlux(), Atum.RefanovFlux(0.0))
# sf = (Atum.RefanovFlux(1.0), Atum.RefanovFlux(1.0), Atum.RefanovFlux(1.0))
linearized_vf = Atum.LinearizedKennedyGruberFlux()
linearized_sf = Atum.LinearizedRefanovFlux(0.0)

dt = 200.0
timeend = 20
aux = sphere_auxiliary.(Ref(law), x⃗, test_state)
hs_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), test_state, aux)
dg_fs.auxstate .= aux
dg_sd.auxstate .= aux
odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
println("starting simulation")
tic = time()
solve!(test_state, timeend, odesolver, adjust_final=false)
toc = time()
println("The time to simulate ", timeend / 60, " minutes is ", toc - tic, " seconds")
println("maximum x-velocity ", maximum(ρu ./ ρ))
println("maximum y-velocity ", maximum(ρv ./ ρ))
println("maximum z-velocity ", maximum(ρw ./ ρ))

println(sum(isnan.(ρ)))
=#



#=
dt = 95.0
timeend = 20 # 60 NaNs
aux = sphere_auxiliary.(Ref(law), x⃗, stable_state)
hs_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), test_state, aux)
dg_fs.auxstate .= aux
dg_sd.auxstate .= aux
odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
println("starting simulation")
tic = time()
solve!(test_state, timeend, odesolver, adjust_final=false)
toc = time()
println("The time to simulate ", timeend / 60, " minutes is ", toc - tic, " seconds")
println("maximum x-velocity ", maximum(ρu ./ ρ))
println("maximum y-velocity ", maximum(ρv ./ ρ))
println("maximum z-velocity ", maximum(ρw ./ ρ))
=#
#=
dt = 200.0
timeend = 60 * 60 * 1
for i in 1:3
    aux = sphere_auxiliary.(Ref(law), x⃗, test_state)
    dg_fs.auxstate .= aux
    dg_sd.auxstate .= aux
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
    println("starting simulation ", i)
    tic = time()
    solve!(test_state, timeend, odesolver, adjust_final=false)
    toc = time()
    println("The time to simulate ", timeend / 60, " minutes is ", toc - tic, " seconds")
    println("maximum x-velocity ", maximum(ρu ./ ρ))
    println("maximum y-velocity ", maximum(ρv ./ ρ))
    println("maximum z-velocity ", maximum(ρw ./ ρ))
end
=#

#=
dt = 200.0
timeend = 60 * 60 * 1
aux = sphere_auxiliary.(Ref(law), x⃗, test_state)
hs_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), test_state, aux)
dg_fs.auxstate .= aux
dg_sd.auxstate .= aux
odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
println("starting simulation")
tic = time()
solve!(test_state, timeend, odesolver, adjust_final=false)
toc = time()
println("The time to simulate ", timeend / 60, " minutes is ", toc - tic, " seconds")
println("maximum x-velocity ", maximum(ρu ./ ρ))
println("maximum y-velocity ", maximum(ρv ./ ρ))
println("maximum z-velocity ", maximum(ρw ./ ρ))
=#

#=
aux = sphere_auxiliary.(Ref(law), x⃗, test_state)
dg_fs.auxstate .= aux
dg_sd.auxstate .= aux
odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
println("starting simulation")
tic = time()
solve!(test_state, timeend, odesolver, adjust_final=false)
toc = time()
println("The time to simulate ", timeend / 60, " minutes is ", toc - tic, " seconds")

println("maximum x-velocity ", maximum(ρu ./ ρ))
println("maximum y-velocity ", maximum(ρv ./ ρ))
println("maximum z-velocity ", maximum(ρw ./ ρ))
=#

toc_global = time()

println("In total the simulation took ", toc_global - tic_global, " seconds")

