vcfl = 120.0
hcfl = 0.4 
Δx = min_node_distance(grid, dims = 1)
Δy = min_node_distance(grid, dims = 2)
Δz = min_node_distance(grid, dims = 3)
vdt = vcfl * Δz / c_max 
hdt = hcfl * Δx / c_max
dt = min(vdt, hdt)

test_state .= old_state
# test_state .= state
endday = 10
tic = time()
partitions = 1:24*endday*2
for i in partitions
    aux = sphere_auxiliary.(Ref(law), x⃗, test_state)
    dg_fs.auxstate .= aux
    dg_sd.auxstate .= aux
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
    timeend = 60 * 60 * 24 * endday/ partitions[end]
    # solve!(test_state, timeend, odesolver; after_step=do_output)
    solve!(test_state, timeend, odesolver)
    if i%4==0
        println("--------")
        println("done with ", timeend)
        println("partition ", i)
        ρ, ρu, ρv, ρw, _ = components(test_state)
        println("maximum x-velocity ", maximum(ρu ./ ρ))
        println("maximum y-velocity ", maximum(ρv ./ ρ))
        println("maximum z-velocity ", maximum(ρw ./ ρ))
        bw_pressure = Atum.EulerTotalEnergy.pressure.(Ref(law), test_state, aux)
        bw_density = components(test_state)[1]
        bw_soundspeed =  Atum.EulerTotalEnergy.soundspeed.(Ref(law), bw_density, bw_pressure)
        c_max = maximum(bw_soundspeed)
        println("The maximum soundspeed is ", c_max)
        println("-----")
    end
end
toc = time()
println("The time for the simulation is ", toc - tic)