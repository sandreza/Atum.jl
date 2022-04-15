
test_state .= old_state
state .= test_state
dg_sd = SingleDirection(; law, grid, volume_form=linearized_vf, surface_numericalflux=linearized_sf)
dg_fs = FluxSource(; law, grid, volume_form=vf, surface_numericalflux=sf)
aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, test_state)
dg_fs.auxstate .= aux
dg_sd.auxstate .= aux
odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
dt = 0.01
println("----")
println("for Nq⃗=", Nq⃗, ", Kh = ", Kh, ", Kv =", Kv)
createlu = @benchmark begin
    odesolver = ARK23(dg_fs, dg_sd, fieldarray(test_state), dt; split_rhs=false, paperversion=false)
end

solvetimes = @benchmark begin
    solve!(test_state, dt/2, odesolver, adjust_final=false)
    test_state .= old_state
    odesolver.time = 0.0
end
meanlu = median(createlu.times)
meansolvetimes = median(solvetimes.times)
println("the time in nanoseconds to do the lufactorization of linear system is ", meanlu)
println("the time in nanoseconds to take one timestep is ", meansolvetimes)
println("the ratio of lu construction to solving one step is ", meanlu / meansolvetimes)

rhstimes = @benchmark begin
    dg_fs(state, test_state, 0.0, increment=false)
    dg_sd(state, test_state, 0.0, increment=false)
end

ldivtimes = @benchmark begin
    ldiv!(test_state, odesolver.fac, state)
end
meanldiv = median(ldivtimes.times)
meanrhs = median(rhstimes.times)
println("the time in nanoseconds to solve the factorized linear system is ", meanldiv)
println("the time in nanoseconds to evaluate the explicit rhs is ", meanrhs)
println("the ratio of implicit to explicit is ", meanldiv / meanrhs)
current_timings = [meanlu, meansolvetimes, meanldiv, meanrhs]
println("----")
##
plevels = range(1e5 * exp(-3e4 / 8e3), 1e5, length=8)
zlevels = reverse(@. -log(plevels / 1e5) * 8e3)
