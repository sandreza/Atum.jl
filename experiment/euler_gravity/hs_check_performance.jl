using UnPack


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
    solve!(test_state, dt / 2, odesolver, adjust_final=false)
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
ark = odesolver;

@unpack time, dt = ark;
@unpack rhs!, linrhs! = ark;
@unpack ex_rka, ex_rkb, ex_rkc = ark;
@unpack im_rka, im_rkb, im_rkc = ark;
@unpack ex_K, im_K = ark;
@unpack Qhat, fac = ark;

q = state
Q = (q, ark.qstages...)

@benchmark begin
    # Compute first explicit stage
    ex_stagetime = time + ex_rkc[1] * dt
    if isnothing(rhs!)
        fill!.(components(ex_K[1]), 0)
    else
        rhs!(ex_K[1], Q[1], ex_stagetime; increment=false)
    end

    # Compute first implicit stage
    im_stagetime = time + im_rkc[1] * dt
    if isnothing(linrhs!)
        @assert ark.dt === ark.dt_fac
        fill!.(components(im_K[1]), 0)
    else
        linrhs!(im_K[1], Q[1], im_stagetime; increment=false)
    end

    if !ark.split_rhs
        ex_K[1] .-= im_K[1]
    end

    Nstages = length(ex_rkc)
    for i = 2:Nstages
        # q̂ = q + dt * \sum_{k=1}^{i-1} (a_{ik} * f(Q^{(k)}) + ã_{ik} * L * Q^{(k)})
        Qhat .= q
        for k = 1:i-1
            @. Qhat += dt * (ex_rka[i, k] * ex_K[k] + im_rka[i, k] * im_K[k])
        end

        # Q^{(i)} = (I + dt ã_{ii} * L) \ Qhat
        if isnothing(fac)
            Q[i] .= Qhat
        else
            ldiv!(Q[i], fac, Qhat)
        end

        # Compute explicit state i
        ex_stagetime = time + ex_rkc[i] * dt
        if isnothing(rhs!)
            fill!.(components(ex_K[i]), 0)
        else
            rhs!(ex_K[i], Q[i], ex_stagetime; increment=false)
        end

        # Compute implicit state i
        im_stagetime = time + im_rkc[i] * dt
        if isnothing(linrhs!)
            fill!.(components(im_K[i]), 0)
        else
            linrhs!(im_K[i], Q[i], im_stagetime; increment=false)
        end

        if !ark.split_rhs
            ex_K[i] .-= im_K[i]
        end

    end

    # q += dt * \sum_{i=1}^{s} (b_{i} * f(Q^{(i)}) + b̃_{i} * L * Q^{(i)})
    for i = 1:Nstages
        @. q += dt * (ex_rkb[i] * ex_K[i] + im_rkb[i] * im_K[i])
    end

    # Advance time
    ark.time += dt

end

# new one
@benchmark begin
    cQhat = parent(components(Qhat)[1])
    cq = parent(components(q)[1])
    # Compute first explicit stage
    ex_stagetime = time + ex_rkc[1] * dt
    if isnothing(rhs!)
        fill!.(components(ex_K[1]), 0)
    else
        rhs!(ex_K[1], Q[1], ex_stagetime; increment=false)
    end

    # Compute first implicit stage
    im_stagetime = time + im_rkc[1] * dt
    if isnothing(linrhs!)
        @assert ark.dt === ark.dt_fac
        fill!.(components(im_K[1]), 0)
    else
        linrhs!(im_K[1], Q[1], im_stagetime; increment=false)
    end

    if !ark.split_rhs
        cex_K = parent(components(ex_K[1])[1])
        cim_K = parent(components(im_K[1])[1])
        cex_K .-= cim_K
    end

    Nstages = length(ex_rkc)
    for i = 2:3
        # q̂ = q + dt * \sum_{k=1}^{i-1} (a_{ik} * f(Q^{(k)}) + ã_{ik} * L * Q^{(k)})
        cQhat .= cq
        for k = 1:i-1
            cex_K = parent(components(ex_K[k])[1])
            cim_K = parent(components(im_K[k])[1])
            @. cQhat += dt * (ex_rka[i, k] * cex_K + im_rka[i, k] * cim_K)
        end

        # Q^{(i)} = (I + dt ã_{ii} * L) \ Qhat
        if isnothing(fac)
            cQ = parent(components(Q[i])[1])
            cQ .= cQhat
        else
            ldiv!(Q[i], fac, Qhat)
        end

        # Compute explicit state i
        ex_stagetime = time + ex_rkc[i] * dt
        if isnothing(rhs!)
            fill!.(components(ex_K[i]), 0)
        else
            rhs!(ex_K[i], Q[i], ex_stagetime; increment=false)
        end

        # Compute implicit state i
        im_stagetime = time + im_rkc[i] * dt
        if isnothing(linrhs!)
            fill!.(components(im_K[i]), 0)
        else
            linrhs!(im_K[i], Q[i], im_stagetime; increment=false)
        end

        if !ark.split_rhs
            cex_K = parent(components(ex_K[i])[1])
            cim_K = parent(components(im_K[i])[1])
            cex_K .-= cim_K
        end

    end

    # q += dt * \sum_{i=1}^{s} (b_{i} * f(Q^{(i)}) + b̃_{i} * L * Q^{(i)})
    for i = 1:3
        cex_K = parent(components(ex_K[i])[1])
        cim_K = parent(components(im_K[i])[1])
        @. cq += dt * (ex_rkb[i] * cex_K + im_rkb[i] * cim_K)
    end

    # Advance time
    ark.time += dt

end

##
@benchmark CUDA.@sync ex_K[1] .-= im_K[1]

@benchmark CUDA.@sync begin
    cex_K = parent(components(ex_K[1])[1])
    cim_K = parent(components(im_K[1])[1])
    cex_K .-= cim_K
end