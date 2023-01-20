using KernelAbstractions
using KernelAbstractions.Extras: @unroll

@kernel function surface_withflux!(law,
    dq,
    q,
    ::Val{faceoffsets},
    ::Val{dir},
    numericalflux,
    MJI,
    faceix⁻,
    faceix⁺,
    faceMJ,
    facenormal,
    boundaryfaces,
    auxstate,
    ::Val{dim}) where {faceoffsets,dir,dim}
    @uniform begin
        if dir == 1
            faces = 1:2
        elseif dir == 2
            faces = 3:4
        elseif dir == 3
            faces = 5:6
        end
    end

    e⁻ = @index(Group, Linear)
    i, fi = @index(Local, NTuple)

    @inbounds begin
        face = faces[fi]
        j = i + faceoffsets[face]
        id⁻ = faceix⁻[j, e⁻]

        n⃗ = facenormal[j, e⁻]
        fMJ = faceMJ[j, e⁻]

        aux⁻ = auxstate[id⁻]
        q⁻ = q[id⁻]

        boundarytag = boundaryfaces[face, e⁻]

        if boundarytag == 0
            id⁺ = faceix⁺[j, e⁻]
            q⁺ = q[id⁺]
            aux⁺ = auxstate[id⁺]
        else
            q⁺, aux⁺ = Atum.boundarystate(law, n⃗, q⁻, aux⁻, boundarytag)
        end
        nf = Atum.surfaceflux(numericalflux, law, n⃗, q⁻, aux⁻, q⁺, aux⁺)
        ## to here
        dq[id⁻] -= fMJ * nf * MJI[id⁻]
    end
end

@kernel function surface_noflux!(law,
    dq,
    q,
    ::Val{faceoffsets},
    ::Val{dir},
    numericalflux,
    MJI,
    faceix⁻,
    faceix⁺,
    faceMJ,
    facenormal,
    boundaryfaces,
    auxstate,
    ::Val{dim}) where {faceoffsets,dir,dim}
    @uniform begin
        if dir == 1
            faces = 1:2
        elseif dir == 2
            faces = 3:4
        elseif dir == 3
            faces = 5:6
        end
    end

    e⁻ = @index(Group, Linear)
    i, fi = @index(Local, NTuple)

    @inbounds begin
        face = faces[fi]
        j = i + faceoffsets[face]
        id⁻ = faceix⁻[j, e⁻]

        n⃗ = facenormal[j, e⁻]
        fMJ = faceMJ[j, e⁻]

        aux⁻ = auxstate[id⁻]
        q⁻ = q[id⁻]

        boundarytag = boundaryfaces[face, e⁻]

        if boundarytag == 0
            id⁺ = faceix⁺[j, e⁻]
            q⁺ = q[id⁺]
            aux⁺ = auxstate[id⁺]
        else
            q⁺, aux⁺ = Atum.boundarystate(law, n⃗, q⁻, aux⁻, boundarytag)
        end
        # nf = Atum.surfaceflux(numericalflux, law, n⃗, q⁻, aux⁻, q⁺, aux⁺)
        ## to here
        dq[id⁻] -= dq[id⁻] * 0
    end
end

##
@kernel function volume_withflux!(law,
    dq,
    q,
    D,
    volume_numericalflux,
    metrics,
    MJ,
    MJI,
    auxstate,
    add_source,
    ::Val{dir},
    ::Val{dim},
    ::Val{Nq1},
    ::Val{Nq2},
    ::Val{Nq3},
    ::Val{Ns},
    ::Val{Naux},
    ::Val{increment}) where {dir,dim,Nq1,Nq2,Nq3,Ns,Naux,increment}
    @uniform begin
        FT = eltype(law)
        if dir == 1
            Nqd = Nq1
        elseif dir == 2
            Nqd = Nq2
        elseif dir == 3
            Nqd = Nq3
        end
    end

    dqijk = @private FT (Ns,)

    q1 = @private FT (Ns,)
    aux1 = @private FT (Naux,)

    l_g = @localmem FT (Nq1 * Nq2 * Nq3, 3)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq1 * (j - 1 + Nq2 * (k - 1))

        MJijk = MJ[ijk, e]
        @unroll for d in 1:dim
            l_g[ijk, d] = MJijk * metrics[ijk, e].g[dir, d]
        end

        fill!(dqijk, -zero(FT))

        @unroll for s in 1:Ns
            q1[s] = q[ijk, e][s]
        end
        @unroll for s in 1:Naux
            aux1[s] = auxstate[ijk, e][s]
        end

        add_source && source!(law, dqijk, q1, aux1, dim, (dir,))

        @synchronize

        ijk = i + Nq1 * (j - 1 + Nq2 * (k - 1))

        MJIijk = MJI[ijk, e]

        @unroll for n in 1:Nqd
            if dir == 1
                id = i
                ild = n + Nq1 * ((j - 1) + Nq2 * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq1 * ((n - 1) + Nq2 * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq1 * ((j - 1) + Nq2 * (n - 1))
            end

            q2 = q[ild, e]
            aux2 = auxstate[ild, e]

            f = Atum.twopointflux(volume_numericalflux, law, q1, aux1, q2, aux2)

            Ddn = MJIijk * D[id, n]
            Dnd = MJIijk * D[n, id]
            @unroll for s in 1:Ns
                @unroll for d in 1:dim
                    dqijk[s] -= Ddn * l_g[ijk, d] * f[d, s]
                    dqijk[s] += Dnd * l_g[ild, d] * f[d, s]
                end
            end
        end

        if increment
            dq[ijk, e] += dqijk
        else
            dq[ijk, e] = dqijk[:]
        end
    end
end

@kernel function volume_withloop2!(law,
    dq,
    q,
    D,
    volume_numericalflux,
    metrics,
    MJ,
    MJI,
    auxstate,
    add_source,
    ::Val{dir},
    ::Val{dim},
    ::Val{Nq1},
    ::Val{Nq2},
    ::Val{Nq3},
    ::Val{Ns},
    ::Val{Naux},
    ::Val{increment}) where {dir,dim,Nq1,Nq2,Nq3,Ns,Naux,increment}
    @uniform begin
        FT = eltype(law)
        if dir == 1
            Nqd = Nq1
        elseif dir == 2
            Nqd = Nq2
        elseif dir == 3
            Nqd = Nq3
        end
    end

    dqijk = @private FT (Ns,)

    q1 = @private FT (Ns,)
    aux1 = @private FT (Naux,)

    l_g = @localmem FT (Nq1 * Nq2 * Nq3, 3)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq1 * (j - 1 + Nq2 * (k - 1))

        MJijk = MJ[ijk, e]
        @unroll for d in 1:dim
            l_g[ijk, d] = MJijk * metrics[ijk, e].g[dir, d]
        end

        fill!(dqijk, -zero(FT))

        @unroll for s in 1:Ns
            q1[s] = q[ijk, e][s]
        end
        @unroll for s in 1:Naux
            aux1[s] = auxstate[ijk, e][s]
        end

        add_source && source!(law, dqijk, q1, aux1, dim, (dir,))

        @synchronize

        ijk = i + Nq1 * (j - 1 + Nq2 * (k - 1))

        MJIijk = MJI[ijk, e]

        @unroll for n in 1:Nqd
            if dir == 1
                id = i
                ild = n + Nq1 * ((j - 1) + Nq2 * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq1 * ((n - 1) + Nq2 * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq1 * ((j - 1) + Nq2 * (n - 1))
            end

            q2 = q[ild, e]
            aux2 = auxstate[ild, e]

            Ddn = MJIijk * D[id, n]
            Dnd = MJIijk * D[n, id]
            @unroll for s in 1:Ns
                @unroll for d in 1:dim
                    dqijk[s] -= Ddn * l_g[ijk, d]
                    dqijk[s] += Dnd * l_g[ild, d]
                end
            end
        end

        if increment
            dq[ijk, e] += dqijk
        else
            dq[ijk, e] = dqijk[:]
        end
    end
end



##
@kernel function volume_noflux!(law,
    dq,
    q,
    D,
    volume_numericalflux,
    metrics,
    MJ,
    MJI,
    auxstate,
    add_source,
    ::Val{dir},
    ::Val{dim},
    ::Val{Nq1},
    ::Val{Nq2},
    ::Val{Nq3},
    ::Val{Ns},
    ::Val{Naux},
    ::Val{increment}) where {dir,dim,Nq1,Nq2,Nq3,Ns,Naux,increment}
    @uniform begin
        FT = eltype(law)
        if dir == 1
            Nqd = Nq1
        elseif dir == 2
            Nqd = Nq2
        elseif dir == 3
            Nqd = Nq3
        end
    end

    dqijk = @private FT (Ns,)

    q1 = @private FT (Ns,)
    aux1 = @private FT (Naux,)

    l_g = @localmem FT (Nq1 * Nq2 * Nq3, 3)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq1 * (j - 1 + Nq2 * (k - 1))

        MJijk = MJ[ijk, e]
        @unroll for d in 1:dim
            l_g[ijk, d] = MJijk * metrics[ijk, e].g[dir, d]
        end

        fill!(dqijk, -zero(FT))

        @unroll for s in 1:Ns
            q1[s] = q[ijk, e][s]
        end
        @unroll for s in 1:Naux
            aux1[s] = auxstate[ijk, e][s]
        end

        add_source && source!(law, dqijk, q1, aux1, dim, (dir,))

        @synchronize

        ijk = i + Nq1 * (j - 1 + Nq2 * (k - 1))

        MJIijk = MJI[ijk, e]

        @unroll for n in 1:Nqd
            if dir == 1
                id = i
                ild = n + Nq1 * ((j - 1) + Nq2 * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq1 * ((n - 1) + Nq2 * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq1 * ((j - 1) + Nq2 * (n - 1))
            end

            q2 = q[ild, e]
            aux2 = auxstate[ild, e]

        end

        if increment
            dq[ijk, e] += dqijk
        else
            dq[ijk, e] = dqijk[:]
        end
    end
end

##
@kernel function volume_functional3!(volumeflux!,
    dq,
    q,
    D,
    metrics,
    MJ,
    MJI,
    auxstate,
    add_source,
    ::Val{dir},
    ::Val{dim},
    ::Val{Nq1},
    ::Val{Nq2},
    ::Val{Nq3},
    ::Val{Ns},
    ::Val{Naux},
    ::Val{increment}) where {dir,dim,Nq1,Nq2,Nq3,Ns,Naux,increment}
    @uniform begin
        FT = Float64
        if dir == 1
            Nqd = Nq1
        elseif dir == 2
            Nqd = Nq2
        elseif dir == 3
            Nqd = Nq3
        end
        f = MArray{Tuple{3,Ns},FT}(undef)
    end

    dqijk = @private FT (Ns,)

    q1 = @private FT (Ns,)
    aux1 = @private FT (Naux,)

    l_g = @localmem FT (Nq1 * Nq2 * Nq3, 3)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq1 * (j - 1 + Nq2 * (k - 1))

        MJijk = MJ[ijk, e]
        @unroll for d in 1:dim
            l_g[ijk, d] = MJijk * metrics[ijk, e].g[dir, d]
        end

        fill!(dqijk, -zero(FT))

        @unroll for s in 1:Ns
            q1[s] = q[ijk, e][s]
        end
        @unroll for s in 1:Naux
            aux1[s] = auxstate[ijk, e][s]
        end

        # add_source && source!(law, dqijk, q1, aux1, dim, (dir,))

        @synchronize

        ijk = i + Nq1 * (j - 1 + Nq2 * (k - 1))

        MJIijk = MJI[ijk, e]

        @unroll for n in 1:Nqd
            if dir == 1
                id = i
                ild = n + Nq1 * ((j - 1) + Nq2 * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq1 * ((n - 1) + Nq2 * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq1 * ((j - 1) + Nq2 * (n - 1))
            end

            q2 = q[ild, e]
            aux2 = auxstate[ild, e]
            fill!(f, -zero(FT))
            volumeflux!(f, q1, aux1, q2, aux2)

            Ddn = MJIijk * D[id, n]
            Dnd = MJIijk * D[n, id]
            @unroll for s in 1:Ns
                @unroll for d in 1:dim
                    dqijk[s] -= Ddn * l_g[ijk, d] * f[d, s]
                    dqijk[s] += Dnd * l_g[ild, d] * f[d, s]
                end
            end
        end

        if increment
            dq[ijk, e] += dqijk
        else
            dq[ijk, e] = dqijk[:]
        end
    end
end

## functional programming style 
@kernel function surface_functional4!(surfaceflux!,
    dq,
    q,
    ::Val{faceoffsets},
    ::Val{dir},
    MJI,
    faceix⁻,
    faceix⁺,
    faceMJ,
    facenormal,
    boundaryfaces,
    auxstate,
    ::Val{dim}) where {faceoffsets,dir,dim}
    @uniform begin
        if dir == 1
            faces = 1:2
        elseif dir == 2
            faces = 3:4
        elseif dir == 3
            faces = 5:6
        end
        nf = MArray{Tuple{5},Float64}(undef)
    end

    e⁻ = @index(Group, Linear)
    i, fi = @index(Local, NTuple)

    @inbounds begin
        face = faces[fi]
        j = i + faceoffsets[face]
        id⁻ = faceix⁻[j, e⁻]

        n⃗ = facenormal[j, e⁻]
        fMJ = faceMJ[j, e⁻]

        aux⁻ = auxstate[id⁻]
        q⁻ = q[id⁻]

        boundarytag = boundaryfaces[face, e⁻]

        if boundarytag == 0
            id⁺ = faceix⁺[j, e⁻]
            q⁺ = q[id⁺]
            aux⁺ = auxstate[id⁺]
        else
            q⁺, aux⁺ = q⁻, aux⁻
        end
        fill!(nf, -zero(Float64))
        surfaceflux!(nf, n⃗, q⁻, aux⁻, q⁺, aux⁺)
        ## to here
        dq[id⁻] -= fMJ * nf * MJI[id⁻]
        # dq[id⁻] -= fMJ * dq[id⁻]
    end
end

#=
function dostep!(
    Q,
    lsrk::LowStorageRungeKutta2N,
    p,
    time,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    in_slow_scaling = nothing,
)
    dt = lsrk.dt

    RKA, RKB, RKC = lsrk.RKA, lsrk.RKB, lsrk.RKC
    rhs!, dQ = lsrk.rhs!, lsrk.dQ

    rv_Q = realview(Q)
    rv_dQ = realview(dQ)

    groupsize = 256

    for s in 1:length(RKA)
        rhs!(dQ, Q, p, time + RKC[s] * dt, increment = true)

        slow_scaling = nothing
        if s == length(RKA)
            slow_scaling = in_slow_scaling
        end
        # update solution and scale RHS
        event = Event(array_device(Q))
        event = update!(array_device(Q), groupsize)(
            rv_dQ,
            rv_Q,
            RKA[s % length(RKA) + 1],
            RKB[s],
            dt,
            slow_δ,
            slow_rv_dQ,
            slow_scaling;
            ndrange = length(rv_Q),
            dependencies = (event,),
        )
        wait(array_device(Q), event)
    end
end

@kernel function update!(dQ, Q, rka, rkb, dt, slow_δ, slow_dQ, slow_scaling)
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            dQ[i] += slow_δ * slow_dQ[i]
        end
        Q[i] += rkb * dt * dQ[i]
        dQ[i] *= rka
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end
    end
end
=#
#=
@kernel function update!(Q, @Const(dQ), rkb, dt)
    i = @index(Global, Linear)
    @inbounds begin
        Q[i] += rkb * dt * dQ[i]
    end
end

@kernel function scalar_mul2!(dQ, rka)
    i = @index(Global, Linear)
    @inbounds begin
        dQ[i] *= rka
    end
end
=#
#=
@benchmark let

    device = Atum.getdevice(dg)
    comp_stream = Event(device)
    #=
    event = update!(device, 256)(
    q,
    q0,
    3.0,
    1.0;
    ndrange=length(q0),
    dependencies=comp_stream
    )
    wait(event)
    =#
    event = scalar_mul2!(device, 256)(
        tmpq,
        3.0;
        ndrange=length(tmpq),
        dependencies=(comp_stream)
    )

    wait(event)
    #=
    @. q *= 0.1
    @. q += 3.0 * 1.0 * q0
    =#
end
=#

    #=
    # Central Flux part 
    ρ_1, ρu_1, ρe_1 = unpackstate(law, q⁻)
    p_1 = linearized_pressure(law, q⁻, aux⁻)
    ρᵣ_1, ρuᵣ_1, ρeᵣ_1 = unpackrefstate(law, aux⁻)
    pᵣ_1 = reference_pressure(law, aux⁻)
  
    ρ_2, ρu_2, ρe_2 = unpackstate(law, q⁺)
    p_2 = linearized_pressure(law, q⁺, aux⁺)
    ρᵣ_2, ρuᵣ_2, ρeᵣ_2 = unpackrefstate(law, aux⁺)
    pᵣ_2 = reference_pressure(law, aux⁺)
  
    # calculate u_1, e_1, and reference states
    u_1 = ρu_1 / ρᵣ_1 - ρ_1 * ρuᵣ_1 / (ρᵣ_1^2)
    e_1 = ρe_1 / ρᵣ_1 - ρ_1 * ρeᵣ_1 / (ρᵣ_1^2)
  
    uᵣ_1 = ρuᵣ_1 / ρᵣ_1
    eᵣ_1 = ρeᵣ_1 / ρᵣ_1
  
    ## State 2 Stuff 
    # calculate u_2, e_2, and reference states
    u_2 = ρu_2 / ρᵣ_2 - ρ_2 * ρuᵣ_2 / (ρᵣ_2^2)
    e_2 = ρe_2 / ρᵣ_2 - ρ_2 * ρeᵣ_2 / (ρᵣ_2^2)
  
    uᵣ_2 = ρuᵣ_2 / ρᵣ_2
    eᵣ_2 = ρeᵣ_2 / ρᵣ_2
  
    # construct averages for perturbation variables
    ρ_avg = avg(ρ_1, ρ_2)
    u_avg = avg(u_1, u_2)
    e_avg = avg(e_1, e_2)
    p_avg = avg(p_1, p_2)
  
    # construct averages for reference variables
    ρᵣ_avg = avg(ρᵣ_1, ρᵣ_2)
    uᵣ_avg = avg(uᵣ_1, uᵣ_2)
    eᵣ_avg = avg(eᵣ_1, eᵣ_2)
    pᵣ_avg = avg(pᵣ_1, pᵣ_2)
  
    fρ = ρᵣ_avg * u_avg + ρ_avg * uᵣ_avg
    fρu⃗ = p_avg * I + ρᵣ_avg .* (uᵣ_avg .* u_avg' + u_avg .* uᵣ_avg')
    fρu⃗ += (ρ_avg .* uᵣ_avg) .* uᵣ_avg'
    fρe = (ρᵣ_avg * eᵣ_avg + pᵣ_avg) * u_avg
    fρe += (ρᵣ_avg * e_avg + ρ_avg * eᵣ_avg + p_avg) * uᵣ_avg
  
    f = hcat(fρ, fρu⃗, fρe)
    =#
