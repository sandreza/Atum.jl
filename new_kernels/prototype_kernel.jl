@kernel function flux_differencing_volume_fused_2!(
    law, dq, q, D, volume_numericalflux,
    metrics, MJ, MJI, auxstate, ::Val{increment},
    ::Val{dim}, ::Val{Nq}, ::Val{Ns}, ::Val{Naux}, ::Val{add_source}
) where {increment,dim,Nq,Ns,Naux,add_source}
    @uniform begin
        FT = eltype(D)
    end

    dqijk = @private FT (Ns,)

    q1 = @private FT (Ns,)
    q2 = @private FT (Ns,)
    aux1 = @private FT (Ns,)
    aux2 = @private FT (Ns,)

    flux = @private FT (dim, Ns)

    l_g = @localmem FT (Nq * Nq * Nq, dim, dim)
    shared_q = @localmem FT (Nq * Nq * Nq, Ns)
    shared_aux = @localmem FT (Nq * Nq * Nq, Naux)
    shared_D = @localmem FT (Nq, Nq)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)
    ijk = @index(Local, Linear)

    MJijk = MJ[ijk, e]
    MJIijk = MJI[ijk, e]

    @unroll for d in 1:dim
        @unroll for dir in 1:dim
            l_g[ijk, d, dir] = MJijk * metrics[ijk, e].g[dir, d]
        end
    end

    @unroll for n in 1:Nq
        @unroll for nn in 1:Nq
            shared_D[nn, n] = D[nn, n]
        end
    end

    @unroll for s in 1:Ns
        shared_q[ijk, s] = q[ijk, e][s]
        q1[s] = shared_q[ijk, s]
    end

    @unroll for s in 1:Naux
        shared_aux[ijk, s] = auxstate[ijk, e][s]
        aux1[s] = shared_aux[ijk, s]
    end

    if add_source
        modified_source!(law, dqijk, q1, aux1)
    end

    @unroll for dir in 1:dim

        # fill!(dqijk, -zero(FT))
        # @synchronize

        @unroll for n in 1:Nq
            if dir == 1
                id = i
                ild = n + Nq * ((j - 1) + Nq * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq * ((n - 1) + Nq * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq * ((j - 1) + Nq * (n - 1))
            end
        
            @unroll for s in 1:Ns
                q2[s] = shared_q[ild, s]
            end
        
            @unroll for s in 1:Naux
                aux2[s] = shared_aux[ild, s]
            end
        
            @views modified_volumeflux!(volume_numericalflux, law, flux[:, :], q1, aux1, q2, aux2)
        
            Ddn = MJIijk * shared_D[id, n]
            Dnd = MJIijk * shared_D[n, id]
            @unroll for s in 1:Ns
                d = 1
                dqijk[s] = -Ddn * l_g[ijk, d, dir] * flux[d, s]
                dqijk[s] += Dnd * l_g[ild, d, dir] * flux[d, s]
                @unroll for d in 2:dim
                    dqijk[s] -= Ddn * l_g[ijk, d, dir] * flux[d, s]
                    dqijk[s] += Dnd * l_g[ild, d, dir] * flux[d, s]
                end
            end
        end
        # always increment directions 2 and 3
        # only increment direction 1 if told to
        if increment | (dir != 1)
            dq[ijk, e] += dqijk
        else
            dq[ijk, e] = dqijk[:]
        end

    end

end