# FT needs to be defined

@kernel function gradient_kernel2!(
    ∇q, q, D, metrics,
    ::Val{dir}, ::Val{dim}, ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{Ns},
    ::Val{increment}
) where {dir,dim,Nq1,Nq2,Nq3,Ns,increment}
    @uniform begin
        FT = eltype(D)
        if dir == 1
            Nqd = Nq1
        elseif dir == 2
            Nqd = Nq2
        elseif dir == 3
            Nqd = Nq3
        end
    end

    dqijk = @private FT (Ns,)

    q_shared = @localmem FT (Nq1 * Nq2 * Nq3, Ns)
    shared_metrics = @localmem FT (dim, Nq1 * Nq2 * Nq3)
    shared_D = @localmem FT (Nqd, Nqd)

    e = @index(Group, Linear)

    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    if dir == 1
        id = i
    elseif dir == 2
        id = j
    else
        id = k
    end

    @unroll for n in 1:Nqd
        shared_D[id, n] = D[id, n]
    end

    @unroll for s in 1:Ns
        q_shared[ijk, s] = q[ijk, s, e]
    end

    @unroll for d in 1:dim
        shared_metrics[d, ijk] = metrics[ijk, dir+dim*(d-1), e]
    end

    fill!(dqijk, -zero(FT))

    @synchronize

    @unroll for s in 1:Ns
        @unroll for idid in 1:Nqd
            if dir == 1
                id = i
                l = idid + Nq1 * (j - 1 + Nq2 * (k - 1))
            elseif dir == 2
                id = j
                l = i + Nq1 * (idid - 1 + Nq2 * (k - 1))
            else
                id = k
                l = i + Nq1 * (j - 1 + Nq2 * (idid - 1))
            end
            dqijk[s] += shared_D[id, idid] * q_shared[l, s]
        end
    end

    if increment
        @unroll for s in 1:Ns
            @unroll for d in 1:dim
                ∇q[ijk, d+dim*(s-1), e] += shared_metrics[d, ijk] * dqijk[s] # order on metric flipped since filling in three cartesian components at once
            end
        end
    else
        @unroll for s in 1:Ns
            @unroll for d in 1:dim
                ∇q[ijk, d+dim*(s-1), e] = shared_metrics[d, ijk] * dqijk[s] # order on metric flipped since filling in three cartesian components at once
            end
        end
    end

end


##


@kernel function gradient_kernel_fused3!(
    ∇q, q, D, metrics,
    ::Val{dim}, ::Val{Nq}, ::Val{Ns},
) where {dim,Nq,Ns}
    @uniform begin
        FT = eltype(D)
    end

    dqijk = @private FT (Ns,)

    q_shared = @localmem FT (Nq * Nq * Nq, Ns)
    shared_metrics = @localmem FT (dim, Nq * Nq * Nq)
    shared_D = @localmem FT (Nq, Nq)

    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    @unroll for n in 1:Nq
        @unroll for nn in 1:Nq
            shared_D[n, nn] = D[n, nn]
        end
    end

    @unroll for s in 1:Ns
        q_shared[ijk, s] = q[ijk, s, e]
    end

    @unroll for dir in 1:dim

        if dir == 1
            id = i
        elseif dir == 2
            id = j
        else
            id = k
        end

        @unroll for d in 1:dim
            shared_metrics[d, ijk] = metrics[ijk, dir+dim*(d-1), e]
        end

        fill!(dqijk, -zero(FT))

        @synchronize

        @unroll for s in 1:Ns
            @unroll for idid in 1:Nq
                if dir == 1
                    id = i
                    l = idid + Nq * (j - 1 + Nq * (k - 1))
                elseif dir == 2
                    id = j
                    l = i + Nq * (idid - 1 + Nq * (k - 1))
                else
                    id = k
                    l = i + Nq * (j - 1 + Nq * (idid - 1))
                end
                dqijk[s] += shared_D[id, idid] * q_shared[l, s]
            end
        end

        if dir == 1
            @unroll for s in 1:Ns
                @unroll for d in 1:dim
                    ∇q[ijk, d+dim*(s-1), e] = shared_metrics[d, ijk] * dqijk[s] # order on metric flipped since filling in three cartesian components at once
                end
            end
        else
            @unroll for s in 1:Ns
                @unroll for d in 1:dim
                    ∇q[ijk, d+dim*(s-1), e] += shared_metrics[d, ijk] * dqijk[s] # order on metric flipped since filling in three cartesian components at once
                end
            end
        end
    end

end

##

@kernel function gradient_kernel_fused_second!(
    ∇q, q, D, metrics,
    ::Val{dim}, ::Val{Nq}, ::Val{Ns},
) where {dim,Nq,Ns}
    @uniform begin
        FT = eltype(D)
    end

    dqijk = @private FT (Ns,)

    q_shared = @localmem FT (Nq * Nq * Nq, Ns)
    shared_metrics = @localmem FT (dim, Nq * Nq * Nq, dim)
    shared_D = @localmem FT (Nq, Nq)

    e = @index(Group, Linear)
    ijk = @index(Local, Linear)
    i, j, k = @index(Local, NTuple)

    @unroll for nn in 1:Nq
        @unroll for n in 1:Nq
            shared_D[n, nn] = D[n, nn]
        end
    end

    @unroll for s in 1:Ns
        q_shared[ijk, s] = q[ijk, s, e]
    end

    @unroll for d in 1:dim
        @unroll for dir in 1:dim
            shared_metrics[d, ijk, dir] = metrics[ijk, dir+dim*(d-1), e]
        end
    end

    @unroll for dir in 1:dim

        fill!(dqijk, -zero(FT))

        @synchronize

        @unroll for s in 1:Ns
            @unroll for idid in 1:Nq
                if dir == 1
                    id = i
                    l = idid + Nq * (j - 1 + Nq * (k - 1))
                elseif dir == 2
                    id = j
                    l = i + Nq * (idid - 1 + Nq * (k - 1))
                else
                    id = k
                    l = i + Nq * (j - 1 + Nq * (idid - 1))
                end
                dqijk[s] += shared_D[id, idid] * q_shared[l, s]
            end
        end

        if dir == 1
            @unroll for s in 1:Ns
                @unroll for d in 1:dim
                    ∇q[ijk, d+dim*(s-1), e] = shared_metrics[d, ijk, dir] * dqijk[s] # order on metric flipped since filling in three cartesian components at once
                end
            end
        else
            @unroll for s in 1:Ns
                @unroll for d in 1:dim
                    ∇q[ijk, d+dim*(s-1), e] += shared_metrics[d, ijk, dir] * dqijk[s] # order on metric flipped since filling in three cartesian components at once
                end
            end
        end
    end

end
