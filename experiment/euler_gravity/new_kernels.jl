using KernelAbstractions
using Adapt
using KernelAbstractions.Extras: @unroll

struct NewFluxArraySource{L,G,A1,A2,A3,A4,VF,SNF}
    law::L
    grid::G
    MJ::A1
    MJI::A2
    faceMJ::A3
    auxstate::A4
    volume_form::VF
    surface_numericalflux::SNF
end

Bennu.referencecell(dg::NewFluxArraySource) = referencecell(dg.grid)

function Adapt.adapt_structure(to, dg::NewFluxArraySource)
    names = fieldnames(NewFluxArraySource)
    args = ntuple(j -> adapt(to, getfield(dg, names[j])), length(names))
    NewFluxArraySource{typeof.(args)...}(args...)
end

function NewFluxArraySource(; law, grid, surface_numericalflux,
    volume_form=WeakForm(), auxstate=nothing)
    cell = referencecell(grid)
    M = mass(cell)
    _, J = components(metrics(grid))
    MJ = M * J
    MJI = 1 ./ MJ

    faceM = facemass(cell)
    _, faceJ = components(facemetrics(grid))

    faceMJ = faceM * faceJ

    if isnothing(auxstate)
        auxstate = Atum.auxiliary.(Ref(law), points(grid))
    end

    dim = ndims(cell)

    if volume_form isa Tuple
        @assert length(volume_form) == dim
        volume_form = volume_form
    else
        volume_form = ntuple(i -> volume_form, dim)
    end

    if surface_numericalflux isa Tuple
        @assert length(surface_numericalflux) == dim
        surface_numericalflux = surface_numericalflux
    else
        surface_numericalflux = ntuple(i -> surface_numericalflux, dim)
    end

    args = (law, grid, MJ, MJI, faceMJ, auxstate,
        volume_form, surface_numericalflux)
    NewFluxArraySource{typeof.(args)...}(args...)
end
getdevice(dg::NewFluxArraySource) = Bennu.device(arraytype(referencecell(dg)))

function (dg::NewFluxArraySource)(dq, q, time; increment=true)
    cell = referencecell(dg)
    grid = dg.grid
    device = getdevice(dg)
    dim = ndims(cell)

    Ne = length(dg.grid)
    Nq⃗ = size(cell)

    # @assert(length(eltype(q)) == numberofstates(dg.law))

    comp_stream = Event(device)

    # Naux = eltype(eltype(dg.auxstate)) === Nothing ? 0 : length(eltype(dg.auxstate))

    workgroup = ntuple(i -> i <= dim ? Nq⃗[i] : 1, 3)
    ndrange = (Ne * workgroup[1], Base.tail(workgroup)...)

    volume_form = dg.volume_form
    # surface_flux = dg.surface_numericalflux
    Naux = 9

    for dir in 1:dim
        comp_stream = Atum.volume_term_dir!(device, workgroup)(
            dg.law, dq, q, derivatives_1d(cell)[dir],
            volume_form[dir],
            metrics(dg.grid), dg.MJ, dg.MJI,
            dg.auxstate,
            dir == 0, # dont add_source
            Val(dir), Val(dim), Val(workgroup[1]), Val(workgroup[2]), Val(workgroup[3]),
            Val(numberofstates(dg.law)), Val(Naux),
            Val(dir == 1 ? increment : true);
            ndrange,
            dependencies=comp_stream
        )
    end

    #=
        for dir in 1:dim
            Nfp = round(Int, prod(Nq⃗) / Nq⃗[dir])
            workgroup_face = (Nfp, 2)
            ndrange = (Nfp * Ne, 2)

            comp_stream = Atum.surface_term_dir!(device, workgroup_face)(
                dg.law, dq, q,
                Val(Bennu.connectivityoffsets(cell, Val(2))),
                Val(dir),
                surface_flux[dir],
                dg.MJI, faceix⁻, faceix⁺, dg.faceMJ, facenormal, boundaryfaces(grid),
                dg.auxstate,
                Val(dim);
                ndrange,
                dependencies=comp_stream
            )
        end
    =#
    wait(comp_stream)
end

function (dg::NewFluxArraySource)(dq::CuArray, q::CuArray, auxstate::CuArray; increment=true)
    cell = referencecell(dg)
    grid = dg.grid
    device = getdevice(dg)
    dim = ndims(cell)

    Ne = length(dg.grid)
    Nq⃗ = size(cell)

    # @assert(length(eltype(q)) == numberofstates(dg.law))

    comp_stream = Event(device)

    # Naux = eltype(eltype(dg.auxstate)) === Nothing ? 0 : length(eltype(dg.auxstate))

    workgroup = ntuple(i -> i <= dim ? Nq⃗[i] : 1, 3)
    ndrange = (Ne * workgroup[1], Base.tail(workgroup)...)

    volume_form = dg.volume_form
    Naux = size(pauxstate)[2]
    for dir in 1:dim
        comp_stream = volume_term_dir27!(device, workgroup)(
            dg.law, dq, q, derivatives_1d(cell)[dir],
            volume_form[dir],
            metrics(dg.grid), dg.MJ, dg.MJI,
            auxstate,
            dir == 0, # dont add_source
            Val(dir), Val(dim), Val(workgroup[1]), Val(workgroup[2]), Val(workgroup[3]),
            Val(numberofstates(dg.law)), Val(Naux),
            Val(dir == 1 ? increment : true);
            ndrange,
            dependencies=comp_stream
        )
    end

    wait(comp_stream)
end

##
@kernel function volume_term_dir27!(
    law, dq, q, D, volume_numericalflux,
    metrics, MJ, MJI, auxstate, add_source,
    ::Val{dir}, ::Val{dim}, ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{Ns},
    ::Val{Naux}, ::Val{increment}
) where {dir,dim,Nq1,Nq2,Nq3,Ns,Naux,increment}
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

    q2 = @private FT (Ns,)
    aux2 = @private FT (Naux,)

    f = @private FT (dim, Ns)

    l_g = @localmem FT (Nq1 * Nq2 * Nq3, dim)

    shared_D = @localmem FT (Nqd, Nqd)

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    @inbounds begin
        ijk = i + Nq1 * (j - 1 + Nq2 * (k - 1))

        MJijk = MJ[ijk, e]

        @unroll for d in 1:dim
            l_g[ijk, d] = MJijk * metrics[ijk, e].g[dir, d]
        end

        @unroll for n in 1:Nqd
            if dir == 1
                id = i
            elseif dir == 2
                id = j
            elseif dir == 3
                id = k
            end
            shared_D[id, n] = D[id, n]
        end

        fill!(dqijk, -zero(FT))

        @unroll for s in 1:Ns
            q1[s] = q[ijk, s, e]
        end
        @unroll for s in 1:Naux
            aux1[s] = auxstate[ijk, s, e]
        end

        if add_source
            # new_source!(dqijk, q1, aux1)
        end

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

    @unroll for s in 1:Ns
        q2[s] = q[ild, s, e]
    end

    @unroll for s in 1:Naux
        aux2[s] = auxstate[ild, s, e]
    end

    # fill!(f, -zero(FT))
    # @views new_flux_5!(f[:, :], q1[:], q2[:], aux1[:], aux2[:])
    # f = MArray{Tuple{3,Ns},Float64}(undef)
    new_flux_8!(f[:, :], q1[:], q2[:], aux1[:], aux2[:])

    Ddn = MJIijk * shared_D[id, n]
    Dnd = MJIijk * shared_D[n, id]
    @unroll for s in 1:Ns
        @unroll for d in 1:dim
            dqijk[s] -= Ddn * l_g[ijk, d] * f[d, s]
            dqijk[s] += Dnd * l_g[ild, d] * f[d, s]
        end
    end

end

        if increment
            @unroll for s in 1:Ns
                dq[ijk, s, e] += dqijk[s]
            end
        else
            @unroll for s in 1:Ns
                dq[ijk, s, e] = dqijk[s]
            end
        end

    end
end

##
@kernel function volume_term_dir_new_5!(
    law, tendency, state_prognostic, D, volume_numericalflux,
    metrics, MJ, MJI, state_auxiliary, add_source,
    ::Val{dir}, ::Val{dim}, ::Val{Nq1}, ::Val{Nq2}, ::Val{Nq3}, ::Val{num_state},
    ::Val{num_aux}, ::Val{increment}
) where {dir,dim,Nq1,Nq2,Nq3,num_state,num_aux,increment}
    @uniform begin

        FT = Float64

        local_H = MArray{Tuple{3,num_state},FT}(undef)

        state_2 = MArray{Tuple{num_state},FT}(undef)
        aux_2 = MArray{Tuple{num_aux},FT}(undef)

        local_source = MArray{Tuple{num_state},FT}(undef)

        if dir == 1
            Nqd = Nq1
        elseif dir == 2
            Nqd = Nq2
        elseif dir == 3
            Nqd = Nq3
        end
    end

    e = @index(Group, Linear)
    i, j, k = @index(Local, NTuple)

    state_1 = @private FT (num_state,)
    aux_1 = @private FT (num_aux,)
    local_tendency = @private FT (num_state,)
    local_MI = @private FT (1,)
    shared_G = @localmem FT (Nq1 * Nq2 * Nq3, 3)
    shared_D = @localmem FT (Nqd, Nqd)

    @inbounds begin
        ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

        # generalization for different polynomial orders
        @unroll for l in 1:Nqd
            if dir == 1
                id = i
                ild = l + Nq1 * ((j - 1) + Nq2 * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq1 * ((l - 1) + Nq2 * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq1 * ((j - 1) + Nq2 * (l - 1))
            end
            shared_D[id, l] = D[id, l]
        end

        MJijk = MJ[ijk, e]
        @unroll for d in 1:dim
            shared_G[ijk, d] = MJijk * metrics[ijk, e].g[dir, d]
        end

        local_MI[1] = MJI[ijk, e]

        # Get the volume tendency (scaling by β done below)
        @unroll for s in 1:num_state
            local_tendency[s] = tendency[ijk, s, e]
        end

        @unroll for s in 1:num_state
            state_1[s] = state_prognostic[ijk, s, e]
        end
        @unroll for s in 1:num_aux
            aux_1[s] = state_auxiliary[ijk, s, e]
        end

        fill!(local_source, -zero(eltype(local_source)))
        # new_source!(local_source, q1, aux1)
        
        @unroll for s in 1:num_state
            local_source[s] += exp(state_1[s])
        end
        
        # add_source && Atum.source!(law, local_source, q1, aux1, dim, (dir,))

        @synchronize

        ijk = i + Nq1 * ((j - 1) + Nq2 * (k - 1))

        @views for l in 1:Nqd
            if dir == 1
                id = i
                ild = l + Nq1 * ((j - 1) + Nq2 * (k - 1))
            elseif dir == 2
                id = j
                ild = i + Nq1 * ((l - 1) + Nq2 * (k - 1))
            elseif dir == 3
                id = k
                ild = i + Nq1 * ((j - 1) + Nq2 * (l - 1))
            end

            @unroll for s in 1:num_state
                state_2[s] = state_prognostic[ild, s, e]
            end
            @unroll for s in 1:num_aux
                aux_2[s] = state_auxiliary[ild, s, e]
            end
            fill!(local_H, -zero(FT))
            #=
            numerical_volume_flux_first_order!(
                volume_numerical_flux_first_order,
                balance_law,
                local_H,
                state_1[:],
                aux_1[:],
                state_2,
                aux_2,
            )
            =#
            @unroll for s in 1:num_state
                local_tendency[s] -=
                    local_MI[1] *
                    shared_D[id, l] *
                    (
                        shared_G[ijk, 1] * local_H[1, s] +
                        shared_G[ijk, 2] * local_H[2, s] +
                        (
                            dim == 3 ? shared_G[ijk, 3] * local_H[3, s] :
                            -zero(FT)
                        )
                    )

                local_tendency[s] +=
                    local_MI[1] *
                    (
                        local_H[1, s] * shared_G[ild, 1] +
                        local_H[2, s] * shared_G[ild, 2] +
                        (
                            dim == 3 ? local_H[3, s] * shared_G[ild, 3] :
                            -zero(FT)
                        )
                    ) *
                    shared_D[l, id]
            end
        end

        @unroll for s in 1:num_state
            tendency[ijk, s, e] = local_tendency[s]
        end
    end
end
