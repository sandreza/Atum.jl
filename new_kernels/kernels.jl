using KernelAbstractions, Adapt
using KernelAbstractions.Extras: @unroll
# fix source and modifed boundary state
struct ModifiedFluxDifferencing{L,G,A1,A2,A3,A4,VF,SNF}
    law::L
    grid::G
    MJ::A1
    MJI::A2
    faceMJ::A3
    auxstate::A4
    volume_form::VF
    surface_numericalflux::SNF
end

# Always needs the following functions defined 
@inline modified_source!(args...) = nothing
@inline modified_volumeflux!(args...) = nothing
@inline modified_surfaceflux!(args...) = nothing
@inline modified_boundarystate(::Atum.AbstractBalanceLaw, n⃗, q⁻, aux⁻, tag) = q⁻, aux⁻

auxiliary(args...) = SVector(nothing)
Bennu.referencecell(dg::ModifiedFluxDifferencing) = referencecell(dg.grid)

function Adapt.adapt_structure(to, dg::ModifiedFluxDifferencing)
    names = fieldnames(ModifiedFluxDifferencing)
    args = ntuple(j -> adapt(to, getfield(dg, names[j])), length(names))
    ModifiedFluxDifferencing{typeof.(args)...}(args...)
end

function ModifiedFluxDifferencing(; law, grid, surface_numericalflux,
    volume_form=WeakForm(), auxstate=auxiliary.(Ref(law), points(grid)))
    cell = referencecell(grid)
    M = mass(cell)
    _, J = components(metrics(grid))
    MJ = M * J
    MJI = 1 ./ MJ

    faceM = facemass(cell)
    _, faceJ = components(facemetrics(grid))

    faceMJ = faceM * faceJ

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
    ModifiedFluxDifferencing{typeof.(args)...}(args...)
end
getdevice(dg::ModifiedFluxDifferencing) = Bennu.device(arraytype(referencecell(dg)))

function (dg::ModifiedFluxDifferencing)(dq, q, time; increment=true)
    cell = referencecell(dg)
    grid = dg.grid
    device = getdevice(dg)
    dim = ndims(cell)

    Ne = length(dg.grid)
    Nq⃗ = size(cell)

    @assert(length(eltype(q)) == numberofstates(dg.law))

    comp_stream = Event(device)

    Naux = eltype(eltype(dg.auxstate)) === Nothing ? 0 : length(eltype(dg.auxstate))

    workgroup = ntuple(i -> i <= dim ? Nq⃗[i] : 1, 3)
    ndrange = (Ne * workgroup[1], Base.tail(workgroup)...)

    volume_form = dg.volume_form
    surface_flux = dg.surface_numericalflux


    for dir in 1:dim
        comp_stream = flux_differencing_volume!(device, workgroup)(
            dg.law, dq, q, derivatives_1d(cell)[dir],
            volume_form[dir],
            metrics(dg.grid), dg.MJ, dg.MJI,
            dg.auxstate,
            dir == 1, # add_source
            Val(dir), Val(dim), Val(workgroup[1]), Val(workgroup[2]), Val(workgroup[3]),
            Val(numberofstates(dg.law)), Val(Naux),
            Val(dir == 1 ? increment : true);
            ndrange,
            dependencies=comp_stream
        )
    end



    faceix⁻, faceix⁺ = faceindices(grid)
    facenormal, _ = components(facemetrics(grid))
    
    for dir in 1:dim
        Nfp = round(Int, prod(Nq⃗) / Nq⃗[dir])
        workgroup_face = (Nfp, 2)
        ndrange = (Nfp * Ne, 2)

        comp_stream = flux_differencing_surface!(device, workgroup_face)(
            dg.law, dq, q,
            Val(Bennu.connectivityoffsets(cell, Val(2))),
            Val(dir),
            Val(numberofstates(dg.law)),
            surface_flux[dir],
            dg.MJI, faceix⁻, faceix⁺, dg.faceMJ, facenormal, boundaryfaces(grid),
            dg.auxstate,
            Val(dim);
            ndrange,
            dependencies=comp_stream
        )
    end
    

    #=
    Nfp = Nq⃗[1] * Nq⃗[2]
    workgroup_face = (Nfp, 2)
    ndrange = (Nfp * Ne, 2)

    comp_stream = flux_differencing_surface_together!(device, workgroup_face)(
        dg.law, dq, q,
        Val(Bennu.connectivityoffsets(cell, Val(2))),
        Val(numberofstates(dg.law)),
        surface_flux[1],
        dg.MJI, faceix⁻, faceix⁺, dg.faceMJ, facenormal, boundaryfaces(grid),
        dg.auxstate,
        Val(dim);
        ndrange,
        dependencies=comp_stream
    )
    =#

    wait(comp_stream)
end

## Kernels 
@kernel function flux_differencing_volume!(
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

    flux = @private FT (dim, Ns)

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
            q1[s] = q[ijk, e][s]
        end
        @unroll for s in 1:Naux
            aux1[s] = auxstate[ijk, e][s]
        end

        add_source && modified_source!(law, dqijk, q1, aux1)

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

            @views modified_volumeflux!(volume_numericalflux, law, flux[:, :], q1, aux1, q2, aux2)
            # modified_volumeflux!(volume_numericalflux, law, flux, q1, aux1, q2, aux2)

            Ddn = MJIijk * shared_D[id, n]
            Dnd = MJIijk * shared_D[n, id]
            @unroll for s in 1:Ns
                @unroll for d in 1:dim
                    dqijk[s] -= Ddn * l_g[ijk, d] * flux[d, s]
                    dqijk[s] += Dnd * l_g[ild, d] * flux[d, s]
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
@kernel function flux_differencing_surface!(law,
    dq,
    q,
    ::Val{faceoffsets},
    ::Val{dir},
    ::Val{Ns},
    numericalflux,
    MJI,
    faceix⁻,
    faceix⁺,
    faceMJ,
    facenormal,
    boundaryfaces,
    auxstate,
    ::Val{dim}) where {faceoffsets,dir,Ns,dim}
    @uniform begin
        FT = eltype(law)
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
    flux = @private FT (Ns,)

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
            q⁺, aux⁺ = modified_boundarystate(law, n⃗, q⁻, aux⁻, boundarytag)
        end

        @views modified_surfaceflux!(numericalflux, law, flux[:], n⃗, q⁻, aux⁻, q⁺, aux⁺)

        dq[id⁻] -= fMJ * flux[:] * MJI[id⁻]
    end
end

##
@kernel function flux_differencing_surface_together!(law,
    dq,
    q,
    ::Val{faceoffsets},
    ::Val{Ns},
    numericalflux,
    MJI,
    faceix⁻,
    faceix⁺,
    faceMJ,
    facenormal,
    boundaryfaces,
    auxstate,
    ::Val{dim}) where {faceoffsets,Ns,dim}
    @uniform begin
        FT = eltype(law)
    end

    e⁻ = @index(Group, Linear)
    i, fi = @index(Local, NTuple)
    flux = @private FT (Ns,)

    faces = 1:2
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
            q⁺, aux⁺ = modified_boundarystate(law, n⃗, q⁻, aux⁻, boundarytag)
        end

        @views modified_surfaceflux!(numericalflux, law, flux[:], n⃗, q⁻, aux⁻, q⁺, aux⁺)

        dq[id⁻] -= fMJ * flux[:] * MJI[id⁻]
    end

    @synchronize

    faces = 3:4
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
            q⁺, aux⁺ = modified_boundarystate(law, n⃗, q⁻, aux⁻, boundarytag)
        end

        @views modified_surfaceflux!(numericalflux, law, flux[:], n⃗, q⁻, aux⁻, q⁺, aux⁺)

        dq[id⁻] -= fMJ * flux[:] * MJI[id⁻]
    end

    @synchronize

    if dim ==3 
        faces = 5:6
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
                q⁺, aux⁺ = modified_boundarystate(law, n⃗, q⁻, aux⁻, boundarytag)
            end

            @views modified_surfaceflux!(numericalflux, law, flux[:], n⃗, q⁻, aux⁻, q⁺, aux⁺)

            dq[id⁻] -= fMJ * flux[:] * MJI[id⁻]
        end
    end

end
