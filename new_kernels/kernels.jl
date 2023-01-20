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
@inline function modified_boundaryflux!(law, flux, n⃗, q⁻, aux⁻, boundarytag)
    S = length(q⁻)
    @unroll for s in 1:S
        flux[s] = 0.0
    end
end

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

        comp_stream = flux_differencing_surface_direct!(device, workgroup_face)(
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
    @assert Nq⃗[1] == Nq⃗[2] == Nq⃗[3]

    dir = 1
    comp_stream = flux_differencing_volume_fused2!(device, workgroup)(
        dg.law, dq, q, derivatives_1d(cell)[dir],
        volume_form[dir],
        metrics(dg.grid), dg.MJ, dg.MJI,
        dg.auxstate,
        Val(increment), # increment
        Val(dim), Val(workgroup[1]),
        Val(numberofstates(dg.law)), Val(Naux),
        Val(true); # add source
        ndrange,
        dependencies=comp_stream
    )



    Nfp = Nq⃗[1] * Nq⃗[2]
    workgroup_face = (Nfp, 2)
    ndrange = (Nfp * Ne, 2)
    faceix⁻, faceix⁺ = faceindices(grid)
    facenormal, _ = components(facemetrics(grid))

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

    if dim == 3
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

##
@kernel function flux_differencing_surface_direct!(law,
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
            @views modified_surfaceflux!(numericalflux, law, flux[:], n⃗, q⁻, aux⁻, q⁺, aux⁺)
            dq[id⁻] -= fMJ * flux[:] * MJI[id⁻]
        else
            @views modified_boundaryflux!(law, flux[:], n⃗, q⁻, aux⁻, boundarytag)
            dq[id⁻] -= fMJ * flux[:] * MJI[id⁻] 
        end

    end
end

##
@kernel function flux_differencing_volume_fused2!(
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

    @unroll for dir in 1:dim
        #=
        if dir==1 & add_source
            modified_source!(law, dqijk, q1, aux1)
        else
            fill!(dqijk, -zero(FT))
        end
        =#
        fill!(dqijk, -zero(FT))
        @synchronize
    
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
        
            # qq2 = q[ijk, e]
            # aaux2 = auxstate[ijk,e]
        
            # fill!(flux, -zero(FT))
            # @views modified_volumeflux!(volume_numericalflux, law, flux[:, :], q1, aux1, q2, aux2)
        
            # ρ₁, ρu⃗₁, ρe₁ = unpackstate(law, q1)
            # ρ₂, ρu⃗₂, ρe₂ = unpackstate(law, q2)
            ρ₁ = q1[1]
            ρu⃗₁ = SVector(q1[2], q1[3], q1[4])
            ρe₁ = q1[5]
        
            ρ₂ = q2[1]
            ρu⃗₂ = SVector(q2[2], q2[3], q2[4])
            ρe₂ = q2[5]
        
            Φ₁ = 9.81 * aux1[3]
            u⃗₁ = ρu⃗₁ / ρ₁
            e₁ = ρe₁ / ρ₁
            p₁ = pressure(law, ρ₁, ρu⃗₁, ρe₁, Φ₁)
        
            Φ₂ = 9.81 * aux2[3]
            u⃗₂ = ρu⃗₂ / ρ₂
            e₂ = ρe₂ / ρ₂
            p₂ = pressure(law, ρ₂, ρu⃗₂, ρe₂, Φ₂)
        
            ρ_avg = avg(ρ₁, ρ₂)
            u⃗_avg = avg(u⃗₁, u⃗₂)
            e_avg = avg(e₁, e₂)
            p_avg = avg(p₁, p₂)
        
            # fluctuation
            α = ρ_avg / 2
        
            fρ = u⃗_avg * ρ_avg
            fρu⃗ = u⃗_avg * fρ' + (p_avg - α * (Φ₁ - Φ₂)) * I
            fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)
        
            @unroll for d in 1:dim
                flux[d, 1] = fρ[d]
                @unroll for dd in 1:dim
                    flux[d, dd+1] = fρu⃗[d, dd]
                end
                flux[d, 5] = fρe[d]
            end
        
            Ddn = MJIijk * shared_D[id, n]
            Dnd = MJIijk * shared_D[n, id]
            @unroll for s in 1:Ns
                @unroll for d in 1:dim
                    dqijk[s] -= Ddn * l_g[ijk, d, dir] * flux[d, s]
                    dqijk[s] += Dnd * l_g[ild, d, dir] * flux[d, s]
                end
            end
        
        end # end of matrix multiplication

        # always increment directions 2 and 3
        # only increment direction 1 if told to

        if increment | (dir != 1)
            dq[ijk, e] += dqijk
        else
            dq[ijk, e] = dqijk[:]
        end

    end # end of direction loop

end