cell = referencecell(new_dg)
grid = new_dg.grid
ldevice = Atum.getdevice(dg)
dim = ndims(cell)
Ne = length(new_dg.grid)
Nq⃗ = size(cell)
@assert(length(eltype(q)) == numberofstates(new_dg.law))
comp_stream = Event(ldevice)
Naux = eltype(eltype(new_dg.auxstate)) === Nothing ? 0 : length(eltype(new_dg.auxstate))

volume_form = new_dg.volume_form
surface_flux = new_dg.surface_numericalflux
increment = true

q .= initial_condition.(Ref(law), points(grid));
dq = fieldarray(undef, law, grid);
dq .= initial_condition.(Ref(law), points(grid));
mets = contravariant_basis(grid)

volume_timing = @benchmark CUDA.@sync begin
    comp_stream = Event(ldevice)
    workgroup = ntuple(i -> i <= dim ? Nq⃗[i] : 1, 3)
    ndrange = (Ne * workgroup[1], Base.tail(workgroup)...)
    for dir in 1:dim
        comp_stream = flux_differencing_volume!(ldevice, workgroup)(
            new_dg.law, dq, q, derivatives_1d(cell)[dir],
            volume_form[dir],
            metrics(new_dg.grid), new_dg.MJ, new_dg.MJI,
            new_dg.auxstate,
            dir == 1, # add_source
            Val(dir), Val(dim), Val(workgroup[1]), Val(workgroup[2]), Val(workgroup[3]),
            Val(numberofstates(new_dg.law)), Val(Naux),
            Val(dir == 1 ? increment : true);
            ndrange,
            dependencies=comp_stream
        )
    end
    wait(comp_stream)
end

q .= initial_condition.(Ref(law), points(grid));
dq .= initial_condition.(Ref(law), points(grid));

volume_fused_timing =  @benchmark CUDA.@sync begin
    comp_stream = Event(ldevice)
    workgroup = ntuple(i -> i <= dim ? Nq⃗[i] : 1, 3)
    ndrange = (Ne * workgroup[1], Base.tail(workgroup)...)
    dir = 1
    comp_stream = flux_differencing_volume_fused_2!(ldevice, workgroup)(
        new_dg.law, dq, q, derivatives_1d(cell)[dir],
        volume_form[dir],
        metrics(new_dg.grid), new_dg.MJ, new_dg.MJI,
        new_dg.auxstate,
        Val(true), # increment
        Val(dim), Val(workgroup[1]), 
        Val(numberofstates(new_dg.law)), Val(Naux),
        Val(true); # add source
        ndrange,
        dependencies=comp_stream
    )
    wait(comp_stream)
end



faceix⁻, faceix⁺ = faceindices(grid)
facenormal, _ = components(facemetrics(grid))

surface_timing = @benchmark CUDA.@sync begin
    comp_stream = Event(ldevice)
    for dir in 1:dim
        Nfp = round(Int, prod(Nq⃗) / Nq⃗[dir])
        workgroup_face = (Nfp, 2)
        ndrange = (Nfp * Ne, 2)

        comp_stream = flux_differencing_surface_direct!(ldevice, workgroup_face)(
            new_dg.law, dq, q,
            Val(Bennu.connectivityoffsets(cell, Val(2))),
            Val(dir),
            Val(numberofstates(new_dg.law)),
            surface_flux[dir],
            new_dg.MJI, faceix⁻, faceix⁺, new_dg.faceMJ, facenormal, boundaryfaces(grid),
            new_dg.auxstate,
            Val(dim);
            ndrange,
            dependencies=comp_stream
        )
    end
    wait(comp_stream)
end

surface_fused_timing = @benchmark CUDA.@sync begin
    Nfp = Nq⃗[1] * Nq⃗[2]
    workgroup_face = (Nfp, 2)
    ndrange = (Nfp * Ne, 2)
    comp_stream = Event(ldevice)
    comp_stream = flux_differencing_surface_together!(ldevice, workgroup_face)(
        new_dg.law, dq, q,
        Val(Bennu.connectivityoffsets(cell, Val(2))),
        Val(numberofstates(new_dg.law)),
        surface_flux[1],
        new_dg.MJI, faceix⁻, faceix⁺, new_dg.faceMJ, facenormal, boundaryfaces(grid),
        new_dg.auxstate,
        Val(dim);
        ndrange,
        dependencies=comp_stream
    )
    wait(comp_stream)

end
