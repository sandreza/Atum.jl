export SingleDirection

struct SingleDirection{L,G,A1,A2,A3,A4,VF,SNF}
    law::L
    grid::G
    MJ::A1
    MJI::A2
    faceMJ::A3
    auxstate::A4
    volume_form::VF
    surface_numericalflux::SNF
end

Bennu.referencecell(dg::SingleDirection) = referencecell(dg.grid)

function Adapt.adapt_structure(to, dg::SingleDirection)
    names = fieldnames(SingleDirection)
    args = ntuple(j -> adapt(to, getfield(dg, names[j])), length(names))
    SingleDirection{typeof.(args)...}(args...)
end

function SingleDirection(; law, grid, surface_numericalflux,
    volume_form=WeakForm())
    cell = referencecell(grid)
    M = mass(cell)
    _, J = components(metrics(grid))
    MJ = M * J
    MJI = 1 ./ MJ

    faceM = facemass(cell)
    _, faceJ = components(facemetrics(grid))

    faceMJ = faceM * faceJ

    auxstate = auxiliary.(Ref(law), points(grid))

    args = (law, grid, MJ, MJI, faceMJ, auxstate,
        volume_form, surface_numericalflux)
    SingleDirection{typeof.(args)...}(args...)
end
getdevice(dg::SingleDirection) = Bennu.device(arraytype(referencecell(dg)))

function (dg::SingleDirection)(dq, q, time; increment=true, dir = 3)
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

    comp_stream = volume_term_dir!(device, workgroup)(
        dg.law, dq, q, derivatives_1d(cell)[dir],
        dg.volume_form.volume_numericalflux,
        metrics(dg.grid), dg.MJ, dg.MJI,
        dg.auxstate,
        false, # don't add_source
        Val(dir), Val(dim), Val(workgroup[1]), Val(workgroup[2]), Val(workgroup[3]),
        Val(numberofstates(dg.law)), Val(Naux),
        Val(increment);
        ndrange,
        dependencies=comp_stream
    )

    faceix⁻, faceix⁺ = faceindices(grid)
    facenormal, _ = components(facemetrics(grid))


    Nfp = round(Int, prod(Nq⃗) / Nq⃗[dir])
    workgroup_face = (Nfp, 2)
    ndrange = (Nfp * Ne, 2)

    comp_stream = surface_term_dir!(device, workgroup_face)(
        dg.law, dq, q,
        Val(Bennu.connectivityoffsets(cell, Val(2))),
        Val(dir),
        dg.surface_numericalflux,
        dg.MJI, faceix⁻, faceix⁺, dg.faceMJ, facenormal, boundaryfaces(grid),
        dg.auxstate,
        Val(dim);
        ndrange,
        dependencies=comp_stream
    )

    wait(comp_stream)
end