
##
@benchmark let
    cell = referencecell(dg)
    Nq = size(cell)[1]
    @assert all(size(cell) .== Nq)
    increment = true
    @assert(length(eltype(q)) == numberofstates(dg.law))
    cell = referencecell(dg)
    grid = dg.grid
    device = Atum.getdevice(dg)
    dim = ndims(cell)


    comp_stream = Atum.Event(device)


    comp_stream = Atum.launch_volumeterm(dg.volume_form, dqq, qq, dg;
        increment, dependencies=comp_stream)
    
    Nfp = Nq^(dim - 1)
    workgroup_face = (Nfp,)
    ndrange = (Nfp * length(grid),)
    faceix⁻, faceix⁺ = faceindices(grid)
    facenormal, _ = components(facemetrics(grid))

    comp_stream = Atum.surfaceterm!(device, workgroup_face)(
        dg.law,
        dqq,
        qq,
        Val(Bennu.connectivityoffsets(cell, Val(2))),
        dg.surface_numericalflux,
        dg.MJI,
        faceix⁻,
        faceix⁺,
        dg.faceMJ,
        facenormal,
        boundaryfaces(grid),
        dg.auxstate,
        Val(Atum.directions(dg));
        ndrange,
        dependencies=comp_stream
    )
    
    wait(comp_stream)
end
##

@benchmark let
    cell = referencecell(dg)
    grid = dg.grid
    device = Atum.getdevice(dg)
    dim = ndims(cell)

    Ne = length(dg.grid)
    Nq⃗ = size(cell)
    increment = true
    @assert(length(eltype(q)) == numberofstates(dg.law))

    Naux = eltype(eltype(dg.auxstate)) === Nothing ? 0 : length(eltype(dg.auxstate))

    workgroup = ntuple(i -> i <= dim ? Nq⃗[i] : 1, 3)
    ndrange = (Ne * workgroup[1], Base.tail(workgroup)...)

    comp_stream = Event(device)

    for dir in 1:3
        comp_stream = volume_withflux!(device, workgroup)(
            dg.law, dqq, qq, derivatives_1d(cell)[dir],
            dg.volume_form.volume_numericalflux,
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

    
    comp_stream = Atum.Event(device)
    faceix⁻, faceix⁺ = faceindices(grid)
    facenormal, _ = components(facemetrics(grid))
    for dir in 1:dim
        Nfp = round(Int, prod(Nq⃗) / Nq⃗[dir])
        workgroup_face = (Nfp, 2)
        ndrange = (Nfp * Ne, 2)

        comp_stream = surface_withflux!(device, workgroup_face)(
            dg.law, dqq, qq,
            Val(Bennu.connectivityoffsets(cell, Val(2))),
            Val(dir),
            RoeFlux(),
            dg.MJI, faceix⁻, faceix⁺, dg.faceMJ, facenormal, boundaryfaces(grid),
            dg.auxstate,
            Val(dim);
            ndrange,
            dependencies=comp_stream
        )
    end
    
    wait(comp_stream)
end

##
@benchmark let
    cell = referencecell(dg)
    Nq = size(cell)[1]
    @assert all(size(cell) .== Nq)
    increment = true
    @assert(length(eltype(q)) == numberofstates(dg.law))
    cell = referencecell(dg)
    grid = dg.grid
    device = Atum.getdevice(dg)
    dim = ndims(cell)


    comp_stream = Atum.Event(device)


    comp_stream = Atum.launch_volumeterm(dg.volume_form, q0, q, dg;
        increment, dependencies=comp_stream)

    Nfp = Nq^(dim - 1)
    workgroup_face = (Nfp,)
    ndrange = (Nfp * length(grid),)
    faceix⁻, faceix⁺ = faceindices(grid)
    facenormal, _ = components(facemetrics(grid))

    comp_stream = Atum.surfaceterm!(device, workgroup_face)(
        dg.law,
        q0,
        q,
        Val(Bennu.connectivityoffsets(cell, Val(2))),
        dg.surface_numericalflux,
        dg.MJI,
        faceix⁻,
        faceix⁺,
        dg.faceMJ,
        facenormal,
        boundaryfaces(grid),
        dg.auxstate,
        Val(Atum.directions(dg));
        ndrange,
        dependencies=comp_stream
    )

    wait(comp_stream)
end
##

@benchmark let
    cell = referencecell(dg)
    grid = dg.grid
    device = Atum.getdevice(dg)
    dim = ndims(cell)

    Ne = length(dg.grid)
    Nq⃗ = size(cell)
    increment = true
    @assert(length(eltype(q)) == numberofstates(dg.law))

    Naux = eltype(eltype(dg.auxstate)) === Nothing ? 0 : length(eltype(dg.auxstate))

    workgroup = ntuple(i -> i <= dim ? Nq⃗[i] : 1, 3)
    ndrange = (Ne * workgroup[1], Base.tail(workgroup)...)

    comp_stream = Event(device)

    for dir in 1:3
        comp_stream = volume_withloop2!(device, workgroup)(
            dg.law, dqq, qq, derivatives_1d(cell)[dir],
            dg.volume_form.volume_numericalflux,
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


    comp_stream = Atum.Event(device)
    faceix⁻, faceix⁺ = faceindices(grid)
    facenormal, _ = components(facemetrics(grid))
    for dir in 1:dim
        Nfp = round(Int, prod(Nq⃗) / Nq⃗[dir])
        workgroup_face = (Nfp, 2)
        ndrange = (Nfp * Ne, 2)

        comp_stream = surface_withflux!(device, workgroup_face)(
            dg.law, dqq, qq,
            Val(Bennu.connectivityoffsets(cell, Val(2))),
            Val(dir),
            RoeFlux(),
            dg.MJI, faceix⁻, faceix⁺, dg.faceMJ, facenormal, boundaryfaces(grid),
            dg.auxstate,
            Val(dim);
            ndrange,
            dependencies=comp_stream
        )
    end

    wait(comp_stream)
end

##

mypressure(ρ, ρu⃗, ρe, Φ) = 0.4 * (ρe - ρu⃗' * ρu⃗ / 2ρ - ρ * Φ)

function myvolumeflux!(vflux,
    q₁, aux₁, q₂, aux₂)
    ρ₁, ρu⃗₁, ρe₁ = q₁[1], @SVector[q₁[2], q₁[3], q₁[4]], q₁[5]
    ρ₂, ρu⃗₂, ρe₂ = q₂[1], @SVector[q₂[2], q₂[3], q₂[4]], q₂[5]

    Φ₁ = 9.81 * aux₁[3]
    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = mypressure(ρ₁, ρu⃗₁, ρe₁, Φ₁)

    Φ₂ = 9.81 * aux₂[3]
    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = mypressure(ρ₂, ρu⃗₂, ρe₂, Φ₂)

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    e_avg = Atum.avg(e₁, e₂)
    p_avg = Atum.avg(p₁, p₂)

    fρ = u⃗_avg * ρ_avg
    fρu⃗ = u⃗_avg * fρ' + p_avg * I
    fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)

    # fluctuation
    α = ρ_avg / 2
    fρu⃗ -= α * (Φ₁ - Φ₂) * I

    vflux .+= hcat(fρ, fρu⃗, fρe)
    return nothing
end

function mysurfaceflux2!(nf, n⃗, q⁻, aux⁻, q⁺, aux⁺)
    return nothing
end
function mysurfaceflux3!(nf, n⃗, q⁻, aux⁻, q⁺, aux⁺)
    Φ = 9.81 * aux⁻[3]

    ρ⁻, ρu⃗⁻, ρe⁻ = q⁻[1], @SVector[q⁻[2], q⁻[3], q⁻[4]], q⁻[5]
    ρ⁺, ρu⃗⁺, ρe⁺ = q⁺[1], @SVector[q⁺[2], q⁺[3], q⁺[4]], q⁺[5]

    ρ₁, ρu⃗₁, ρe₁ = ρ⁻, ρu⃗⁻, ρe⁻
    ρ₂, ρu⃗₂, ρe₂ = ρ⁺, ρu⃗⁺, ρe⁺

    Φ₁ = 9.81 * aux⁻[3]
    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = mypressure(ρ₁, ρu⃗₁, ρe₁, Φ₁)

    Φ₂ = 9.81 * aux⁺[3]
    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = mypressure(ρ₂, ρu⃗₂, ρe₂, Φ₂)

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    e_avg = Atum.avg(e₁, e₂)
    p_avg = Atum.avg(p₁, p₂)

    uₙ_avg = u⃗_avg' * n⃗
    fρ = uₙ_avg * ρ_avg
    fρu⃗ = uₙ_avg * u⃗_avg * ρ_avg + p_avg * n⃗
    fρe = uₙ_avg * (ρ_avg * e_avg + p_avg)

    fⁿ = SVector(fρ, fρu⃗..., fρe)

    u⃗⁻ = ρu⃗⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    p⁻ = mypressure(ρ⁻, ρu⃗⁻, ρe⁻, Φ₁)
    h⁻ = e⁻ + p⁻ / ρ⁻
    c⁻ = sqrt(1.4 * p⁻ / ρ⁻)

    u⃗⁺ = ρu⃗⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    p⁺ = mypressure(ρ⁺, ρu⃗⁺, ρe⁺, Φ₂)
    h⁺ = e⁺ + p⁺ / ρ⁺
    c⁺ = sqrt(1.4 * p⁺ / ρ⁺)

    ρ = Atum.avg(ρ⁻ , ρ⁺)
    u⃗ = Atum.avg(u⃗⁻, u⃗⁺)
    h = Atum.avg(h⁻, h⁺)
    c = Atum.avg(c⁻, c⁺)

    uₙ = u⃗' * n⃗

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu⃗ = u⃗⁺ - u⃗⁻
    Δuₙ = Δu⃗' * n⃗

    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² * 0.5
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² * 0.5
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ

    fp_ρ = (w1 + w2 + w3) * 0.5
    fp_ρu⃗ = (w1 * (u⃗ - c * n⃗) +
              w2 * (u⃗ + c * n⃗) +
              w3 * u⃗ +
              w4 * (Δu⃗ - Δuₙ * n⃗)) * 0.5
    fp_ρe = (w1 * (h - c * uₙ) +
             w2 * (h + c * uₙ) +
             w3 * (u⃗' * u⃗ * 0.5 + Φ) +
             w4 * (u⃗' * Δu⃗ - uₙ * Δuₙ)) * 0.5

    nf .+= fⁿ - SVector(fp_ρ, fp_ρu⃗..., fp_ρe)
    return nothing
end
##
q₁ = q[1]
q₂ = q[2]
aux₁ = x⃗[1]
aux₂ = x⃗[2]
vflux = MArray{Tuple{3,5},Float64}(undef)
fill!(vflux, 0.0)
myvolumeflux!(vflux, q₁, aux₁, q₂, aux₂)

nf = MArray{Tuple{5},Float64}(undef)
fill!(nf, 0.0)
n⃗ = @SVector[1.0, 0.0, 0.0]
mysurfaceflux!(nf, n⃗, q₁, aux₁, q₂, aux₂)

##

@benchmark let
    cell = referencecell(dg)
    grid = dg.grid
    device = Atum.getdevice(dg)
    dim = ndims(cell)

    Ne = length(dg.grid)
    Nq⃗ = size(cell)
    increment = true
    @assert(length(eltype(q)) == numberofstates(dg.law))

    Naux = eltype(eltype(dg.auxstate)) === Nothing ? 0 : length(eltype(dg.auxstate))

    workgroup = ntuple(i -> i <= dim ? Nq⃗[i] : 1, 3)
    ndrange = (Ne * workgroup[1], Base.tail(workgroup)...)

    comp_stream = Event(device)

    for dir in 1:3
        comp_stream = volume_functional3!(device, workgroup)(
            myvolumeflux!, dqq, qq, derivatives_1d(cell)[dir],
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

    
    comp_stream = Atum.Event(device)
    faceix⁻, faceix⁺ = faceindices(grid)
    facenormal, _ = components(facemetrics(grid))
    for dir in 1:dim
        Nfp = round(Int, prod(Nq⃗) / Nq⃗[dir])
        workgroup_face = (Nfp, 2)
        ndrange = (Nfp * Ne, 2)

        comp_stream = surface_functional4!(device, workgroup_face)(
            mysurfaceflux3!, dqq, qq,
            Val(Bennu.connectivityoffsets(cell, Val(2))),
            Val(dir),
            dg.MJI, faceix⁻, faceix⁺, dg.faceMJ, facenormal, boundaryfaces(grid),
            dg.auxstate,
            Val(dim);
            ndrange,
            dependencies=comp_stream
        )
    end
    
    wait(comp_stream)
end

# @benchmark dg(dq, q, 0.0)