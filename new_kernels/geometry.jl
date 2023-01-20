# det(J⃗[1]) * J[1]

function contravariant_basis(grid)
    J⃗, J = components(metrics(grid))
    m, _ = size(J⃗[1])
    if m == 1
        a⃗¹ = contravariant_basis.(J⃗, Ref(1))
        return a⃗¹
    elseif m == 2
        a⃗¹ = contravariant_basis.(J⃗, Ref(1))
        a⃗² = contravariant_basis.(J⃗, Ref(2))
        return a⃗¹, a⃗²
    else
        a⃗¹ = contravariant_basis.(J⃗, Ref(1))
        a⃗² = contravariant_basis.(J⃗, Ref(2))
        a⃗³ = contravariant_basis.(J⃗, Ref(3))
        return a⃗¹, a⃗², a⃗³
    end
end

function contravariant_basis(J⃗, i)
    return SVector(J⃗[i, :])
end


#=
M = mass(cell)
_, J = components(metrics(grid))
MJ = M * J
MJI = 1 ./ MJ

faceM = facemass(cell)
_, faceJ = components(facemetrics(grid))

faceMJ = faceM * faceJ

derivatives_1d(cell)[dir]

# in kernel 
metrics(dg.grid)

##
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

faceix⁻, faceix⁺ = faceindices(grid)
facenormal, _ = components(facemetrics(grid))
Val(Bennu.connectivityoffsets(cell, Val(2)))

Val(Bennu.connectivityoffsets(cell, Val(2))),
dg.MJI, faceix⁻, faceix⁺, dg.faceMJ, facenormal, boundaryfaces(grid)

metrics[ijk, e].g[dir, d]


=#

function construct_metric_terms()
    metrics(grid)
end