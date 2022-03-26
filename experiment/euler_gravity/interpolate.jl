"""
    baryweights(r)
returns the barycentric weights associated with the array of points `r`
Reference:
  [Berrut2004](@cite)
"""
function baryweights(r::AbstractVector{T}) where {T}
    Np = length(r)
    wb = ones(T, Np)

    for j in 1:Np
        for i in 1:Np
            if i != j
                wb[j] = wb[j] * (r[j] - r[i])
            end
        end
        wb[j] = T(1) / wb[j]
    end
    wb
end
# baryweights(cell.points_1d[1][:])

function checkgl(x, rx)
    for i in eachindex(rx)
        if abs(x - rx[i]) ≤ eps(1e8)
            return i
        end
    end
    return 0
end

function lagrange_pole(ξ, r, ω, i, ξcheck)
    if ξcheck == 0
        Δξ = (ξ - r[i])
        pole = ω[i] / Δξ
        ii = i
        return pole, ii, i
    else
        pole = 1.0
        i = eachindex(r)[end]
        ii = ξcheck
        return pole, ii, i
    end
end

#=
@assert length(x⃗) == length(r) == length(ω)
index_check = [checkgl(x⃗[i], r[i]) for eachindex(r)]
for rx⃗ in eachindex(r) 
    for i in eachindex(rx⃗)
        pole, index = lagrange_pole(ξ, r, ω, i, ξcheck)
    end
end
=#

function lagrange_eval(f, newx, newy, newz, rx, ry, rz, ωx, ωy, ωz)
    icheck = checkgl(newx, rx)
    jcheck = checkgl(newy, ry)
    kcheck = checkgl(newz, rz)
    numerator = zeros(1)
    denominator = zeros(1)
    for k in eachindex(rz)
        polez, kk, k = lagrange_pole(newz, rz, ωz, k, kcheck)
        for j in eachindex(ry)
            poley, jj, j = lagrange_pole(newy, ry, ωy, j, jcheck)
            for i in eachindex(rx)
                polex, ii, i = lagrange_pole(newx, rx, ωx, i, icheck)
                numerator[1] += f[ii, jj, kk] * polex * poley * polez
                denominator[1] += polex * poley * polez
            end
        end
    end
    return numerator[1] / denominator[1]
end

function lagrange_eval_2(f, ξ, r, ω)
    newx, newy, newz = ξ
    rx, ry, rz = r
    ωx, ωy, ωz = ω
    icheck = checkgl(newx, rx)
    jcheck = checkgl(newy, ry)
    kcheck = checkgl(newz, rz)
    numerator = zeros(1)
    denominator = zeros(1)
    for k in eachindex(rz)
        polez, kk, k = lagrange_pole(newz, rz, ωz, k, kcheck)
        for j in eachindex(ry)
            poley, jj, j = lagrange_pole(newy, ry, ωy, j, jcheck)
            for i in eachindex(rx)
                polex, ii, i = lagrange_pole(newx, rx, ωx, i, icheck)
                numerator[1] += f[ii, jj, kk] * polex * poley * polez
                denominator[1] += polex * poley * polez
            end
        end
    end
    return numerator[1] / denominator[1]
end

function lagrange_eval_3(f, ξ, r, ω, Nq)
    newx, newy, newz = ξ
    rx, ry, rz = r
    ωx, ωy, ωz = ω
    icheck = checkgl(newx, rx)
    jcheck = checkgl(newy, ry)
    kcheck = checkgl(newz, rz)
    numerator = zeros(1)
    denominator = zeros(1)
    for k in eachindex(rz)
        polez, kk, k = lagrange_pole(newz, rz, ωz, k, kcheck)
        for j in eachindex(ry)
            poley, jj, j = lagrange_pole(newy, ry, ωy, j, jcheck)
            for i in eachindex(rx)
                polex, ii, i = lagrange_pole(newx, rx, ωx, i, icheck)
                I = ii + Nq[1] * (jj - 1 + Nq[2] * (kk - 1))
                poles = polex * poley * polez
                @inbounds numerator[1] += f[I] * poles
                denominator[1] += poles
            end
        end
    end
    return numerator[1] / denominator[1]
end

function lagrange_eval_nb(f, ξ, r, ω)
    icheck = checkgl(newx, rx)
    jcheck = checkgl(newy, ry)
    kcheck = checkgl(newz, rz)
    numerator = zeros(1)
    denominator = zeros(1)
    for k in eachindex(rz)
        if kcheck == 0
            Δz = (newz .- rz[k])
            polez = ωz[k] ./ Δz
            kk = k
        else
            polez = 1.0
            k = eachindex(rz)[end]
            kk = kcheck
        end
        for j in eachindex(ry)
            if jcheck == 0
                Δy = (newy .- ry[j])
                poley = ωy[j] ./ Δy
                jj = j
            else
                poley = 1.0
                j = eachindex(ry)[end]
                jj = jcheck
            end
            for i in eachindex(rx)
                if icheck == 0
                    Δx = (newx .- rx[i])
                    polex = ωx[i] ./ Δx
                    ii = i
                else
                    polex = 1.0
                    i = eachindex(rx)[end]
                    ii = icheck
                end
                numerator[1] += f[ii, jj, kk] * polex * poley * polez
                denominator[1] += polex * poley * polez
            end
        end
    end
    return numerator[1] / denominator[1]
end


## goal, figure out data structures to write kernel
e_num = 4
newgrid = zeros(e_num, e_num, e_num) # number of elements in grid
newf = 0 * newgrid
ξlist = -ones(3, length(newgrid)) # same point in each element 
elist = collect(1:(e_num^3))

r = [cell.points_1d[i][:] for i in eachindex(cell.points_1d)]
ω = [baryweights(cell.points_1d[i][:]) for i in eachindex(cell.points_1d)]
# ω = [cell.weights_1d[i][:] for i in eachindex(cell.weights_1d)]
bw[1]
ξ = [0.0, 0.0, 0.0]
# ξ = [-1.0, -1.0, -1.0]
icheck = checkgl(0.0, r[1])
#=
@benchmark let
    for I in eachindex(newf)
        e = elist[I]
        ξ = ξlist[:, I]
        newf[I] = lagrange_eval(reshape(x[:, e], Nq, Nq, Nq), ξ..., r..., ω...)
    end
end
=#
##
using KernelAbstractions
@kernel function first_pass!(newf, oldf, elist, ξlist, r, ω, ::Val{Nq}) where {Nq}
    I = @index(Global, Linear)
    oldfijk = @private Nq
    ξ = ξlist[I]
    e = elist[I]
    for II in eachindex(oldfijk)
        oldfijk[II] = oldf[II, e]
    end
    newf[I] = lagrange_eval_2(oldfijk, ξ, r, ω)
end

@kernel function second_pass!(newf, oldf, elist, ξlist, r, ω, ::Val{Nq}) where {Nq}
    I = @index(Global, Linear)
    ξ = ξlist[I]
    e = elist[I]
    oldfijk = view(oldf, :, e)
    newf[I] = lagrange_eval_3(oldfijk, ξ, r, ω, Nq)
end

function interpolate_field!(newf, oldf, elist, ξlist, r, ω, Nq)
    kernel = second_pass!(CPU(), 16)
    event = kernel(newf, oldf, elist, ξlist, r, ω, Val(Nq), ndrange=size(newf))
    wait(event)
    return nothing
end
