using KernelAbstractions, CUDAKernels

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
    numerator = 0.0
    denominator = 0.0
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

@kernel function a7_pass!(newf, oldf, elist, ξlist, r, ω, ::Val{Nq}) where {Nq}
    I = @index(Global, Linear)
    ξ = ξlist[I]
    e = elist[I]
    oldfijk = view(oldf, :, e)
    newx, newy, newz = ξ
    rx, ry, rz = r
    ωx, ωy, ωz = ω
    icheck = checkgl(newx, rx)
    jcheck = checkgl(newy, ry)
    kcheck = checkgl(newz, rz)
    numerator = 0.0
    denominator = 0.0
    for k in eachindex(rz)
        polez, kk, k = lagrange_pole(newz, rz, ωz, k, kcheck)
        for j in eachindex(ry)
            poley, jj, j = lagrange_pole(newy, ry, ωy, j, jcheck)
            for i in eachindex(rx)
                polex, ii, i = lagrange_pole(newx, rx, ωx, i, icheck)
                II = ii + Nq[1] * (jj - 1 + Nq[2] * (kk - 1))
                poles = polex * poley * polez
                @inbounds numerator += oldfijk[II] * poles
                denominator += poles
            end
        end
    end
    newf[I] = numerator / denominator
end

function interpolate_field!(newf, oldf, elist, ξlist, r, ω, Nq; arch=CUDADevice(), blocksize=256)
    kernel = a7_pass!(arch, blocksize)
    event = kernel(newf, oldf, elist, ξlist, r, ω, Val(Nq), ndrange=size(newf))
    wait(event)
    return nothing
end

function get_element(xnew, ynew, znew, xinfo, yinfo, zinfo)
    xmin, xmax, nex = xinfo
    ymin, ymax, ney = yinfo
    zmin, zmax, nez = zinfo
    ex = ceil(Int, (xnew - xmin) / (xmax - xmin) * nex)
    ex = min(max(ex, 1), nex)
    ey = ceil(Int, (ynew - ymin) / (ymax - ymin) * ney)
    ey = min(max(ey, 1), ney)
    ez = ceil(Int, (znew - zmin) / (zmax - zmin) * nez)
    ez = min(max(ez, 1), nez)
    e = ex + nex * (ey - 1 + ney * (ez - 1))
    return e
end

rescale(x, xmin, xmax) = 2 * (x - xmin) / (xmax - xmin) - 1

function get_reference(x, y, z, oldx, oldy, oldz)
    xmin, xmax = extrema(oldx)
    ymin, ymax = extrema(oldy)
    zmin, zmax = extrema(oldz)

    ξ1 = rescale(x, xmin, xmax)
    ξ2 = rescale(y, ymin, ymax)
    ξ3 = rescale(z, zmin, zmax)

    return @SVector[ξ1, ξ2, ξ3]
end

@kernel function cube_kernel!(ξlist, elist, newgrid, x, xinfo, y, yinfo, z, zinfo)
    I = @index(Global, Linear)
    xnew, ynew, znew = newgrid[I]
    e = get_element(xnew, ynew, znew, xinfo, yinfo, zinfo)
    oldx = view(x, :, e)
    oldy = view(y, :, e)
    oldz = view(z, :, e)
    ξ = get_reference(xnew, ynew, znew, oldx, oldy, oldz)
    ξlist[I] = ξ
    elist[I] = e
end

function cube_interpolate(newgrid, oldgrid; arch=CUDADevice())
    if arch isa CUDADevice
        ξlist = CuArray([@SVector[0.0, 0.0, 0.0] for i in eachindex(newgrid)])
        elist = CuArray(zeros(Int, length(newgrid)))
    end
    x, y, z = components(grid.points)
    nex, ney, nez = (size(oldgrid.vertices) .- 1)
    xinfo = (extrema(x)..., nex)
    yinfo = (extrema(y)..., ney)
    zinfo = (extrema(z)..., nez)
    if arch isa CUDADevice
        kernel! = cube_kernel!(arch, 256)
        event = kernel!(ξlist, elist, newgrid, x, xinfo, y, yinfo, z, zinfo, ndrange=size(elist))
        wait(event)
    end

    return ξlist, elist
end
