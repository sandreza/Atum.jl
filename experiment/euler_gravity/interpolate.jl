using KernelAbstractions, CUDAKernels, StaticArrays

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

@kernel function interpolate_field_kernel!(newf, oldf, elist, ξlist, r, ω, ::Val{Nq}) where {Nq}
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
    kernel = interpolate_field_kernel!(arch, blocksize)
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
    else
        ξlist = Array([@SVector[0.0, 0.0, 0.0] for i in eachindex(newgrid)])
        elist = Array(zeros(Int, length(newgrid)))
    end
    x, y, z = components(oldgrid.points)
    nex, ney, nez = (size(oldgrid.vertices) .- 1)
    xinfo = (extrema(x)..., nex)
    yinfo = (extrema(y)..., ney)
    zinfo = (extrema(z)..., nez)

    if arch isa CUDADevice
        kernel! = cube_kernel!(arch, 256)
        event = kernel!(ξlist, elist, newgrid, x, xinfo, y, yinfo, z, zinfo, ndrange=size(elist))
        wait(event)
    else
        kernel! = cube_kernel!(arch, 16)
        event = kernel!(ξlist, elist, newgrid, x, xinfo, y, yinfo, z, zinfo, ndrange=size(elist))
        wait(event)
    end

    return ξlist, elist
end

function cube_single_element_index(cell_coords, Kv, Kh)
    ev = cell_coords[1] # vertical element
    ehi = cell_coords[2] # local face i index
    ehj = cell_coords[3] # local face j index
    ehf = cell_coords[4] # local face index
    return ev + Kv * (ehi - 1 + Kh * (ehj - 1 + Kh * (ehf - 1)))
end

sphericaltocartesian(θ, ϕ, r) = SVector(r * cos(θ) * sin(ϕ), r * sin(θ) * sin(ϕ), r * cos(ϕ))
