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

@kernel function interpolate_field_kernel_2D!(newf, oldf, elist, ξlist, r, ω, ::Val{Nq}) where {Nq}
    I = @index(Global, Linear)
    ξ = ξlist[I]
    e = elist[I]
    oldfijk = view(oldf, :, e)
    newx, newy = ξ
    rx, ry = r
    ωx, ωy = ω
    icheck = checkgl(newx, rx)
    jcheck = checkgl(newy, ry)
    numerator = 0.0
    denominator = 0.0
    for j in eachindex(ry)
        poley, jj, j = lagrange_pole(newy, ry, ωy, j, jcheck)
        for i in eachindex(rx)
            polex, ii, i = lagrange_pole(newx, rx, ωx, i, icheck)
            II = ii + Nq[1] * (jj - 1)
            poles = polex * poley
            @inbounds numerator += oldfijk[II] * poles
            denominator += poles
        end
    end
    newf[I] = numerator / denominator
end

function interpolate_field_2D!(newf, oldf, elist, ξlist, r, ω, Nq; arch=CUDADevice(), blocksize=256)
    kernel = interpolate_field_kernel_2D!(arch, blocksize)
    event = kernel(newf, oldf, elist, ξlist, r, ω, Val(Nq), ndrange=size(newf))
    wait(event)
    return nothing
end

rescale(x, xmin, xmax) = 2 * (x - xmin) / (xmax - xmin) - 1

function get_element(xnew, ynew, xinfo, yinfo)
    xmin, xmax, nex = xinfo
    ymin, ymax, ney = yinfo
    ex = ceil(Int, (xnew - xmin) / (xmax - xmin) * nex)
    ex = min(max(ex, 1), nex)
    ey = ceil(Int, (ynew - ymin) / (ymax - ymin) * ney)
    ey = min(max(ey, 1), ney)
    e = ex + nex * (ey - 1)
    return e
end

function get_reference(x, y, oldx, oldy)
    xmin, xmax = extrema(oldx)
    ymin, ymax = extrema(oldy)

    ξ1 = rescale(x, xmin, xmax)
    ξ2 = rescale(y, ymin, ymax)

    return @SVector[ξ1, ξ2]
end

# kernels 
@kernel function cube_kernel_2D!(ξlist, elist, newgrid, x, xinfo, y, yinfo)
    I = @index(Global, Linear)
    xnew, ynew = newgrid[I]
    e = get_element(xnew, ynew, xinfo, yinfo)
    oldx = view(x, :, e)
    oldy = view(y, :, e)
    ξ = get_reference(xnew, ynew, oldx, oldy)
    ξlist[I] = ξ
    elist[I] = e
end

function cube_interpolate_2D(newgrid, oldgrid; arch=CUDADevice())
    if arch isa CUDADevice
        ξlist = CuArray([@SVector[0.0, 0.0] for i in eachindex(newgrid)])
        elist = CuArray(zeros(Int, length(newgrid)))
    else
        ξlist = Array([@SVector[0.0, 0.0] for i in eachindex(newgrid)])
        elist = Array(zeros(Int, length(newgrid)))
    end
    x, y = components(oldgrid.points)
    nex, ney = (size(oldgrid.vertices) .- 1)
    xinfo = (extrema(x)..., nex)
    yinfo = (extrema(y)..., ney)
    if arch isa CUDADevice
        kernel! = cube_kernel_2D!(arch, 256)
        event = kernel!(ξlist, elist, newgrid, x, xinfo, y, yinfo, ndrange=size(elist))
        wait(event)
    else
        kernel! = cube_kernel_2D!(arch, 16)
        event = kernel!(ξlist, elist, newgrid, x, xinfo, y, yinfo, ndrange=size(elist))
        wait(event)
    end

    return ξlist, elist
end
