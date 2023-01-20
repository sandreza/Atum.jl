using Atum
using Bennu: fieldarray
using BenchmarkTools
using CUDA
using StaticArrays: SVector
using LinearAlgebra
using StructArrays


include("geometry.jl")
include("gradient_kernels.jl")

function xdir(x⃗)
    x, y, z = x⃗
    return SVector(x)
end

function ydir(x⃗)
    x, y, z = x⃗
    return SVector(y)
end

function zdir(x⃗)
    x, y, z = x⃗
    return SVector(z)
end

function xyzdir(x⃗)
    x, y, z = x⃗
    return SVector(x + 2 * y + 3 * z)
end

function graddir(x⃗)
    x, y, z = x⃗
    return x⃗
end


## Specify the Grid 
A = CuArray
FT = Float64
N = 3
K = 50

Nq = N + 1

cell = LobattoCell{FT,A}(Nq, Nq, Nq)
vx = range(FT(-1), stop=FT(1), length=K + 1)
vy = range(FT(-1), stop=FT(1), length=K + 1)
vz = range(FT(-1), stop=FT(1), length=K + 1)
grid = brickgrid(cell, (vx, vy, vz); periodic=(false, false, false))

x⃗ = points(grid)

ϕ = xyzdir.(x⃗)
∇ϕ = graddir.(x⃗)

D = derivatives_1d(cell)[1]
ϕ_A = components(ϕ)[1]
fill!(∇ϕ, SVector(0.0, 0.0, 0.0))
dxϕ_A = components(∇ϕ)[1]
dyϕ_A = components(∇ϕ)[2]
dzϕ_A = components(∇ϕ)[3]
a = randn(2, 2)
b = randn(2, 2)
sab = StructArray((a, b))
aa = randn(2, 2, 2)
saa_view = StructArray((view(aa, :, :, 1), view(aa, :, :, 2)))
tmp = StructArray{SVector{2,Float64}}(undef, 2, 2)
fill!(tmp, SVector(1.0, 1.0))
otmp = StructArray{SVector{2,Float64}}((view(aa, :, :, 1), view(aa, :, :, 2)))

function grab_array(a::StructArray{S,T,W}) where {S,T,W<:Tuple}
    parent(components(a)[1])
end

grab_array(otmp)

(a⃗¹, a⃗², a⃗³) = contravariant_basis(grid)
J⃗, _ = components(metrics(grid))

# x-direction
fill!(∇ϕ, SVector(0.0, 0.0, 0.0))
#=
for e in 1:8
    for i in 1:Nq, j in 1:Nq, k in 1:Nq
        ijk = i + Nq * (j - 1 + Nq * (k - 1))
        for ii in 1:Nq
            l = ii + Nq * (j - 1 + Nq * (k - 1))
            dxϕ_A[ijk, e] += D[i, ii] * ϕ_A[l, e]
        end
    end
end

# y-direction
# fill!(∇ϕ, SVector(0.0, 0.0, 0.0))
for e in 1:8
    for i in 1:Nq, j in 1:Nq, k in 1:Nq
        ijk = i + Nq * (j - 1 + Nq * (k - 1))
        for jj in 1:Nq
            l = i + Nq * (jj - 1 + Nq * (k - 1))
            dyϕ_A[ijk, e] += D[j, jj] * ϕ_A[l, e]
        end
    end
end

# z-direction
# fill!(∇ϕ, SVector(0.0, 0.0, 0.0))
for e in 1:8
    for i in 1:Nq, j in 1:Nq, k in 1:Nq
        ijk = i + Nq * (j - 1 + Nq * (k - 1))
        for kk in 1:Nq
            l = i + Nq * (j - 1 + Nq * (kk - 1))
            dzϕ_A[ijk, e] += D[k, kk] * ϕ_A[l, e]
        end
    end
end
=#

function change_basis(J⃗, ∇ϕ)
    return J⃗ * ∇ϕ
end

∇ϕ .= change_basis.(J⃗, ∇ϕ)


Nq⃗ = (Nq, Nq, Nq)
ldevice = Bennu.device(dg)
dim = length(Nq⃗)
workgroup = ntuple(i -> i <= dim ? Nq⃗[i] : 1, 3)
Ne = size(ϕ)[2]
ndrange = (Ne * workgroup[1], Base.tail(workgroup)...)

ϕ_A = A(zeros(size(ϕ)))
Nq_total, Ne = size(ϕ)
Ns = length(ϕ[1, 1])

∇ϕ2 = A(zeros(Nq_total, dim * Ns, Ne))
ϕ2 = A(zeros(Nq_total, Ns, Ne))
s_ϕ2 = StructArray{SVector{1,Float64}}((view(ϕ2, :, 1, :),))
s_∇ϕ2 = StructArray{SVector{3,Float64}}(Tuple([view(∇ϕ2, :, i, :) for i in 1:dim*Ns]))



nJ = length(components(J⃗))
metric_terms = A(zeros(Nq_total, nJ, Ne))
for i in 1:nJ
    metric_terms[:, i, :] .= components(J⃗)[i]
end
# s_metric_terms = StructArray{SMatrix{3,3,Float64,9}}(Tuple([view(∇ϕ2, :, i, :) for i in 1:dim*Ns])) figure out how to do this

s_ϕ2 .= ϕ
fill!(s_∇ϕ2, SVector(0.0, 0.0, 0.0))
timing = @benchmark CUDA.@sync begin
    comp_stream = Event(ldevice)

    for dir in 1:3
        comp_stream = gradient_kernel2!(ldevice, workgroup)(
            ∇ϕ2, ϕ2, D, metric_terms,
            Val(dir), Val(dim), Val(Nq⃗[1]), Val(Nq⃗[2]), Val(Nq⃗[3]), Val(Ns),
            Val(dir != 1),
            ndrange=ndrange,
            dependencies=comp_stream,
        )
    end
    fill!(s_∇ϕ2, SVector(0.0, 0.0, 0.0))
    wait(comp_stream)
end

fill!(s_∇ϕ2, SVector(0.0, 0.0, 0.0))
timing2 = @benchmark CUDA.@sync begin
    comp_stream = Event(ldevice)

    compstream = gradient_kernel_fused3!(ldevice, workgroup)(
        ∇ϕ2, ϕ2, D, metric_terms,
        Val(dim), Val(Nq⃗[1]), Val(Ns),
        ndrange=ndrange, dependencies=comp_stream)

    fill!(s_∇ϕ2, SVector(0.0, 0.0, 0.0))
    wait(comp_stream)
end


timing3 = @benchmark CUDA.@sync begin
    comp_stream = Event(ldevice)
    compstream = gradient_kernel_fused_second!(ldevice, workgroup)(
        ∇ϕ2, ϕ2, D, metric_terms,
        Val(dim), Val(Nq⃗[1]), Val(Ns),
        ndrange=ndrange, dependencies=comp_stream)
    fill!(s_∇ϕ2, SVector(0.0, 0.0, 0.0))
    wait(comp_stream)
end

timing4 = @benchmark CUDA.@sync begin
    comp_stream = Event(ldevice)
    fill!(s_∇ϕ2, SVector(0.0, 0.0, 0.0))
    wait(comp_stream)
end

#=
gradient_kernel!(
    ∇q, q, D, metrics,
    Val(1), Val(3), Val(Nq), Val(Nq), Val(Nq), Val(1),
    Val(true)
)
=#