using Atum
using StaticArrays
using BenchmarkTools
using CUDA
using KernelAbstractions
using CUDAKernels

A = CuArray
FT = Float64
N = 4
K = 16

println("DOFs = ", (N + 1) * K)
Nq = N + 1
# pp = 2
cell = LobattoCell{FT,A}(Nq, Nq, Nq)
v1d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v2d = range(FT(-1.5e3), stop=FT(1.5e3), length=K + 1)
v3d = range(FT(0), stop=FT(3.0e3), length=K + 1)
grid = brickgrid(cell, (v1d, v2d, v3d); periodic=(true, true, false))
setzero(x) = SVector(0.0, 0.0, 0.0, 0.0, 0.0)
tmp = fieldarray(undef, SVector{5,FT}, grid)
tmp2 = fieldarray(undef, SVector{5,FT}, grid)
tmp3 = setzero.(points(grid))
tmp4 = setzero.(points(grid))

tmp .= setzero.(tmp) # SVector(0.0, 0.0, 0.0, 0.0, 0.0)
tmp2 .= tmp

q = copy(parent(components(tmp)[1])) .+ randn()
dq = similar(q)
c_device = CUDADevice()

@benchmark CUDA.@sync begin
    comp_stream = Event(c_device)
    $tmp2 .= $tmp .+ $tmp
    for i in 1:9
        $tmp2 .= $tmp .+ $tmp
    end
    wait(comp_stream)
end

@benchmark CUDA.@sync begin
    $tmp2 .= $tmp .+ $tmp
    for i in 1:9
        $tmp2 .= $tmp .+ $tmp
    end
end

@benchmark CUDA.@sync begin
    $tmp3 .= $tmp4 .+ $tmp4
    for i in 1:9
        $tmp3 .= $tmp4 .+ $tmp4
    end
end

@benchmark CUDA.@sync begin
    comp_stream = Event(c_device)
    q2 = parent(components(tmp)[1])
    dq2 = parent(components(tmp2)[1])
    dq2 .= q2 .+ q2
    for i in 1:9
        q2 = parent(components(tmp)[1])
        dq2 = parent(components(tmp2)[1])
        dq2 .= q2 .+ q2
    end
    wait(comp_stream)

end

@benchmark CUDA.@sync begin
    comp_stream = Event(c_device)
    dq .= q .+ q
    for i in 1:9
        dq .= q .+ q
    end
    wait(comp_stream)
end

@benchmark CUDA.@sync begin
    comp_stream = Event(c_device)
    for i in 1:9
        $dq .= $q .+ $q
    end
    wait(comp_stream)
end


##
@kernel function add_two_2!(Q, @Const(dQ))
    i = @index(Global, Linear)
    @inbounds begin
        Q[i] = dQ[i] + dQ[i]
    end
end

@kernel function add_two!(Q, @Const(dQ), rkb, dt)
    i = @index(Global, Linear)
    @inbounds begin
        Q[i] = dQ[i] + dQ[i]
    end
end


@benchmark CUDA.@sync begin
    c_device = CUDADevice()
    comp_stream = Event(c_device)

    event = add_two_2!(c_device)(
        $tmp,
        $tmp2,
        ndrange=length($tmp),
        dependencies=comp_stream
    )
    wait(event)
    for i in 1:9
        event = add_two_2!(c_device)(
            $tmp,
            $tmp2,
            ndrange=length($tmp),
            dependencies=comp_stream
        )
    end
    wait(event)
end

@benchmark CUDA.@sync begin
    c_device = CUDADevice()
    comp_stream = Event(c_device)

    event = add_two_2!(c_device)(
        $q,
        $dq,
        ndrange=length($q),
        dependencies=comp_stream
    )
    wait(event)
    for i in 1:9
        event = add_two_2!(c_device)(
            $q,
            $dq,
            ndrange=length($q),
            dependencies=comp_stream
        )
    end
    wait(event)
end



@benchmark begin
    c_device = CUDADevice()
    comp_stream = Event(c_device)
    q2 = parent(components($tmp)[1])
    dq2 = parent(components($tmp2)[1])
    event = add_two_2!(c_device)(
        q2,
        dq2,
        ndrange=length(dq2),
        dependencies=comp_stream
    )
    for i in 1:9

        q2 = parent(components($tmp)[1])
        dq2 = parent(components($tmp2)[1])
        event = add_two_2!(c_device)(
            q2,
            dq2,
            ndrange=length(dq2),
            dependencies=event
        )
    end
    wait(event)
end

@benchmark begin
    c_device = CUDADevice()
    comp_stream = Event(c_device)

    event = add_two_2!(c_device)(
        q,
        dq,
        ndrange=length(q),
        dependencies=comp_stream
    )
    for i in 1:9
        event = add_two_2!(c_device)(
            q,
            dq,
            ndrange=length(q),
            dependencies=event
        )
    end
    wait(event)

end