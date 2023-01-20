Base.@propagate_inbounds function new_source!(local_source, q1, aux1)
    local_source[1] += exp(q1[1]) + aux1[1]
    local_source[2] += exp(q1[2]) + aux1[2]
    local_source[3] += exp(q1[3]) + aux1[3]
    local_source[4] += exp(q1[4]) + aux1[4]
    local_source[5] += exp(q1[5]) + aux1[5]
end
##
Base.@propagate_inbounds function mypressure(ρ, ρu⃗, ρe, Φ)
    0.4 * (ρe - 0.5 * ρu⃗' * ρu⃗ / ρ - ρ * Φ)
end
##
Base.@propagate_inbounds function new_flux_8!(f, q₁, q₂, aux₁, aux₂)
    ρ₁, ρu⃗₁, ρe₁ = q₁[1], SVector(q₁[2], q₁[3], q₁[4]), q₁[5]
    ρ₂, ρu⃗₂, ρe₂ = q₂[1], SVector(q₂[2], q₂[3], q₂[4]), q₂[5]

    Φ₁ = aux₁[9]
    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = mypressure(ρ₁, ρu⃗₁, ρe₁, Φ₁)

    Φ₂ = aux₂[9]
    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = mypressure(ρ₂, ρu⃗₂, ρe₂, Φ₂)

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    e_avg = Atum.avg(e₁, e₂)
    p_avg = Atum.avg(p₁, p₂)

    # fluctuation
    α = ρ_avg / 2
    @unroll for d in 1:3
        f[d, 1] = u⃗_avg[d] * ρ_avg
        @unroll for dd in 1:3
            f[d, dd+1] = (ρ_avg * u⃗_avg[d]) * u⃗_avg[dd]
        end
        f[d, 5] = u⃗_avg[d] * (ρ_avg * e_avg + p_avg)
        f[d, d+1] -= α * (Φ₁ - Φ₂) - p_avg
    end

end

##
Base.@propagate_inbounds function new_flux_7!(f, q₁, q₂, aux₁, aux₂)
    ρ₁, ρu⃗₁, ρe₁ = q₁[1], SVector(q₁[2], q₁[3], q₁[4]), q₁[5]
    ρ₂, ρu⃗₂, ρe₂ = q₂[1], SVector(q₂[2], q₂[3], q₂[4]), q₂[5]

    Φ₁ = aux₁[9]
    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = mypressure(ρ₁, ρu⃗₁, ρe₁, Φ₁)

    Φ₂ = aux₂[9]
    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = mypressure(ρ₂, ρu⃗₂, ρe₂, Φ₂)

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    e_avg = Atum.avg(e₁, e₂)
    p_avg = Atum.avg(p₁, p₂)

    # fluctuation
    α = ρ_avg / 2
    f[1:3, 1] = u⃗_avg * ρ_avg
    f[1:3, 2:4] = (ρ_avg * u⃗_avg) * u⃗_avg' + p_avg * I
    f[1:3, 2:4] -= α * (Φ₁ - Φ₂) * I
    f[1:3, 5] = u⃗_avg * (ρ_avg * e_avg + p_avg)

end

Base.@propagate_inbounds function new_flux_6!(f, q₁, q₂, aux₁, aux₂)
    ρ₁, ρu⃗₁, ρe₁ = q₁[1], SVector(q₁[2], q₁[3], q₁[4]), q₁[5]
    ρ₂, ρu⃗₂, ρe₂ = q₂[1], SVector(q₂[2], q₂[3], q₂[4]), q₂[5]

    Φ₁ = aux₁[9]
    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = mypressure(ρ₁, ρu⃗₁, ρe₁, Φ₁)

    Φ₂ = aux₂[9]
    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = mypressure(ρ₂, ρu⃗₂, ρe₂, Φ₂)

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    e_avg = Atum.avg(e₁, e₂)
    p_avg = Atum.avg(p₁, p₂)

    # fluctuation
    α = ρ_avg / 2
    fρ = u⃗_avg * ρ_avg
    fρu⃗ = (ρ_avg * u⃗_avg) * u⃗_avg' + p_avg * I
    fρu⃗ -= α * (Φ₁ - Φ₂) * I
    fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)
    @unroll for d in 1:3
        f[d, 1] = fρ[d]
        @unroll for dd in 1:3
            f[d, dd+1] = fρu⃗[d, dd]
        end
        f[d, 5] = fρe[d]
    end

end

Base.@propagate_inbounds function new_flux_5!(f, q₁, q₂, aux₁, aux₂, dims)
    ρ₁, ρu⃗₁, ρe₁ = q₁[1], SVector(q₁[2], q₁[3], q₁[4]), q₁[5]
    ρ₂, ρu⃗₂, ρe₂ = q₂[1], SVector(q₂[2], q₂[3], q₂[4]), q₂[5]

    Φ₁ = aux₁[9]
    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = mypressure(ρ₁, ρu⃗₁, ρe₁, Φ₁)

    Φ₂ = aux₂[9]
    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = mypressure(ρ₂, ρu⃗₂, ρe₂, Φ₂)

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    e_avg = Atum.avg(e₁, e₂)
    p_avg = Atum.avg(p₁, p₂)

    # fluctuation
    α = ρ_avg / 2
    fρ = u⃗_avg * ρ_avg
    fρu⃗ = (ρ_avg * u⃗_avg) * u⃗_avg' + p_avg * I
    fρu⃗ -= α * (Φ₁ - Φ₂) * I
    fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)
    @unroll for d in 1:dims
        f[d, 1] = fρ[d]
        @unroll for dd in 1:dims
            f[d, dd+1] = fρu⃗[d, dd]
        end
        f[d, 5] = fρe[d]
    end

end

Base.@propagate_inbounds function new_flux_4!(f, q₁, q₂, aux₁, aux₂)
    ρ₁, ρu⃗₁, ρe₁ = q₁[1], SVector(q₁[2], q₁[3], q₁[4]), q₁[5]
    ρ₂, ρu⃗₂, ρe₂ = q₂[1], SVector(q₂[2], q₂[3], q₂[4]), q₂[5]

    Φ₁ = aux₁[9]
    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = mypressure(ρ₁, ρu⃗₁, ρe₁, Φ₁)

    Φ₂ = aux₂[9]
    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = mypressure(ρ₂, ρu⃗₂, ρe₂, Φ₂)

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    e_avg = Atum.avg(e₁, e₂)
    p_avg = Atum.avg(p₁, p₂)

    # fluctuation
    α = ρ_avg / 2
    fρ = u⃗_avg * ρ_avg
    fρu⃗ = (ρ_avg * u⃗_avg) * u⃗_avg' + p_avg * I
    fρu⃗ -= α * (Φ₁ - Φ₂) * I
    fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)
    return hcat(fρ, fρu⃗, fρe)
end

# this way compiler can infer the dimensions
Base.@propagate_inbounds function new_flux_3!(f, q₁, q₂, aux₁, aux₂, ::Val{dims}) where {dims}
    ρ₁, ρu⃗₁, ρe₁ = q₁[1], SVector(q₁[2], q₁[3], q₁[4]), q₁[5]
    ρ₂, ρu⃗₂, ρe₂ = q₂[1], SVector(q₂[2], q₂[3], q₂[4]), q₂[5]

    Φ₁ = aux₁[9]
    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = mypressure(ρ₁, ρu⃗₁, ρe₁, Φ₁)

    Φ₂ = aux₂[9]
    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = mypressure(ρ₂, ρu⃗₂, ρe₂, Φ₂)

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    e_avg = Atum.avg(e₁, e₂)
    p_avg = Atum.avg(p₁, p₂)

    # fluctuation
    α = ρ_avg / 2
    fρ = u⃗_avg * ρ_avg
    fρu⃗ = (ρ_avg * u⃗_avg) * u⃗_avg' + p_avg * I
    fρu⃗ -= α * (Φ₁ - Φ₂) * I
    fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)
    @unroll for d in 1:dims
        f[d, 1] = fρ[d]
        @unroll for dd in 1:dims
            f[d, dd+1] = fρu⃗[d, dd]
        end
        f[d, 5] = fρe[d]
    end

end


##
f = zeros(3, 5)
f2 = zeros(3, 5)
f3 = zeros(3, 5)
q1 = randn(5)
q2 = randn(5)
aux1 = randn(9)
aux2 = randn(9)


new_flux_7!(f2, q1, q2, aux1, aux2)
new_flux_8!(f, q1, q2, aux1, aux2)
new_flux_6!(f3, q1, q2, aux1, aux2)

norm(f2 - f)
norm(f3 - f)
f3 .= 0.0
new_flux_5!(f3, q1, q2, aux1, aux2, 3)
norm(f3 - f)
f3 .= 0.0
new_flux_3!(f3, q1, q2, aux1, aux2, Val(3))
norm(f3 - f)
f = new_flux_4!(f3, q1, q2, aux1, aux2)