function mean_variables(law, state, aux)
    ρ, ρu⃗, _ = EulerTotalEnergy.unpackstate(law, state)
    x = aux[1]
    y = aux[2]
    z = aux[3]
    # spherical vectors
    r⃗ = SVector(x, y, z)
    ϕ⃗ = SVector(x * z, y * z, -(x^2 + y^2))
    λ⃗ = SVector(-y, x, 0)
    # normalize (using nested functions gives error)
    r⃗_norm = sqrt(r⃗' * r⃗)
    r⃗_norm = r⃗_norm ≈ 0.0 ? 1.0 : r⃗_norm
    ϕ⃗_norm = sqrt(ϕ⃗' * ϕ⃗)
    ϕ⃗_norm = ϕ⃗_norm ≈ 0.0 ? 1.0 : ϕ⃗_norm
    λ⃗_norm = sqrt(λ⃗' * λ⃗)
    λ⃗_norm = λ⃗_norm ≈ 0.0 ? 1.0 : λ⃗_norm
    u⃗ = ρu⃗ / ρ
    u = (λ⃗' * u⃗) / λ⃗_norm
    v = (ϕ⃗' * u⃗) / ϕ⃗_norm
    w = (r⃗' * u⃗) / r⃗_norm
    p = Atum.EulerTotalEnergy.pressure(law, state, aux)
    T = p / (ρ * 287)
    
    SVector(ρ, u, v, w, p, T)
end

function second_moment_variables(mvar)
    ρ, u, v, w, p, T = mvar
    uu = u * u
    vv = v * v
    ww = w * w
    uv = u * v
    uw = u * w 
    vw = v * w 
    uT = u * T 
    vT = v * T 
    wT = w * T 
    ρρ = ρ * ρ 
    pp = p * p
    TT = T * T
    SVector(uu, vv, ww, uv, uw, vw, uT, vT, wT, ρρ, pp, TT)
end

function second_moment_variables2!(smvar, mvar)
    ρ, u, v, w, p, T = mvar
    uu, vv, ww, uv, uw, vw, uT, vT, wT, ρρ, pp, TT = smvar
    @. uu = u * u
    @. vv = v * v
    @. ww = w * w
    @. uv = u * v
    @. uw = u * w
    @. vw = v * w
    @. uT = u * T
    @. vT = v * T
    @. wT = w * T
    @. ρρ = ρ * ρ
    @. pp = p * p
    @. TT = T * T
    return nothing
end
