import ..Atum
using ..Atum: avg, logavg, roe_avg, constants
using StaticArrays
using LinearAlgebra: I, norm

struct LorenzAttractor{FT, D, S, C} <: Atum.AbstractBalanceLaw{FT, D, S, C}
    function LorenzAttractor{FT,D}(; ρ = 28, σ = 10, β = 8/3) where {FT,D}
        S = 1
        C = (ρ=FT(ρ), σ=FT(σ), β = FT(β))
        new{FT,D,S,C}()
    end
end

function modified_surfaceflux!(::Atum.RoeFlux, law::LorenzAttractor{FT,D,S,C}, flux, n⃗, q₁, aux₁, q₂, aux₂) where {FT,D,S,C}
    ρ₁ = q₁[1]
    ρ₂ = q₂[1]
    x₁, y₁, z₁ = aux₁[1], aux₁[2], aux₁[3]
    x₂, y₂, z₂ = aux₂[1], aux₂[2], aux₂[3]
    ρ, σ, β = C
    u⃗₁ = SVector(-σ * (x₁ - y₁), -y₁ - x₁ * z₁ + ρ * x₁, -β * z₁ + x₁ * y₁)
    u⃗₂ = SVector(-σ * (x₂ - y₂), -y₂ - x₂ * z₂ + ρ * x₂, -β * z₂ + x₂ * y₂)

    c = max(abs(u⃗₁' * n⃗), abs(u⃗₂' * n⃗))
    Δρ = ρ₂ - ρ₁
    flux[1] = c * Δρ * 0.5 
    fρ = avg(u⃗₁, u⃗₂) * avg(ρ₁, ρ₂)

    @unroll for d in 1:D
        flux[1] += n⃗[d] * fρ[d]
    end

end

function modified_volumeflux!(::Atum.KennedyGruberFlux, law::LorenzAttractor{FT,D,S,C},
    flux, q₁, aux₁, q₂, aux₂) where {FT,D,S,C}
    ρ₁ = q₁[1]
    ρ₂ = q₂[1]
    x₁, y₁, z₁ = aux₁[1], aux₁[2], aux₁[3]
    x₂, y₂, z₂ = aux₂[1], aux₂[2], aux₂[3]
    ρ, σ, β = C
    u⃗₁ = SVector(-σ * (x₁ - y₁), -y₁ - x₁ * z₁ + ρ * x₁, -β * z₁ + x₁ * y₁)
    u⃗₂ = SVector(-σ * (x₂ - y₂), -y₂ - x₂ * z₂ + ρ * x₂, -β * z₂ + x₂ * y₂)

    fρ = avg(u⃗₁, u⃗₂) * avg(ρ₁, ρ₂)

    @unroll for d in 1:D
        flux[d, 1] = fρ[d]
    end

end
