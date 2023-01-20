import ..Atum
using ..Atum: avg, logavg, roe_avg, constants
using StaticArrays
using LinearAlgebra: I, norm

struct Gradient{FT, D, S, C} <: Atum.AbstractBalanceLaw{FT, D, S, C}
    function Gradient{FT,D}(; S = 1, C = nothing) where {FT,D}
        new{FT,D,S,C}()
    end
end

function modified_surfaceflux!(::Atum.CentralFlux, law::Gradient{FT,D,S,C}, flux, n⃗, q₁, aux₁, q₂, aux₂) where {FT,D,S,C}
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

function volume_gradient!(::Atum.KennedyGruberFlux, law::Gradient{FT,D,S,C},
    flux, q₁, aux₁, q₂, aux₂) where {FT,D,S,C}

    @unroll for s in 1:S 
        @unroll for d in 1:D
            flux[d, s] = (q₁[s] + q₂[s])/2
        end
    end

end
