#=
module NewShallowWater
    export NewShallowWaterLaw
=#

import ..Atum
using ..Atum: avg, roe_avg, constants
using StaticArrays
using LinearAlgebra: I

struct NewShallowWaterLaw{FT, D, S, C} <: Atum.AbstractBalanceLaw{FT, D, S, C}
function NewShallowWaterLaw{FT, D}(; grav = 10) where {FT, D}
    S = 2 + D
    C = (grav = FT(grav),)
    new{FT, D, S, C}()
end
end

function varsindices(law::NewShallowWaterLaw)
    S = Atum.numberofstates(law)
    ix_ρ = 1
    ix_ρu⃗ = StaticArrays.SUnitRange(2, S - 1)
    ix_ρθ = S
    return ix_ρ, ix_ρu⃗, ix_ρθ
end

@inline function unpackstate(law::NewShallowWaterLaw, q)
    ix_ρ, ix_ρu⃗, ix_ρθ = varsindices(law)
    @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρθ]
end

function Atum.wavespeed(law::NewShallowWaterLaw, n⃗, q, aux)
    ρ, ρu⃗, ρθ = unpackstate(law, q)

    u⃗ = ρu⃗ / ρ
    abs(n⃗' * u⃗) + sqrt(constants(law).grav * ρ)
end

Base.@propagate_inbounds function modified_surfaceflux!(::Atum.RoeFlux, law::NewShallowWaterLaw{FT,D,S,C}, flux, n⃗, q⁻, aux⁻, q⁺, aux⁺) where {FT,D,S,C}
    g = constants(law).grav

    ρ⁻, ρu⃗⁻, ρθ⁻ = unpackstate(law, q⁻)
    u⃗⁻ = ρu⃗⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    p⁻ = g * ρ⁻^2 / 2
    c⁻ = sqrt(g * ρ⁻)

    ρ⁺, ρu⃗⁺, ρθ⁺ = unpackstate(law, q⁺)
    u⃗⁺ = ρu⃗⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    p⁺ = g * ρ⁺^2 / 2
    c⁺ = sqrt(g * ρ⁺)

    ρ = sqrt(ρ⁻ * ρ⁺)
    u⃗ = roe_avg(ρ⁻, ρ⁺, u⃗⁻, u⃗⁺)
    θ = roe_avg(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_avg(ρ⁻, ρ⁺, c⁻, c⁺)

    uₙ = u⃗' * n⃗

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu⃗ = u⃗⁺ - u⃗⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu⃗' * n⃗

    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² / 2
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² / 2
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    fp_ρ = (w1 + w2 + w3) / 2
    fp_ρu⃗ = (w1 * (u⃗ - c * n⃗) + w2 * (u⃗ + c * n⃗) + w3 * u⃗ + w4 * (Δu⃗ - Δuₙ * n⃗)) / 2
    fp_ρθ = ((w1 + w2) * θ + w5) / 2

    # penalty
    flux[1] = -fp_ρ
    @unroll for d in 1:D
        flux[d+1] = -fp_ρu⃗[d]
    end
    flux[S] = -fp_ρθ

    # Centralish
    u⃗⁺ = ρu⃗⁺ / ρ⁺
    p⁺ = constants(law).grav * ρ⁺^2 / 2
    fρ⁺ = ρu⃗⁺
    fρu⃗⁺ = ρu⃗⁺ * u⃗⁺' + p⁺ * I
    fρθ⁺ = u⃗⁺ * ρθ⁺

    u⃗⁻ = ρu⃗⁻ / ρ⁻
    p⁻ = constants(law).grav * ρ⁻^2 / 2
    fρ⁻ = ρu⃗⁻
    fρu⃗⁻ = ρu⃗⁻ * u⃗⁻' + p⁻ * I
    fρθ⁻ = u⃗⁻ * ρθ⁻

    @unroll for d in 1:D
        flux[1] += n⃗[d] * (fρ⁺ + fρ⁻)[d] / 2 
        @unroll for dd in 1:D
            flux[d+1] += n⃗[dd] *(fρu⃗⁻ + fρu⃗⁺)[d,dd] /2
        end
        flux[S] += n⃗[d] *(fρθ⁺ + fρθ⁻)[d]/2
    end


    #=
    ρ_avg = Atum.avg(ρ⁺, ρ⁻)
    ρ²_avg = Atum.avg(ρ⁺^2, ρ⁻^2)
    u⃗_avg = Atum.avg(u⃗⁺, u⃗⁻)
    ρu⃗_avg = Atum.avg(ρu⃗⁺, ρu⃗⁻)
    θ_avg = Atum.avg(θ⁺, θ⁻)

    @unroll for d in 1:D
        ρuₙ = ρu⃗_avg[d] * n⃗[d]
        flux[1] += ρuₙ
        @unroll for dd in 1:D
            flux[dd+1] += ρuₙ * u⃗_avg[dd] + n⃗[dd] * constants(law).grav * (ρ_avg^2 - ρ²_avg / 2)
        end
        flux[S] += ρuₙ * θ_avg
    end
    =#

end

Base.@propagate_inbounds function modified_volumeflux!(::Atum.EntropyConservativeFlux,
    law::NewShallowWaterLaw{FT,D,S,C},
    flux,
    q₁, _, q₂, _) where {FT,D,S,C}

    ρ₁, ρu⃗₁, ρθ₁ = unpackstate(law, q₁)
    ρ₂, ρu⃗₂, ρθ₂ = unpackstate(law, q₂)

    u⃗₁ = ρu⃗₁ / ρ₁
    θ₁ = ρθ₁ / ρ₁

    u⃗₂ = ρu⃗₂ / ρ₂
    θ₂ = ρθ₂ / ρ₂

    ρ_avg = Atum.avg(ρ₁, ρ₂)
    ρ²_avg = Atum.avg(ρ₁^2, ρ₂^2)
    u⃗_avg = Atum.avg(u⃗₁, u⃗₂)
    ρu⃗_avg = Atum.avg(ρu⃗₁, ρu⃗₂)
    θ_avg = Atum.avg(θ₁, θ₂)

    @unroll for d in 1:D
        flux[d, 1] = ρu⃗_avg[d]
        @unroll for dd in 1:D
            flux[d, dd+1] = ρu⃗_avg[d] * u⃗_avg[dd]

        end
        flux[d, S] = ρu⃗_avg[d] * θ_avg
        flux[d, d+1] += constants(law).grav * (ρ_avg^2 - ρ²_avg / 2)
    end

end
#=
end
=#
