module EulerGravity
  export EulerGravityLaw, γ, grav

  import ..Atum
  using ..Atum: roe_avg
  using StaticArrays
  using LinearAlgebra: I

  struct EulerGravityLaw{γ, grav, FT, D, S} <: Atum.AbstractBalanceLaw{FT, D, S}
    function EulerGravityLaw{FT, D}(; γ = 7 // 5, grav = 981 // 100) where {FT, D}
      S = 2 + D
      new{FT(γ), FT(grav), FT, D, S}()
    end
  end

  γ(::EulerGravityLaw{_γ}) where {_γ} = _γ
  grav(::EulerGravityLaw{_γ, _grav}) where {_γ, _grav} = _grav

  function varsindices(law::EulerGravityLaw)
    S = Atum.numberofstates(law)
    ix_ρ = 1
    ix_ρu⃗ = StaticArrays.SUnitRange(2, S - 1)
    ix_ρe = S
    return ix_ρ, ix_ρu⃗, ix_ρe
  end

  function unpackstate(law::EulerGravityLaw, q)
    ix_ρ, ix_ρu⃗, ix_ρe = varsindices(law)
    @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρe]
  end

  function pressure(law::EulerGravityLaw, ρ, ρu⃗, ρe, Φ)
    (γ(law) - 1) * (ρe - ρu⃗' * ρu⃗ / 2ρ - ρ * Φ)
  end
  function energy(law::EulerGravityLaw, ρ, ρu⃗, p, Φ)
    p / (γ(law) - 1) + ρu⃗' * ρu⃗ / 2ρ + ρ * Φ
  end
  function soundspeed(law::EulerGravityLaw, ρ, p)
    sqrt(γ(law) * p / ρ)
  end
  function soundspeed(law::EulerGravityLaw, ρ, ρu⃗, ρe, Φ)
    soundspeed(law, ρ, pressure(law, ρ, ρu⃗, ρe, Φ))
  end

  function Atum.flux(law::EulerGravityLaw, q, x⃗)
    ρ, ρu⃗, ρe = unpackstate(law, q)

    z = last(x⃗)
    Φ = grav(law) * z

    u⃗ = ρu⃗ / ρ
    p = pressure(law, ρ, ρu⃗, ρe, Φ)

    fρ = ρu⃗
    fρu⃗ = ρu⃗ * u⃗' + p * I
    fρe = u⃗ * (ρe + p)

    hcat(fρ, fρu⃗, fρe)
  end
  
  function Atum.source!(law::EulerGravityLaw, dq, q, x⃗)
    ix_ρ, ix_ρu⃗, _ = varsindices(law)
    @inbounds dq[ix_ρu⃗[end]] -= q[ix_ρ] * grav(law)
  end

  function Atum.wavespeed(law::EulerGravityLaw, n⃗, q, x⃗)
    ρ, ρu⃗, ρe = unpackstate(law, q)
    
    z = last(x⃗)
    Φ = grav(law) * z

    u⃗ = ρu⃗ / ρ
    abs(n⃗' * u⃗) + soundspeed(law, ρ, ρu⃗, ρe, Φ)
  end

  function (::Atum.RoeFlux)(law::EulerGravityLaw, n⃗, x⃗, q⁻, q⁺)
    z = last(x⃗)
    Φ = grav(law) * z

    f⁻ = Atum.flux(law, q⁻, x⃗)
    f⁺ = Atum.flux(law, q⁺, x⃗)

    ρ⁻, ρu⃗⁻, ρe⁻ = unpackstate(law, q⁻)
    u⃗⁻ = ρu⃗⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    p⁻ = pressure(law, ρ⁻, ρu⃗⁻, ρe⁻, Φ)
    h⁻ = e⁻ + p⁻ / ρ⁻
    c⁻ = soundspeed(law, ρ⁻, p⁻)

    ρ⁺, ρu⃗⁺, ρe⁺ = unpackstate(law, q⁺)
    u⃗⁺ = ρu⃗⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    p⁺ = pressure(law, ρ⁺, ρu⃗⁺, ρe⁺, Φ)
    h⁺ = e⁺ + p⁺ / ρ⁺
    c⁺ = soundspeed(law, ρ⁺, p⁺)

    ρ = sqrt(ρ⁻ * ρ⁺)
    u⃗ = roe_avg(ρ⁻, ρ⁺, u⃗⁻, u⃗⁺)
    h = roe_avg(ρ⁻, ρ⁺, h⁻, h⁺)
    c = roe_avg(ρ⁻, ρ⁺, c⁻, c⁺)

    uₙ = u⃗' * n⃗

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu⃗ = u⃗⁺ - u⃗⁻
    Δuₙ = Δu⃗' * n⃗

    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² / 2
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² / 2
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ

    fp_ρ = (w1 + w2 + w3) / 2
    fp_ρu = (w1 * (u⃗ - c * n⃗) +
             w2 * (u⃗ + c * n⃗) +
             w3 * u⃗ +
             w4 * (Δu⃗ - Δuₙ * n⃗)) / 2
    fp_ρe = (w1 * (h - c * uₙ) +
             w2 * (h + c * uₙ) +
             w3 * (u⃗' * u⃗ / 2 + Φ) +
             w4 * (u⃗' * Δu⃗ - uₙ * Δuₙ)) / 2

    (f⁻ + f⁺)' * n⃗ / 2 - vcat(fp_ρ, fp_ρu, fp_ρe)
  end
end
