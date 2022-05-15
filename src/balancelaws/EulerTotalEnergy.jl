module EulerTotalEnergy
  export EulerTotalEnergyLaw

  import ..Atum
  using ..Atum: avg, logavg, roe_avg, constants
  using StaticArrays
  using LinearAlgebra: I, norm

  struct EulerTotalEnergyLaw{FT, D, S, C} <: Atum.AbstractBalanceLaw{FT, D, S, C}
    function EulerTotalEnergyLaw{FT, D}(; γ = 7 // 5,
                                      grav = 981 // 100,
                                      pde_level_balance = false) where {FT, D}
      S = 2 + D
      C = (γ = FT(γ), grav = FT(grav), pde_level_balance = pde_level_balance)
      new{FT, D, S, C}()
    end
  end

  function varsindices(law::EulerTotalEnergyLaw)
    S = Atum.numberofstates(law)
    ix_ρ = 1
    ix_ρu⃗ = StaticArrays.SUnitRange(2, S - 1)
    ix_ρe = S
    return ix_ρ, ix_ρu⃗, ix_ρe
  end

  function auxindices(law::EulerTotalEnergyLaw)
    S = Atum.numberofstates(law)
    n_u⃗ = S - 2
    ix_x = 1
    ix_y = 2
    ix_z = 3
    ix_ρ = 1 + 3
    ix_ρu⃗ = StaticArrays.SUnitRange(5, 4 + n_u⃗)
    ix_ρe = 4 + n_u⃗ + 1
    ix_geo = 4 + n_u⃗ + 2
    return ix_x, ix_y, ix_z, ix_ρ, ix_ρu⃗, ix_ρe, ix_geo
  end

  function unpackstate(law::EulerTotalEnergyLaw, q)
      ix_ρ, ix_ρu⃗, ix_ρe = varsindices(law)
      @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρe]
  end

  function unpackrefstate(law::EulerTotalEnergyLaw, aux)
    ix_x, ix_y, ix_z, ix_ρ, ix_ρu⃗, ix_ρe, ix_geo = auxindices(law)
    @inbounds aux[ix_ρ], aux[ix_ρu⃗], aux[ix_ρe]
  end

  function unpackaux(law::EulerTotalEnergyLaw, q)
      ix_x, ix_y, ix_z, ix_ρ, ix_ρu⃗, ix_ρe, ix_geo = auxindices(law)
      @inbounds q[ix_x], q[ix_y], q[ix_z], q[ix_ρ], q[ix_ρu⃗], q[ix_ρe], q[ix_geo]
  end

  # These are incorrect for a sphere: 
  function Atum.auxiliary(law::EulerTotalEnergyLaw, x⃗)
    ix_x, ix_y, ix_z, ix_ρ, ix_ρu⃗, ix_ρe, ix_geo = auxindices(law)
    ixρu = StaticArrays.SUnitRange(1, 3)
    @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], x⃗[ix_z], x⃗[ixρu]..., x⃗[ix_z], 9.81 * x⃗[ix_z])
  end

  # These are incorrect for a sphere: 
  function Atum.auxiliary(law::EulerTotalEnergyLaw, x⃗, state)
    ix_ρ, ix_ρu⃗, ix_ρe = varsindices(law)
    ix_x, ix_y, ix_z, _, _, _, _ = auxindices(law)
    @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], state[ix_ρ], state[ix_ρu⃗]..., state[ix_ρe], 9.81 * x⃗[ix_z])
  end

  function coordinates(law::EulerTotalEnergyLaw, aux)
    aux[1:3]
  end

  function geopotential(law::EulerTotalEnergyLaw, aux)
    aux[end]
  end

  function pressure(law::EulerTotalEnergyLaw, ρ, ρu⃗, ρe, Φ)
    γ = constants(law).γ
    return (γ - 1) * (ρe - ρu⃗' * ρu⃗ / 2ρ - ρ * Φ)
  end

  function pressure(law::EulerTotalEnergyLaw, state, aux)
    ρ, ρu⃗, ρe = unpackstate(law, state)
    Φ = geopotential(law, aux)
    γ = constants(law).γ
    return (γ - 1) * (ρe - ρu⃗' * ρu⃗ / 2ρ - ρ * Φ)
  end

  function linearized_pressure(law::EulerTotalEnergyLaw, state, aux)
    ρ, ρu, ρe = unpackstate(law, state)
    x, y, z, ρᵣ, ρuᵣ, ρeᵣ, Φ = unpackaux(law, aux)
    γ = constants(law).γ
    (γ - 1) * (ρe - StaticArrays.dot(ρuᵣ, ρu) / ρᵣ + ρ * StaticArrays.dot(ρuᵣ, ρuᵣ) / (2 * ρᵣ^2) - ρ * Φ)
  end 

  function reference_pressure(law::EulerTotalEnergyLaw, aux)
    x, y, z, ρᵣ, ρuᵣ, ρeᵣ, Φ = unpackaux(law, aux)
    γ = constants(law).γ
    pressure(law, ρᵣ, ρuᵣ, ρeᵣ, Φ)
  end 

  function soundspeed(law::EulerTotalEnergyLaw, ρ, p)
    γ = constants(law).γ
    sqrt(γ * p / ρ)
  end

  function reference_soundspeed(law::EulerTotalEnergyLaw, aux)
    x, y, z, ρᵣ, ρuᵣ, ρeᵣ, Φ = unpackaux(law, aux)
    γ = constants(law).γ
    pᵣ = reference_pressure(law, aux)
    return sqrt(γ * pᵣ / ρᵣ)
  end 

  function Atum.flux(law::EulerTotalEnergyLaw, q, aux)
    ρ, ρu⃗, ρe = unpackstate(law, q)

    Φ = geopotential(law, aux)

    u⃗ = ρu⃗ / ρ
    p = pressure(law, ρ, ρu⃗, ρe, Φ)

    δp = p
    if constants(law).pde_level_balance
      δp -= reference_p(law, aux)
    end

    fρ = ρu⃗
    fρu⃗ = ρu⃗ * u⃗' + δp * I
    fρe = u⃗ * (ρe + p)

    hcat(fρ, fρu⃗, fρe)
  end

  function Atum.surfaceflux(::Atum.RoeFlux, law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
    Φ = geopotential(law, aux⁻)
  
    # f⁻ = Atum.flux(law, q⁻, aux⁻)
    # f⁺ = Atum.flux(law, q⁺, aux⁺)
    # main_flux = (f⁻ + f⁺)' * n⃗ * 0.5
    kg_flux = Atum.twopointflux(Atum.KennedyGruberFlux(), law, q⁻, aux⁻, q⁺, aux⁺)
    main_flux = kg_flux' * n⃗
  
  
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
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² * 0.5
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² * 0.5
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
  
    fp_ρ = (w1 + w2 + w3) * 0.5
    fp_ρu⃗ = (w1 * (u⃗ - c * n⃗) +
              w2 * (u⃗ + c * n⃗) +
              w3 * u⃗ +
              w4 * (Δu⃗ - Δuₙ * n⃗)) * 0.5
    fp_ρe = (w1 * (h - c * uₙ) +
             w2 * (h + c * uₙ) +
             w3 * (u⃗' * u⃗ * 0.5 + Φ) +
             w4 * (u⃗' * Δu⃗ - uₙ * Δuₙ)) * 0.5
  
    main_flux - SVector(fp_ρ, fp_ρu⃗..., fp_ρe)
  end
  
  function Atum.surfaceflux(::Atum.KennedyGruberFlux, law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
    f = Atum.twopointflux(Atum.KennedyGruberFlux(), law, q⁻, aux⁻, q⁺, aux⁺)
    f' * n⃗ 
  end

  function Atum.surfaceflux(::Atum.LinearizedKennedyGruberFlux, law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
    f = Atum.twopointflux(Atum.LinearizedKennedyGruberFlux(), law, q⁻, aux⁻, q⁺, aux⁺)
    f' * n⃗ 
  end

  function Atum.surfaceflux(rf::Atum.RefanovFlux, law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)

    # Flux
    f = Atum.twopointflux(Atum.KennedyGruberFlux(), law, q⁻, aux⁻, q⁺, aux⁺)

    # Penalty
    c⁻ = reference_soundspeed(law, aux⁻)
    c⁺ = reference_soundspeed(law, aux⁺)
    c = max(c⁻, c⁺) * rf.scale

    # - states
    ρ⁻, ρu⁻, ρe⁻ = unpackstate(law, q⁻)

    # + states
    ρ⁺, ρu⁺, ρe⁺ = unpackstate(law, q⁺)

    Δρ = ρ⁺ - ρ⁻
    Δρu = ρu⁺ - ρu⁻
    Δρe = ρe⁺ - ρe⁻

    fp_ρ = c * Δρ * 0.5
    fp_ρu = c * Δρu * 0.5
    fp_ρe = c * Δρe * 0.5

    f' * n⃗ - SVector(fp_ρ, fp_ρu..., fp_ρe)
  end


  function Atum.surfaceflux(rf::Atum.LinearizedRefanovFlux, law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)

    # Flux
    f = Atum.twopointflux(Atum.LinearizedKennedyGruberFlux(), law, q⁻, aux⁻, q⁺, aux⁺)

    # Penalty
    c⁻ = reference_soundspeed(law, aux⁻)
    c⁺ = reference_soundspeed(law, aux⁺)
    c = max(c⁻, c⁺) * rf.scale

    # - states
    ρ⁻, ρu⁻, ρe⁻ = unpackstate(law, q⁻)

    # + states
    ρ⁺, ρu⁺, ρe⁺ = unpackstate(law, q⁺)

    Δρ = ρ⁺ - ρ⁻
    Δρu = ρu⁺ - ρu⁻
    Δρe = ρe⁺ - ρe⁻

    fp_ρ = c * Δρ * 0.5
    fp_ρu = c * Δρu * 0.5
    fp_ρe = c * Δρe * 0.5

    f' * n⃗ - SVector(fp_ρ, fp_ρu..., fp_ρe)
  end

  function Atum.twopointflux(::Atum.KennedyGruberFlux,
    law::EulerTotalEnergyLaw,
    q₁, aux₁, q₂, aux₂)
    FT = eltype(law)
    ρ₁, ρu⃗₁, ρe₁ = unpackstate(law, q₁)
    ρ₂, ρu⃗₂, ρe₂ = unpackstate(law, q₂)

    Φ₁ = geopotential(law, aux₁)
    u⃗₁ = ρu⃗₁ / ρ₁
    e₁ = ρe₁ / ρ₁
    p₁ = pressure(law, ρ₁, ρu⃗₁, ρe₁, Φ₁)

    Φ₂ = geopotential(law, aux₂)
    u⃗₂ = ρu⃗₂ / ρ₂
    e₂ = ρe₂ / ρ₂
    p₂ = pressure(law, ρ₂, ρu⃗₂, ρe₂, Φ₂)

    ρ_avg = avg(ρ₁, ρ₂)
    u⃗_avg = avg(u⃗₁, u⃗₂)
    e_avg = avg(e₁, e₂)
    p_avg = avg(p₁, p₂)

    fρ = u⃗_avg * ρ_avg
    fρu⃗ = u⃗_avg * fρ' + p_avg * I
    fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)

    # fluctuation
    α = ρ_avg / 2
    fρu⃗ -= α * (Φ₁ - Φ₂) * I

    hcat(fρ, fρu⃗, fρe)
  end

  function Atum.twopointflux(::Atum.LinearizedKennedyGruberFlux,
    law::EulerTotalEnergyLaw,
    state_1, aux_1, state_2, aux_2)

    ρ_1, ρu_1, ρe_1 = unpackstate(law, state_1)
    p_1 = linearized_pressure(law, state_1, aux_1)
    ρᵣ_1, ρuᵣ_1, ρeᵣ_1 = unpackrefstate(law, aux_1)
    pᵣ_1 = reference_pressure(law, aux_1)
    Φ₁ = geopotential(law, aux_1)


    ρ_2, ρu_2, ρe_2 = unpackstate(law, state_2)
    p_2 = linearized_pressure(law, state_2, aux_2)
    ρᵣ_2, ρuᵣ_2, ρeᵣ_2 = unpackrefstate(law, aux_2)
    pᵣ_2 = reference_pressure(law, aux_2)
    Φ₂ = geopotential(law, aux_2)

    # calculate u_1, e_1, and reference states
    u_1 = ρu_1 / ρᵣ_1 - ρ_1 * ρuᵣ_1 / (ρᵣ_1^2)
    e_1 = ρe_1 / ρᵣ_1 - ρ_1 * ρeᵣ_1 / (ρᵣ_1^2)

    uᵣ_1 = ρuᵣ_1 / ρᵣ_1
    eᵣ_1 = ρeᵣ_1 / ρᵣ_1

    ## State 2 Stuff 
    # calculate u_2, e_2, and reference states
    u_2 = ρu_2 / ρᵣ_2 - ρ_2 * ρuᵣ_2 / (ρᵣ_2^2)
    e_2 = ρe_2 / ρᵣ_2 - ρ_2 * ρeᵣ_2 / (ρᵣ_2^2)

    uᵣ_2 = ρuᵣ_2 / ρᵣ_2
    eᵣ_2 = ρeᵣ_2 / ρᵣ_2

    # construct averages for perturbation variables
    ρ_avg = avg(ρ_1, ρ_2)
    u_avg = avg(u_1, u_2)
    e_avg = avg(e_1, e_2)
    p_avg = avg(p_1, p_2)

    # construct averages for reference variables
    ρᵣ_avg = avg(ρᵣ_1, ρᵣ_2)
    uᵣ_avg = avg(uᵣ_1, uᵣ_2)
    eᵣ_avg = avg(eᵣ_1, eᵣ_2)
    pᵣ_avg = avg(pᵣ_1, pᵣ_2)

    fρ = ρᵣ_avg * u_avg + ρ_avg * uᵣ_avg
    fρu⃗ = p_avg * I + ρᵣ_avg .* (uᵣ_avg .* u_avg' + u_avg .* uᵣ_avg')
    fρu⃗ += (ρ_avg .* uᵣ_avg) .* uᵣ_avg'
    fρe = (ρᵣ_avg * eᵣ_avg + pᵣ_avg) * u_avg
    fρe += (ρᵣ_avg * e_avg + ρ_avg * eᵣ_avg + p_avg) * uᵣ_avg

    # product rule gravity
    α = ρ_avg * 0.5
    fρu⃗ -= α * (Φ₁ - Φ₂) * I

    hcat(fρ, fρu⃗, fρe)
  end


  function Atum.twopointflux(::Atum.CentralFlux,
    law::EulerTotalEnergyLaw,
    state_1, aux_1, state_2, aux_2)
  
    ρ_1, ρu_1, ρe_1 = unpackstate(law, state_1)
    Φ₁ = geopotential(law, aux_1)
  
    ρ_2, ρu_2, ρe_2 = unpackstate(law, state_2)
    Φ₂ = geopotential(law, aux_2)
  
    # construct averages 
    p_1 = pressure(law, ρ_1, ρu_1, ρe_1, Φ₁)
    p_2 = pressure(law, ρ_2, ρu_2, ρe_2, Φ₂)
  
    fρ = avg(ρu_1, ρu_2)
    fρu⃗ = avg(ρu_1 * ρu_1' / ρ_1, ρu_2 * ρu_2' / ρ_2) + avg(p_1, p_2) * I
    fρe = avg(ρu_1 / ρ_1 * (ρe_1 + p_1), ρu_2 / ρ_2 * (ρe_2 + p_2))
  
    # product rule gravity
    α = avg(ρ_1, ρ_2) * 0.5
    fρu⃗ -= α * (Φ₁ - Φ₂) * I
  
    hcat(fρ, fρu⃗, fρe)
  end

  function Atum.twopointflux(::Atum.LinearizedCentralFlux,
    law::EulerTotalEnergyLaw,
    state_1, aux_1, state_2, aux_2)
  
    ρ_1, ρu_1, ρe_1 = unpackstate(law, state_1)
    p_1 = linearized_pressure(law, state_1, aux_1)
    ρᵣ_1, ρuᵣ_1, ρeᵣ_1 = unpackrefstate(law, aux_1)
    pᵣ_1 = reference_pressure(law, aux_1)
    Φ₁ = geopotential(law, aux_1)
  
  
    ρ_2, ρu_2, ρe_2 = unpackstate(law, state_2)
    p_2 = linearized_pressure(law, state_2, aux_2)
    ρᵣ_2, ρuᵣ_2, ρeᵣ_2 = unpackrefstate(law, aux_2)
    pᵣ_2 = reference_pressure(law, aux_2)
    Φ₂ = geopotential(law, aux_2)
  
    # calculate u_1, e_1, and reference states
    u_1 = ρu_1 / ρᵣ_1 - ρ_1 * ρuᵣ_1 / (ρᵣ_1^2)
  
    uᵣ_1 = ρuᵣ_1 / ρᵣ_1
  
    ## State 2 Stuff 
    # calculate u_2, e_2, and reference states
    u_2 = ρu_2 / ρᵣ_2 - ρ_2 * ρuᵣ_2 / (ρᵣ_2^2)
  
    uᵣ_2 = ρuᵣ_2 / ρᵣ_2
  
    # construct averages for perturbation variables
    ρ_avg = avg(ρ_1, ρ_2)
    ρu_avg = avg(ρu_1, ρu_2)
    p_avg = avg(p_1, p_2)
  
    fρ = ρu_avg
    fρu⃗ = avg(ρu_1 * uᵣ_1' + ρuᵣ_1 * u_1' , ρu_2 * uᵣ_2' + ρuᵣ_2 * u_2') + p_avg * I
    fρe = avg(u_1 * (ρeᵣ_1 + pᵣ_1) + uᵣ_1 * (ρe_1 + p_1), u_2 * (ρeᵣ_2 + pᵣ_2) + uᵣ_2 * (ρe_2 + p_2))
  
    # product rule gravity
    α = ρ_avg * 0.5
    fρu⃗ -= α * (Φ₁ - Φ₂) * I
  
    hcat(fρ, fρu⃗, fρe)
  end

  function Atum.surfaceflux(rf::Atum.LinearizedCentralRefanovFlux, law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
  
    # Flux
    f = Atum.twopointflux(Atum.LinearizedCentralFlux(), law, q⁻, aux⁻, q⁺, aux⁺)
  
    # Penalty
    c⁻ = reference_soundspeed(law, aux⁻)
    c⁺ = reference_soundspeed(law, aux⁺)
    c = max(c⁻, c⁺) * rf.scale
  
    # - states
    ρ⁻, ρu⁻, ρe⁻ = unpackstate(law, q⁻)
  
    # + states
    ρ⁺, ρu⁺, ρe⁺ = unpackstate(law, q⁺)
  
    Δρ = ρ⁺ - ρ⁻
    Δρu = ρu⁺ - ρu⁻
    Δρe = ρe⁺ - ρe⁻
  
    fp_ρ = c * Δρ * 0.5
    fp_ρu = c * Δρu * 0.5
    fp_ρe = c * Δρe * 0.5
  
    f' * n⃗ - SVector(fp_ρ, fp_ρu..., fp_ρe)
  end

  function Atum.surfaceflux(rf::Atum.CentralRefanovFlux, law::EulerTotalEnergyLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
  
    # Flux
    f = Atum.twopointflux(Atum.CentralFlux(), law, q⁻, aux⁻, q⁺, aux⁺)
  
    # Penalty
    c⁻ = reference_soundspeed(law, aux⁻)
    c⁺ = reference_soundspeed(law, aux⁺)
    c = max(c⁻, c⁺) * rf.scale
  
    # - states
    ρ⁻, ρu⁻, ρe⁻ = unpackstate(law, q⁻)
  
    # + states
    ρ⁺, ρu⁺, ρe⁺ = unpackstate(law, q⁺)
  
    Δρ = ρ⁺ - ρ⁻
    Δρu = ρu⁺ - ρu⁻
    Δρe = ρe⁺ - ρe⁻
  
    fp_ρ = c * Δρ * 0.5
    fp_ρu = c * Δρu * 0.5
    fp_ρe = c * Δρe * 0.5
  
    f' * n⃗ - SVector(fp_ρ, fp_ρu..., fp_ρe)
  end

end # end of module