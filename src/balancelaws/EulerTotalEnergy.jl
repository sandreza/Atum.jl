module EulerTotalEnergy
  export EulerTotalEnergy

  import ..Atum
  using ..Atum: avg, logavg, roe_avg, constants
  using StaticArrays
  using LinearAlgebra: I, norm

  struct EulerTotalEnergy{FT, D, S, C} <: Atum.AbstractBalanceLaw{FT, D, S, C}
    function EulerTotalEnergy{FT, D}(; γ = 7 // 5,
                                      grav = 981 // 100,
                                      pde_level_balance = false) where {FT, D}
      S = 2 + D
      C = (γ = FT(γ), grav = FT(grav), pde_level_balance = pde_level_balance)
      new{FT, D, S, C}()
    end
  end

  function varsindices(law::EulerTotalEnergy)
    S = Atum.numberofstates(law)
    ix_ρ = 1
    ix_ρu⃗ = StaticArrays.SUnitRange(2, S - 1)
    ix_ρe = S
    return ix_ρ, ix_ρu⃗, ix_ρe
  end

  function varsaux(law::EulerTotalEnergy)
      S = Atum.numberofstates(law)
      ix_x = 1
      ix_y = 2
      ix_z = 3
      ix_ρ = 1 + 3
      ix_ρu⃗ = StaticArrays.SUnitRange(2 + 3, 3 - 1 + 3)
      ix_ρe  = 3 + 3
      ix_geo = 3 + 4
      return ix_x, ix_y, ix_z, ix_ρ, ix_ρu⃗, ix_ρe, ix_geo
  end

  function unpackstate(law::EulerTotalEnergy, q)
      ix_ρ, ix_ρu⃗, ix_ρe = auxindices(law)
      @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρe]
  end

  function unpackaux(law::EulerTotalEnergy, q)
      ix_x, ix_y, ix_z, ix_ρ, ix_ρu⃗, ix_ρe, ix_geo = varsindices(law)
      @inbounds q[ix_x], q[ix_y], q[ix_z], q[ix_ρ], q[ix_ρu⃗], q[ix_ρe], q[ix_geo]
  end

  function Atum.auxiliary(law::EulerTotalEnergy, q)
      unpackaux(law, q)
  end

  function coordinates(law::EulerTotalEnergy, aux)
    aux[1:3]
  end

  function geopotential(law, aux)
    aux[end]
  end

  function linear_pressure(law::EulerTotalEnergy, state, aux)
      ρ, ρu, ρe = unpackstate(law, state)
      x, y, z, ρᵣ, ρuᵣ, ρeᵣ, Φ = unpackaux(law, aux)
      γ = constants(law).γ
      (γ - 1) * (ρe - dot(ρuᵣ, ρu) / ρᵣ + ρ * dot(ρuᵣ, ρuᵣ) / (2 * ρᵣ^2) - ρ * Φ)
  end 

  function reference_soundspeed(law::EulerTotalEnergy, aux)
      x, y, z, ρᵣ, ρuᵣ, ρeᵣ, Φ = unpackaux(law, aux)
      γ = constants(law).γ
      pᵣ = (γ - 1) * (ρeᵣ - dot(ρuᵣ, ρuᵣ) / ρᵣ - ρᵣ * Φ)
      return sqrt(γ * pᵣ / ρᵣ)
  end 

  function Atum.flux(law::EulerTotalEnergy, q, aux)
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

  function Atum.surfaceflux(::Atum.RefanovFlux, law::EulerTotalEnergy, n⃗, q⁻, aux⁻, q⁺, aux⁺)
  
      # f⁻ = Atum.flux(law, q⁻, aux⁻)
      # f⁺ = Atum.flux(law, q⁺, aux⁺)
      f = Atum.twopointflux(Atum.LinearizedKennedyGruberFlux(), law,
                             state_1, aux_1, state_2, aux_2)
  
      c⁻ = reference_soundspeed(law, aux⁻)
      c⁺ = reference_soundspeed(law, aux⁺)
      c = max(c⁻, c⁺)
  
      # - states
      ρ⁻, ρu⁻, ρe⁻ = unpackstate(law, q⁻)
  
      # + states
      ρ⁺, ρu⁺, ρe⁺ = unpackstate(law, q⁺)
  
      Δρ = ρ⁺ - ρ⁻
      Δρu = ρu⁺ - ρu⁻
      Δρe = ρe⁺ - ρe⁻
  
      fp_ρ -= c * Δρ
      fp_ρu -= c * Δρu
      fp_ρe -= c * Δρe
  
      f' * n⃗ / 2 - SVector(fp_ρ, fp_ρu..., fp_ρe)
  end


  function Atum.twopointflux(::Atum.LinearizedKennedyGruberFlux,
                             law::EulerTotalEnergy,
                             state_1, aux_1, state_2, aux_2)
    ρ_1 = state_1.ρ
    ρu_1 = state_1.ρu
    ρe_1 = state_1.ρe

    # grab reference state
    ρᵣ_1  = aux_1.ref_state.ρ
    ρuᵣ_1 = aux_1.ref_state.ρu
    ρeᵣ_1 = aux_1.ref_state.ρe
    pᵣ_1  = aux_1.ref_state.p

    # calculate pressure perturbation
    p_1 = linearized_pressure(law, state_1, aux_1, parameters)

    # calculate u_1, e_1, and reference states
    u_1 = ρu_1 / ρᵣ_1 - ρ_1 * ρuᵣ_1 / (ρᵣ_1^2)
    e_1 = ρe_1 / ρᵣ_1 - ρ_1 * ρeᵣ_1 / (ρᵣ_1^2)

    uᵣ_1 = ρuᵣ_1 / ρᵣ_1
    eᵣ_1 = ρeᵣ_1 / ρᵣ_1

    ## State 2 Stuff 
    # unpack the state perubation
    ρ_2 = state_2.ρ
    ρu_2 = state_2.ρu
    ρe_2 = state_2.ρe

    # grab reference state
    ρᵣ_2 = aux_2.ref_state.ρ
    ρuᵣ_2 = aux_2.ref_state.ρu
    ρeᵣ_2 = aux_2.ref_state.ρe
    pᵣ_2 = aux_2.ref_state.p

    # calculate pressure perturbation
    p_2 = linearized_pressure(law, state_2, aux_2, parameters)

    # calculate u_2, e_2, and reference states
    u_2 = ρu_2 / ρᵣ_2 - ρ_2 * ρuᵣ_2 / (ρᵣ_2^2)
    e_2 = ρe_2 / ρᵣ_2 - ρ_2 * ρeᵣ_2 / (ρᵣ_2^2)

    uᵣ_2 = ρuᵣ_2 / ρᵣ_2
    eᵣ_2 = ρeᵣ_2 / ρᵣ_2

    # construct averages for perturbation variables
    ρ_avg = ave(ρ_1, ρ_2)
    u_avg = ave(u_1, u_2)
    e_avg = ave(e_1, e_2)
    p_avg = ave(p_1, p_2)

    # construct averages for reference variables
    ρᵣ_avg = ave(ρᵣ_1, ρᵣ_2)
    uᵣ_avg = ave(uᵣ_1, uᵣ_2)
    eᵣ_avg = ave(eᵣ_1, eᵣ_2)
    pᵣ_avg = ave(pᵣ_1, pᵣ_2)

    fρ = ρᵣ_avg * u_avg + ρ_avg * uᵣ_avg
    fρu⃗ = p_avg * I + ρᵣ_avg .* (uᵣ_avg .* u_avg' + u_avg .* uᵣ_avg')
    fρu⃗ += (ρ_avg .* uᵣ_avg) .* uᵣ_avg'
    fρe = (ρᵣ_avg * eᵣ_avg + pᵣ_avg) * u_avg
    fρe += (ρᵣ_avg * e_avg + ρ_avg * eᵣ_avg + p_avg) * uᵣ_avg

    hcat(fρ, fρu⃗, fρe)
  end
