module EulerWithTracer
export EulerWithTracerLaw

import ..Atum
using ..Atum: avg, logavg, roe_avg, constants
using StaticArrays
using LinearAlgebra: I, norm

struct EulerWithTracerLaw{FT,D,S,C} <: Atum.AbstractBalanceLaw{FT,D,S,C}
  function EulerWithTracerLaw{FT,D}(; γ=5 // 3,
    grav=0 // 100,
    pde_level_balance=false) where {FT,D}
    S = 3 + D
    C = (γ=FT(γ), grav=FT(grav), pde_level_balance=pde_level_balance)
    new{FT,D,S,C}()
  end
end

function varsindices(law::EulerWithTracerLaw{FT,D,S,C}) where {FT,D,S,C}
  ix_ρ = 1
  ix_ρu⃗ = StaticArrays.SUnitRange(2, 2 + D - 1)
  ix_ρe = S - 1
  ix_ρc = S
  return ix_ρ, ix_ρu⃗, ix_ρe, ix_ρc
end

function auxindices(law::EulerWithTracerLaw)
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

function unpackstate(law::EulerWithTracerLaw, q)
  ix_ρ, ix_ρu⃗, ix_ρe, ix_ρc = varsindices(law)
  @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρe], q[ix_ρc]
end

function unpackrefstate(law::EulerWithTracerLaw, aux)
  ix_x, ix_y, ix_z, ix_geo = auxindices(law)
  @inbounds aux[ix_x], aux[ix_y], aux[ix_z], aux[ix_geo]
end

function unpackaux(law::EulerWithTracerLaw, q)
  ix_x, ix_y, ix_z, ix_ρ, ix_ρu⃗, ix_ρe, ix_geo = auxindices(law)
  @inbounds q[ix_x], q[ix_y], q[ix_z], q[ix_ρ], q[ix_ρu⃗], q[ix_ρe], q[ix_geo]
end

# These are incorrect for a sphere: 
function Atum.auxiliary(law::EulerWithTracerLaw, x⃗)
  ix_x, ix_y, ix_z, ix_ρ, ix_ρu⃗, ix_ρe, ix_geo = auxindices(law)
  ixρu = StaticArrays.SUnitRange(1, 3)
  @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], x⃗[ix_z], x⃗[ixρu]..., x⃗[ix_z], 9.81 * x⃗[ix_z])
end

# These are incorrect for a sphere: 
function Atum.auxiliary(law::EulerWithTracerLaw, x⃗, state)
  ix_x, ix_y, ix_z, ix_Φ = auxindices(law)
  @inbounds SVector(x⃗[ix_x], x⃗[ix_y], x⃗[ix_z], 0.0 * x⃗[ix_z])
end

function coordinates(law::EulerWithTracerLaw, aux)
  aux[1:3]
end

function geopotential(law::EulerWithTracerLaw, aux)
  0.0 * aux[4]
end

function Atum.entropyvariables(law::EulerWithTracerLaw, q, aux)
  ρ, ρu⃗, ρe, ρc = unpackstate(law, q)
  Φ = geopotential(law, aux)
  γ = constants(law).γ
  p = pressure(law, ρ, ρu⃗, ρe, Φ)
  s = log(p / ρ^γ)
  b = ρ / 2p
  u⃗ = ρu⃗ / ρ
  c = ρc / ρ # check this
  vρ = (γ - s) / (γ - 1) - (u⃗' * u⃗ - 2Φ) * b
  vρu⃗ = 2b * u⃗
  vρe = -2b
  vρc = -2b * c # check this

  SVector(vρ, vρu⃗..., vρe, vρc)
end


function pressure(law::EulerWithTracerLaw, ρ, ρu⃗, ρe, Φ)
  γ = constants(law).γ
  return (γ - 1) * (ρe - ρu⃗' * ρu⃗ / 2ρ - ρ * Φ)
end

function pressure(law::EulerWithTracerLaw, state, aux)
  ρ, ρu⃗, ρe = unpackstate(law, state)
  Φ = geopotential(law, aux)
  γ = constants(law).γ
  return (γ - 1) * (ρe - ρu⃗' * ρu⃗ / 2ρ - ρ * Φ)
end

function linearized_pressure(law::EulerWithTracerLaw, state, aux)
  ρ, ρu, ρe = unpackstate(law, state)
  x, y, z, ρᵣ, ρuᵣ, ρeᵣ, Φ = unpackaux(law, aux)
  γ = constants(law).γ
  (γ - 1) * (ρe - StaticArrays.dot(ρuᵣ, ρu) / ρᵣ + ρ * StaticArrays.dot(ρuᵣ, ρuᵣ) / (2 * ρᵣ^2) - ρ * Φ)
end

function reference_pressure(law::EulerWithTracerLaw, aux)
  x, y, z, ρᵣ, ρuᵣ, ρeᵣ, Φ = unpackaux(law, aux)
  γ = constants(law).γ
  pressure(law, ρᵣ, ρuᵣ, ρeᵣ, Φ)
end

function soundspeed(law::EulerWithTracerLaw, ρ, p)
  γ = constants(law).γ
  sqrt(γ * p / ρ)
end

function reference_soundspeed(law::EulerWithTracerLaw, aux)
  x, y, z, ρᵣ, ρuᵣ, ρeᵣ, Φ = unpackaux(law, aux)
  γ = constants(law).γ
  pᵣ = reference_pressure(law, aux)
  return sqrt(γ * pᵣ / ρᵣ)
end

function Atum.flux(law::EulerWithTracerLaw, q, aux)
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

function Atum.surfaceflux(::Atum.RoeFlux, law::EulerWithTracerLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
  Φ = geopotential(law, aux⁻)

  # f⁻ = Atum.flux(law, q⁻, aux⁻)
  # f⁺ = Atum.flux(law, q⁺, aux⁺)
  # main_flux = (f⁻ + f⁺)' * n⃗ * 0.5
  kg_flux = Atum.twopointflux(Atum.KennedyGruberFlux(), law, q⁻, aux⁻, q⁺, aux⁺)
  main_flux = kg_flux' * n⃗

  ρ⁻, ρu⃗⁻, ρe⁻, ρθ⁻ = unpackstate(law, q⁻)
  u⃗⁻ = ρu⃗⁻ / ρ⁻
  e⁻ = ρe⁻ / ρ⁻
  θ⁻ = ρθ⁻ / ρ⁻
  p⁻ = pressure(law, ρ⁻, ρu⃗⁻, ρe⁻, Φ)
  h⁻ = e⁻ + p⁻ / ρ⁻
  c⁻ = soundspeed(law, ρ⁻, p⁻)

  ρ⁺, ρu⃗⁺, ρe⁺, ρθ⁺ = unpackstate(law, q⁺)
  u⃗⁺ = ρu⃗⁺ / ρ⁺
  e⁺ = ρe⁺ / ρ⁺
  θ⁺ = ρθ⁺ / ρ⁺
  p⁺ = pressure(law, ρ⁺, ρu⃗⁺, ρe⁺, Φ)
  h⁺ = e⁺ + p⁺ / ρ⁺
  c⁺ = soundspeed(law, ρ⁺, p⁺)

  ρ = sqrt(ρ⁻ * ρ⁺)
  u⃗ = roe_avg(ρ⁻, ρ⁺, u⃗⁻, u⃗⁺)
  h = roe_avg(ρ⁻, ρ⁺, h⁻, h⁺)
  c = roe_avg(ρ⁻, ρ⁺, c⁻, c⁺)
  θ = roe_avg(ρ⁻, ρ⁺, θ⁻, θ⁺)

  uₙ = u⃗' * n⃗

  Δρ = ρ⁺ - ρ⁻
  Δp = p⁺ - p⁻
  Δu⃗ = u⃗⁺ - u⃗⁻
  Δuₙ = Δu⃗' * n⃗
  Δρθ = ρθ⁺ - ρθ⁻

  c⁻² = 1 / c^2
  w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² * 0.5
  w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² * 0.5
  w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
  w4 = abs(uₙ) * ρ
  w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

  fp_ρ = (w1 + w2 + w3) * 0.5
  fp_ρu⃗ = (w1 * (u⃗ - c * n⃗) +
            w2 * (u⃗ + c * n⃗) +
            w3 * u⃗ +
            w4 * (Δu⃗ - Δuₙ * n⃗)) * 0.5
  fp_ρe = (w1 * (h - c * uₙ) +
           w2 * (h + c * uₙ) +
           w3 * (u⃗' * u⃗ * 0.5 + Φ) +
           w4 * (u⃗' * Δu⃗ - uₙ * Δuₙ)) * 0.5
  fp_ρθ = ((w1 + w2) * θ + w5) / 2

  main_flux - SVector(fp_ρ, fp_ρu⃗..., fp_ρe, fp_ρθ)
end

function Atum.surfaceflux(::Atum.KennedyGruberFlux, law::EulerWithTracerLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
  f = Atum.twopointflux(Atum.KennedyGruberFlux(), law, q⁻, aux⁻, q⁺, aux⁺)
  f' * n⃗
end

function Atum.surfaceflux(::Atum.LinearizedKennedyGruberFlux, law::EulerWithTracerLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
  f = Atum.twopointflux(Atum.LinearizedKennedyGruberFlux(), law, q⁻, aux⁻, q⁺, aux⁺)
  f' * n⃗
end

function Atum.twopointflux(::Atum.KennedyGruberFlux, law::EulerWithTracerLaw, q₁, aux₁, q₂, aux₂)

  ρ₁, ρu⃗₁, ρe₁, ρc₁ = unpackstate(law, q₁)
  ρ₂, ρu⃗₂, ρe₂, ρc₂ = unpackstate(law, q₂)

  Φ₁ = geopotential(law, aux₁)
  u⃗₁ = ρu⃗₁ / ρ₁
  e₁ = ρe₁ / ρ₁
  c₁ = ρc₁ / ρ₁
  p₁ = pressure(law, ρ₁, ρu⃗₁, ρe₁, Φ₁)

  Φ₂ = geopotential(law, aux₂)
  u⃗₂ = ρu⃗₂ / ρ₂
  e₂ = ρe₂ / ρ₂
  c₂ = ρc₂ / ρ₂
  p₂ = pressure(law, ρ₂, ρu⃗₂, ρe₂, Φ₂)

  ρ_avg = avg(ρ₁, ρ₂)
  u⃗_avg = avg(u⃗₁, u⃗₂)
  e_avg = avg(e₁, e₂)
  c_avg = avg(c₁, c₂)
  p_avg = avg(p₁, p₂)

  fρ = u⃗_avg * ρ_avg
  fρu⃗ = u⃗_avg * fρ' + p_avg * I
  fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)
  fρc = u⃗_avg * (ρ_avg * c_avg)

  # fluctuation
  #=
  α = ρ_avg / 2
  fρu⃗ -= α * (Φ₁ - Φ₂) * I
  =#

  hcat(fρ, fρu⃗, fρe, fρc)
end

# Entropy Conservative with Kinetic-Energy preserving tracer 
function Atum.twopointflux(::Atum.EntropyConservativeFlux, law::EulerWithTracerLaw, q₁, aux₁, q₂, aux₂)


  γ = constants(law).γ
  ρ₁, ρu⃗₁, ρe₁, ρc₁ = unpackstate(law, q₁)
  ρ₂, ρu⃗₂, ρe₂, ρc₂ = unpackstate(law, q₂)

  Φ₁ = geopotential(law, aux₁)
  u⃗₁ = ρu⃗₁ / ρ₁
  c₁ = ρc₁ / ρ₁
  p₁ = pressure(law, ρ₁, ρu⃗₁, ρe₁, Φ₁)
  b₁ = ρ₁ / 2p₁

  Φ₂ = geopotential(law, aux₂)
  u⃗₂ = ρu⃗₂ / ρ₂
  c₂ = ρc₂ / ρ₂
  p₂ = pressure(law, ρ₂, ρu⃗₂, ρe₂, Φ₂)
  b₂ = ρ₂ / 2p₂

  ρ_avg = avg(ρ₁, ρ₂)
  u⃗_avg = avg(u⃗₁, u⃗₂)
  c_avg = avg(c₁, c₂)
  b_avg = avg(b₁, b₂)
  Φ_avg = avg(Φ₁, Φ₂)

  u²_avg = avg(u⃗₁' * u⃗₁, u⃗₂' * u⃗₂)
  ρ_log = logavg(ρ₁, ρ₂)
  b_log = logavg(b₁, b₂)

  fρ = u⃗_avg * ρ_log
  fρu⃗ = u⃗_avg * fρ' + ρ_avg / 2b_avg * I
  fρe = (1 / (2 * (γ - 1) * b_log) - u²_avg / 2 + Φ_avg) * fρ + fρu⃗ * u⃗_avg
  # fρc = c_avg * fρ' # check this 
  fρc = u⃗_avg * (ρ_avg * c_avg) # use kennedy gruber flux instead


  hcat(fρ, fρu⃗, fρe, fρc)
end


function Atum.surfaceflux(::Atum.MatrixFlux, law::EulerWithTracerLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
  FT = eltype(law)
  γ = constants(law).γ
  ecflux = Atum.surfaceflux(Atum.EntropyConservativeFlux(), law, n⃗, q⁻, aux⁻, q⁺, aux⁺) # defined in surface fluxes automagically

  ρ⁻, ρu⃗⁻, ρe⁻, ρθ⁻ = unpackstate(law, q⁻)
  Φ⁻ = geopotential(law, aux⁻)
  u⃗⁻ = ρu⃗⁻ / ρ⁻
  e⁻ = ρe⁻ / ρ⁻
  θ⁻ = ρθ⁻ / ρ⁻
  p⁻ = pressure(law, ρ⁻, ρu⃗⁻, ρe⁻, Φ⁻)
  b⁻ = ρ⁻ / 2p⁻

  ρ⁺, ρu⃗⁺, ρe⁺, ρθ⁺ = unpackstate(law, q⁺)
  Φ⁺ = geopotential(law, aux⁺)
  u⃗⁺ = ρu⃗⁺ / ρ⁺
  e⁺ = ρe⁺ / ρ⁺
  θ⁺ = ρθ⁺ / ρ⁺
  p⁺ = pressure(law, ρ⁺, ρu⃗⁺, ρe⁺, Φ⁺)
  b⁺ = ρ⁺ / 2p⁺

  Φ_avg = avg(Φ⁻, Φ⁺)
  ρ_log = logavg(ρ⁻, ρ⁺)
  b_log = logavg(b⁻, b⁺)
  u⃗_avg = avg(u⃗⁻, u⃗⁺)
  θ_avg = avg(θ⁻, θ⁺)
  p_avg = avg(ρ⁻, ρ⁺) / 2avg(b⁻, b⁺)
  u²_bar = 2 * norm(u⃗_avg) - avg(norm(u⃗⁻), norm(u⃗⁺))
  h_bar = γ / (2 * b_log * (γ - 1)) + u²_bar / 2 + Φ_avg
  c_bar = sqrt(γ * p_avg / ρ_log)

  u⃗mc = u⃗_avg - c_bar * n⃗
  u⃗pc = u⃗_avg + c_bar * n⃗
  u_avgᵀn = u⃗_avg' * n⃗

  v⁻ = Atum.entropyvariables(law, q⁻, aux⁻)
  v⁺ = Atum.entropyvariables(law, q⁺, aux⁺)
  Δv = v⁺ - v⁻

  λ1 = abs(u_avgᵀn - c_bar) * ρ_log / 2γ
  λ2 = abs(u_avgᵀn) * ρ_log * (γ - 1) / γ
  λ3 = abs(u_avgᵀn + c_bar) * ρ_log / 2γ
  λ4 = abs(u_avgᵀn) * p_avg

  Δv_ρ, Δv_ρu⃗, Δv_ρe, Δv_ρc = unpackstate(law, Δv)
  u⃗ₜ = u⃗_avg - u_avgᵀn * n⃗

  w1 = λ1 * (Δv_ρ + u⃗mc' * Δv_ρu⃗ + (h_bar - c_bar * u_avgᵀn) * Δv_ρe)
  w2 = λ2 * (Δv_ρ + u⃗_avg' * Δv_ρu⃗ + (u²_bar / 2 + Φ_avg) * Δv_ρe)
  w3 = λ3 * (Δv_ρ + u⃗pc' * Δv_ρu⃗ + (h_bar + c_bar * u_avgᵀn) * Δv_ρe)

  Dρ = w1 + w2 + w3

  Dρu⃗ = (w1 * u⃗mc +
          w2 * u⃗_avg +
          w3 * u⃗pc +
          λ4 * (Δv_ρu⃗ - n⃗' * (Δv_ρu⃗) * n⃗ + Δv_ρe * u⃗ₜ))

  Dρe = (w1 * (h_bar - c_bar * u_avgᵀn) +
         w2 * (u²_bar / 2 + Φ_avg) +
         w3 * (h_bar + c_bar * u_avgᵀn) +
         λ4 * (u⃗ₜ' * Δv_ρu⃗ + Δv_ρe * (u⃗_avg' * u⃗_avg - u_avgᵀn^2)))

  c = c_bar
  ρ = ρ_log
  c⁻² = 1 / c^2
  uₙ = u_avgᵀn

  Δp = p⁺ - p⁻
  Δu⃗ = u⃗⁺ - u⃗⁻
  Δuₙ = Δu⃗' * n⃗
  Δρθ = ρθ⁺ - ρθ⁻

  ww1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² * 0.5
  ww3 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² * 0.5
  ww5 = abs(uₙ) * (Δρθ - θ_avg  * Δp * c⁻²)
  Dρc = (ww1 + ww3) * θ_avg + ww5


  ecflux - SVector(Dρ, Dρu⃗..., Dρe, Dρc) / 2
end


end # end of module