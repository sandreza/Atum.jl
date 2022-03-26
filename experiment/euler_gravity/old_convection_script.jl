using CLIMAParameters#: AbstractEarthParameterSet
struct PlanetParameterSet <: AbstractEarthParameterSet end
get_planet_parameter(p::Symbol) = getproperty(CLIMAParameters.Planet, p)(PlanetParameterSet())

parameters = (
    R_d = get_planet_parameter(:R_d),
    pₒ = 1e5, # get_planet_parameter(:MSLP),
    κ = get_planet_parameter(:kappa_d),
    g = get_planet_parameter(:grav),
    cp_d = get_planet_parameter(:cp_d),
    cv_d = get_planet_parameter(:cv_d),
    γ = get_planet_parameter(:cp_d) / get_planet_parameter(:cv_d),
    T_0 = 0 * get_planet_parameter(:T_0),
    xmax = 3e3,
    ymax = 3e3,
    zmax = 3e3,
    θₐ = 2.0,
    cₛ = 340,
    Tₛ = 300.0, # 300.0,
    ϵ = 1e-0,
    ρₛ = 1.27,
    scale_height = 8e3,
    Δθ = 10.0,
    Q₀ = 100.0, # W/m²
    r_ℓ = 100.0, # radiation length scale
    s_ℓ = 100.0, # sponge exponential decay length scale
    λ = 1 / 10.0, # sponge relaxation timescale
)

# set up grid
x_domain = IntervalDomain(min = 0.0, max = parameters.xmax, periodic = true)
y_domain = IntervalDomain(min = 0.0, max = parameters.ymax, periodic = true)
z_domain = IntervalDomain(min = 0.0, max = parameters.zmax, periodic = false)
discretized_domain = DiscretizedDomain(
    domain = x_domain × y_domain × z_domain,
    discretization = (
        elements = (25, 25, 25),
        polynomial_order = (7, 7, 7),
        grid_stretching = nothing,),
)

## Γ = g / c_p, Tₛ = 300
## T(z) = -g/cₚ z + Tₛ
## p = R_d ρ T and dp/dz = -g ρ => ∂ᶻ( R_d ρ (-g/cₚ z + Tₛ) ) = - g ρ
## -g R_d / cₚ ρ + (-g/cₚ z  + Tₛ) R_d ∂ᶻ(ρ) = - g ρ => (-g/cₚ z  + Tₛ) ∂ᶻ(ρ) = -g( 1/R_d + 1/cₚ) ρ
## ∂ᶻ ln(ρ) = -g( 1/R_d + 1/cₚ) / (-g/cₚ z  + Tₛ) => ln(ρ) = -( cₚ /R_d + 1) ln(-g/cₚ z  + Tₛ) + C
## ρ = ρ₀*(-g/cₚ z  + Tₛ)^(-( cₚ /R_d + 1))
# set up initial condition 

# temperature drops by about 70 Kelvin in 10km (linearish)
# pressure drops by about 28% in 8.85km (exponentialish), 7km scale height
# potential temperature increases by about 10 degrees over 2km , 15 + 273 at ground

#=
R = 287
cp = 1004
p0 = 1e5
g = 9.8
Γ = 10 / 2e3
θ0 = 300
θnew = @. θ0 - Γ * z
pnew = @. p0 * (g / (Γ * cp) * log(θnew / θ0) + 1)^(cp / R)
scatter(pnew, z)
Tnew = @. (pnew / p0)^(R / cp) * θnew
ρnew = @. pnew / (R * Tnew)
ρnew_check = @. 1/(-g) * p0 * (cp / R) * (g / (Γ * cp) * log(θnew / θ0) + 1)^(cp / R - 1) * (g / (Γ * cp) * -Γ/θnew)
=#

θ₀(p, x, y, z) = p.Tₛ + p.Δθ / p.zmax * z
p₀(p, x, y, z) = p.pₒ * (p.g / (-p.Δθ / p.zmax * p.cp_d) * log(θ₀(p, x, y, z) / p.Tₛ) + 1)^(p.cp_d / p.R_d)
T₀(p, x, y, z) = (p₀(p, x, y, z) / p.pₒ)^(p.R_d / p.cp_d) * θ₀(p, x, y, z)
ρ₀(p, x, y, z) = p₀(p, x, y, z) / (p.R_d * T₀(p, x, y, z))

ρu₀(p, x, y, z) = 0.01 * @SVector [randn(), randn(), randn()]

e_pot(p, x, y, z) = p.g * z
e_int(p, x, y, z) = p.cv_d * (T₀(p, x, y, z) - p.T_0)
e_kin(p, x, y, z) = 0.5 * (ρu₀(p, x, y, z)' * ρu₀(p, x, y, z)) / ρ₀(p, x, y, z)^2

ρe₀(p, x, y, z) = ρ₀(p, x, y, z) * (e_kin(p, x, y, z) + e_int(p, x, y, z) + e_pot(p, x, y, z))

#=
π_exn(p, x, y, z) = 1.0 - p.g / (p.cp_d * θ₀(p, x, y, z)) * z

e_pot(p, x, y, z) = p.g * z
e_int(p, x, y, z) = p.cv_d * (θ₀(p, x, y, z) * π_exn(p, x, y, z) - p.T_0)
e_kin(p, x, y, z) = 0.0

ρ₀(p, x, y, z) = p.pₒ / (p.R_d * θ₀(p, x, y, z)) * (π_exn(p, x, y, z))^(p.cv_d / p.R_d)
ρu₀(p, x, y, z) = @SVector [0.0, 0.0, 0.0]
ρe₀(p, x, y, z) = ρ₀(p, x, y, z) * (e_kin(p, x, y, z) + e_int(p, x, y, z) + e_pot(p, x, y, z))
=#

# Radiative Forcing 
struct ConvectiveForcing{S} <: AbstractForcing
    parameters::S
end

convective_parameters = (;
    Q₀ = parameters.Q₀,
    r_ℓ = parameters.r_ℓ,
    s_ℓ = parameters.s_ℓ,
    λ = parameters.λ,
    L = parameters.zmax,
)

function calc_source!(
    source,
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    convective_forcing::ConvectiveForcing,
    state,
    aux,
)
    FT = eltype(state)

    # Extract the state
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ = aux.Φ

    x = aux.x
    y = aux.y
    z = aux.z
    coord = @SVector [x, y, z]

    Q₀  = 100.0# convective_forcing.Q₀
    r_ℓ = 100.0# convective_forcing.r_ℓ
    s_ℓ = 100.0# convective_forcing.s_ℓ
    λ   = 1/10.0# convective_forcing.λ
    L   = 3e3# convective_forcing.L

    radiation_profile = Q₀ / r_ℓ * exp(-z / r_ℓ)

    damping_profile = -exp(-(L - z) / s_ℓ)

    # Apply convective forcing
    source.ρu += λ * damping_profile * ρu
    source.ρe += ρ * radiation_profile

    return nothing
end

# set up model

model = ModelSetup(
    equations = ThreeDimensionalEuler(
        thermodynamic_variable = TotalEnergy(),
        equation_of_state = DryIdealGas(),
        pressure_convention = Compressible(),
        sources = (
            # coriolis = DeepShellCoriolis(),
            gravity = Gravity(),
            forcing = ConvectiveForcing(convective_parameters),
        ),
        ref_state = NoReferenceState(),
    ),
    boundary_conditions = (DefaultBC(), DefaultBC(), DefaultBC(), DefaultBC(), HackBC(), DefaultBC()),
    initial_conditions = (
        ρ = ρ₀, ρu = ρu₀, ρe = ρe₀,
    ), parameters = parameters,
)

end_time = 60 * 60 * 6 # 60 * 60 * 8 could probably be divided by 2

numerical_grid = create_grid(backend, discretized_domain);
cₛ = 330
Δxᵥ = min_node_distance(numerical_grid, VerticalDirection())
Δxₕ = min_node_distance(numerical_grid, HorizontalDirection())
Δt = (Δxᵥ / cₛ) * 1.8 # (Δxᵥ / cₛ) * 0.1
println("The timestep is ", Δt)
vCFL = Δt / (Δxᵥ / cₛ)
hCFL = Δt / (Δxₕ / cₛ)

iteration_partition = 10 # for videos 1000
iterations = floor(Int, end_time / Δt / iteration_partition)


println("vcfl is ", vCFL)
println("hcfl is ", hCFL)

jl_cb = JLD2State(iteration = iterations, filepath = filename)

# methods = (LSRK144NiegemannDiehlBusch, SSPRK22Heuns)
# cfl (1.8, 0.2)
# set up simulation
simulation = Simulation(
    backend = backend,
    discretized_domain = discretized_domain,
    model = model,
    timestepper = (
        method = LSRK144NiegemannDiehlBusch,
        start = 0.0,
        finish = end_time,
        timestep = Δt,
    ),
    callbacks = (
        Info(),
        jl_cb,
        CFL(),
    ),
)

# run the simulation
tic = Base.time()
initialize!(simulation)
evolve!(simulation)
toc = Base.time()
println("The amount of time for the simulation is ", (toc - tic) / 60, " minutes")