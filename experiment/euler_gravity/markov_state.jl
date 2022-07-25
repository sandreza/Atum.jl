ρ_m, ρu_m, ρv_m, ρw_m, ρe_m = components(test_state)
ρ_m2, ρu_m2, ρv_m2, ρw_m2, ρe_m2 = components(state)
ρₙ = maximum(abs.(ρ_m))   # 1.3ish
ρuₙ = maximum(abs.(ρu_m)) # 60ishh
ρvₙ = maximum(abs.(ρv_m)) # 60ish
ρwₙ = maximum(abs.(ρw_m)) # 30ish
ρeₙ = maximum(abs.(ρe_m)) # 2.3e6ish

ρₙ2 = maximum(abs.(ρ_m2))   # 1.3ish
ρuₙ2 = maximum(abs.(ρu_m2)) # 60ishh
ρvₙ2 = maximum(abs.(ρv_m2)) # 60ish
ρwₙ2 = maximum(abs.(ρw_m2)) # 30ish
ρeₙ2 = maximum(abs.(ρe_m2)) # 2.3e6ish

sum(ρ_m .* dg_fs.MJ) / sum(dg_fs.MJ)
sum(ρ_m2 .* dg_fs.MJ) / sum(dg_fs.MJ)

# This distance function emphasizes the difference between the two states.
# It is not a good measure of the difference between density or total energy
# roughly by a factor of 10
function distance_gpu(x, y, dg_fs; normalization=(1.3 / 3, 60.0, 60.0, 60.0, 2.3e6 / 3))
    ρ_m, ρu_m, ρv_m, ρw_m, ρe_m = components(x)
    ρ_m2, ρu_m2, ρv_m2, ρw_m2, ρe_m2 = components(y)
    MJ = dg_fs.MJ
    error_ρ = sum((ρ_m - ρ_m2) .^ 2 .* MJ) / sum(MJ) / normalization[1]^2
    error_ρu = sum((ρu_m - ρu_m2) .^ 2 .* MJ) / sum(MJ) / normalization[2]^2
    error_ρv = sum((ρv_m - ρv_m2) .^ 2 .* MJ) / sum(MJ) / normalization[3]^2
    error_ρw = sum((ρw_m - ρw_m2) .^ 2 .* MJ) / sum(MJ) / normalization[4]^2
    error_ρe = sum((ρe_m - ρe_m2) .^ 2 .* MJ) / sum(MJ) / normalization[5]^2
    error_total = sqrt.((error_ρ, error_ρu, error_ρv, error_ρw, error_ρe))
    return error_total
end

function distance_cpu(x, y, dg_fs; normalization=(1.3 / 3, 60.0, 60.0, 60.0, 2.3e6 / 3))
    ρ_m, ρu_m, ρv_m, ρw_m, ρe_m = Array.(components(x))
    ρ_m2, ρu_m2, ρv_m2, ρw_m2, ρe_m2 = Array.(components(y))
    MJ = Array(dg_fs.MJ)
    error_ρ = sum((ρ_m - ρ_m2) .^ 2 .* MJ) / sum(MJ) / normalization[1]^2
    error_ρu = sum((ρu_m - ρu_m2) .^ 2 .* MJ) / sum(MJ) / normalization[2]^2
    error_ρv = sum((ρv_m - ρv_m2) .^ 2 .* MJ) / sum(MJ) / normalization[3]^2
    error_ρw = sum((ρw_m - ρw_m2) .^ 2 .* MJ) / sum(MJ) / normalization[4]^2
    error_ρe = sum((ρe_m - ρe_m2) .^ 2 .* MJ) / sum(MJ) / normalization[5]^2
    error_total = sqrt.((error_ρ, error_ρu, error_ρv, error_ρw, error_ρe))
    return error_total
end

function convert_gpu_to_cpu(state)
    return Array.(components(state))
end

function distance(x, y, metric; normalization=(1.3 / 3, 60.0, 60.0, 60.0, 2.3e6 / 3))
    ρ_m, ρu_m, ρv_m, ρw_m, ρe_m = x
    ρ_m2, ρu_m2, ρv_m2, ρw_m2, ρe_m2 = y
    error_ρ = sum((ρ_m - ρ_m2) .^ 2 .* metric) / sum(metric) / normalization[1]^2
    error_ρu = sum((ρu_m - ρu_m2) .^ 2 .* metric) / sum(metric) / normalization[2]^2
    error_ρv = sum((ρv_m - ρv_m2) .^ 2 .* metric) / sum(metric) / normalization[3]^2
    error_ρw = sum((ρw_m - ρw_m2) .^ 2 .* metric) / sum(metric) / normalization[4]^2
    error_ρe = sum((ρe_m - ρe_m2) .^ 2 .* metric) / sum(metric) / normalization[5]^2
    error_total = sqrt.((error_ρ, error_ρu, error_ρv, error_ρw, error_ρe))
    return sum(error_total)
end


# Check distance in a few timesteps
totes_sim = 2000
save_radius = []
state .= test_state
for i in 1:totes_sim
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = 5 * dt
    solve!(test_state, end_time, odesolver, adjust_final=false)
    distances = distance_gpu(test_state, state, dg_explicit)
    push!(save_radius, distances)
    if i % 10 == 0
        println("currently at timestep ", i, " out of ", totes_sim)
        println("current distances are ", save_radius[i])
    end
end

fig_d = Figure()
ax1 = Axis(fig_d[1, 1])
ax2 = Axis(fig_d[1, 2])
ax3 = Axis(fig_d[2, 1])
ax4 = Axis(fig_d[2, 2])
ax5 = Axis(fig_d[1, 3])
ax6 = Axis(fig_d[2, 3])
r1 = [radius[1] for radius in save_radius]
r2 = [radius[2] for radius in save_radius]
r3 = [radius[3] for radius in save_radius]
r4 = [radius[4] for radius in save_radius]
r5 = [radius[5] for radius in save_radius]
r6 = [sum(radius) for radius in save_radius]
lines!(ax1, r1)
lines!(ax2, r2)
lines!(ax3, r3)
lines!(ax4, r4)
lines!(ax5, r5)
lines!(ax6, r6)

hist(r6[1000:end], bins=100)
distance_threshold = quantile(r6[1000:end], 0.15)
# once in statistically steady state evolve for a bit 
# choose threshold for quantiles as a distance, 0.18 seems legit

# (5.896364107365401e-5, 0.0005940753758554695, 0.0005707376275348086, 0.0006037918041617984, 5.4708253453281114e-5)

totes_sim = 400 * 100 # 100 is about 30 days
markov_states = []
current_state = []
states_in_time = Int64[]
push!(markov_states, convert_gpu_to_cpu(test_state))
push!(states_in_time, 1)

MJ = Array(dg_fs.MJ)
for i in 1:totes_sim
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = 25 * dt
    solve!(test_state, end_time, odesolver, adjust_final=false)
    candidate_state = convert_gpu_to_cpu(test_state)
    distances = [distance(markov_state, candidate_state, MJ) for markov_state in markov_states]
    push!(save_radius, distances)
    if all(distances .>= distance_threshold)
        push!(markov_states, candidate_state)
        push!(current_state, length(markov_states))
    else
        push!(current_state, argmin(distances))
    end

    push!(states_in_time, length(markov_states))
    if i % 10 == 0
        println("currently at timestep ", i, " out of ", totes_sim)
        println("current number of states are ", states_in_time[i])
    end
end


# get rid of bad last state
if current_state[end] == length(markov_states)
    while current_state[end] == length(markov_states)
        pop!(current_state)
    end
    count_matrix = zeros(length(markov_states) - 1, length(markov_states) - 1)
else
    count_matrix = zeros(length(markov_states), length(markov_states))
end


for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end
perron_frobenius = count_matrix ./ sum(count_matrix, dims=1)
ll, vv = eigen(perron_frobenius)
p = real.(vv[:, end] / sum(vv[:, end]))
println("The entropy is ", -sum(p .* log.(p) ./log(length(p))))