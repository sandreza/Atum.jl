using ProgressBars, HDF5

filename = "part3_markov_model_even_time_nstate_" * string(length(markov_states)) * ".h5"
fid = h5open(filename, "w")
markov_array = zeros(size(markov_states[1][1])..., 5)
save_ic = convert_gpu_to_cpu(test_state)
for j in 1:5
    markov_array[:, :, j] .= save_ic[j]
end
fid["initial condition"] = markov_array

function not_distance(x, y, metric; normalization=(1.3, 60.0, 60.0, 60.0, 2.3e6))
    ρ_m, ρu_m, ρv_m, ρw_m, ρe_m = x
    ρ_m2, ρu_m2, ρv_m2, ρw_m2, ρe_m2 = y
    powa = 1 / 2
    error_ρ = sum(abs.(ρ_m .- ρ_m2) .^ powa .* metric) / sum(metric) / normalization[1]^powa
    error_ρu = sum(abs.(ρu_m .- ρu_m2) .^ powa .* metric) / sum(metric) / normalization[2]^powa
    error_ρv = sum(abs.(ρv_m .- ρv_m2) .^ powa .* metric) / sum(metric) / normalization[3]^powa
    error_ρw = sum(abs.(ρw_m .- ρw_m2) .^ powa .* metric) / sum(metric) / normalization[4]^powa
    error_ρe = sum(abs.(ρe_m .- ρe_m2) .^ powa .* metric) / sum(metric) / normalization[5]^powa

    error_total = (error_ρ, error_ρu, error_ρv, error_ρw, error_ρe) .^ (1 / powa)
    return sum(error_total)
end

function distance1(x, y, metric; normalization=(1.3, 60.0, 60.0, 60.0, 2.3e6))
    ρ_m, ρu_m, ρv_m, ρw_m, ρe_m = x
    ρ_m2, ρu_m2, ρv_m2, ρw_m2, ρe_m2 = y
    error_ρ = sum(abs.(ρ_m .- ρ_m2) .* metric) / sum(metric) / normalization[1]
    error_ρu = sum(abs.(ρu_m .- ρu_m2) .* metric) / sum(metric) / normalization[2]
    error_ρv = sum(abs.(ρv_m .- ρv_m2) .* metric) / sum(metric) / normalization[3]
    error_ρw = sum(abs.(ρw_m .- ρw_m2) .* metric) / sum(metric) / normalization[4]
    error_ρe = sum(abs.(ρe_m .- ρe_m2) .* metric) / sum(metric) / normalization[5]
    error_total = (error_ρ, error_ρu, error_ρv, error_ρw, error_ρe)
    return sum(error_total)
end

function distance2(x, y, metric; normalization=(1.3, 60.0, 60.0, 60.0, 2.3e6))
    ρ_m, ρu_m, ρv_m, ρw_m, ρe_m = x
    ρ_m2, ρu_m2, ρv_m2, ρw_m2, ρe_m2 = y
    powa = 2
    error_ρ = sum(abs.(ρ_m .- ρ_m2) .^ powa .* metric) / sum(metric) / normalization[1]^powa
    error_ρu = sum(abs.(ρu_m .- ρu_m2) .^ powa .* metric) / sum(metric) / normalization[2]^powa
    error_ρv = sum(abs.(ρv_m .- ρv_m2) .^ powa .* metric) / sum(metric) / normalization[3]^powa
    error_ρw = sum(abs.(ρw_m .- ρw_m2) .^ powa .* metric) / sum(metric) / normalization[4]^powa
    error_ρe = sum(abs.(ρe_m .- ρe_m2) .^ powa .* metric) / sum(metric) / normalization[5]^powa

    error_total = (error_ρ, error_ρu, error_ρv, error_ρw, error_ρe) .^ (1 / powa)
    return sum(error_total)
end

function assign_index(markov_states, candidate_state, MJ)
    m_distances = zeros(length(markov_states))
    Threads.@threads for j in eachindex(markov_states)
        m_distances[j] = not_distance(markov_states[j], candidate_state, MJ)
    end
    argmin(m_distances)
end

function assign_index1(markov_states, candidate_state, MJ)
    m_distances = zeros(length(markov_states))
    Threads.@threads for j in eachindex(markov_states)
        m_distances[j] = distance1(markov_states[j], candidate_state, MJ)
    end
    argmin(m_distances)
end

function assign_index2(markov_states, candidate_state, MJ)
    m_distances = zeros(length(markov_states))
    Threads.@threads for j in eachindex(markov_states)
        m_distances[j] = distance2(markov_states[j], candidate_state, MJ)
    end
    argmin(m_distances)
end

function assign_index_together(markov_states, candidate_state, MJ)
    m_distances = zeros(length(markov_states), 3)
    Threads.@threads for j in eachindex(markov_states)
        m_distances[j, 1] = not_distance(markov_states[j], candidate_state, MJ)
        m_distances[j, 2] = distance1(markov_states[j], candidate_state, MJ)
        m_distances[j, 3] = distance2(markov_states[j], candidate_state, MJ)
    end
    return [argmin(m_distances[:, i]) for i in 1:3]
end

totes_sim = 10 * 12 * 100 * 10 * 10 # 100 is about 30 days for timejump = 1

markov_chain_p2 = Int64[]
markov_chain_1 = Int64[]
markov_chain_2 = Int64[]

markov_chain = []

time_jump = 5 # 5 is a good compromise between speed and accuracy
MJ = Array(dg_fs.MJ)
for i in ProgressBar(1:totes_sim)
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = time_jump * dt
    solve!(test_state, end_time, odesolver, adjust_final=false)
    candidate_state = convert_gpu_to_cpu(test_state)

    # markov embedding
    # push!(markov_chain_p2, assign_index(markov_states, candidate_state, MJ))
    # push!(markov_chain_1, assign_index1(markov_states, candidate_state, MJ))
    push!(markov_chain_2, assign_index2(markov_states, candidate_state, MJ))
    # push!(markov_chain, assign_index_together(markov_states, candidate_state, MJ))
end

if length(markov_chain) > 0
    markov_chain_p2 = [markov_chain[i][1] for i in 1:length(markov_chain)]
    markov_chain_1 = [markov_chain[i][2] for i in 1:length(markov_chain)]
    markov_chain_2 = [markov_chain[i][3] for i in 1:length(markov_chain)]
end

# memo to self. Record initial condition and the final condition. [DONE]
save_fc = convert_gpu_to_cpu(test_state)
for j in 1:5
    markov_array[:, :, j] .= save_fc[j]
end
fid["final condition"] = markov_array
fid["markov embedding 0p5"] = markov_chain_p2
fid["markov embedding 1"] = markov_chain_1
fid["markov embedding 2"] = markov_chain_2
fid["time jump "] = time_jump
fid["dt"] = dt
fid["small planet factor"] = X
close(fid)


#=
fig = Figure()
ax11 = Axis(fig[1, 1]; title="p = 2")
ax12 = Axis(fig[1, 2]; title="p = 1")
ax13 = Axis(fig[1, 3]; title="p = 1/2")

ax21 = Axis(fig[2, 1]; title="p = 2")
ax22 = Axis(fig[2, 2]; title="p = 1")
ax23 = Axis(fig[2, 3]; title="p = 1/2")

scatter!(ax11, markov_chain_p2[1:500], color="red")
scatter!(ax12, markov_chain_1[1:500], color="blue")
scatter!(ax13, markov_chain_2[1:500], color="orange")

scatter!(ax21, markov_chain_p2[end-500:end], color="red")
scatter!(ax22, markov_chain_1[end-500:end], color="blue")
scatter!(ax23, markov_chain_2[end-500:end], color="orange")

cp2 = count_operator(markov_chain_p2)
c1 = count_operator(markov_chain_1)
c2 = count_operator(markov_chain_2)

pf_p2 = cp2 ./ sum(cp2, dims=1)
pf_1 = c1 ./ sum(c1, dims=1)
pf_2 = c2 ./ sum(c2, dims=1)

scp2 = [sum(markov_chain_p2 .== i)/length(markov_chain_1) for i in eachindex(markov_states)]
sc1 = [sum(markov_chain_1 .== i)/length(markov_chain_1) for i in eachindex(markov_states)]
sc2 = [sum(markov_chain_2 .== i) / length(markov_chain_1) for i in eachindex(markov_states)]

for steady_distribution in [scp2, sc1, sc2]
    entropy = -sum(steady_distribution .* log.(steady_distribution)) / log(length(markov_states))
    println("The entropy is $entropy")
end
=#
#=
stragglers = setdiff(1:maximum(current_state), union(current_state))
current_state = [current_state..., stragglers..., current_state[1]]
count_matrix = zeros(length(markov_states), length(markov_states));
for i in 1:length(current_state)-1
    count_matrix[current_state[i+1], current_state[i]] += 1
end

column_sum = sum(count_matrix, dims=1) .> 0;
row_sum = sum(count_matrix, dims=2) .> 0;
if all(column_sum[:] .== row_sum[:])
    reduced_count_matrix = count_matrix[column_sum[:], column_sum[:]]
    reduced_markov_states = markov_states[column_sum[:]]
else
    println("think harder")
end

perron_frobenius = reduced_count_matrix ./ sum(reduced_count_matrix, dims=1);
ll, vv = eigen(perron_frobenius);
p = real.(vv[:, end] / sum(vv[:, end]));
println("The entropy is ", -sum(p .* log.(p) ./ log(length(p))))
λₜ = real.(1 / (log(ll[end-1]) / (time_jump * dt)) * X / 86400)
println("the slowest decaying statistical scale is ", λₜ, " days")

# ht = holding_times(current_state, length(markov_states); γ=dt);
# Q = transition_rate_matrix(current_state, length(markov_states); γ= time_jump * dt);
# Q̂ = Q[column_sum[:], column_sum[:]];
Q̂ = (perron_frobenius - I) ./ (time_jump * dt)
Λ, V = eigen(Q̂);
p = real.(V[:, end] ./ sum(V[:, end]));
V⁻¹ = inv(V);
=#
##
# markov = [markov_state[2][1]/markov_state[1][1] * markov_state[2][end]/markov_state[1][end] for markov_state in reduced_markov_states]
# time_series = [state_tuple[1+1]/state_tuple[1] * state_tuple[6+1]/state_tuple[6] for state_tuple in totes_timeseries]
#=
markov = [markov_state[3][1] for markov_state in markov_states];
# markov2 = [markov_state[3][1] for markov_state in markov_states];
time_series = [state_tuple[3] for state_tuple in totes_timeseries];

ensemble_mean = sum(p .* markov);
temporal_mean = mean(time_series);
ensemble_variance = sum(p .* markov .^ 2) - sum(p .* markov)^2;
temporal_variance = mean(time_series .^ 2) - mean(time_series)^2;
println("The ensemble mean is ", ensemble_mean)
println("The temporal mean is ", temporal_mean)
println("The mean markov state is ", mean(markov2))
println("The ens# emble variance is ", ensemble_variance)
println("The temporal variance is ", temporal_variance)
println("The variance of the markov state is ", var(markov2))
println("The absolute error between the ensemble and temporal means is ", abs(ensemble_mean - temporal_mean))
println("The relative error between the ensemble and temporal variances are ", 100 * abs(ensemble_variance - temporal_variance) / temporal_variance, " percent")

##
xs_m, ys_m = histogram2(markov, normalization=p, bins=20, custom_range=extrema(time_series))
xs_t, ys_t = histogram2(time_series, bins=20, custom_range=extrema(time_series))
xs_mu, ys_mu = histogram2(markov2, bins=20, custom_range=extrema(time_series))
fig = Figure()
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics")
ax2 = Axis(fig[1, 2]; title="Temporal Statistics")
ax3 = Axis(fig[1, 3]; title="Uniform Ensemble Statistics")
barplot!(ax1, xs_m, ys_m, color=:red)
barplot!(ax2, xs_t, ys_t, color=:blue)
barplot!(ax3, xs_mu, ys_mu, color=:purple)
for ax in [ax1, ax2, ax3]
    x_min = minimum([minimum(xs_m), minimum(xs_t)])
    x_max = maximum([maximum(xs_m), maximum(xs_t)])
    y_min = minimum([minimum(ys_m), minimum(ys_t)])
    y_max = maximum([maximum(ys_m), maximum(ys_t)])
    xlims!(ax, (x_min, x_max))
    ylims!(ax, (y_min, y_max))
end
display(fig)
=#