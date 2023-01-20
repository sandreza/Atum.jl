using ProgressBars
function distance(x, y, metric; normalization=(1.3, 60.0, 60.0, 60.0, 2.3e6))
    ρ_m, ρu_m, ρv_m, ρw_m, ρe_m = x
    ρ_m2, ρu_m2, ρv_m2, ρw_m2, ρe_m2 = y
    powa = 1 / 2
    error_ρ = sum(abs.(ρ_m - ρ_m2) .^ powa .* metric) / sum(metric) / normalization[1]^powa
    error_ρu = sum(abs.(ρu_m - ρu_m2) .^ powa .* metric) / sum(metric) / normalization[2]^powa
    error_ρv = sum(abs.(ρv_m - ρv_m2) .^ powa .* metric) / sum(metric) / normalization[3]^powa
    error_ρw = sum(abs.(ρw_m - ρw_m2) .^ powa .* metric) / sum(metric) / normalization[4]^powa
    error_ρe = sum(abs.(ρe_m - ρe_m2) .^ powa .* metric) / sum(metric) / normalization[5]^powa
    #=
    error_ρ = sum((sum((ρ_m - ρ_m2) .* metric, dims = 1) ./ sum(metric, dims = 1)  ) .^2 )/ normalization[1]^2
    error_ρu = sum((sum((ρu_m - ρu_m2) .* metric, dims = 1) ./ sum(metric, dims = 1) ) .^2) / normalization[2]^2
    error_ρv = sum(abs.(sum((ρv_m - ρv_m2) .* metric, dims = 1) ./ sum(metric, dims = 1)) .^2) / normalization[3]^2
    error_ρw = sum(abs.(sum((ρw_m - ρw_m2) .* metric, dims = 1) ./ sum(metric, dims = 1)) .^2) / normalization[4]^2
    error_ρe = sum(abs.(sum((ρe_m - ρe_m2) .* metric, dims = 1) ./ sum(metric, dims = 1)) .^2) / normalization[5]^2
    =#
    error_total = (error_ρ, error_ρu, error_ρv, error_ρw, error_ρe) .^ (1 / powa)
    return sum(error_total)
end

totes_sim = 10 * 12 * 100000 # 100 is about 30 days
current_state = Int64[]
save_radius = []

time_jump = 1
MJ = Array(dg_fs.MJ)
for i in ProgressBar(1:totes_sim)
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = time_jump * dt
    solve!(test_state, end_time, odesolver, adjust_final=false)
    candidate_state = convert_gpu_to_cpu(test_state)
    distances = [distance(markov_state, candidate_state, MJ) for markov_state in markov_states]
    # push!(save_radius, distances[1:10])
    push!(current_state, argmin(distances))
    if i % 10 == 0
        println("currently at timestep ", i, " out of ", totes_sim)
        println(length(union(current_state)), " states have been revisited")
        if length(distances) < 10
            println("current distances are ", distances)
        else
            println("current distances are ", distances[1:10])
        end
    end
end


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
##
# markov = [markov_state[2][1]/markov_state[1][1] * markov_state[2][end]/markov_state[1][end] for markov_state in reduced_markov_states]
# time_series = [state_tuple[1+1]/state_tuple[1] * state_tuple[6+1]/state_tuple[6] for state_tuple in totes_timeseries]

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
