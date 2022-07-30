function spot_check(state)
    candidate_state = convert_gpu_to_cpu(state)
    reaction_coordinates = (candidate_state[1][1], candidate_state[2][1], candidate_state[3][1], candidate_state[4][1], candidate_state[5][1])
    reaction_coordinates = (reaction_coordinates..., candidate_state[1][end], candidate_state[2][end], candidate_state[3][end], candidate_state[4][end], candidate_state[5][end])
    return reaction_coordinates
end

totes_sim = 2000
totes_timeseries = []
state .= test_state
push!(timeseries, reaction_coordinate(state))
for i in 1:totes_sim
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = 25 * dt
    solve!(test_state, end_time, odesolver, adjust_final=false)
    push!(totes_timeseries, spot_check(test_state))
    if i % 10 == 0
        println("currently at timestep ", i, " out of ", totes_sim)
    end
end
##

# 1.21, 0.00048781
markov = [markov_state[3][1] for markov_state in markov_states[1:length(p)]]
timeseries = [state_tuple[3] for state_tuple in totes_timeseries]
ensemble_mean = sum(p .* markov)
temporal_mean = mean(timeseries)
ensemble_variance = sum(p .* markov .^ 2) - sum(p .* markov)^2
temporal_variance = mean(timeseries .^ 2) - mean(timeseries)^2
println("The ensemble mean is ", ensemble_mean)
println("The temporal mean is ", temporal_mean)
println("The mean markov state is ", mean(markov))
println("The ensemble variance is ", ensemble_variance)
println("The temporal variance is ", temporal_variance)
println("The variance of the markov state is ", var(markov))
println("The absolute error between the ensemble and temporal means is ", abs(ensemble_mean - temporal_mean))
println("The relative error between the ensemble and temporal variances are ", 100 * abs(ensemble_variance - temporal_variance) / temporal_variance, " percent")

##
xs_m, ys_m = histogram2(markov, normalization=p, bins=20, custom_range=extrema(timeseries))
xs_t, ys_t = histogram2(timeseries, bins=20, custom_range=extrema(timeseries))
fig = Figure()
ax1 = Axis(fig[1, 1]; title="Ensemble Statistics")
ax2 = Axis(fig[1, 2]; title="Temporal Statistics")
barplot!(ax1, xs_m, ys_m, color=:red)
barplot!(ax2, xs_t, ys_t, color=:blue)
for ax in [ax1, ax2]
    x_min = minimum([minimum(xs_m), minimum(xs_t)])
    x_max = maximum([maximum(xs_m), maximum(xs_t)])
    y_min = minimum([minimum(ys_m), minimum(ys_t)])
    y_max = maximum([maximum(ys_m), maximum(ys_t)])
    xlims!(ax, (x_min, x_max))
    ylims!(ax, (y_min, y_max))
end
display(fig)
