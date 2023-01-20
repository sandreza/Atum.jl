function holding_times(markov_chain, number_of_states; γ=1)
    # dynamic container for holding times
    holding_times = [[] for n in 1:number_of_states]
    # dynamic container for states
    state_list = []

    push!(holding_times[markov_chain[1]], γ)
    push!(state_list, markov_chain[1])

    M = length(markov_chain)
    # loop through steps
    for i in 2:M
        current_state = markov_chain[i]
        previous_state = markov_chain[i-1]
        if current_state == previous_state
            holding_times[current_state][end] += γ
        else
            push!(state_list, current_state)
            push!(holding_times[current_state], γ)
        end
    end
    return holding_times
end

function transition_rate_matrix(markov_chain, number_of_states; γ=1)
    # dynamic container for holding times
    holding_times = [[] for n in 1:number_of_states]
    # dynamic container for states
    state_list = []

    push!(holding_times[markov_chain[1]], γ)
    push!(state_list, markov_chain[1])
 
    M = length(markov_chain)
    # loop through steps
    for i in 2:M
        current_state = markov_chain[i]
        previous_state = markov_chain[i-1]
        if current_state == previous_state
            holding_times[current_state][end] += γ
        else
            push!(state_list, current_state)
            push!(holding_times[current_state], γ)
        end
    end

    for i in 1:number_of_states
        if length(holding_times[i]) == 0
            holding_times[i] = [0]
        end
    end

    # construct transition matrix from state list 
    constructed_T = zeros(number_of_states, number_of_states)

    # first count how often state=i transitions to state=j 
    # the current state corresponds to a column of the matrix 
    # the next state corresponds to the row of the matrix
    number_of_unique_states = length(state_list)
    for i in 1:number_of_unique_states-1
        local current_state = state_list[i]
        local next_state = state_list[i+1]
        constructed_T[next_state, current_state] += 1
    end

    # now we need to normalize
    normalization = sum(constructed_T, dims=1)
    normalized_T = constructed_T ./ normalization

    # now we account for holding times 
    holding_scale = 1 ./ Statistics.mean.(holding_times)
    for i in 1:number_of_states
        normalized_T[i, i] = -1.0
        normalized_T[:, i] *= holding_scale[i]
    end
    return normalized_T
end


##
Λ, V = eigen(Q)
p = real.(V[:, end] ./ sum(V[:, end]))
V⁻¹ = inv(V)

totes = 100
auto_markov = zeros(totes)
println("the largest decaying mode is ", 1/real(Λ[end-1]) * X / 86400, " days")
##
auto_timeseries = zeros(totes)

for s in 0:totes-1
    @inbounds auto_timeseries[s+1] = mean(time_series[1:end-s] .* time_series[s+1:end]) 
end
auto_timeseries .= auto_timeseries  .- mean(time_series) .^ 2
##
times = Float64[]
for i in 0:1:totes-1
    τ = i * 25 * dt
    push!(times, τ * X / 86400)
    Pτ = real.(V * Diagonal(exp.(Λ .* τ)) * V⁻¹)
    val = markov
    aa = val' * Pτ * (p .* val)
    auto_markov[i+1] = aa
end
auto_markov .= auto_markov  .- sum(p .* markov) ^ 2
##
fig_auto = Figure()
ax1 = Axis(fig_auto[1, 1]; title="Ensemble Statistics")
l1 = lines!(ax1, times, auto_markov, color=:red)
l2 = lines!(ax1, times, auto_timeseries, color=:blue)
Legend(fig_auto[1, 2], [l1, l2], ["Markov", "Timeseries"])
display(fig_auto)