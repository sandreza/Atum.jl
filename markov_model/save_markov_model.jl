using HDF5
begin
# memo to self. Record initial condition and the final condition.
# 
filename = "markov_model_even_time_nstate_" * string(length(markov_states)) * ".h5"
fid = h5open(filename, "w")
fid["markov embedding 0p5"] = markov_chain_p2
fid["markov embedding 1"] = markov_chain_1
fid["markov embedding 2"] = markov_chain_2
fid["time jump "] = time_jump
fid["dt"] = dt
fid["small planet factor"] = X
markov_array = zeros(size(markov_states[1][1])..., 5)
for (i, markov_state) in enumerate(markov_states)
    for j in 1:5
        markov_array[:, :, j] .= markov_state[j]
    end
    fid["markov state " * string(i)] = markov_array
end
close(fid)
end