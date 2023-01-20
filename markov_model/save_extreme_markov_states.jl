filename = "markov_model_extreme_nstate_" * string(100) * ".h5"
fid = h5open(filename, "w")
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
fid["end index of extreme"] = 10
fid["extreme temperature cutoff"] = 290 
fid["extreme temperature gridpoint "] = [1,1]
fid["geopotential"] = geopot
close(fid)
