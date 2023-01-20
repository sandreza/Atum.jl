using ProgressBars, HDF5
save_it = true
load_it = true 
if load_it
    filename = "markov_model_even_time_nstate_" * string(100) * "_extreme.h5"
    fid = h5open(filename, "r")
    markov_array = read(fid["initial condition"])
    close(fid)
    state_components = components(test_state)
    for i in 1:5
        state_components[i] .= CuArray(markov_array[:, :, i])
    end
    filename =  "markov_model_extreme_nstate_" * string(100) * ".h5"
    fid = h5open(filename, "r")
    markov_states = Vector{Matrix{Float64}}[]
    for i in 1:100
        markov_array = read(fid["markov state " * string(i)])
        push!(markov_states, [markov_array[:,:, i] for i in 1:5])
    end
    close(fid)
end
if save_it
    println("saving the file")
    filename = "p2_markov_model_even_time_nstate_" * string(length(markov_states)) * "_extreme.h5"
    fid = h5open(filename, "w")
    markov_array = zeros(size(markov_states[1][1])..., 5)
    save_ic = convert_gpu_to_cpu(test_state)
    for j in 1:5
        markov_array[:, :, j] .= save_ic[j]
    end
    fid["initial condition"] = markov_array
end

function distance(x, y, metric; normalization=(1.3, 60.0, 60.0, 60.0, 2.3e6))
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

function isextreme(state, geopot)
    temperature_from_markov(state, geopot)[1,1] > 290 
end

function assign_index(markov_states, candidate_state, MJ, geopot)
    m_distances = zeros(length(markov_states))
    if isextreme(candidate_state, geopot)
        
        Threads.@threads for j in 1:10
            m_distances[j] = distance(markov_states[j], candidate_state, MJ)
        end
        return argmin(m_distances[1:10])
        
        # return argmin([distance(markov_states[j], candidate_state, MJ) for j in 1:10])
    else
        
        Threads.@threads for j in 11:100
            m_distances[j] = distance(markov_states[j], candidate_state, MJ)
        end
        return argmin(m_distances[11:100]) + 10
        
        # return argmin([distance(markov_states[j], candidate_state, MJ) for j in 11:100]) + 10
    end
end

# checkit = [assign_index(markov_states, markov_states[i], A_MJ, geopot) for i in 1:100]
geopot = Array(components(aux)[end])
function temperature_from_markov(markov_state, geopot)
    ρ = markov_state[1]
    ρu = markov_state[2]
    ρv = markov_state[3]
    ρw = markov_state[4]
    ρe = markov_state[5]
    γ = 1.4
    R_d = 287
    p = (γ - 1) * (ρe .- 0.5 * (ρu .* ρu .+ ρv .* ρv .+ ρw .* ρw) ./ ρ .- ρ .* geopot)
    return p ./ (ρ * R_d)
end

totes_sim = 10 * 12 * 100 * 10 * 10 # 100 is about 30 days for timejump = 1

markov_chain = Int64[]

time_jump = 5 # 5 is a good compromise between speed and accuracy
MJ = Array(dg_explicit.MJ)
observable_list = []
for i in ProgressBar(1:totes_sim)
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = time_jump * dt
    solve!(test_state, end_time, odesolver, adjust_final=false)
    candidate_state = convert_gpu_to_cpu(test_state)
    # markov embedding
    push!(markov_chain, assign_index(markov_states, candidate_state, MJ, geopot))
    ##
    oldlist = components(mean_variables.(Ref(law), test_state, aux))
    push!(observable_list, [oldlist[i][1, 1] for i in eachindex(oldlist)])
end

# memo to self. Record initial condition and the final condition. [DONE]
if save_it
    println("saving the rest")
    save_fc = convert_gpu_to_cpu(test_state)
    for j in 1:5
        markov_array[:, :, j] .= save_fc[j]
    end
    fid["final condition"] = markov_array
    fid["markov embedding"] = markov_chain
    fid["time jump "] = time_jump
    fid["dt"] = dt
    fid["small planet factor"] = X
    close(fid)
end

