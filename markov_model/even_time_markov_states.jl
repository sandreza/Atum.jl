Tday = 25 * dt * 7 # is 1 day

total_markov_states = 400
totes_sim = 10 * 12 * 100 # 100 is about 30 days
markov_states = []
current_state = Int64[]
markov_chain = Int64[]
save_radius = []
push!(markov_states, convert_gpu_to_cpu(test_state))
push!(markov_chain, 1)

for i in ProgressBar(2:total_markov_states )
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = Tday * 15 # 30 days is safe
    solve!(test_state, end_time, odesolver, adjust_final=false)
    candidate_state = convert_gpu_to_cpu(test_state)
    push!(markov_states, candidate_state)
    markov_index = i
    push!(markov_chain, markov_index)
end