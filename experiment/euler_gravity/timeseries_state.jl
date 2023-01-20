function convert_gpu_to_cpu(state)
    return Array.(components(state))
end

markov_states = []
number_of_states = 400
for i in 1:number_of_states
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = 86400 / X * 60
    solve!(test_state, end_time, odesolver, adjust_final=false)
    candidate_state = convert_gpu_to_cpu(test_state)
    push!(markov_states, candidate_state)
    println("currently at timestep ", i, " out of ", number_of_states)
end