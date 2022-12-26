time_jump = 1
MJ = Array(dg_fs.MJ)
for i in ProgressBar(1:totes_sim)
    aux = sphere_auxiliary.(Ref(law), Ref(hs_p), x⃗, state)
    dg_explicit.auxstate .= aux
    odesolver = LSRK144(dg_explicit, test_state, dt)
    end_time = time_jump * dt
    solve!(test_state, end_time, odesolver, adjust_final=false)
    candidate_state = convert_gpu_to_cpu(test_state)
    # distances = [distance(markov_state, candidate_state, MJ) for markov_state in markov_states]
    # push!(current_state, argmin(distances))
    # push!(save_radius, distances[1:10])

    push!(current_state, assign_index(markov_states, candidate_state, MJ))

    
    if i % 100 == 0
        println("currently at timestep ", i, " out of ", totes_sim)
        println(length(union(current_state)), " states have been revisited")
        #=
        if length(distances) < 10
            println("current distances are ", distances)
        else
            println("current distances are ", distances[1:10])
        end
        =#
    end
    
end