include("sphere_statistics_functions.jl")
println("saving statistics")
last_step_mvar = mean_variables.(Ref(law), test_state, aux)

# just to initialize for saving
cpu_aux = sphere_auxiliary.(Ref(law), Ref(hs_p), cpu_x⃗, cpu_state)
cpu_mvar = mean_variables.(Ref(law), cpu_state, cpu_aux)
cpu_svar = second_moment_variables.(cpu_mvar)

fmnames = ("ρ", "u", "v", "w", "p", "T")
smnames = ("uu", "vv", "ww", "uv", "uw", "vw", "uT", "vT", "wT", "ρρ", "pp")

filepath = "HeldSuarezStatistics_" * "Nev" * string(Kv) * "_Neh" * string(Kh) * "_Nq" * string(Nq⃗[1]) * ".jld2"
file = jldopen(filepath, "a+")
JLD2.Group(file, "instantaneous")
JLD2.Group(file, "firstmoment")
JLD2.Group(file, "secondmoment")
JLD2.Group(file, "grid")

# first moment
gpu_components = components(fmvar)
cpu_components = components(cpu_mvar)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end
for (i, statename) in enumerate(fmnames)
    file["firstmoment"][statename] = cpu_components[i]
end

# instantaneous
gpu_components = components(last_step_mvar)
cpu_components = components(cpu_mvar)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end
for (i, statename) in enumerate(fmnames)
    file["instantaneous"][statename] = cpu_components[i]
end

# Second moment
gpu_components = components(smvar)
cpu_components = components(cpu_svar)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end
for (i, statename) in enumerate(smnames)
    file["secondmoment"][statename] = cpu_components[i]
end

file["grid"]["vertical_coordinate"] = vert_coord
file["grid"]["gauss_lobatto_points"] = Nq⃗
file["grid"]["vertical_element_number"] = Kv
file["grid"]["horizontal_element_number"] = Kh
file["parameters"] = hs_p

close(file)
println("done")