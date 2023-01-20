state .*= 1 / averaging_counter
gpu_components = components(test_state)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end
statenames = ("ρ", "ρu", "ρv", "ρw", "ρe")

filepath = "HeldSuarezDeep_" * "Nev" * string(Kv) * "_Neh" * string(Kh) * "_Nq" * string(Nq⃗[1]) * ".jld2"
file = jldopen(filepath, "a+")
JLD2.Group(file, "state")
JLD2.Group(file, "averagedstate")
JLD2.Group(file, "grid")
for (i, statename) in enumerate(statenames)
    file["state"][statename] = cpu_components[i]
end

gpu_components = components(state)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end

for (i, statename) in enumerate(statenames)
    file["averagedstate"][statename] = cpu_components[i]
end

file["grid"]["vertical_coordinate"] = vert_coord
file["grid"]["gauss_lobatto_points"] = Nq⃗
file["grid"]["vertical_element_number"] = Kv
file["grid"]["horizontal_element_number"] = Kh

close(file)