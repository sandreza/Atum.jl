
gpu_components = components(test_state)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end
statenames = ("ρ", "ρu", "ρv", "ρw", "ρe")

filepath = "HeldSuarezDeepNewBadState_" * "Nev" * string(Kv) * "_Neh" * string(Kv) * "_Nq" * string(Nq⃗[1]) * ".jld2"
file = jldopen(filepath, "a+")
JLD2.Group(file, "state")
JLD2.Group(file, "averagedstate")
JLD2.Group(file, "stablestate")
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

gpu_components = components(stable_state)
for i in eachindex(gpu_components)
    cpu_components[i] .= Array(gpu_components[i])
end

for (i, statename) in enumerate(statenames)
    file["stablestate"][statename] = cpu_components[i]
end
close(file)
