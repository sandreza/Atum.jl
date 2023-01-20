oldlist = components(mean_variables.(Ref(law), test_state, aux))
newlist = [A(zeros(size(xlist))) for i in 1:length(oldlist)]

for (newf, oldf) in zip(newlist, oldlist)
    interpolate_field!(newf, oldf, d_elist, d_ξlist, r, ω, Nq⃗, arch=CUDADevice())
end

# ρ, u, v, w, p, T
plist = mean(Array(newlist[end-1][:, :, :]), dims=(1,2))[1, 1, :]
##
fig2 = Figure(resolution=(1600, 800))
axlist = []
for (i, state_variable) in enumerate(newlist)
    push!(axlist, Axis(fig2[1, i]))
    field = mean(Array(state_variable[:, :, :]), dims=1)[1, :, :]
    heatmap!(axlist[i], ϕlist, plist, field, interpolate=true)
    println("extrema: ", extrema(field))
    axlist[i].yreversed = true
end

##
fig = Figure()
ax11 = Axis(fig[1, 1])
ax12 = Axis(fig[1, 2])
ax13 = Axis(fig[1, 3])
field1 = T_observable[1]
field2 = T_observable[end]
heatmap!(ax11, field1)
heatmap!(ax12, field2)
maxΔT = maximum(abs.(field1 - field2))
heatmap!(ax13, field1 - field2, colormap=:balance, colorrange=(-maxΔT, maxΔT))


##
ΔT = T_observable[1] - T_observable[2]
argmax(ΔT)
Tlist = [T_val[argmax(ΔT)] for T_val in T_observable]
scatter(Tlist)
μT = mean(Tlist)
##
function autocort(Tlist; N = 300)
    μT = mean(Tlist)
    return [mean(Tlist[1:end-i+1] .* Tlist[i:end]) for i in 1:N] .- μT^2
end
##autocorT=  [mean(Tlist[1:end-i+1] .* Tlist[i:end]) for i in 1:300] .- μT^2
##
fig = Figure()
ax11 = Axis(fig[1, 1])
sl_x = Slider(fig[2, 1], range=eachindex(T_observable), startvalue=1)
obsi = sl_x.value
field = @lift T_observable[$obsi]
heatmap!(ax11, field, colormap = :plasma, colorrange = extrema(T_observable[1]))