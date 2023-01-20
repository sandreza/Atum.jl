fig = Figure()
ax = Axis(fig[1, 1])
colormap = :balance
slize_zonal = U̅
colorrange, contour_levels, s_string = plot_helper("u", slice_zonal)

hm = heatmap!(ax, ϕ, p_coord, slice_zonal, colorrange=colorrange, colormap=colormap, interpolate=true)
cplot = contour!(ax, ϕ, p_coord, slice_zonal, levels=contour_levels, color=:black)

ax.limits = (extrema(ϕ)..., extrema(p_coord)...)
ax.xlabel = "Latitude [ᵒ]"
ax.ylabel = "Stretched Height"
ax.xlabelsize = 25
ax.ylabelsize = 25
ax.xticks = ([-80, -60, -30, 0, 30, 60, 80], ["80S", "60S", "30S", "0", "30N", "60N", "80N"])
pressure_levels = [1000, 850, 700, 550, 400, 250, 100, 10]
ax.yticks = (pressure_levels .* 1e2, string.(pressure_levels))
ax.yreversed = true
add_labels = true
random_seed = 1234
heuristic = 1
# hack 
    Random.seed!(random_seed)
    if add_labels
        list_o_stuff = []
        labeled_contours = contour_levels[1:1:end]
        for level in labeled_contours
            local fig_t, ax_t, cp_t = contour(ϕ, p_coord, slice_zonal, levels = [-1e13, level], linewidth = 0)
            local segments = cp_t.plots[1][1][]
            local index_vals = []
            local beginnings = []
            for (i, p) in enumerate(segments)
                # the segments are separated by NaN, which signals that a new contour starts
                if isnan(p)
                    push!(beginnings, segments[i-1])
                    push!(index_vals, i)
                end
            end
            push!(list_o_stuff, (; segments, beginnings, index_vals))
        end

        for contour_index = 1:length(labeled_contours)

            local contour_val = labeled_contours[contour_index]
            local segments = list_o_stuff[contour_index].segments

            local indices = [0, list_o_stuff[contour_index].index_vals[1:end]...]
            for i = 1:length(indices)-1
                # heuristics for choosing where on line
                local index1 = rand(indices[i]+1:indices[i+1]-1) # choose random point on segment
                local index2 = round(Int, 0.5 * indices[i] + 0.5 * indices[i+1]) # choose point in middle
                β = (rand() - 0.5) * 0.9 + 0.5 # α ∈ [0,1]
                # α = (contour_index-1) / (length(labeled_contours)-1)
                α = contour_index % 2 == 0 ? 0.15 : 0.85
                α = rand([α, β])
                local index3 = round(Int, α * (indices[i] + 1) + (1 - α) * (indices[i+1] - 1)) # choose point in middle
                if heuristic == 3
                    local index = index3 # rand([index1, index2]) # choose between random point and point in middle
                elseif heuristic == 1
                    local index = index1
                elseif heuristic == 2
                    local index = index2
                end
                # end of heuristics
                local location = Point3(segments[index]..., 2.0f0)
                local sc = scatter!(ax, location, markersize = 20, align = (:center, :center), color = (:white, 0.1), strokecolor = :white)
                local anno = text!(ax, [("$contour_val", location)], align = (:center, :center), textsize = 20, color = :black)

                delete!(ax, sc)
                delete!(ax, cplot)
                delete!(ax, anno)

                push!(ax.scene, anno)
                push!(ax.scene, sc)
                push!(ax.scene, cplot)
            end
        end
    end # end of adding labels
    