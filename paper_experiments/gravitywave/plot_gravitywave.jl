include("gravitywave.jl")
include("../helpers.jl")

using FileIO
using JLD2: @load
using PyPlot
using PGFPlotsX
using LaTeXStrings

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

function compute_errors(dg, diag, diag_exact)
  w, δT = components(diag)
  w_exact, δT_exact = components(diag_exact)

  err_w = weightednorm(dg, w .- w_exact) / sqrt(sum(dg.MJ))
  err_T = weightednorm(dg, δT .- δT_exact) / sqrt(sum(dg.MJ))

  err_w, err_T
end
function convergence_plot(outputdir, convergence_data)
  dxs = convergence_data[3].dxs
  @pgf begin
    plotsetup = {
              xlabel = "Δx [km]",
              grid = "major",
              xmode = "log",
              ymode = "log",
              xticklabel="{\\pgfmathparse{exp(\\tick)/1000}\\pgfmathprintnumber[fixed,precision=3]{\\pgfmathresult}}",
              #xmax = 1,
              xtick = dxs,
              #ymin = 10. ^ -10 / 5,
              #ymax = 5,
              #ytick = 10. .^ -(0:2:10),
              legend_pos="south east",
              group_style= {group_size="2 by 2",
                            vertical_sep="2cm",
                            horizontal_sep="2cm"},
            }

    Ns = sort(collect(keys(convergence_data)))
    ylabel = L"L_{2}" * " error"
    fig = GroupPlot(plotsetup)
    for s in ('T', 'w')
        labels = []
        plots = []
        title = "$s convergence"
        for N in Ns
          @show N
          dxs = convergence_data[N].dxs
          if N == 2
            Tcoeff = 2e-4
            wcoeff = 3e-4
          elseif N == 3
            Tcoeff = 2e-5
            wcoeff = 3e-5
          elseif N == 4
            Tcoeff = 5e-6
            wcoeff = 8e-6
          end
          ordl = (dxs ./ dxs[end]) .^ (N + 1)
          if s === 'T'
            errs = convergence_data[N].T_errors
            ordl *= Tcoeff
          else
            errs = convergence_data[N].w_errors
            ordl *= wcoeff
          end
          coords = Coordinates(dxs, errs)
          ordc = Coordinates(dxs, ordl)
          plot = PlotInc({}, coords)
          plotc = Plot({dashed}, ordc)
          push!(plots, plot, plotc)
          #push!(labels, "N$N " * @sprintf("(%.2f)", rates[end]))
          push!(labels, "N$N")
          push!(labels, "order $(N+1)")
        end
        legend = Legend(labels)
        push!(fig, {title=title, ylabel=ylabel}, plots..., legend)
      end
      savepath = joinpath(outputdir,
                          "gw_convergence.pdf")
      pgfsave(savepath, fig)
  end
end

function contour_plot(root, diag_points, diag, diag_exact)
  x, z = components(diag_points)
  FT = eltype(x)
  w, δT = components(diag)
  w_exact, δT_exact = components(diag_exact)

  # convert coordiantes to km
  x ./= 1e3
  z ./= 1e3

  ioff()
  xticks = range(0, 300, length = 7)
  fig, ax = subplots(2, 1, figsize=(14, 14))

  for a in ax
    a.set_xlim([xticks[1], xticks[end]])
    a.set_xticks(xticks)
    a.set_xlabel(L"x" * " [km]")
    a.set_ylabel(L"z" * " [km]")
    a.set_aspect(15)
  end

  ΔT = FT(0.0001)
  scaling = 1e2 * _ΔT
  ll = 0.0036 * scaling
  sl = 0.0006 * scaling
  levels = vcat(-ll:sl:-sl, sl:sl:ll)
  ax[1].set_title("T perturbation [K]")
  norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
  cset = ax[1].contourf(x', z', δT', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
  ax[1].contour(x', z', δT_exact', levels=levels, colors=("k",))
  cbar = colorbar(cset, ax = ax[1])

  ax[2].set_title("w [m/s]")
  #norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
  cset = ax[2].contourf(x', z', w', cmap=ColorMap("PuOr"), levels=levels)
  ax[2].contour(x', z', w_exact', levels=levels, colors=("k",))
  cbar = colorbar(cset, ax = ax[2])

  tight_layout()
  savefig(joinpath(root, "gw_T_perturbation.pdf"))
  close(fig)

  k = findfirst(z[1, :] .>= 5)
  @show z[1, k]
  x_k = x[:, k]
  w_k = w[:, k]
  w_exact_k = w_exact[:, k]
  #linedata[(N, K...)] = (x_k, w_k, w_exact_k)
  @pgf begin
    fig = @pgf GroupPlot({group_style= {group_size="1 by 1", vertical_sep="1.5cm"},
                         xmin=0,
                         xmax= 300})
    #x, w, w_exact = linedata[(3, 40, 10)]

    ytick = [scaling * (-3 + i) * 1e-3 for i in 0:7]
    xtick = [50 * i for i in 0:6]
    p1 = Plot({color="blue"}, Coordinates(x_k, w_k))
    p2 = Plot({}, Coordinates(x_k, w_exact_k))
    push!(fig, {xlabel="x [km]",
                ylabel="w [m/s]",
                ytick = ytick,
                xtick = xtick,
                width="10cm",
                height="5cm"},
               p1, p2)
    #x, w, w_exact = linedata[(3, 40, 10)]
    #p1 = Plot({color="blue"}, Coordinates(x, w))
    #p2 = Plot({}, Coordinates(x, w_exact))
    #push!(fig, {xlabel="x [km]",
    #            ylabel="w [m/s]",
    #            width="10cm",
    #            height="5cm"},
    #           p1, p2)
    pgfsave(joinpath(root, "gw_line.pdf"), fig)
  end
end

function calculate_diagnostics(law, q, x⃗)
  ρ, ρu, ρw, ρe = q
  ρu⃗ = SVector(ρu, ρw)

  x, z = x⃗
  Φ = grav(law) * z

  R_d = gas_constant(law)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  T = p / (R_d * ρ)

  ρ_ref, p_ref = referencestate(law, x⃗)
  T_ref = p_ref / (R_d * ρ_ref)

  w = ρw / ρ
  δT = T - T_ref

  SVector(w, δT)
end

let
  outputdir = joinpath("paper_output", "gravitywave")

  convergence_data = Dict()
  for (root, dir, files) in walkdir(outputdir)
    jldfiles = filter(s->endswith(s, "jld2"), files)
    length(jldfiles) == 0 && continue
    @assert length(jldfiles) == 1
    data = load(joinpath(root, jldfiles[1]))

    law = data["law"]
    dg = data["dg"]
    grid = dg.grid
    cell = referencecell(grid)
    q = data["q"]
    qexact = data["qexact"]
    N = size(cell)[1] - 1
    KX = size(grid)[1]

    diag = calculate_diagnostics.(Ref(law), q, points(grid))
    diag_exact = calculate_diagnostics.(Ref(law), qexact, points(grid))

    dx = _L / size(grid)[1] / 1e3

    err_w, err_T = compute_errors(dg, diag, diag_exact)
    @show dx, err_w, err_T

    if N in keys(convergence_data)
      push!(convergence_data[N].w_errors, err_w)
      push!(convergence_data[N].T_errors, err_T)
      push!(convergence_data[N].dxs, dx)
    else
      convergence_data[N] = (dxs = Float64[], w_errors=Float64[], T_errors=Float64[])
      push!(convergence_data[N].w_errors, err_w)
      push!(convergence_data[N].T_errors, err_T)
      push!(convergence_data[N].dxs, dx)
    end

    diag_points, diag = interpolate_equidistant(diag, grid)
    _, diag_exact = interpolate_equidistant(diag_exact, grid)

    contour_plot(root, diag_points, diag, diag_exact)
  end

  # sort convergence_data
  for N in keys(convergence_data)
    tpl = convergence_data[N]
    tpl = sort(collect(zip(tpl...)))
    convergence_data[N] = (dxs=[t[1] for t in tpl],
                           w_errors = [t[2] for t in tpl],
                           T_errors = [t[3] for t in tpl])
  end
  convergence_plot(outputdir, convergence_data)
end