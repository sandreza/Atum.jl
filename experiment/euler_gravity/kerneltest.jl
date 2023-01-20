using BenchmarkTools

struct_state = held_suarez_init.(x⃗, Ref(hs_p))
old_struct_state = held_suarez_init.(x⃗, Ref(hs_p))
@benchmark CUDA.@sync begin
    dg_fs($state, $old_state, 0.0, increment=false)
end

@benchmark CUDA.@sync begin
    dg_fs($struct_state, $old_struct_state, 0.0, increment=false)
end

dg_fs2 = NewFluxArraySource(; law, grid, volume_form=vf, surface_numericalflux=sf);

@benchmark CUDA.@sync begin
    dg_fs2($struct_state, $old_struct_state, 0.0, increment=false)
end

pstate = parent(components(state)[1])
poldstate = parent(components(old_state)[1])
pstate .= poldstate
aux_state = fieldarray(undef, SVector{length(dg_fs.auxstate[1]),FT}, grid)
aux_state .= dg_fs2.auxstate
pauxstate = parent(components(aux_state)[1])

@benchmark CUDA.@sync begin
    dg_fs2($pstate, $poldstate, $pauxstate, increment=false)
end
dg_fs2(pstate, poldstate, pauxstate, increment=false)
##
@benchmark begin
    dg_fs2($struct_state, $old_struct_state, 0.0, increment=false)
end

@benchmark begin
    dg_fs2($pstate, $poldstate, $pauxstate, increment=false)
end

dg_fs2(pstate, poldstate, pauxstate, increment=false)