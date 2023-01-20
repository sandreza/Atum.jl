using HDF5
filename = "even_high_rez_hs.h5"
fid = h5open(filename, "w")
create_group(fid, "T")
create_group(fid, "rho")
create_group(fid, "u")
create_group(fid, "v")
create_group(fid, "grid")
fid["grid"]["θlist"] = collect(θlist)
fid["grid"]["ϕlist"] = collect(ϕlist)
fid["grid"]["rlist"] = collect(rlist)
tic = time()
for i in eachindex(T_timeseries)
    fid["T"][string(i)] = T_timeseries[i]
    fid["rho"][string(i)] = rho_timeseries[i]
    fid["u"][string(i)] = u_timeseries[i]
    fid["v"][string(i)] = v_timeseries[i]
    toc = time()
    if toc - tic > 1
        println("currently at timestep ", i, " out of ", length(T_timeseries))
        tic = toc
    end
end
close(fid)