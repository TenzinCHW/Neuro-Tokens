import DrWatson
DrWatson.@quickactivate
import Plots, MEFK


function save_plot_raster(pattinds, loc, label)
    t = 1:length(pattinds) |> collect
    y = pattinds .|> log10
    Plots.scatter(t, y, label=label, xlabel="timestep", ylabel="log pattern first appearance", ms=1)
    Plots.savefig(loc)
end


function sortindsbyoccurrence(inds)
    ref = Dict()
    i = 1
    for ind in inds
        if !(ind in keys(ref))
            ref[ind] = i
            i += 1
        end
    end
    [ref[ind] for ind in inds]
end


function plot_raster_all(basedir, dset, ext)
    files = readdir(basedir)
    for fn in files
        params = DrWatson.parse_savename(fn)[2]
        binsz, winsz = params["binsz"], params["winsz"]
        f = DrWatson.savename(Dict("winsz"=>winsz, "binsz"=>binsz))
        data = DrWatson.wload(joinpath(basedir, fn))["1"]
        ininds, outinds = [data["ininds"], data["outinds"]] .|> sortindsbyoccurrence
        loc = DrWatson.plotsdir(dset, "raw", "$(f).$(ext)")
        save_plot_raster(ininds, loc, "raw")
        loc = DrWatson.plotsdir(dset, "model", "$(f).$(ext)")
        save_plot_raster(outinds, loc, "model")
    end
end


dset = ARGS[1]
#binsz = ARGS[2]
#winsz = ARGS[3]
ext = ARGS[end]
basedir = DrWatson.datadir("exp_pro", "matrix_hm", dset, "full", "complete")
plot_raster_all(basedir, dset, ext)
#f = DrWatson.savename(Dict("winsz"=>winsz, "binsz"=>binsz, "maxiter"=>100))
#fname = "$(f).jld2"
#basedir = DrWatson.datadir("exp_pro", "testnew", dset, "full")
#inpdata = DrWatson.wload(joinpath(basedir, "complete", fname))
#nulldata = DrWatson.wload(joinpath(basedir, "null", fname))
#data = inpdata["1"]
#ininds = data["ininds"] |> sortindsbyoccurrence
#outinds = data["outinds"] |> sortindsbyoccurrence
#loc = DrWatson.plotsdir("$(f)_raw.$ext")
#save_plot_raster(ininds, loc, f)
#loc = DrWatson.plotsdir("$(f)_model.$ext")
#save_plot_raster(outinds, loc, f)
