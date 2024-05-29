import DrWatson
DrWatson.@quickactivate
import Plots, MEFK


function save_plot_raster(pattinds, loc, savetype, winsz)
    t = 1:length(pattinds) |> collect
    if savetype == "log"
        y = pattinds .|> log10
        ylab = "log "
    elseif savetype == "linear"
        y = pattinds
        ylab = ""
    else
        error("savetype $(savetype) must be log or linear")
    end
    Plots.scatter(t, y, xlabel="timestep", ylabel="$(ylab)index pattern first appearance", title="L=$(winsz)", ms=1, legend=false)
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


function plot_raster_all(basedir, dset, savetype, ext)
    files = readdir(basedir)
    for fn in files
        params = DrWatson.parse_savename(fn)[2]
        binsz, winsz = params["binsz"], params["winsz"]
        f = DrWatson.savename(Dict("winsz"=>winsz, "binsz"=>binsz))
        data = DrWatson.wload(joinpath(basedir, fn))["1"]
        ininds, outinds = [data["ininds"], data["outinds"]] .|> sortindsbyoccurrence
        loc = DrWatson.plotsdir("pattern_index_raster", savetype, dset, "raw", "$(f).$(ext)")
        save_plot_raster(ininds, loc, savetype, winsz)
        loc = DrWatson.plotsdir("pattern_index_raster", savetype, dset, "model", "$(f).$(ext)")
        save_plot_raster(outinds, loc, savetype, winsz)
    end
end


dset = ARGS[1]
savetype = ARGS[2] # log or linear
ext = ARGS[end]
basedir = DrWatson.datadir("exp_pro", "matrix_hm", dset, "full", "complete")
plot_raster_all(basedir, dset, savetype, ext)

#dset = ARGS[1]
#binsz = ARGS[2]
#winsz = ARGS[3]
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
