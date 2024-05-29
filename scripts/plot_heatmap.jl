import DrWatson
DrWatson.@quickactivate
import Plots, Statistics


function save_plot_hm(hm, loc, xr, yr, binrange, winrange, min_val, max_val)
    hm = sum(hm, dims=3) / size(hm)[3]
    hm = hm[yr, xr, 1]
    Plots.heatmap(winrange, binrange, hm, xlabel="window size / number of timesteps",  ylabel="bin size / μs")#, clim=(min_val, max_val))
    Plots.savefig(loc)
end


function save_plot_hm_var(hm, loc, xr, yr, binrange, winrange, min_val, max_val)
    hm = Statistics.std(hm, dims=3)
    hm = hm[yr, xr, 1]
    Plots.heatmap(winrange, binrange, hm, xlabel="window size / number of timesteps",  ylabel="bin size / μs")#, clim=(min_val, max_val))
    Plots.savefig(loc)
end


fpath = ARGS[1]
data = DrWatson.wload(DrWatson.datadir(fpath))
ext = ARGS[2]
fn = split(fpath, "/")[end]
f = split(fn, ".")[1]
est_type = "MM correction"
winrange = 10:5:60
binrange = 500:500:5000
xr = 1:10
yr = 1:9
data = data[est_type]
raw_ent = data["raw"]
model_ent = data["complete"]
null_ent = data["null"]
min_val = min(min(raw_ent...), min(model_ent...), min(null_ent...))
max_val = max(max(raw_ent...), max(model_ent...), max(null_ent...))

loc = DrWatson.plotsdir("$(f)_raw_std.$ext")
save_plot_hm_var(raw_ent, loc, xr, yr, binrange, winrange, min_val, max_val)
loc = DrWatson.plotsdir("$(f)_model_std.$ext")
save_plot_hm_var(model_ent, loc, xr, yr, binrange, winrange, min_val, max_val)
loc = DrWatson.plotsdir("$(f)_null_std.$ext")
save_plot_hm_var(null_ent, loc, xr, yr, binrange, winrange, min_val, max_val)

#loc = DrWatson.plotsdir("$(f)_raw.$ext")
#save_plot_hm(raw_ent, loc, xr, yr, binrange, winrange, min_val, max_val)
#loc = DrWatson.plotsdir("$(f)_model.$ext")
#save_plot_hm(model_ent, loc, xr, yr, binrange, winrange, min_val, max_val)
#loc = DrWatson.plotsdir("$(f)_null.$ext")
#save_plot_hm(null_ent, loc, xr, yr, binrange, winrange, min_val, max_val)
