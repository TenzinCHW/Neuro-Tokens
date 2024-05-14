import DrWatson
DrWatson.@quickactivate
import Plots


function save_plot_hm(hm, loc, min_val, max_val)
    hm = sum(hm, dims=3) / size(hm)[3]
    hm = hm[:, :, 1]
    Plots.heatmap(10:5:60, 500:500:6000, hm, xlabel="window size / number of timesteps",  ylabel="bin size / μs", clim=(min_val, max_val))
    Plots.savefig(loc)
end


fpath = ARGS[1]
data = DrWatson.wload(DrWatson.datadir(fpath))
ext = ARGS[2]
fn = split(fpath, "/")[end]
f = split(fn, ".")[1]
est_type = "MM correction"
data = data[est_type]
raw_ent = data["raw"]
model_ent = data["complete"]
null_ent = data["null"]
min_val = min(min(raw_ent...), min(model_ent...), min(null_ent...))
max_val = max(max(raw_ent...), max(model_ent...), max(null_ent...))
loc = DrWatson.plotsdir("$(f)_raw.$ext")
save_plot_hm(raw_ent, loc, min_val, max_val)
loc = DrWatson.plotsdir("$(f)_model.$ext")
save_plot_hm(model_ent, loc, min_val, max_val)
loc = DrWatson.plotsdir("$(f)_null.$ext")
save_plot_hm(null_ent, loc, min_val, max_val)
