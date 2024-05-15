import DrWatson
DrWatson.@quickactivate
import Plots


function save_plot_ent(ent, loc, min_val, max_val, label)
    ent = sum(ent, dims=3) / size(ent)[3]
    ent = ent[1, :, 1]
    Plots.plot(1:length(ent) |> collect, ent, label=label, xlabel="window size", ylabel="entropy / nats")
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
save_plot_ent(raw_ent, loc, min_val, max_val, f)
loc = DrWatson.plotsdir("$(f)_model.$ext")
save_plot_ent(model_ent, loc, min_val, max_val, f)
loc = DrWatson.plotsdir("$(f)_null.$ext")
save_plot_ent(null_ent, loc, min_val, max_val, f)

