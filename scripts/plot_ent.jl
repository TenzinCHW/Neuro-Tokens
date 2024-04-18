import DrWatson
DrWatson.@quickactivate
import Plots


function save_plot_hm(hm, loc, min_val, max_val)
    hm = sum(hm, dims=3) / size(hm)[3]
    hm = hm[:, :, 1]
    Plots.heatmap(10:5:50, 500:500:5000, hm, xlabel="window size / number of timesteps",  ylabel="bin size / ms", clim=(min_val, max_val))
    Plots.savefig(loc)
end


data = DrWatson.wload(DrWatson.datadir("entropy_winbin.jld2"))
raw_ent = data["raw_entropy"]
model_ent = data["model_entropy"]
min_val = min(min(raw_ent...), min(model_ent...))
max_val = max(max(raw_ent...), max(model_ent...))
loc = DrWatson.plotsdir("raw_entropy.pdf")
save_plot_hm(raw_ent, loc, min_val, max_val)
loc = DrWatson.plotsdir("model_entropy.pdf")
save_plot_hm(model_ent, loc, min_val, max_val)
