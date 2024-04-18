include("blanche_expt.jl")
import Plots


function calc_entropy(dist)
    prob = dist ./ sum(dist)
    sum(-prob .* log2.(prob))
end


function combine_counts(data, counts)
    patt_cnt = DataStructures.DefaultDict{Vector{UInt8}, Float32}(0)
    for i in 1:size(data)[1]
        patt_cnt[data[i, :]] += counts[i]
    end
    out = reduce(hcat, patt_cnt |> keys |> collect)' |> Array
    cnt = patt_cnt |> values |> collect
    out, cnt
end


function unique_ind(data)
    unique_patterns = unique(data, dims=1)
    patt2ids = Dict(unique_patterns[i, :]=>i for i in 1:size(unique_patterns)[1])
    inds = [patt2ids[data[i, :]] for i in 1:size(data)[1]]
    patt2ids, inds
end


unique_counts(arr) = [count(==(e), arr) for e in unique(arr)]


function raster_patterns(patterns)
    _, y = unique_ind(patterns)
    num_patterns = size(patterns)[1]
    x = [i for i in 1:num_patterns]
    x, y
end


function converge_dynamics(model, data)
    out_ = MEFK.dynamics(model, data)
    out = MEFK.dynamics(model, out_)
    while !all(out .== out_)
        out_ .= out
        out = MEFK.dynamics(model, out)
    end
    out
end


if abspath(PROGRAM_FILE) == @__FILE__
    numwin = 10
    numbin = 10
    winszs = [5i for i in 2:numwin]
    binszs = [500i for i in 1:numbin]
    maxiter = 100
    num_split = 10
    base_dir = ["exp_pro", "matrix", "split", "complete"]
    ents = zeros(length(binszs), length(winszs), num_split)
    raw_ents = zeros(size(ents))
    for (i, binsz) in enumerate(binszs)
        for (j, winsz) in enumerate(winszs)
            println("win $winsz bin $binsz")
            params = Dict("winsz"=>winsz, "binsz"=>binsz, "maxiter"=>maxiter)
            save_name = "$(DrWatson.savename(params)).jld2"
            data = DrWatson.wload(DrWatson.datadir(base_dir..., save_name))

            for k in 1:num_split
                input = data["$k"]["input"]
                output = data["$k"]["output"]

                in_patt, in_ids = unique_ind(input)
                out_patt, out_ids = unique_ind(output)
                in_count = unique_counts(in_ids)
                out_count = unique_counts(out_ids)
                raw_ents[i, j, k] = calc_entropy(in_count)
                println("raw entropy: $(raw_ents[i, j, k])")
                ents[i, j, k] = calc_entropy(out_count)
                println("converged entropy: $(ents[i, j, k])")

                #x, y = raster_patterns(out)
                #y = log10.(y)
                #p = Plots.scatter(x, y)
                #Plots.savefig(DrWatson.plotsdir("converged_patterns_$(winsz)_order2.png"))
            end
        end
    end
    DrWatson.wsave(DrWatson.datadir("entropy_winbin.jld2"), Dict("model_entropy"=>ents, "raw_entropy"=>raw_ents))
end

