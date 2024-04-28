import DrWatson
DrWatson.@quickactivate
import MEFK, UnicodePlots


function restack(line::Vector, length::Int, width::Int)
    reshape(line, (length, width))
end


function restack(output::Matrix, winsz::Int)
    outlength, outwidth = size(output)
    num_neuron = outwidth / winsz |> Int
    inlength = outlength + winsz - 1
    res = zeros(Int8, inlength, num_neuron, winsz)
    for i in 1:outlength
        rs = restack(output[i, :], winsz, num_neuron)
        for j in 1:winsz
            res[i+j-1, :, j] = rs[j, :]  # place signal along diagonal
        end
    end
    res[winsz:end-winsz+1, :, :]  # truncate first winsz-1 and last winsz-1 since they won't have winsz outputs
end


function restack_splits(save_dir::String, fname::String)
    fpath = joinpath(save_dir, fname)
    data = DrWatson.wload(fpath)
    params = DrWatson.parse_savename(fname)[2]
    winsz = params["winsz"]
    trial_outputs = Dict()
    for i in keys(data)
        trial = data[i]
        trial_outputs[i] = restack(trial["output"], winsz)
    end
    trial_outputs
end


function frac_most_pop(trial_output)
    winsz = size(trial_output)[end]
    num_ts = size(trial_output)[1]
    popular_patt = []
    counts = Int[]
    for i in 1:num_ts
        ts_out = @view trial_output[i, :, :]
        ts_patterns = unique(ts_out, dims=2)
        ts_patterns = Dict(ts_patterns[:, j]=>0 for j in 1:size(ts_patterns)[2])
        for j in 1:winsz
            for (patt, _) in ts_patterns
                if patt == @view ts_out[:, j]
                    ts_patterns[patt] += 1
                    break
                end
            end
        end
        cnt, pop_patt = findmax(ts_patterns)
        push!(popular_patt, pop_patt)
        push!(counts, cnt)
    end
    popular_patt = reduce(hcat, popular_patt)' |> Array
    probs = counts ./ winsz
    popular_patt, probs
end


if abspath(PROGRAM_FILE) == @__FILE__
    save_dir = DrWatson.datadir("exp_pro", "matrix", "split", "complete")#"../data/exp_pro/matrix/split/complete/"
    for f in readdir(save_dir)[1:10]
        params = DrWatson.parse_savename(f)[2]
        if params["binsz"] != 1000
            continue
        end
        println(f)
        trial_outputs = restack_splits(save_dir, f)
        # TODO analyze output here, not enough memory to store all results
        patt, probs = frac_most_pop(trial_outputs["1"])
        unique_ts_patt = unique(patt, dims=1) 
        println(size(unique_ts_patt))
        plot = UnicodePlots.histogram(probs, nbins=10)
        println(plot)
    end
end

