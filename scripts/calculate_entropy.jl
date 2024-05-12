import DrWatson
DrWatson.@quickactivate
import Plots


function naiveentropy(dist)
    prob = dist ./ sum(dist)
    sum(-prob .* log2.(prob))
end


function NSB(HL, HLm1, L)
    HL_ent = naiveentropy(HL)
    HLm1_ent = naiveentropy(HLm1)
    hmu = HL_ent - HLm1_ent
    HL_ent - hmu * L
end


function MMcorrectedentropy(dist, totnumpatts)
    naiveentropy(dist) + (length(dist) - 1) / (2 * totnumpatts)
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


function uniqueind(data)
    unique_patterns = unique(data, dims=1)
    patt2ids = Dict(unique_patterns[i, :]=>i for i in 1:size(unique_patterns)[1])
    inds = [patt2ids[data[i, :]] for i in 1:size(data)[1]]
    patt2ids, inds
end


"""Get ordered array of uniquedata to indices of uniquedata that make up origdata.
uniquedata might not be in same order as unique(origdata, dims=1)."""
function uniqueinds(uniquedata, origdata)
    inddict = Dict()
    for i in 1:size(uniquedata)[1]
        inddict[uniquedata[i, :]] = i
    end
    [inddict[origdata[i, :]] for i in 1:size(origdata)[1]]
end


uniquecounts(arr) = [count(==(e), arr) for e in unique(arr)]
# TODO figure out why this is so slow compared to the window function in extract_blanche.jl
function uniquecounts(data, dims::Int)
    patt_cnt = DataStructures.DefaultDict{Vector{UInt8}, Float32}(0)
    for i in 1:size(data)[1]
        patt_cnt[@view data[i, :]] += 1
    end
    unique_patt = reduce(hcat, patt_cnt |> keys |> collect)' |> Array
    unique_patt, values(patt_cnt) |> collect
end


function rasterpatterns(patterns)
    _, y = uniqueind(patterns)
    num_patterns = size(patterns)[1]
    x = [i for i in 1:num_patterns]
    x, y
end


function outputentropydict(subdirs, numbin, numwin, numsplit)
    d = Dict(subdir=>zeros(numbin, numwin, numsplit) for subdir in subdirs)
    d["raw"] = zeros(numbin, numwin, numsplit)
    d
end


if abspath(PROGRAM_FILE) == @__FILE__
    #winszs = [5i for i in 2:numwin]
    #binszs = [500i for i in 1:numbin]
    winszs = [i for i in 10:5:60]
    binszs = [10000]
    maxiter = 100
    numsplit = 1
    basedir = ["exp_pro", "matrix_old", "joost_long", "full"]
    subdirs = ["complete", "null"]
    entfuncs = ["MLE plugin", "MM correction"]
    numbin, numwin = length(binszs), length(winszs)
    entropies = Dict(name=>outputentropydict(subdirs, numbin, numwin, numsplit) for name in entfuncs)
    for subdir in subdirs
        for (i, binsz) in enumerate(binszs)
            for (j, winsz) in enumerate(winszs)
                println("win $winsz bin $binsz")
                params = Dict("winsz"=>winsz, "binsz"=>binsz, "maxiter"=>maxiter)
                savename = "$(DrWatson.savename(params)).jld2"
                data = DrWatson.wload(DrWatson.datadir(basedir..., subdir, savename))

                for k in 1:numsplit
                    input = data["$k"]["input"]
                    output = data["$k"]["output"]
                    incnt = data["$k"]["incnt"]
                    outcnt = data["$k"]["outcnt"]
                    if subdir == "complete"
                        entropies["MLE plugin"]["raw"][i, j, k] = naiveentropy(incnt)
                        entropies["MM correction"]["raw"][i, j, k] = MMcorrectedentropy(incnt, size(input)[1])
                    end
                    entropies["MLE plugin"][subdir][i, j, k] = naiveentropy(outcnt)
                    entropies["MM correction"][subdir][i, j, k] = MMcorrectedentropy(outcnt, size(output)[1])

                    #x, y = rasterpatterns(out)
                    #y = log10.(y)
                    #p = Plots.scatter(x, y)
                    #Plots.savefig(DrWatson.plotsdir("converged_patterns_$(winsz)_order2.png"))
                end
            end
        end
    end
    DrWatson.wsave(DrWatson.datadir(basedir..., "entropy_winbin.jld2"), entropies)
end

