import DrWatson
DrWatson.@quickactivate
import MEFK, Distributions, Flux, CUDA, MAT
import ProgressBars: ProgressBar, set_multiline_postfix
import Printf
include("extract_joost.jl")
include("calculate_entropy.jl")


function modeltocpu(model)
    if isa(model, MEFK.MEF3T)
        cpumodel = MEFK.MEF3T(model, Array)
    elseif isa(model, MEFK.MEF2T)
        cpumodel = MEFK.MEF2T(model, Array)
    else
        cpumodel = MEFK.MEFMPNK(model, Array)
    end
    cpumodel
end


function convergeloader(model, loader)
    output = []
    for (batch, _) in loader
        out = MEFK.convergedynamics(model, batch' |> Array) |> Array
        push!(output, out)
    end
    reduce(vcat, output)
end


print2dp(val) = Printf.@sprintf("%.2f", val)


function trainondata(data, maxiter, winsz, batchsize, arraycast, lr, checkevery=100; model=nothing)
    println("windowing")
    traindata, counts = window(data, winsz)
    bs, n = size(traindata)
    batchsize = min(bs, batchsize)
    println(size(traindata))
    # Prep dataloader
    println("prep loader")
    counts = reshape(counts, (length(counts), 1))
    loader = Flux.Data.DataLoader((traindata', counts'); batchsize=batchsize, partial=true)
    if isnothing(model)
        model = MEFK.MEF2T(n; array_cast=arraycast)
    end
    optim = Flux.setup(Flux.Adam(lr), model)
    losses = []
    entropies = Dict()
    uniqueout = MEFK.convergedynamics(model, traindata |> arraycast) |> Array
    pb = ProgressBar(1:maxiter)
    ent = 0
    for i in pb
        loss = 0
        for (d, c) in loader
            d = d' |> Array
            l, grads = model(d, c[:], reset_grad=true)
            loss += l
            Flux.update!(optim, model, grads)
        end
        set_multiline_postfix(pb, "Loss: $(print2dp(loss))\nEntropy: $(ent)")
        push!(losses, loss)
        if i % checkevery == 0
            checkout = convergeloader(model, loader)
            _, cnt = combine_counts(checkout, counts)
            ent = naiveentropy(cnt)
            entropies[i] = ent
            set_multiline_postfix(pb, "Loss: $(print2dp(loss))\nEntropy: $(ent)")
            if all(checkout .== uniqueout)
                break
            else
                uniqueout = checkout
            end
        end
    end
    input = ordered_window(data, winsz)
    # converge on data
    println(size(input))
    inputloader = Flux.Data.DataLoader((input', ones(1, size(input)[1])); batchsize=batchsize, partial=true)
    output = convergeloader(model, inputloader)
    # instead of returning input and output, return index of traindata and uniqueout
    uniqueout = MEFK.convergedynamics(model, traindata |> arraycast) |> Array
    ininds = uniqueinds(traindata, input)
    combout, comboutcnt = combine_counts(uniqueout, counts)
    outinds = uniqueinds(combout, output)
    cpumodel = modeltocpu(model)
    inspkcnt = sum(traindata, dims=2)
    comboutspkcnt = sum(combout, dims=2)
    cpumodel, ininds, traindata, inspkcnt, counts, outinds, combout, comboutspkcnt, comboutcnt, losses, entropies
end


function generatebernoulli(data)
    np = size(data)[1]
    p = sum(data, dims=1) ./ np
    b = [rand(Distributions.Bernoulli(i), np) for i in p]
    reduce(hcat, b) .|> UInt8
end


function runexperiment(binsz, winszs, maxiter, path, batchsize, arraycast, params, basedir, numsplit, extract_fn, lr)
    for winsz in winszs
        params["winsz"] = winsz
        # Only doing matrix for now
        saveloc = joinpath(basedir, "complete", "$(DrWatson.savename(params)).jld2")
        nullloc = joinpath(basedir, "null", "$(DrWatson.savename(params)).jld2")
        cdmdir = joinpath(basedir, "cdmentropy")

        data = extract_fn(path, binsz)
        data = trim_recording(data, 0.15)#094)
        println(size(data))
        data_split = recordingsplittrain(data, numsplit)
        println("splits $numsplit total $(size(data))")

        savedata = Dict()
        savenull = Dict()
        for i in 1:numsplit
            if !isfile(saveloc)
                println("processing $saveloc")
                model, ininds, uniqueinput, inspkcnt, counts, outinds, combout, comboutspkcnt, comboutcnt, losses, entropies =
                    trainondata(data_split[i], maxiter, winsz, batchsize, arraycast, lr)
                savedata["$i"] = Dict("ininds"=>ininds, "outinds"=>outinds, "input"=>uniqueinput, "output"=>combout, "net"=>model, "inspikecnt"=>inspkcnt, "incount"=>counts, "outspikecnt"=>comboutspkcnt, "outcount"=>comboutcnt, "loss"=>losses, "entropy"=>entropies)
                #MAT.matwrite(joinpath(cdmdir, "input", "$(DrWatson.savename(params))_$(i).mat"),
                #             Dict("inspikecnt"=>inspkcnt, "counts"=>counts))
                #MAT.matwrite(joinpath(cdmdir, "complete", "$(DrWatson.savename(params))_$(i).mat"),
                #             Dict("cells"=>model.n, "spike_counts"=>comboutspkcnt, "counts"=>comboutcnt))
            end
            # train null model, calculate means of each location and generate Bernoulli for dataset
            if !isfile(nullloc)
                println("processing $nullloc")
                nulldata = generatebernoulli(data_split[i])
                model, ininds, uniqueinput, inspkcnt, counts, outinds, combout, comboutspkcnt, comboutcnt, losses, entropies =
                    trainondata(nulldata, maxiter, winsz, batchsize, arraycast, lr)
                savenull["$i"] = Dict("ininds"=>ininds, "outinds"=>outinds, "input"=>uniqueinput, "output"=>combout, "net"=>model, "inspikecnt"=>inspkcnt, "incount"=>counts, "outspikecnt"=>comboutspkcnt, "outcount"=>comboutcnt, "loss"=>losses, "entropy"=>entropies)
                #MAT.matwrite(joinpath(cdmdir, "null", "$(DrWatson.savename(params))_$(i).mat"),
                #             Dict("cells"=>model.n, "spike_counts"=>comboutspkcnt, "counts"=>comboutcnt))
            end
        end

        if !isfile(saveloc)
            DrWatson.wsave(saveloc, savedata)
        end
        if !isfile(nullloc)
            DrWatson.wsave(nullloc, savenull)
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    params = Dict("binsz"=>parse(Int, ARGS[1]), "maxiter"=>parse(Int, ARGS[2]))
    st = 1
    en = 1
    winszs = [i for i in st:en]
    #winszs = vcat(winszs[1:5], winszs[end-4:end])
    println(winszs)
    binsz = params["binsz"]

    # For Blanche's data
    #path = DrWatson.datadir("exp_raw", "pvc3", "crcns_pvc3_cat_recordings", "spont_activity", "spike_data_area18")
    #extract_fn = extract_bin_spikes_blanche
    #basedir = DrWatson.datadir("exp_pro", "matrix", "blanche", "full")
    #lr = 0.1
    # For Joost's data
    path = DrWatson.datadir("exp_raw", "joost_data", "Long_recordings-stability_MaxEnt_and_CFP", "long_1_spontaneous_activity.jld2")
    extract_fn = extract_bin_spikes_joost
    basedir = DrWatson.datadir("exp_pro", "matrix", "joost_long", "full")
    lr = 0.01


    maxiter = params["maxiter"]
    numsplit = parse(Int, ARGS[3])

    dev = parse(Int, ARGS[4])
    CUDA.device!(dev)
    arraycast = CUDA.cu
    batchsize = parse(Int, ARGS[5])

    #k = 2
    #params["k"] = k
    #model = MEFK.MEFMPNK(n, k; array_cast=arraycast)
    #model = MEFK.MEF3T(n; array_cast=arraycast)
    println("starting")

    runexperiment(binsz, winszs, maxiter, path, batchsize, arraycast, params, basedir, numsplit, extract_fn, lr)
end

