import DrWatson
DrWatson.@quickactivate
import MEFK, Distributions, Flux, CUDA, MAT
include("extract_blanche.jl")
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


function trainondata(data, maxiter, winsz, batchsize, arraycast)
    println("windowing")
    traindata, counts = window(data, winsz)
    bs, n = size(traindata)
    batchsize = min(bs, batchsize)
    println(size(traindata))
    # Prep dataloader
    println("prep loader")
    counts = reshape(counts, (length(counts), 1))
    loader = Flux.Data.DataLoader((traindata', counts'); batchsize=batchsize, partial=true)
    model = MEFK.MEF2T(n; array_cast=arraycast)
    optim = Flux.setup(Flux.Adam(0.01), model) # TODO refactor learning rate as parameter?
    losses = []
    for i in 1:maxiter
        loss = 0
        grads = [0, 0, 0]
        for (d, c) in loader
            d = d' |> Array
            l, _ = model(d, c[:])
            loss += l
        end
        println(loss)
        push!(losses, loss)
        # TODO set stopping criterion here
        grads = MEFK.retrieve_reset_gradients!(model; reset_grad=true)
        Flux.update!(optim, model, grads)
    end
    input = ordered_window(data, winsz)
    # converge on data
    output = MEFK.convergedynamics(model, input |> arraycast) |> Array
    cpumodel = modeltocpu(model)
    inspkcnt = sum(traindata, dims=2)
    uniqueout = MEFK.convergedynamics(model, traindata |> arraycast) |> Array
    combout, comboutcnt = combine_counts(uniqueout, counts)
    comboutspkcnt = sum(combout, dims=2)
    cpumodel, input, output, inspkcnt, counts, comboutspkcnt, comboutcnt, losses
end


function generatebernoulli(data)
    np = size(data)[1]
    p = sum(data, dims=1) ./ np
    b = [rand(Distributions.Bernoulli(i), np) for i in p]
    reduce(hcat, b) .|> UInt8
end


function runexperiment(binsz, winszs, maxiter, path, batchsize, arraycast, params, basedir, numsplit, extract_fn)
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
                model, input, output, inspkcnt, counts, comboutspkcnt, comboutcnt, losses =
                    trainondata(data_split[i], maxiter, winsz, batchsize, arraycast)
                savedata["$i"] = Dict("input"=>input, "output"=>output, "net"=>model, "inspikecnt"=>inspkcnt, "incnt"=>counts, "outspikecnt"=>comboutspkcnt, "outcnt"=>comboutcnt, "loss"=>losses)
                MAT.matwrite(joinpath(cdmdir, "input", "$(DrWatson.savename(params))_$(i).mat"),
                             Dict("inspikecnt"=>inspkcnt, "incnt"=>counts))
                MAT.matwrite(joinpath(cdmdir, "complete", "$(DrWatson.savename(params))_$(i).mat"),
                             Dict("cells"=>model.n, "spike_counts"=>comboutspkcnt, "counts"=>comboutcnt))
            end
            # train null model, calculate means of each location and generate Bernoulli for dataset
            if !isfile(nullloc)
                println("processing $nullloc")
                nulldata = generatebernoulli(data_split[i])
                model, input, output, inspkcnt, counts, comboutspkcnt, comboutcnt, losses =
                    trainondata(nulldata, maxiter, winsz, batchsize, arraycast)
                savenull["$i"] = Dict("input"=>input, "output"=>output, "net"=>model, "inspikecnt"=>inspkcnt, "incnt"=>counts, "outspikecnt"=>comboutspkcnt, "outcnt"=>comboutcnt, "loss"=>losses)
                MAT.matwrite(joinpath(cdmdir, "null", "$(DrWatson.savename(params))_$(i).mat"),
                             Dict("cells"=>model.n, "spike_counts"=>comboutspkcnt, "counts"=>comboutcnt))
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
    st = 36
    en = 45
    winszs = [i for i in st:en]
    winszs = vcat(winszs[1:5], winszs[end-4:end])
    println(winszs)
    binsz = params["binsz"]

    # For Blanche's data
    #path = DrWatson.datadir("exp_raw", "pvc3", "crcns_pvc3_cat_recordings", "spont_activity", "spike_data_area18")
    #extract_fn = extract_bin_spikes_blanche
    # For Joost's data
    path = DrWatson.datadir("exp_raw", "joost_data", "Long_recordings-stability_MaxEnt_and_CFP", "long_1_spontaneous_activity.jld2")
    extract_fn = extract_bin_spikes_joost


    #basedir = DrWatson.datadir("exp_pro", "matrix", "blanche", "full")
    basedir = DrWatson.datadir("exp_pro", "matrix", "joost_long", "full")
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

    runexperiment(binsz, winszs, maxiter, path, batchsize, arraycast, params, basedir, numsplit, extract_fn)
end

