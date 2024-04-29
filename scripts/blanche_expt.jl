import DrWatson
DrWatson.@quickactivate
import MEFK, Distributions, Flux, CUDA
#include("extract_blanche.jl")
include("extract_joost.jl")


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


function trainondata(data, max_iter, winsz, batchsize, array_cast)
    println("windowing")
    train_data, counts = window(data, winsz)
    n = size(train_data)[2]
    println(size(train_data))
    # Prep dataloader
    println("prep loader")
    counts = reshape(counts, (length(counts), 1))
    loader = Flux.Data.DataLoader((train_data', counts'); batchsize=batchsize, partial=true)
    model = MEFK.MEF2T(n; array_cast=array_cast)
    optim = Flux.setup(Flux.Adam(0.01), model) # TODO refactor learning rate as parameter?
    losses = []
    for i in 1:max_iter
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
    output = convergedynamics(model, input |> array_cast) |> Array
    cpumodel = modeltocpu(model)
    cpumodel, input, output, losses
end


function recordingsplittrain(data, num_split::Int)
    split_sz = size(data)[1] / num_split
    [data[ceilint(i*split_sz)+1:ceilint((i+1)*split_sz), :] for i in 0:num_split-1]
end


function convergedynamics(model, data)
    out_ = MEFK.dynamics(model, data)
    out = MEFK.dynamics(model, out_)
    while !all(out .== out_)
        out_ .= out
        out = MEFK.dynamics(model, out)
    end
    out
end


function generatebernoulli(data)
    np = size(data)[1]
    p = sum(data, dims=1) ./ np
    b = [rand(Distributions.Bernoulli(i), np) for i in p]
    reduce(hcat, b) .|> UInt8
end


function runexperiment(binszs, winsz, max_iter, path, batchsize, array_cast, params, basedir, numsplit)
    #for winsz in winszs
        #params["winsz"] = winsz
    for binsz in binszs
        params["binsz"] = binsz
        # Only doing matrix for now
        save_loc = joinpath(basedir, "complete", "$(DrWatson.savename(params)).jld2")
        null_loc = joinpath(basedir, "null", "$(DrWatson.savename(params)).jld2")

        data = extract_bin_spikes_joost(path, binsz)
        data = trim_recording(data, 0.15)#094)
        println(size(data))
        data_split = recordingsplittrain(data, numsplit)
        println("splits $numsplit total $(size(data))")

        save_data = Dict()
        save_null = Dict()
        for i in 1:numsplit
            if !isfile(save_loc)
                println("processing $save_loc")
                model, input, output, losses = trainondata(data_split[i], max_iter, winsz, batchsize, array_cast)
                save_data["$i"] = Dict("input"=>input, "output"=>output, "net"=>model, "loss"=>losses)
            end
            # train null model, calculate means of each location and generate Bernoulli for dataset
            if !isfile(null_loc)
                println("processing $null_loc")
                null_data = generatebernoulli(data_split[i])
                model, input, output, lossess = trainondata(null_data, max_iter, winsz, batchsize, array_cast)
                save_null["$i"] = Dict("input"=>input, "output"=>output, "net"=>model, "loss"=>losses)
            end
        end

        if !isfile(save_loc)
            DrWatson.wsave(save_loc, save_data)
        end
        if !isfile(null_loc)
            DrWatson.wsave(null_loc, save_null)
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    #params = Dict("binsz"=>parse(Int, ARGS[1]), "maxiter"=>parse(Int, ARGS[2]))
    #winszs = [i for i in 10:5:60]
    #binsz = params["binsz"]

    # For Blanche's data
    #dir = DrWatson.datadir("exp_raw", "pvc3", "crcns_pvc3_cat_recordings", "spont_activity", "spike_data_area18")
    # For Joost's data
    dir = DrWatson.datadir("exp_raw", "joost_data", "Long_recordings-stability_MaxEnt_and_CFP", "long_1_spontaneous_activity.jld2")
    params = Dict("winsz"=>parse(Int, ARGS[1]), "maxiter"=>parse(Int, ARGS[2]))
    winsz = params["winsz"]
    binszs = [10000]
    basedir = DrWatson.datadir("exp_pro", "matrix", "joost_long", "full")
    max_iter = params["maxiter"]
    numsplit = parse(Int, ARGS[3])

    dev = parse(Int, ARGS[4])
    CUDA.device!(dev)
    array_cast = CUDA.cu
    batchsize = parse(Int, ARGS[5])

    #k = 2
    #params["k"] = k
    #model = MEFK.MEFMPNK(n, k; array_cast=array_cast)
    #model = MEFK.MEF3T(n; array_cast=array_cast)
    println("starting")

    # TODO do binszs and winsz instead
    runexperiment(binszs, winsz, max_iter, path, batchsize, array_cast, params, basedir, numsplit)
end

