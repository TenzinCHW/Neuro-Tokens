include("train_network.jl")


function continueexperiment(binsz, winszs, startiter, maxiter, path, batchsize, arraycast, params, basedir, numsplit, extract_fn, lr)
    # TODO same setup as runexperiment but load model from start point
    for winsz in winszs
        params["winsz"] = winsz
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

        loadparams = Dict(k=>v for (k, v) in params)
        loadparams["maxiter"] = startiter

        for i in 1:numsplit
            if !isfile(saveloc)
                println("processing $saveloc")
                loadmodel = DrWatson.wload(joinpath(basedir, "complete", "$(DrWatson.savename(loadparams)).jld2"))["1"]["net"]
                model, ininds, uniqueinput, inspkcnt, counts, outinds, combout, comboutspkcnt, comboutcnt, losses, entropies =
                    trainondata(data_split[i], maxiter, winsz, batchsize, arraycast, lr; model=loadmodel, startiter=startiter)
                savedata["$i"] = Dict("ininds"=>ininds, "outinds"=>outinds, "input"=>uniqueinput, "output"=>combout, "net"=>model, "inspikecnt"=>inspkcnt, "incount"=>counts, "outspikecnt"=>comboutspkcnt, "outcount"=>comboutcnt, "loss"=>losses, "entropy"=>entropies)
                MAT.matwrite(joinpath(cdmdir, "input", "$(DrWatson.savename(params))_$(i).mat"),
                             Dict("inspikecnt"=>inspkcnt, "counts"=>counts))
                MAT.matwrite(joinpath(cdmdir, "complete", "$(DrWatson.savename(params))_$(i).mat"),
                             Dict("cells"=>model.n, "spike_counts"=>comboutspkcnt, "counts"=>comboutcnt))
            end
            if !isfile(nullloc)
                println("processing $nullloc")
                #loadmodel = DrWatson.wload(joinpath(basedir, "null", "$(DrWatson.savename(loadparams)).jld2"))["1"]["net"]
                nulldata = generatebernoulli(data_split[i])
                model, ininds, uniqueinput, inspkcnt, counts, outinds, combout, comboutspkcnt, comboutcnt, losses, entropies =
                    trainondata(nulldata, maxiter, winsz, batchsize, arraycast, lr)
                savenull["$i"] = Dict("ininds"=>ininds, "outinds"=>outinds, "input"=>uniqueinput, "output"=>combout, "net"=>model, "inspikecnt"=>inspkcnt, "incount"=>counts, "outspikecnt"=>comboutspkcnt, "outcount"=>comboutcnt, "loss"=>losses, "entropy"=>entropies)
                MAT.matwrite(joinpath(cdmdir, "null", "$(DrWatson.savename(params))_$(i).mat"),
                             Dict("cells"=>model.n, "spike_counts"=>comboutspkcnt, "counts"=>comboutcnt))
            end
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    binsz, startiter, maxiter, numsplit, dev, batchsize = parse.(Int, ARGS)
    params = Dict("binsz"=>binsz, "maxiter"=>maxiter)
    st = 25
    en = 25
    step = 5
    winszs = [i for i in st:step:en]
    println(winszs)

    # For Blanche's data
    #path = DrWatson.datadir("exp_raw", "pvc3", "crcns_pvc3_cat_recordings", "spont_activity", "spike_data_area18")
    #extract_fn = extract_bin_spikes_blanche
    #basedir = DrWatson.datadir("exp_pro", "matrix", "blanche", "full")
    #lr = 0.1
    # For Joost's data
    path = DrWatson.datadir("exp_raw", "joost_data", "Long_recordings-stability_MaxEnt_and_CFP", "long_1_spontaneous_activity.jld2")
    extract_fn = extract_bin_spikes_joost
    basedir = DrWatson.datadir("exp_pro", "matrix_new", "joost_long", "full")
    lr = 0.01

    CUDA.device!(dev)
    arraycast = CUDA.cu

    println("starting")
    continueexperiment(binsz, winszs, startiter, maxiter, path, batchsize, arraycast, params, basedir, numsplit, extract_fn, lr)
end
