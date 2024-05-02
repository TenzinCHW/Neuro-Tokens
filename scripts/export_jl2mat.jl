include("extract_joost.jl")
include("calculate_entropy.jl")
import MAT, MEFK, CUDA


#function cntspikecell(dirpath, inout)
#    out = Dict{String, Dict{String, Any}}()
#    for f in readdir(dirpath)
#        fn = split(f, ".")[1]
#        data = DrWatson.wload(joinpath(dirpath, f))
#        data = data["1"][inout] # only have 1 split
#        unique_patt = unique(data, dims=1)
#        if size(data)[1] == size(unique_patt)[1]
#            cnts = ones(size(unique_patt)[1])
#        else
#            unique_patt, cnts = uniquecounts(data, 1)
#        end
#        cell = size(data)[2] |> Float32
#        cnts = reshape(cnts, (length(cnts), 1))
#        spk = sum(unique_patt, dims=2) .|> Float32
#        o = Dict("cells"=>cell, "counts"=>cnts, "spike_counts"=>spk)
#        out[f] = o
#    end
#    out
#end


function writemat(vals, basedir, subdir)
    for (f, v) in vals
        fn = split(f, ".")[1]
        fp = joinpath(basedir, "cdmentropy", subdir, "$(fn).mat")
        println(fp)
        println(typeof(v))
        MAT.matwrite(fp, v)
    end
end


"""Gets all pattern counts and spike counts for each unique pattern and pattern dimension"""
function cntspikecell(dirpath, datapath, extractfn)
    inp = Dict()
    out = Dict()
    null = Dict()
    for f in readdir(joinpath(dirpath, "complete"))
        modeldata = DrWatson.wload(joinpath(dirpath, "complete", f))
        # extract model
        model = MEFK.MEF2T(modeldata["1"]["net"], CUDA.cu)
        # get win/bin params
        params = DrWatson.parse_savename(f)[2]
        winsz = params["winsz"]
        binsz = params["binsz"]
        # window data
        spikedata = extractfn(datapath, binsz)
        spikedata = trim_recording(spikedata, 0.15)#094)
        data_split = recordingsplittrain(spikedata, 1)
        windowedspikes, counts = window(data_split[1], winsz)
        spkcnt = sum(windowedspikes, dims=2)
        # get outputs for data
        output = convergedynamics(model, windowedspikes |> CUDA.cu) |> Array
        # combine output count
        output, outcounts = combine_counts(output, counts)
        outspkcnt = sum(output, dims=2)

        nullmodeldata = DrWatson.wload(joinpath(dirpath, "null", f))
        nullout = nullmodeldata["1"]["output"]
        nullpatt, nullcnts = uniquecounts(nullout, 1)
        nullspkcnt = sum(nullpatt, dims=2)

        o = Dict("cells"=>model.n, "counts"=>outcounts, "spike_counts"=>outspkcnt)
        i = Dict("cells"=>model.n, "counts"=>counts, "spike_counts"=>spkcnt)
        nl = Dict("cells"=>model.n, "counts"=>nullcnts, "spike_counts"=>nullspkcnt)
        out[f] = o
        inp[f] = i
        null[f] = nl
    end
    inp, out, null
end


if abspath(PROGRAM_FILE) == @__FILE__
    maindir = ARGS[1]
    basedir = DrWatson.datadir("exp_pro", "$maindir")
    datapath = DrWatson.datadir("exp_raw", "pvc3", "crcns_pvc3_cat_recordings", "spont_activity", "spike_data_area18")
    extractfn = extract_bin_spikes_blanche
    #datapath = DrWatson.datadir("exp_raw", "joost_data", "Long_recordings-stability_MaxEnt_and_CFP", "long_1_spontaneous_activity.jld2")
    #extractfn = extract_bin_spikes_joost
    #subdirs = ["complete", "null"]

    # Extract input counts (assuming same for all)
    #inpdir = joinpath(basedir, subdirs[1])
    #inpvals = cntspikecell(inpdir, "input")
    #writemat(inpvals, basedir, "input")


    # Extract output counts
    #for subdir in subdirs
    #    outdir = joinpath(basedir, subdir)
    #    outvals = cntspikecell(outdir, "output")
    #    writemat(outvals, basedir, subdir)
    #end
    
    inp, out, null = cntspikecell(basedir, datapath, extractfn)
    subdirs = Dict("input"=>inp, "complete"=>out, "null"=>null)
    for (dn, v) in subdirs
        writemat(v, basedir, dn)
    end
end
