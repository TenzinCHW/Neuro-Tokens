include("extract_joost.jl")
include("calculate_entropy.jl")
import MAT, MEFK, CUDA


# only the split of the old version doesn't have loss. Have to extract the data and then the indices
function from_v1(basedir, dstdir)
    # TODO extract data, get indices and counts and spike counts
    for dir in readdir(basedir)
        dp = joinpath(basedir, dir)
        if dir == "cdmentropy" || isfile(dp)
            continue
        end
        Threads.@threads for f in readdir(dp)
            inpath = joinpath(dp, f)
            #if !isfile(inpath)
            #    continue
            #end
            params = DrWatson.parse_savename(f)[2]
            if params["binsz"] != 5000
                continue
            end
            println(inpath)
            data = DrWatson.wload(inpath)
            savedata = Dict()
            for i in keys(data)
                dt = data[i]
                input, output, net, loss = dt["input"], dt["output"], dt["net"], dt["loss"]
                uniqueinput, incount = uniquecounts(input, 1)
                uniqueoutput, outcount = uniquecounts(output, 1)
                ininds = uniqueinds(uniqueinput, input)
                outinds = uniqueinds(uniqueoutput, output)
                inspikecnt = sum(uniqueinput, dims=2)
                outspikecnt = sum(uniqueoutput, dims=2)
                savedata[i] = Dict("ininds"=>ininds, "outinds"=>outinds, "input"=>uniqueinput, "output"=>uniqueoutput,"net"=>net, "inspikecnt"=>inspikecnt, "incount"=>incount, "outspikecnt"=>outspikecnt,  "outcount"=>outcount, "loss"=>loss)
            end
            outpath = joinpath(dstdir, dir, f)
            DrWatson.wsave(outpath, savedata)
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    maindir = ARGS[1]
    outdir = ARGS[2]
    basedir = DrWatson.datadir("exp_pro", maindir)
    dstdir = DrWatson.datadir("exp_pro", outdir)
    from_v1(basedir, dstdir)
end
