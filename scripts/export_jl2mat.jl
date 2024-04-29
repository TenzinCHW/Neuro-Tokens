include("calculate_entropy.jl")
import MAT


"""Gets all pattern counts and spike counts for each unique pattern and pattern dimension"""
function cntspikecell(dirpath, inout)
    out = Dict{String, Dict{String, Any}}()
    for f in readdir(inpdir)
        fn = split(f, ".")[1]
        data = DrWatson.wload(joinpath(inpdir, f))
        data = data["1"][inout] # only have 1 split
        cell = size(data)[2] |> Float32
        unique_patt, cnts = unique_counts(data, 1)
        cnts = reshape(cnts, (length(cnts), 1))
        spk = sum(unique_patt, dims=2) .|> Float32
        o = Dict("cells"=>cell, "counts"=>cnts, "spike_counts"=>spk)
        out[f] = o
    end
    out
end


function writemat(vals, basedir, subdir)
    for (f, v) in vals
        fn = split(f, ".")[1]
        fp = joinpath(basedir, "cdmentropy", subdir, "$(fn).mat")
        println(fp)
        println(typeof(v))
        DrWatson.wsave("what.jld2", v)
        MAT.matwrite(fp, v)
    end
end


maindir = ARGS[1]
basedir = DrWatson.datadir("exp_pro", "matrix", "$maindir")
subdirs = ["complete", "null"]

# Extract input counts (assuming same for all)
inpdir = joinpath(basedir, subdirs[1])
inpvals = cntspikecell(inpdir, "input")
writemat(inpvals, basedir, "input")


# Extract output counts
for subdir in subdirs
    outdir = joinpath(basedir, subdir)
    outvals = cntspikecell(outdir, "output")
    writemat(outvals, basedir, subdir)
end

