include("extract_blanche.jl")


function groupspikes(timings, ids)
    spikes = [Int64[] for _ in 1:60]  # we know that the setup uses 60 electrodes but not all are always present in data
    timings = (timings .+ 1) .|> Int64
    for i in 1:length(timings)
        push!(spikes[ids[i] + 1], timings[i])  # ids always range from 0 to 59 (in UInt8)
    end
    spikes
end


function extract_bin_spikes_joost(fpath::String, binsz)
    data = DrWatson.wload(fpath)
    timings = data["Ts"]
    #println(max(timings...) / 1000_000 / 60)
    ids = data["Cs"]
    #println(length(unique(ids)))
    spikes_times = groupspikes(timings, ids)
    bin_spikes(spikes_times, binsz)
end


if abspath(PROGRAM_FILE) == @__FILE__
    basedir = "../data/exp_raw/joost_data/Long_recordings-stability_MaxEnt_and_CFP"
    println(readdir(basedir))
    for f in readdir(basedir)
        println(f)
        spk = extract_bin_spikes_joost(joinpath(basedir, f), 10000)
        println("binned spike size: $(size(spk))")
    end
end

